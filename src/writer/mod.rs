//! How a collection is built. This is the only code here that writes array
//! data.
//!
//! # How a container is assembled
//!
//! A container holds one segment per **variable**, not one per dataset. A
//! segment is a complete `array-format` file holding one array name across the
//! whole collection. Inside it, each dataset's array is stored under the
//! dataset's own name:
//!
//! ```text
//! add_dataset("jan") -> writer[temperature]    temperature/jan
//!                       writer[precipitation]  precipitation/jan
//! add_dataset("feb") -> writer[temperature]    temperature/feb
//!                       writer[precipitation]  precipitation/feb
//! AtlasWriter::finish -> finish each, copy in, footer, trailer, done
//! ```
//!
//! That is what makes a scan of one variable cheap. `array-format` packs
//! neighbouring chunks into one block, so a block holds `temperature` for a
//! run of datasets. One fetch then serves them all, and the block holds one
//! dtype, which compresses far better than a mix.
//!
//! # Staging
//!
//! Each variable builds in an `array-format` writer. Nothing reaches the
//! container until [`AtlasWriter::finish`], because a variable's segment is
//! complete only when every dataset has contributed.
//!
//! Memory stays bounded. The writer packs each chunk into a compressed block
//! as it arrives, and spills every full block to a temporary file. It keeps
//! one open block per variable in memory, and the chunk table beside it.
//!
//! At finish, each writer lands its file in a scratch directory, and the
//! container takes a copy. Local disk therefore holds each variable twice for
//! a moment, and the whole collection once.
//!
//! # Order
//!
//! A [`DatasetWriter`] takes the writer's lock for each define and each write,
//! because the variable writers are shared. Ordinals do not depend on that
//! order. Each dataset carries the number of the [`AtlasWriter::add_dataset`]
//! call that opened it, and the footer sorts on that at the end. Stage a
//! directory twice and every dataset lands at the same ordinal.
//!
//! # Failure
//!
//! Nothing is readable until [`AtlasWriter::finish`] writes the trailer. Drop
//! a [`DatasetWriter`] before you finish it, and that dataset never enters the
//! container. Drop the [`AtlasWriter`], and no valid collection appears.

use std::collections::HashSet;
use std::sync::Arc;

use array_format::{
    ArrayElement, ArrayWriter, DType, FillValue, Lz4Codec, NoCompression, ZstdCodec,
    WriterConfig as SegmentConfig,
};
use bytes::Bytes;
use indexmap::IndexMap;
use object_store::ObjectStore;
use object_store::buffered::BufWriter;
use object_store::path::Path as OsPath;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::Mutex;
use tracing::debug;

use smol_str::SmolStr;

use crate::config::{Codec, WriterConfig};
use crate::format::footer::{CollectionFooter, Interner, VariableEntry};
use crate::format::{self, DATA_FILE, DATASET_ATTRS_VARIABLE, child};
use crate::schema::Attr;
use crate::{Error, Result, validate_name};

/// Bytes the copy moves per turn, from a finished segment into the container.
const COPY_CHUNK_SIZE: usize = 8 * 1024 * 1024;

/// Name of a finished segment inside its scratch directory.
const SEGMENT_FILE: &str = "data.af";

/// The half of a writer the datasets share. The output stream, the variable
/// writers, and the footer under construction.
struct WriterState {
    out: Option<BufWriter>,
    prefix: OsPath,
    config: WriterConfig,
    /// Bytes written so far. This is also the offset of the next segment.
    offset: u64,
    interner: Interner,
    /// One writer per variable, created when a dataset first declares the
    /// array. The insertion order fixes the segment order.
    variables: IndexMap<String, ArrayWriter>,
    /// Each committed dataset: the number of the `add_dataset` call that
    /// opened it, its name, and the schema it interned. Sorting by that number
    /// at the end restores call order, which is what fixes the ordinals.
    entries: Vec<(u64, SmolStr, u32)>,
    /// Every name handed to `add_dataset`. It refuses a repeat, and the hash
    /// keeps that check flat as the collection grows.
    names: HashSet<String>,
    /// Where each finished segment lands before the container takes it.
    scratch: tempfile::TempDir,
    dataset_seq: u64,
}

impl WriterState {
    /// The writer for `array`, created on first use.
    fn variable(&mut self, array: &str) -> &mut ArrayWriter {
        if !self.variables.contains_key(array) {
            self.variables
                .insert(array.to_string(), new_variable_writer(&self.config));
        }
        self.variables
            .get_mut(array)
            .expect("inserted above when absent")
    }

    /// Appends a finished segment file to the container, and returns its byte
    /// range. It streams in [`COPY_CHUNK_SIZE`] pieces, so a large variable
    /// never needs to fit in memory.
    async fn append_segment(&mut self, file: &std::path::Path) -> Result<(u64, u64)> {
        let out = self.out.as_mut().ok_or(Error::WriterFinished)?;
        let mut src = tokio::fs::File::open(file).await?;
        let start = self.offset;
        let mut written = 0u64;
        let mut buf = vec![0u8; COPY_CHUNK_SIZE];
        loop {
            let n = src.read(&mut buf).await?;
            if n == 0 {
                break;
            }
            out.put(Bytes::copy_from_slice(&buf[..n])).await?;
            written += n as u64;
        }
        if written == 0 {
            return Err(Error::Internal(format!(
                "finished segment {} is empty",
                file.display()
            )));
        }
        self.offset += written;
        Ok((start, written))
    }
}

/// Builds one collection, then finishes. A collection never reopens for more
/// data. Rewrite it instead.
///
/// ```no_run
/// use atlas::{AtlasWriter, WriterConfig};
/// use ndarray::Array2;
///
/// # async fn run() -> atlas::Result<()> {
/// let w = AtlasWriter::create_path("/tmp/my_collection", WriterConfig::default()).await?;
/// {
///     let mut ds = w.add_dataset("jan_2024").await?;
///     ds.define_array::<f32>("temperature", vec!["lat".into(), "lon".into()], vec![4, 8], None, None)
///         .await?;
///     let data = Array2::<f32>::from_elem([4, 8], 20.0).into_dyn();
///     ds.write_array("temperature", vec![0, 0], data.view()).await?;
///     ds.finish().await?;
/// }
/// w.finish().await?;
/// # Ok(())
/// # }
/// ```
pub struct AtlasWriter {
    state: Arc<Mutex<WriterState>>,
}

impl AtlasWriter {
    /// Starts a collection under `prefix` in `store`.
    ///
    /// The write starts at once. Nothing at `prefix` reads as a collection
    /// until [`finish`](Self::finish).
    pub async fn create(
        store: Arc<dyn ObjectStore>,
        prefix: OsPath,
        config: WriterConfig,
    ) -> Result<Self> {
        let mut out = BufWriter::new(store, child(&prefix, DATA_FILE));
        out.put(Bytes::copy_from_slice(&format::encode_header()))
            .await?;
        Ok(Self {
            state: Arc::new(Mutex::new(WriterState {
                out: Some(out),
                prefix,
                config,
                offset: format::HEADER_SIZE,
                interner: Interner::default(),
                variables: IndexMap::new(),
                entries: Vec::new(),
                names: HashSet::new(),
                scratch: tempfile::tempdir()?,
                dataset_seq: 0,
            })),
        })
    }

    /// Starts a collection in a local directory. Creates the directory if it
    /// is absent.
    pub async fn create_path(
        path: impl AsRef<std::path::Path>,
        config: WriterConfig,
    ) -> Result<Self> {
        let path = path.as_ref();
        std::fs::create_dir_all(path)?;
        let store = object_store::local::LocalFileSystem::new_with_prefix(path)?;
        Self::create(Arc::new(store), OsPath::default(), config).await
    }

    /// Begins a dataset. Call [`DatasetWriter::finish`] to commit it.
    ///
    /// A name is unique within a collection. A repeat returns
    /// [`Error::DatasetAlreadyExists`]. Every variable segment keys on this
    /// name, so a repeat would also collide inside the segments.
    ///
    /// The name holds from this call, not from [`DatasetWriter::finish`]. An
    /// aborted dataset therefore keeps its name reserved for the life of the
    /// writer.
    pub async fn add_dataset(&self, name: &str) -> Result<DatasetWriter> {
        validate_name(name)?;
        let mut state = self.state.lock().await;
        if state.out.is_none() {
            return Err(Error::WriterFinished);
        }
        if !state.names.insert(name.to_string()) {
            return Err(Error::DatasetAlreadyExists(name.to_string()));
        }
        state.dataset_seq += 1;
        let seq = state.dataset_seq;
        drop(state);
        Ok(DatasetWriter {
            state: Arc::clone(&self.state),
            name: name.to_string(),
            seq,
            arrays: IndexMap::new(),
            global_attrs: Vec::new(),
            array_attrs: IndexMap::new(),
        })
    }

    /// How many datasets the writer has committed so far.
    pub async fn dataset_count(&self) -> usize {
        self.state.lock().await.entries.len()
    }

    /// Finishes every variable, writes the segments, then the footer and the
    /// trailer, and completes the upload.
    ///
    /// After this the collection is readable, and fixed for good. A
    /// [`DatasetWriter`] that is still open fails with
    /// [`Error::WriterFinished`].
    ///
    /// Ordinals follow the order of the [`add_dataset`](Self::add_dataset)
    /// calls, not the order the datasets finished in.
    pub async fn finish(self) -> Result<()> {
        let mut state = self.state.lock().await;
        if state.out.is_none() {
            return Err(Error::WriterFinished);
        }
        // The reserved segment's name is not an array name, so no schema
        // interned it. The footer still stores it as a pool index.
        if state.variables.contains_key(DATASET_ATTRS_VARIABLE) {
            state.interner.intern_string(DATASET_ATTRS_VARIABLE);
        }
        let (string_pool, dtype_pool, schema_pool) =
            std::mem::take(&mut state.interner).into_pools();

        // Back to call order, which fixes the ordinals.
        let mut tagged = std::mem::take(&mut state.entries);
        tagged.sort_by_key(|(seq, _, _)| *seq);
        let datasets: IndexMap<SmolStr, u32> = tagged
            .into_iter()
            .map(|(_, name, schema)| (name, schema))
            .collect();

        let staged = std::mem::take(&mut state.variables);
        let mut variables = Vec::with_capacity(staged.len());
        for (index, (array, writer)) in staged.into_iter().enumerate() {
            let name = string_pool
                .iter()
                .position(|s| *s == array)
                .ok_or_else(|| Error::Internal(format!("variable '{array}' was never interned")))?
                as u32;

            // The writer lands its file on local disk first. It writes a whole
            // object, and a segment is a byte range of one.
            let dir = state.scratch.path().join(format!("v{index}"));
            std::fs::create_dir_all(&dir)?;
            let store: Arc<dyn ObjectStore> =
                Arc::new(object_store::local::LocalFileSystem::new_with_prefix(&dir)?);
            let finished = writer.finish(store, OsPath::from(SEGMENT_FILE)).await?;
            drop(finished);

            let (seg_offset, seg_len) = state.append_segment(&dir.join(SEGMENT_FILE)).await?;
            debug!(variable = %array, seg_offset, seg_len, "appended variable segment");
            variables.push(VariableEntry {
                name,
                seg_offset,
                seg_len,
            });
            // The scratch copy is large. Reclaim it now, not at the end.
            let _ = std::fs::remove_dir_all(&dir);
        }

        let footer = CollectionFooter {
            version: format::FORMAT_VERSION,
            codec: state.config.codec,
            created_unix_ms: chrono::Utc::now().timestamp_millis(),
            string_pool,
            dtype_pool,
            schema_pool,
            variables,
            datasets,
        };
        let bytes = footer.encode()?;
        let footer_size = bytes.len() as u64;
        debug!(
            datasets = footer.datasets.len(),
            variables = footer.variables.len(),
            footer_size,
            "writing collection footer"
        );
        let mut out = state.out.take().expect("checked above");
        out.put(Bytes::from(bytes)).await?;
        out.put(Bytes::copy_from_slice(&format::encode_trailer(footer_size)))
            .await?;
        out.shutdown().await?;
        Ok(())
    }
}

impl Drop for WriterState {
    fn drop(&mut self) {
        if self.out.is_some() {
            // The trailer never landed, so nothing at the target opens as a
            // collection. The backend decides whether a partial object stays.
            // Report that instead of a silent exit.
            tracing::warn!(
                prefix = %self.prefix,
                "atlas writer dropped without finish; no collection was written"
            );
        }
    }
}

/// Builds one dataset inside a collection.
///
/// Declare an array with [`define_array`](Self::define_array). Fill it with
/// [`write_array`](Self::write_array), in any order and in any number of
/// slabs. Each call writes into that variable's shared writer, under this
/// dataset's name. Nothing reaches the container until
/// [`AtlasWriter::finish`].
pub struct DatasetWriter {
    state: Arc<Mutex<WriterState>>,
    name: String,
    /// Which `add_dataset` call opened this one. It fixes the ordinal, so a
    /// dataset that finishes out of turn still lands where it was asked for.
    seq: u64,
    /// Array name to element type, in definition order. Shape and chunking go
    /// straight into the variable's segment, which is where they live.
    arrays: IndexMap<String, DType>,
    global_attrs: Vec<(String, Attr)>,
    array_attrs: IndexMap<String, Vec<(String, Attr)>>,
}

impl DatasetWriter {
    /// The dataset's name. Every variable segment stores this dataset's array
    /// under it.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Names of the arrays this dataset declares, in definition order.
    pub fn list_arrays(&self) -> Vec<String> {
        self.arrays.keys().cloned().collect()
    }

    /// The element type of one declared array.
    pub fn array_dtype(&self, array: &str) -> Option<&DType> {
        self.arrays.get(array)
    }

    /// Declares an array.
    ///
    /// `chunk_shape` defaults to `shape`, which stores the array as one chunk.
    /// A read returns `fill_value` for every element nobody writes.
    ///
    /// The shape, the chunking, the dimension names, and the fill value go
    /// into the variable's segment. The footer records only the name and the
    /// element type.
    pub async fn define_array<T: ArrayElement>(
        &mut self,
        array: &str,
        dimension_names: Vec<String>,
        shape: Vec<usize>,
        chunk_shape: Option<Vec<usize>>,
        fill_value: Option<FillValue>,
    ) -> Result<()> {
        validate_name(array)?;
        if self.arrays.contains_key(array) {
            return Err(Error::ArrayAlreadyExists(array.to_string()));
        }
        let chunk = chunk_shape.unwrap_or_else(|| shape.clone());
        let mut state = self.state.lock().await;
        if state.out.is_none() {
            return Err(Error::WriterFinished);
        }
        // The segment keys on the dataset name, so one file holds this array
        // for every dataset in the collection.
        state.variable(array).define_array::<T>(
            self.name.clone(),
            dimension_names,
            shape,
            Some(chunk),
            fill_value,
        )?;
        drop(state);
        self.arrays.insert(array.to_string(), T::DTYPE);
        Ok(())
    }

    /// Writes `data` into `array` with its origin at `start`.
    ///
    /// The region can span chunks, and needs no chunk alignment. `T` must
    /// match the array's declared type.
    pub async fn write_array<T: ArrayElement>(
        &mut self,
        array: &str,
        start: Vec<usize>,
        data: ndarray::ArrayView<'_, T, ndarray::IxDyn>,
    ) -> Result<()> {
        if !self.arrays.contains_key(array) {
            return Err(Error::ArrayNotFound(array.to_string()));
        }
        let mut state = self.state.lock().await;
        if state.out.is_none() {
            return Err(Error::WriterFinished);
        }
        state.variable(array).write_array(&self.name, start, data)?;
        Ok(())
    }

    /// Attaches a dataset-level attribute. A repeated key replaces the value
    /// before it.
    pub fn set_attribute(&mut self, key: &str, value: Attr) {
        upsert(&mut self.global_attrs, key, value);
    }

    /// Attaches an attribute to one array. Define the array first.
    pub fn set_array_attribute(&mut self, array: &str, key: &str, value: Attr) -> Result<()> {
        if !self.arrays.contains_key(array) {
            return Err(Error::ArrayNotFound(array.to_string()));
        }
        upsert(
            self.array_attrs.entry(array.to_string()).or_default(),
            key,
            value,
        );
        Ok(())
    }

    /// Commits the dataset into the collection.
    ///
    /// The array data already sits in the variable writers. This writes the
    /// attribute values into those same writers, and records what the dataset
    /// declares in the footer. [`AtlasWriter::finish`] does the rest.
    ///
    /// A dataset-level value goes on this dataset's array in the reserved
    /// `_datasets` segment. A per-array value goes on the array's own entry in
    /// that variable's segment.
    pub async fn finish(mut self) -> Result<()> {
        let arrays = std::mem::take(&mut self.arrays);
        let global_attrs = std::mem::take(&mut self.global_attrs);
        // A per-array attribute carries the array's position, not its name,
        // because datasets share the schema.
        let mut array_attrs: Vec<(u32, Vec<(String, Attr)>)> =
            std::mem::take(&mut self.array_attrs)
                .into_iter()
                .filter_map(|(array, attrs)| Some((arrays.get_index_of(&array)? as u32, attrs)))
                .collect();
        array_attrs.sort_by_key(|(position, _)| *position);

        let mut state = self.state.lock().await;
        if state.out.is_none() {
            return Err(Error::WriterFinished);
        }

        // Dataset-level values need an array to sit on, so the reserved
        // segment gets a rank-0 one per dataset. It appears only when some
        // dataset carries a global attribute.
        if !global_attrs.is_empty() {
            let marker = state.variable(DATASET_ATTRS_VARIABLE);
            marker.define_array::<u8>(self.name.clone(), Vec::new(), Vec::new(), None, None)?;
            for (key, value) in &global_attrs {
                marker.set_attribute(&self.name, key, value.clone().into_stored())?;
            }
        }
        for (position, keyed) in &array_attrs {
            let (array, _) = arrays
                .get_index(*position as usize)
                .expect("the position came from this map");
            let writer = state.variable(array);
            for (key, value) in keyed {
                writer.set_attribute(&self.name, key, value.clone().into_stored())?;
            }
        }

        let schema = state
            .interner
            .intern_schema(&arrays, &global_attrs, &array_attrs);
        debug!(dataset = %self.name, arrays = arrays.len(), "committed dataset");
        state
            .entries
            .push((self.seq, std::mem::take(&mut self.name).into(), schema));
        Ok(())
    }
}

/// Creates one variable's writer.
fn new_variable_writer(config: &WriterConfig) -> ArrayWriter {
    let target = config.block_target_size;
    match config.codec {
        Codec::Zstd => ArrayWriter::new(segment_config(ZstdCodec::default(), target)),
        Codec::Lz4 => ArrayWriter::new(segment_config(Lz4Codec, target)),
        Codec::Uncompressed => ArrayWriter::new(segment_config(NoCompression, target)),
    }
}

fn segment_config<C: array_format::CompressionCodec>(
    codec: C,
    block_target_size: usize,
) -> SegmentConfig<C> {
    SegmentConfig {
        codec,
        block_target_size,
    }
}

fn upsert(list: &mut Vec<(String, Attr)>, key: &str, value: Attr) {
    match list.iter_mut().find(|(k, _)| k == key) {
        Some(slot) => slot.1 = value,
        None => list.push((key.to_string(), value)),
    }
}
