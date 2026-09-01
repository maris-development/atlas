//! How a collection is built. This is the only code here that writes array
//! data.
//!
//! # How a container is assembled
//!
//! [`AtlasWriter`] streams one object. Each dataset stages as a complete
//! `array-format` file in a local scratch directory. The writer then copies
//! that file into the stream, byte for byte:
//!
//! ```text
//! add_dataset("jan")  -> scratch/1/data.af   (define, write, define, write, ...)
//!   finish()          -> flush, compact, copy  -> container[8 .. 4_100]
//! add_dataset("feb")  -> scratch/2/data.af
//!   finish()          -> flush, compact, copy  -> container[4_100 .. 9_002]
//! AtlasWriter::finish -> footer, trailer, done
//! ```
//!
//! Local staging bounds the memory, whatever the dataset size. `array-format`
//! spills each compressed chunk to a temporary file on arrival. The copy into
//! the container streams in fixed-size pieces.
//!
//! The order of `flush` then `compact` matters. `flush` commits the buffered
//! writes into a sidecar layer. `compact` merges every layer into one base
//! file. A `compact` without a `flush` first leaves the buffered writes behind.
//! The result is one self-contained file, which is what a segment must be.
//!
//! # Staging several datasets at once
//!
//! A [`DatasetWriter`] touches the container only in
//! [`DatasetWriter::finish`], which takes the writer's lock for the whole
//! append. The costly part, the flush and the compact, runs outside that lock.
//! Several datasets can therefore stage at once, which is what makes a
//! many-file ingest scale. Their bytes land in finish order, and never
//! interleave.
//!
//! Ordinals do not follow that order. Each dataset carries the number of the
//! [`AtlasWriter::add_dataset`] call that opened it, and the footer sorts on
//! that at the end. Stage a directory twice and every dataset lands at the
//! same ordinal, however many threads did the work.
//!
//! # Failure
//!
//! Nothing is readable until [`AtlasWriter::finish`] writes the trailer. Drop
//! a [`DatasetWriter`] before you finish it, and that dataset never enters the
//! container. Drop the [`AtlasWriter`], and no valid collection appears.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use array_format::{
    ArrayElement, ArrayFile, FileConfig, FillValue, Lz4Codec, NoCompression, ZstdCodec,
};
use bytes::Bytes;
use indexmap::IndexMap;
use object_store::ObjectStore;
use object_store::buffered::BufWriter;
use object_store::path::Path as OsPath;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::Mutex;
use tracing::debug;

use crate::config::{Codec, WriterConfig};
use crate::format::footer::{ArrayStatsS, AttrS, CollectionFooter, DatasetEntry, Interner};
use crate::format::{self, DATA_FILE, child};
use crate::schema::{ArraySchema, Attr, DatasetSchema};
use crate::{Error, Result, validate_name};

/// Bytes the copy moves per turn, from a staged segment into the container.
const COPY_CHUNK_SIZE: usize = 8 * 1024 * 1024;

/// The half of a writer the datasets share. The output stream, the footer
/// under construction, and the scratch area.
struct WriterState {
    out: Option<BufWriter>,
    prefix: OsPath,
    config: WriterConfig,
    /// Bytes written so far. This is also the offset of the next segment.
    offset: u64,
    interner: Interner,
    /// Each committed dataset, tagged with the `add_dataset` call that opened
    /// it. A dataset appends as soon as it finishes, so this arrives in
    /// completion order. Sorting by the tag at the end restores call order,
    /// which is what fixes the ordinals.
    entries: Vec<(u64, DatasetEntry)>,
    /// Every name handed to `add_dataset`. It refuses a repeat, and the hash
    /// keeps that check flat as the collection grows.
    names: HashSet<String>,
    scratch: tempfile::TempDir,
    dataset_seq: u64,
}

impl WriterState {
    /// Appends a staged segment file to the container, and returns its byte
    /// range. It streams in [`COPY_CHUNK_SIZE`] pieces, so a large dataset
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
                "staged segment {} is empty",
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
    /// [`Error::DatasetAlreadyExists`]. The writer keeps the names in a hash
    /// set, so the check costs the same for ten datasets and a million.
    ///
    /// The name holds from this call, not from
    /// [`DatasetWriter::finish`]. An aborted dataset therefore keeps its name
    /// reserved for the life of the writer.
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
        let dir = state.scratch.path().join(seq.to_string());
        std::fs::create_dir_all(&dir)?;
        let config = state.config.clone();
        drop(state);
        Ok(DatasetWriter {
            state: Arc::clone(&self.state),
            name: name.to_string(),
            seq,
            dir,
            config,
            file: None,
            arrays: IndexMap::new(),
            global_attrs: Vec::new(),
            array_attrs: IndexMap::new(),
        })
    }

    /// How many datasets the writer has committed so far.
    pub async fn dataset_count(&self) -> usize {
        self.state.lock().await.entries.len()
    }

    /// Writes the footer and trailer, and completes the upload.
    ///
    /// After this the collection is readable, and fixed for good. A
    /// [`DatasetWriter`] that is still open fails with
    /// [`Error::WriterFinished`].
    ///
    /// Ordinals follow the order of the [`add_dataset`](Self::add_dataset)
    /// calls, not the order the datasets finished in. Stage a directory twice
    /// and every dataset lands at the same ordinal, however many threads did
    /// the work.
    pub async fn finish(self) -> Result<()> {
        let mut state = self.state.lock().await;
        let mut out = state.out.take().ok_or(Error::WriterFinished)?;
        let (schema_pool, attr_key_pool) = std::mem::take(&mut state.interner).into_pools();
        // Back to call order. Each entry carries its own byte range, so the
        // segments need no matching order on disk.
        let mut tagged = std::mem::take(&mut state.entries);
        tagged.sort_by_key(|(seq, _)| *seq);
        let datasets: Vec<DatasetEntry> = tagged.into_iter().map(|(_, e)| e).collect();
        let footer = CollectionFooter {
            version: format::FORMAT_VERSION,
            segment_format: format::SEGMENT_FORMAT,
            codec: state.config.codec,
            created_unix_ms: chrono::Utc::now().timestamp_millis(),
            schema_pool,
            attr_key_pool,
            datasets,
        };
        let bytes = footer.encode()?;
        let footer_size = bytes.len() as u64;
        debug!(
            datasets = footer.datasets.len(),
            footer_size, "writing collection footer"
        );
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
/// slabs. Nothing reaches the container until [`finish`](Self::finish).
pub struct DatasetWriter {
    state: Arc<Mutex<WriterState>>,
    name: String,
    /// Which `add_dataset` call opened this one. It fixes the ordinal, so a
    /// dataset that finishes out of turn still lands where it was asked for.
    seq: u64,
    dir: std::path::PathBuf,
    config: WriterConfig,
    file: Option<ArrayFile>,
    arrays: IndexMap<String, ArraySchema>,
    global_attrs: Vec<(String, Attr)>,
    array_attrs: IndexMap<String, Vec<(String, Attr)>>,
}

impl DatasetWriter {
    /// The dataset's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Names of the arrays this dataset declares, in definition order.
    pub fn list_arrays(&self) -> Vec<String> {
        self.arrays.keys().cloned().collect()
    }

    /// The schema of one declared array.
    pub fn array_meta(&self, array: &str) -> Option<&ArraySchema> {
        self.arrays.get(array)
    }

    /// Opens the staging file on first use. A dataset somebody declares and
    /// then abandons therefore costs no I/O.
    async fn staging(&mut self) -> Result<&mut ArrayFile> {
        if self.file.is_none() {
            self.file = Some(create_staging_file(&self.dir, &self.config).await?);
        }
        Ok(self.file.as_mut().expect("just set"))
    }

    /// Declares an array.
    ///
    /// `chunk_shape` defaults to `shape`, which stores the array as one chunk.
    /// A read returns `fill_value` for every element nobody writes.
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
        let ndim = shape.len();
        let chunk = chunk_shape.unwrap_or_else(|| shape.clone());
        // array-format substitutes dim0, dim1, ... when the count is wrong.
        // Record what it stores, not what the caller asked for.
        let dims = if dimension_names.len() == ndim {
            dimension_names.clone()
        } else {
            (0..ndim).map(|i| format!("dim{i}")).collect()
        };
        let schema = ArraySchema {
            dtype: T::DTYPE,
            shape: shape.clone(),
            chunk_shape: chunk.clone(),
            dimension_names: dims,
            fill_value: fill_value.clone().map(Into::into),
        };
        self.staging().await?.define_array::<T>(
            array.to_string(),
            dimension_names,
            shape,
            Some(chunk),
            fill_value,
        )?;
        self.arrays.insert(array.to_string(), schema);
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
        self.staging()
            .await?
            .write_array(array, start, data)
            .await?;
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

    /// Commits the dataset into the container.
    ///
    /// Flushes and compacts the staged file into one segment. Appends that
    /// segment. Then records the dataset in the footer.
    pub async fn finish(mut self) -> Result<()> {
        match self.file.as_mut() {
            Some(file) => {
                // Flush first. compact merges committed layers only, so a
                // pending write must commit before it runs.
                file.flush().await?;
                file.compact().await?;
            }
            None => {
                // A dataset with no arrays still needs a segment. Every
                // ordinal must map to real bytes.
                let mut file = create_staging_file(&self.dir, &self.config).await?;
                file.flush().await?;
            }
        }
        // Take the statistics array-format computed during the flush, while
        // the file is still open. They go into the footer, so a reader gets
        // them without the segment.
        let array_stats: Vec<(u32, ArrayStatsS)> = match self.file.as_ref() {
            Some(file) => self
                .arrays
                .keys()
                .enumerate()
                .filter_map(|(position, array)| {
                    let stats = file.array_stats(array)?;
                    Some((position as u32, ArrayStatsS::from(stats)))
                })
                .collect(),
            None => Vec::new(),
        };

        // Release the file before the read-back. array-format holds an open
        // handle and a cache that nothing needs now.
        self.file = None;

        let arrays = std::mem::take(&mut self.arrays);
        let positions: HashMap<String, u32> = arrays
            .keys()
            .enumerate()
            .map(|(i, k)| (k.clone(), i as u32))
            .collect();

        let segment = self.dir.join("data.af");
        // One lock covers the append and the footer entry. Concurrent
        // datasets therefore interleave neither bytes nor ordinals.
        let mut state = self.state.lock().await;
        let (seg_offset, seg_len) = state.append_segment(&segment).await?;
        debug!(
            dataset = %self.name,
            seg_offset, seg_len, arrays = arrays.len(),
            "appended dataset segment"
        );

        let mut global_attrs = Vec::with_capacity(self.global_attrs.len());
        for (key, value) in self.global_attrs.drain(..) {
            global_attrs.push((state.interner.intern_key(&key), AttrS::from(value)));
        }
        let mut array_attrs = Vec::with_capacity(self.array_attrs.len());
        for (array, attrs) in std::mem::take(&mut self.array_attrs) {
            let position = positions[&array];
            let mut encoded = Vec::with_capacity(attrs.len());
            for (key, value) in attrs {
                encoded.push((state.interner.intern_key(&key), AttrS::from(value)));
            }
            array_attrs.push((position, encoded));
        }
        let schema = state.interner.intern_schema(DatasetSchema { arrays });

        state.entries.push((
            self.seq,
            DatasetEntry {
                name: std::mem::take(&mut self.name),
                schema,
                seg_offset,
                seg_len,
                global_attrs,
                array_attrs,
                array_stats,
            },
        ));
        drop(state);
        // The staging area is large. Reclaim it now, not at the end.
        let _ = std::fs::remove_dir_all(&self.dir);
        Ok(())
    }
}

/// Creates the per-dataset staging file under `dir`.
async fn create_staging_file(dir: &std::path::Path, config: &WriterConfig) -> Result<ArrayFile> {
    let store: Arc<dyn ObjectStore> =
        Arc::new(object_store::local::LocalFileSystem::new_with_prefix(dir)?);
    let path = OsPath::from("data.af");
    let target = config.block_target_size;
    Ok(match config.codec {
        Codec::Zstd => {
            ArrayFile::create(store, path, staging_config(ZstdCodec::default(), target)).await?
        }
        Codec::Lz4 => ArrayFile::create(store, path, staging_config(Lz4Codec, target)).await?,
        Codec::Uncompressed => {
            ArrayFile::create(store, path, staging_config(NoCompression, target)).await?
        }
    })
}

fn upsert(list: &mut Vec<(String, Attr)>, key: &str, value: Attr) {
    match list.iter_mut().find(|(k, _)| k == key) {
        Some(slot) => slot.1 = value,
        None => list.push((key.to_string(), value)),
    }
}

fn staging_config<C: array_format::CompressionCodec>(
    codec: C,
    block_target_size: usize,
) -> FileConfig<C> {
    FileConfig {
        codec,
        block_target_size,
        // One write and one read, by compact, cover the staging file. A large
        // cache would only duplicate the page cache.
        cache_capacity: 16 * 1024 * 1024,
        io_cache_capacity: 8 * 1024 * 1024,
        cache: None,
    }
}
