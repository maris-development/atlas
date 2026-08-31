//! Building a collection. The only code in this crate that writes array data.
//!
//! # How a container is assembled
//!
//! [`AtlasWriter`] streams one object. Each dataset is staged as a complete
//! `array-format` file in a local scratch directory, then copied verbatim into
//! the stream:
//!
//! ```text
//! add_dataset("jan")  -> scratch/1/data.af   (define, write, define, write, ...)
//!   finish()          -> flush, compact, copy  -> container[8 .. 4_100]
//! add_dataset("feb")  -> scratch/2/data.af
//!   finish()          -> flush, compact, copy  -> container[4_100 .. 9_002]
//! AtlasWriter::finish -> footer, trailer, done
//! ```
//!
//! Staging on local disk keeps memory bounded whatever the dataset size:
//! `array-format` spills compressed chunks to a temporary file as they arrive,
//! and the copy into the container streams in fixed-size pieces.
//!
//! `flush` then `compact` is deliberate, and the order matters. `flush` commits
//! buffered writes into a sidecar layer; `compact` merges every layer into one
//! base file. Compacting without flushing first would leave the buffered writes
//! behind. The result is a single self-contained file, which is what a segment
//! has to be.
//!
//! # Staging several datasets at once
//!
//! A [`DatasetWriter`] touches the container only in [`DatasetWriter::finish`],
//! which takes the writer's lock for the whole append. So several datasets may
//! be staged concurrently — useful when ingesting many files — and they land in
//! the container in whatever order they finish, never interleaved.
//!
//! # Failure
//!
//! Nothing is readable until [`AtlasWriter::finish`] writes the trailer. Drop a
//! [`DatasetWriter`] without finishing it and that dataset never enters the
//! container; drop the [`AtlasWriter`] and no valid collection appears at all.

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

/// Bytes moved per turn when copying a staged segment into the container.
const COPY_CHUNK_SIZE: usize = 8 * 1024 * 1024;

/// The half of a writer that datasets share: the output stream, the footer
/// being built, and the scratch area.
struct WriterState {
    out: Option<BufWriter>,
    prefix: OsPath,
    config: WriterConfig,
    /// Bytes written so far, and therefore the offset of the next segment.
    offset: u64,
    interner: Interner,
    entries: Vec<DatasetEntry>,
    names: HashSet<String>,
    scratch: tempfile::TempDir,
    dataset_seq: u64,
}

impl WriterState {
    /// Appends a staged segment file to the container and returns its byte
    /// range. Streams in [`COPY_CHUNK_SIZE`] pieces so a large dataset never
    /// has to fit in memory.
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

/// Builds one collection, then finishes. There is no reopening a collection to
/// add to it; rewrite it instead.
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
    /// Writing begins at once, but nothing at `prefix` reads as a collection
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

    /// Starts a collection in a local directory, creating the directory if
    /// needed.
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
        let dir = state.scratch.path().join(state.dataset_seq.to_string());
        std::fs::create_dir_all(&dir)?;
        let config = state.config.clone();
        drop(state);
        Ok(DatasetWriter {
            state: Arc::clone(&self.state),
            name: name.to_string(),
            dir,
            config,
            file: None,
            arrays: IndexMap::new(),
            global_attrs: Vec::new(),
            array_attrs: IndexMap::new(),
        })
    }

    /// How many datasets have been committed so far.
    pub async fn dataset_count(&self) -> usize {
        self.state.lock().await.entries.len()
    }

    /// Writes the footer and trailer, and completes the upload.
    ///
    /// After this the collection is readable and permanently fixed. Any
    /// [`DatasetWriter`] still open fails with [`Error::WriterFinished`].
    pub async fn finish(self) -> Result<()> {
        let mut state = self.state.lock().await;
        let mut out = state.out.take().ok_or(Error::WriterFinished)?;
        let (schema_pool, attr_key_pool) = std::mem::take(&mut state.interner).into_pools();
        let footer = CollectionFooter {
            version: format::FORMAT_VERSION,
            segment_format: format::SEGMENT_FORMAT,
            codec: state.config.codec,
            created_unix_ms: chrono::Utc::now().timestamp_millis(),
            schema_pool,
            attr_key_pool,
            datasets: std::mem::take(&mut state.entries),
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
            // No trailer was written, so nothing at the target can open as a
            // collection. Whether a partial object lingers is up to the
            // backend; say so rather than leave it silent.
            tracing::warn!(
                prefix = %self.prefix,
                "atlas writer dropped without finish; no collection was written"
            );
        }
    }
}

/// Builds one dataset inside a collection.
///
/// Declare arrays with [`define_array`](Self::define_array) and fill them with
/// [`write_array`](Self::write_array), in any order and any number of slabs.
/// Nothing reaches the container until [`finish`](Self::finish).
pub struct DatasetWriter {
    state: Arc<Mutex<WriterState>>,
    name: String,
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

    /// Names of the arrays declared so far, in definition order.
    pub fn list_arrays(&self) -> Vec<String> {
        self.arrays.keys().cloned().collect()
    }

    /// The schema of an array declared so far.
    pub fn array_meta(&self, array: &str) -> Option<&ArraySchema> {
        self.arrays.get(array)
    }

    /// Opens the staging file on first use, so a dataset that is declared and
    /// then abandoned costs no I/O.
    async fn staging(&mut self) -> Result<&mut ArrayFile> {
        if self.file.is_none() {
            self.file = Some(create_staging_file(&self.dir, &self.config).await?);
        }
        Ok(self.file.as_mut().expect("just set"))
    }

    /// Declares an array.
    ///
    /// `chunk_shape` defaults to `shape`, storing the array as a single chunk.
    /// `fill_value` is what a read returns for elements that are never written.
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
        // Record what it will actually store, not what was asked for.
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
    /// The region may span chunks and need not be chunk-aligned. `T` must match
    /// the array's declared type.
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

    /// Attaches a dataset-level attribute. A repeated key replaces the earlier
    /// value.
    pub fn set_attribute(&mut self, key: &str, value: Attr) {
        upsert(&mut self.global_attrs, key, value);
    }

    /// Attaches an attribute to one array, which must already be defined.
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
    /// Flushes and compacts the staged file into one segment, appends it, and
    /// records the dataset in the footer being built.
    pub async fn finish(mut self) -> Result<()> {
        match self.file.as_mut() {
            Some(file) => {
                // flush first: compact merges committed layers, so pending
                // writes have to be committed before it runs.
                file.flush().await?;
                file.compact().await?;
            }
            None => {
                // A dataset with no arrays still needs a segment, so every
                // ordinal maps to real bytes.
                let mut file = create_staging_file(&self.dir, &self.config).await?;
                file.flush().await?;
            }
        }
        // Harvest the statistics array-format computed during the flush, while
        // the file is still open. They go into the footer, so a reader gets
        // them without touching the segment.
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

        // Release the file before reading it back: array-format holds an open
        // handle and a cache that are no longer wanted.
        self.file = None;

        let arrays = std::mem::take(&mut self.arrays);
        let positions: HashMap<String, u32> = arrays
            .keys()
            .enumerate()
            .map(|(i, k)| (k.clone(), i as u32))
            .collect();

        let segment = self.dir.join("data.af");
        // One lock for the whole append plus the footer entry, so concurrent
        // datasets cannot interleave their bytes or their ordinals.
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

        state.entries.push(DatasetEntry {
            name: std::mem::take(&mut self.name),
            schema,
            seg_offset,
            seg_len,
            global_attrs,
            array_attrs,
            array_stats,
        });
        drop(state);
        // Staging is large; reclaim it now rather than at the end of the write.
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
        // Staging is written once and read once, by compact. A large cache
        // would only duplicate the page cache.
        cache_capacity: 16 * 1024 * 1024,
        io_cache_capacity: 8 * 1024 * 1024,
        cache: None,
    }
}
