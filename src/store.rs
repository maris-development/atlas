use std::sync::Arc;

use array_format::DeltaCache;
use object_store::{ObjectStore, local::LocalFileSystem, path::Path, prefix::PrefixStore};
use parking_lot::Mutex;
use tracing::{debug, info, instrument};

use crate::{
    Error, Result,
    config::{Codec, MetaFormat, StoreConfig},
    dataset::{ArrayCache, DatasetView, GLOBAL_ATTRS_ARRAY, PendingAttrs, open_dataset_view},
    meta::{STORE_FORMAT_VERSION, StoreMeta, load_meta, save_meta},
};

/// Handle to an opened or newly created atlas store.
///
/// Owns the [`object_store`] backend, the in-memory store metadata, a
/// per-array file cache, and the chosen array / metadata codecs. All
/// mutations (`create_dataset`, `delete_dataset`, and everything that
/// flows through a [`DatasetView`]) update in-memory state only —
/// nothing reaches disk until [`Atlas::flush`].
///
/// `Atlas` is `Send + Sync` and safe to share across tasks; each array
/// file is independently guarded by a `tokio::sync::RwLock`.
pub struct Atlas {
    store: Arc<dyn ObjectStore>,
    meta: Arc<Mutex<StoreMeta>>,
    /// Attribute writes buffered until [`Atlas::flush`], keeping mutations
    /// off-disk and attribute setters non-blocking.
    pending_attrs: Arc<Mutex<PendingAttrs>>,
    cache: Arc<ArrayCache>,
    codec: Codec,
    meta_format: MetaFormat,
    meta_compression: Codec,
}

impl Atlas {
    /// Open an existing store at `prefix` within `store`.
    ///
    /// Reads `atlas.json` exactly once. Subsequent mutations only touch the
    /// in-memory meta until [`Atlas::flush`] is called.
    #[instrument(skip(store), fields(prefix = %prefix))]
    pub async fn open(store: Arc<dyn ObjectStore>, prefix: Path) -> Result<Self> {
        let store = prefixed(store, prefix);
        let (meta, meta_format, meta_compression) = load_meta(&store).await?;
        let codec = meta.codec;
        info!(
            datasets = meta.datasets.len(),
            ?codec,
            ?meta_format,
            ?meta_compression,
            "opened atlas store"
        );
        Ok(Self {
            store,
            meta: Arc::new(Mutex::new(meta)),
            pending_attrs: Arc::new(Mutex::new(PendingAttrs::default())),
            cache: default_cache(),
            codec,
            meta_format,
            meta_compression,
        })
    }

    /// Create a new store at `prefix` within `store`.
    #[instrument(skip(store, config), fields(prefix = %prefix, codec = ?config.codec, meta_format = ?config.meta_format, meta_compression = ?config.meta_compression))]
    pub async fn create(store: Arc<dyn ObjectStore>, prefix: Path, config: StoreConfig) -> Result<Self> {
        let store = prefixed(store, prefix);
        let meta = StoreMeta {
            version: STORE_FORMAT_VERSION,
            codec: config.codec,
            ..Default::default()
        };
        save_meta(&store, &meta, config.meta_format, config.meta_compression).await?;
        info!("created atlas store");
        Ok(Self {
            store,
            meta: Arc::new(Mutex::new(meta)),
            pending_attrs: Arc::new(Mutex::new(PendingAttrs::default())),
            cache: default_cache(),
            codec: config.codec,
            meta_format: config.meta_format,
            meta_compression: config.meta_compression,
        })
    }

    /// Open an existing store at the given local filesystem path.
    ///
    /// The metadata format (`atlas.json` / `atlas.msgpack` / `…zst` / `…lz4`)
    /// and array codec are auto-detected from the on-disk files — no
    /// [`StoreConfig`] needed on reopen.
    ///
    /// # Examples
    ///
    /// ```
    /// # tokio::runtime::Runtime::new().unwrap().block_on(async {
    /// use atlas::{Atlas, StoreConfig};
    /// let tmp = tempfile::tempdir().unwrap();
    /// // Create + flush a store so there's something to open.
    /// {
    ///     let mut s = Atlas::create_path(tmp.path(), StoreConfig::default()).await.unwrap();
    ///     s.create_dataset("ds1").await.unwrap();
    ///     s.flush().await.unwrap();
    /// }
    /// let s = Atlas::open_path(tmp.path()).await.unwrap();
    /// assert!(s.dataset_exists("ds1"));
    /// # });
    /// ```
    pub async fn open_path(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let store = Arc::new(LocalFileSystem::new_with_prefix(path.as_ref())?);
        Self::open(store, Path::from("")).await
    }

    /// Create a new store at the given local filesystem path. The directory is created
    /// (recursively, like `mkdir -p`) if it does not already exist.
    ///
    /// # Examples
    ///
    /// ```
    /// # tokio::runtime::Runtime::new().unwrap().block_on(async {
    /// use atlas::{Atlas, StoreConfig};
    /// let tmp = tempfile::tempdir().unwrap();
    /// let s = Atlas::create_path(tmp.path(), StoreConfig::default()).await.unwrap();
    /// assert!(s.list_datasets().is_empty());
    /// # });
    /// ```
    pub async fn create_path(path: impl AsRef<std::path::Path>, config: StoreConfig) -> Result<Self> {
        let path = path.as_ref();
        std::fs::create_dir_all(path)?;
        let store = Arc::new(LocalFileSystem::new_with_prefix(path)?);
        Self::create(store, Path::from(""), config).await
    }

    /// Create a new dataset in this store and return a [`DatasetView`]
    /// for populating it. Errors with [`Error::DatasetAlreadyExists`] if
    /// a dataset with this name is already registered, or
    /// [`Error::InvalidName`] if `name` violates the naming rules
    /// (non-empty, no `/`, no leading `_`, not `.` or `..`).
    #[instrument(skip(self))]
    pub async fn create_dataset(&mut self, name: &str) -> Result<DatasetView> {
        crate::validate_name(name)?;
        {
            let mut meta = self.meta.lock();
            if meta.datasets.contains_key(name) {
                return Err(Error::DatasetAlreadyExists(name.to_string()));
            }
            meta.datasets.insert(name.to_string(), Default::default());
        }
        debug!("created dataset");
        Ok(DatasetView::new(
            self.store.clone(),
            self.cache.clone(),
            name.to_string(),
            self.meta.clone(),
            self.pending_attrs.clone(),
            self.codec.clone(),
        ))
    }

    /// Return a [`DatasetView`] for an existing dataset. Errors with
    /// [`Error::DatasetNotFound`] if no dataset with this name exists.
    /// Cheap — reads the in-memory metadata, never touches disk.
    #[instrument(skip(self))]
    pub async fn open_dataset(&self, name: &str) -> Result<DatasetView> {
        open_dataset_view(
            self.store.clone(),
            self.cache.clone(),
            self.meta.clone(),
            self.pending_attrs.clone(),
            name,
            self.codec.clone(),
        )
        .await
    }

    /// Remove a dataset from this store. Tombstones the dataset's entries
    /// inside every shared array file but does not flush — call
    /// [`Atlas::flush`] to persist the deletion, and optionally
    /// [`Atlas::compact`] afterwards to reclaim the storage.
    /// Errors with [`Error::DatasetNotFound`] if no dataset with this
    /// name exists.
    #[instrument(skip(self))]
    pub async fn delete_dataset(&mut self, name: &str) -> Result<()> {
        let dataset_meta = {
            let mut meta = self.meta.lock();
            meta.unrecord_dataset(name)
                .ok_or_else(|| Error::DatasetNotFound(name.to_string()))?
        };
        // Drop any buffered (not-yet-flushed) attribute writes for this dataset.
        self.pending_attrs.lock().remove_dataset(name);
        debug!(arrays = dataset_meta.arrays.len(), "deleting dataset");
        for (array_name, schema) in &dataset_meta.arrays {
            let handle = self
                .cache
                .get_or_insert(&self.store, array_name, &schema.codec);
            let arc = handle.get().await?;
            let mut guard = arc.write().await;
            guard.delete(name)?;
            // No flush here; persistence happens on Atlas::flush().
        }
        // Tombstone the dataset's entry in the global-attributes file, if it
        // was ever created. `get_existing` never creates the file, so a store
        // that never set a global attribute stays free of `_global/data.af`.
        let global = self
            .cache
            .get_or_insert(&self.store, GLOBAL_ATTRS_ARRAY, &self.codec);
        if let Some(arc) = global.get_existing().await? {
            let mut guard = arc.write().await;
            if guard.get_array(name).is_ok() {
                guard.delete(name)?;
            }
        }
        Ok(())
    }

    /// All dataset names currently registered in this store, in insertion order.
    /// Reads from the in-memory store metadata — no disk I/O.
    pub fn list_datasets(&self) -> Vec<String> {
        let meta = self.meta.lock();
        meta.datasets.keys().cloned().collect()
    }

    /// `true` if a dataset with this name is registered. O(1) hash lookup in
    /// the in-memory store metadata.
    pub fn dataset_exists(&self, name: &str) -> bool {
        let meta = self.meta.lock();
        meta.datasets.contains_key(name)
    }

    /// Distinct array names across all datasets in this store, sorted.
    /// One entry per physical `.af` file — datasets sharing an array name
    /// (the common case) collapse to a single entry here.
    pub fn list_arrays(&self) -> Vec<String> {
        let meta = self.meta.lock();
        let mut arrays: Vec<String> = meta
            .datasets
            .values()
            .flat_map(|d| d.arrays.keys().cloned())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        arrays.sort();
        arrays
    }

    /// The collection-wide **merged schema**: every unique array (with its
    /// dtype widened across datasets and its named dimensions) and every unique
    /// attribute key with its widened type. This is the same summary written
    /// into `atlas.json`; it is descriptive only — reads always use each
    /// dataset's own schema. Computed from the in-memory metadata, no disk I/O.
    pub fn merged_schema(&self) -> crate::MergedSchema {
        crate::meta::compute_merged(&self.meta.lock().datasets)
    }

    /// Returns the dtype of `array` if any dataset in this store declares it.
    /// Used by `read_array_across`'s Python binding to pick the generic
    /// instantiation without round-tripping through a `DatasetView`.
    pub fn array_dtype(&self, array: &str) -> Option<array_format::DType> {
        let meta = self.meta.lock();
        meta.datasets
            .values()
            .find_map(|d| d.arrays.get(array))
            .map(|schema| schema.dtype.clone())
    }

    /// Bulk read the same slice of `array` from many datasets that share its
    /// physical file. Runs at most `num_cpus` reads concurrently — matching
    /// what a well-tuned dask threadpool would do — to keep
    /// `tokio::task::spawn_blocking`'s decompression pool from oversubscribing
    /// the actual CPU cores.
    ///
    /// This exists because `open_as_many_xarray_dataset` over N datasets used to incur N
    /// separate Python → Rust → tokio::block_on transitions plus Python-side
    /// dask graph overhead. One call here replaces all of that and gets the
    /// same parallelism dask was providing — but in pure Rust, with no GIL
    /// involvement until the results return.
    ///
    /// `start` and `shape` follow the same conventions as
    /// [`DatasetView::read_array`]: empty `start` + empty `shape` mean the
    /// full array. Per-dataset entries that don't declare `array` are
    /// returned as `None`.
    #[instrument(skip(self, dataset_names), fields(array = %array, n = dataset_names.len()))]
    pub async fn read_array_across<T: array_format::ArrayElement + Send + Sync + 'static>(
        &self,
        array: &str,
        dataset_names: &[String],
        start: Vec<usize>,
        shape: Vec<usize>,
    ) -> Result<Vec<Option<ndarray::ArcArray<T, ndarray::IxDyn>>>> {
        // Discover the codec for `array` from any dataset that defines it,
        // and pre-flight which dataset names declare it.
        let (codec, present): (Codec, Vec<bool>) = {
            let meta = self.meta.lock();
            let mut codec: Option<Codec> = None;
            let mut present: Vec<bool> = Vec::with_capacity(dataset_names.len());
            for name in dataset_names {
                let has = meta
                    .datasets
                    .get(name)
                    .and_then(|d| d.arrays.get(array))
                    .map(|schema| {
                        codec.get_or_insert(schema.codec);
                        true
                    })
                    .unwrap_or(false);
                present.push(has);
            }
            let codec = codec.ok_or_else(|| Error::ArrayNotFound(array.to_string()))?;
            (codec, present)
        };

        let handle = self.cache.get_or_insert(&self.store, array, &codec);
        let arc = handle.get().await?;

        // Spawn each per-dataset read as a top-level tokio task so the
        // multi-thread runtime distributes them across worker threads.
        // A semaphore caps in-flight tasks at `concurrency` (≈ num_cpus)
        // to keep `tokio::task::spawn_blocking`'s decompression pool from
        // oversubscribing the actual CPU cores.
        let concurrency = num_cpus::get().max(1);
        let sem = Arc::new(tokio::sync::Semaphore::new(concurrency));
        let mut joinset = tokio::task::JoinSet::new();
        for (idx, (name, &has)) in dataset_names.iter().zip(present.iter()).enumerate() {
            if !has {
                continue;
            }
            let permit = Arc::clone(&sem)
                .acquire_owned()
                .await
                .expect("semaphore never closed");
            let arc = Arc::clone(&arc);
            let name = name.clone();
            let start = start.clone();
            let shape = shape.clone();
            joinset.spawn(async move {
                let _permit = permit;
                let guard = arc.read().await;
                let res = guard.read_array::<T>(&name, start, shape).await;
                (idx, res)
            });
        }

        let mut out: Vec<Option<ndarray::ArcArray<T, ndarray::IxDyn>>> =
            (0..dataset_names.len()).map(|_| None).collect();
        while let Some(join_res) = joinset.join_next().await {
            let (idx, read_res) = join_res
                .map_err(|e| Error::ArrayFormat(array_format::Error::Storage(e.to_string())))?;
            out[idx] = Some(read_res?);
        }
        Ok(out)
    }

    /// Like [`Atlas::read_array_across`] but returns one stacked
    /// `(len(dataset_names), *per_dataset_shape)` `ndarray::Array` instead of
    /// a `Vec` of per-dataset arrays.
    ///
    /// The output buffer is pre-allocated once; each parallel read writes its
    /// row in as the task completes, overlapping the serial copy with the
    /// remaining in-flight reads. Saves the ~5.7 GiB of memory copies that
    /// the Python-side `np.stack` on the per-dataset list would do on a
    /// 1000-dataset gridded workload.
    ///
    /// Errors if any listed dataset doesn't declare `array` — the stacked
    /// representation has no positional "missing" sentinel.
    #[instrument(skip(self, dataset_names), fields(array = %array, n = dataset_names.len()))]
    pub async fn read_array_across_stacked<
        T: array_format::ArrayElement + Send + Sync + Clone + 'static,
    >(
        &self,
        array: &str,
        dataset_names: &[String],
        start: Vec<usize>,
        shape: Vec<usize>,
    ) -> Result<ndarray::Array<T, ndarray::IxDyn>> {
        if dataset_names.is_empty() {
            return Err(Error::ArrayNotFound(array.to_string()));
        }

        // Discover the codec and verify ALL listed datasets declare the array.
        let codec: Codec = {
            let meta = self.meta.lock();
            let mut codec: Option<Codec> = None;
            for name in dataset_names {
                let schema = meta
                    .datasets
                    .get(name)
                    .and_then(|d| d.arrays.get(array))
                    .ok_or_else(|| {
                        Error::ArrayNotFound(format!("{array} (in dataset {name})"))
                    })?;
                codec.get_or_insert(schema.codec);
            }
            codec.expect("non-empty dataset_names, all schemas present")
        };

        let handle = self.cache.get_or_insert(&self.store, array, &codec);
        let arc_file = handle.get().await?;

        // Read the first dataset synchronously to discover the per-dataset
        // shape (after `start`/`shape` slicing) so we can pre-allocate the
        // stacked output. Then write its row in.
        let first_arr = {
            let guard = arc_file.read().await;
            guard
                .read_array::<T>(&dataset_names[0], start.clone(), shape.clone())
                .await?
        };
        let per_dataset_shape: Vec<usize> = first_arr.shape().to_vec();
        let n = dataset_names.len();
        let mut out_shape = Vec::with_capacity(per_dataset_shape.len() + 1);
        out_shape.push(n);
        out_shape.extend(&per_dataset_shape);

        // Allocate the output as a flat `Vec<T>` of N * per_dataset_elements
        // entries. We bypass `Array::default` so we don't pay an extra
        // zero-fill memset of the entire buffer — every slot will be written
        // by either the first-dataset read above or a spawned task below.
        let per_dataset_elements: usize = per_dataset_shape.iter().product();
        let total_elements = n * per_dataset_elements;
        let mut buf: Vec<T> = Vec::with_capacity(total_elements);
        // SAFETY: every element is written exactly once before we hand the
        // Vec to `Array::from_shape_vec`. Until then, the uninitialised
        // portion is never read.
        unsafe { buf.set_len(total_elements) };

        // Helper: copy a per-dataset ArcArray into row `idx` of the flat
        // buffer via `copy_from_slice` (memcpy). Both source and destination
        // are C-order contiguous (array-format's `assemble_nd` builds via
        // `Array::from_elem`, our Vec is contiguous by construction).
        fn write_row<T: array_format::ArrayElement + Clone>(
            buf: &mut [T],
            idx: usize,
            per_row: usize,
            src: &ndarray::ArcArray<T, ndarray::IxDyn>,
        ) -> Result<()> {
            let src_slice = src
                .as_slice()
                .ok_or_else(|| Error::ArrayFormat(array_format::Error::Storage(
                    "per-dataset read returned non-contiguous array".into(),
                )))?;
            let dst = &mut buf[idx * per_row..(idx + 1) * per_row];
            dst.clone_from_slice(src_slice);
            Ok(())
        }

        write_row(&mut buf, 0, per_dataset_elements, &first_arr)?;
        drop(first_arr);

        // Spawn the remaining N-1 reads with the same concurrency-capped
        // pattern as `read_array_across`.
        let concurrency = num_cpus::get().max(1);
        let sem = Arc::new(tokio::sync::Semaphore::new(concurrency));
        let mut joinset = tokio::task::JoinSet::new();
        for (idx, name) in dataset_names.iter().enumerate().skip(1) {
            let permit = Arc::clone(&sem)
                .acquire_owned()
                .await
                .expect("semaphore never closed");
            let arc = Arc::clone(&arc_file);
            let name = name.clone();
            let start = start.clone();
            let shape = shape.clone();
            joinset.spawn(async move {
                let _permit = permit;
                let guard = arc.read().await;
                let res = guard.read_array::<T>(&name, start, shape).await;
                (idx, res)
            });
        }

        // As tasks complete, memcpy their row into the pre-allocated buffer.
        // The serial memcpy overlaps with the remaining in-flight parallel
        // reads happening on the runtime's other workers.
        while let Some(join_res) = joinset.join_next().await {
            let (idx, read_res) = join_res
                .map_err(|e| Error::ArrayFormat(array_format::Error::Storage(e.to_string())))?;
            let arr = read_res?;
            write_row(&mut buf, idx, per_dataset_elements, &arr)?;
        }

        ndarray::Array::from_shape_vec(ndarray::IxDyn(&out_shape), buf).map_err(|e| {
            Error::ArrayFormat(array_format::Error::Storage(format!(
                "stacked output shape mismatch: {e}"
            )))
        })
    }

    /// Flush every known array file's pending writes AND persist the in-memory
    /// `atlas.json`. This is the single durability boundary for the store.
    ///
    /// Force-initializes every array referenced in meta, even ones never
    /// touched by a `DatasetView` (lazy-init wins are on the read path, not
    /// on flush).
    #[instrument(skip(self))]
    pub async fn flush(&mut self) -> Result<()> {
        // Apply buffered attribute writes into the array files (creating the
        // `_global` file on demand), then commit every touched file.
        self.drain_pending_attrs().await?;
        self.force_init_all_known_arrays().await?;
        let snapshot = self.all_initialized_files();
        let files = snapshot.len();
        debug!(files, "flushing array files");
        for arc in snapshot {
            arc.write().await.flush().await?;
        }
        let meta_snapshot = self.meta.lock().clone();
        let datasets = meta_snapshot.datasets.len();
        save_meta(&self.store, &meta_snapshot, self.meta_format, self.meta_compression).await?;
        info!(files, datasets, "flushed atlas store");
        Ok(())
    }

    /// Compact every known array file in place (reclaims tombstoned space).
    /// Drains buffered attributes and commits them first, then force-initializes
    /// every array referenced in meta.
    #[instrument(skip(self))]
    pub async fn compact(&mut self) -> Result<()> {
        self.drain_pending_attrs().await?;
        self.force_init_all_known_arrays().await?;
        let snapshot = self.all_initialized_files();
        let files = snapshot.len();
        debug!(files, "compacting array files");
        for arc in snapshot {
            let mut guard = arc.write().await;
            // Commit any pending (incl. just-drained attributes) so the compact
            // merges them into the new base.
            guard.flush().await?;
            guard.compact().await?;
        }
        info!(files, "compacted atlas store");
        Ok(())
    }

    /// Drain the buffered attribute writes into their `.af` files. For each
    /// `(file, dataset)` this ensures the dataset's entry exists (defining an
    /// empty-shape scalar in the `_global` file when needed) and sets every
    /// buffered attribute. The changes land in each file's pending layer and
    /// are committed by the subsequent flush/compact.
    async fn drain_pending_attrs(&self) -> Result<()> {
        let drained = self.pending_attrs.lock().drain_all();
        for ((file, dataset), attrs) in drained {
            let codec = self.file_codec(&file);
            let handle = self.cache.get_or_insert(&self.store, &file, &codec);
            let arc = handle.get().await?;
            let mut guard = arc.write().await;
            // The `_global` file has no data arrays defined up front — create a
            // scalar placeholder entry for this dataset if it's missing.
            if guard.get_array(&dataset).is_err() {
                guard.define_array::<u8>(dataset.clone(), vec![], vec![], None, None)?;
            }
            for (key, value) in attrs {
                guard.set_attribute(&dataset, &key, value.into())?;
            }
        }
        Ok(())
    }

    /// Codec to open a physical file with: the store default for the global
    /// attributes file, otherwise the array's recorded codec (any dataset that
    /// declares it), falling back to the store codec.
    fn file_codec(&self, file: &str) -> Codec {
        if file == GLOBAL_ATTRS_ARRAY {
            return self.codec.clone();
        }
        self.meta
            .lock()
            .array_file_codec(file)
            .unwrap_or_else(|| self.codec.clone())
    }

    /// Every array file currently initialized in the cache (including
    /// `_global` and any read-opened file), deduped by handle.
    fn all_initialized_files(&self) -> Vec<Arc<tokio::sync::RwLock<array_format::ArrayFile>>> {
        self.cache
            .files
            .read()
            .values()
            .filter_map(|a| a.try_get())
            .collect()
    }

    /// Ensures every array referenced by any dataset in meta has an
    /// initialized `ArrayFile` in the cache, and returns the inner locks
    /// (deduped by array name).
    async fn force_init_all_known_arrays(
        &self,
    ) -> Result<Vec<Arc<tokio::sync::RwLock<array_format::ArrayFile>>>> {
        let specs: Vec<(String, Codec)> = {
            let meta = self.meta.lock();
            let mut seen = std::collections::HashSet::new();
            let mut out = Vec::new();
            for ds in meta.datasets.values() {
                for (name, schema) in &ds.arrays {
                    if seen.insert(name.clone()) {
                        out.push((name.clone(), schema.codec.clone()));
                    }
                }
            }
            out
        };
        let mut result = Vec::with_capacity(specs.len());
        for (name, codec) in specs {
            let handle = self.cache.get_or_insert(&self.store, &name, &codec);
            result.push(handle.get().await?);
        }
        Ok(result)
    }
}

fn prefixed(store: Arc<dyn ObjectStore>, prefix: Path) -> Arc<dyn ObjectStore> {
    if prefix.as_ref().is_empty() {
        store
    } else {
        Arc::new(PrefixStore::new(store, prefix))
    }
}

fn default_cache() -> Arc<ArrayCache> {
    let delta = Arc::new(DeltaCache::new(
        256 * 1024 * 1024,
        64 * 1024 * 1024,
    ));
    Arc::new(ArrayCache::new(delta))
}

#[cfg(test)]
mod tests {
    use super::*;
    use object_store::memory::InMemory;

    fn make_store() -> (Arc<dyn ObjectStore>, Path) {
        (Arc::new(InMemory::new()), Path::from(""))
    }

    #[tokio::test]
    async fn empty_store_lists_nothing() {
        let (store, prefix) = make_store();
        let s = Atlas::create(store, prefix, StoreConfig::default()).await.unwrap();
        assert!(s.list_datasets().is_empty());
        assert!(s.list_arrays().is_empty());
    }

    #[tokio::test]
    async fn dataset_exists_false_on_empty_store() {
        let (store, prefix) = make_store();
        let s = Atlas::create(store, prefix, StoreConfig::default()).await.unwrap();
        assert!(!s.dataset_exists("any"));
    }

    #[tokio::test]
    async fn create_dataset_makes_it_visible() {
        let (store, prefix) = make_store();
        let mut s = Atlas::create(store, prefix, StoreConfig::default()).await.unwrap();
        s.create_dataset("ds").await.unwrap();
        assert!(s.dataset_exists("ds"));
        assert!(s.list_datasets().contains(&"ds".to_string()));
    }

    #[tokio::test]
    async fn duplicate_dataset_name_rejected() {
        let (store, prefix) = make_store();
        let mut s = Atlas::create(store, prefix, StoreConfig::default()).await.unwrap();
        s.create_dataset("ds").await.unwrap();
        let err = s.create_dataset("ds").await.err().unwrap();
        assert!(matches!(err, crate::Error::DatasetAlreadyExists(_)));
    }

    #[tokio::test]
    async fn open_nonexistent_dataset_errors() {
        let (store, prefix) = make_store();
        let s = Atlas::create(store, prefix, StoreConfig::default()).await.unwrap();
        let err = s.open_dataset("ghost").await.err().unwrap();
        assert!(matches!(err, crate::Error::DatasetNotFound(_)));
    }

    #[tokio::test]
    async fn delete_nonexistent_dataset_errors() {
        let (store, prefix) = make_store();
        let mut s = Atlas::create(store, prefix, StoreConfig::default()).await.unwrap();
        let err = s.delete_dataset("ghost").await.unwrap_err();
        assert!(matches!(err, crate::Error::DatasetNotFound(_)));
    }

    #[tokio::test]
    async fn delete_dataset_removes_it() {
        let (store, prefix) = make_store();
        let mut s = Atlas::create(store, prefix, StoreConfig::default()).await.unwrap();
        s.create_dataset("to_delete").await.unwrap();
        assert!(s.dataset_exists("to_delete"));
        s.delete_dataset("to_delete").await.unwrap();
        assert!(!s.dataset_exists("to_delete"));
    }

    #[tokio::test]
    async fn list_datasets_returns_all_created() {
        let (store, prefix) = make_store();
        let mut s = Atlas::create(store, prefix, StoreConfig::default()).await.unwrap();
        s.create_dataset("a").await.unwrap();
        s.create_dataset("b").await.unwrap();
        s.create_dataset("c").await.unwrap();
        let mut names = s.list_datasets();
        names.sort();
        assert_eq!(names, vec!["a", "b", "c"]);
    }

    #[tokio::test]
    async fn invalid_dataset_name_rejected() {
        let (store, prefix) = make_store();
        let mut s = Atlas::create(store, prefix, StoreConfig::default()).await.unwrap();
        assert!(matches!(s.create_dataset("").await, Err(crate::Error::InvalidName(_))));
        assert!(matches!(s.create_dataset("a/b").await, Err(crate::Error::InvalidName(_))));
        assert!(matches!(s.create_dataset("_x").await, Err(crate::Error::InvalidName(_))));
        assert!(matches!(s.create_dataset("..").await, Err(crate::Error::InvalidName(_))));
    }

    #[tokio::test]
    async fn list_arrays_deduplicates_shared_names() {
        let (store, prefix) = make_store();
        let mut s = Atlas::create(store.clone(), prefix.clone(), StoreConfig::default()).await.unwrap();

        {
            let mut ds_a = s.create_dataset("a").await.unwrap();
            ds_a.define_array::<f32>("shared", vec!["x".into()], vec![2], None, None)
                .await
                .unwrap();
            ds_a.define_array::<f32>("only_a", vec!["x".into()], vec![2], None, None)
                .await
                .unwrap();
        }

        {
            let mut ds_b = s.create_dataset("b").await.unwrap();
            ds_b.define_array::<f32>("shared", vec!["x".into()], vec![2], None, None)
                .await
                .unwrap();
        }

        s.flush().await.unwrap();

        let s2 = Atlas::open(store, prefix).await.unwrap();
        let arrays = s2.list_arrays();
        assert_eq!(arrays, vec!["only_a", "shared"]);
    }

    #[tokio::test]
    async fn lz4_codec_roundtrip() {
        let (store, prefix) = make_store();
        let config = StoreConfig { codec: Codec::Lz4, ..Default::default() };
        let mut s = Atlas::create(store.clone(), prefix.clone(), config).await.unwrap();

        {
            let mut ds = s.create_dataset("ds").await.unwrap();
            ds.define_array::<f32>("arr", vec!["x".into()], vec![4], None, None)
                .await
                .unwrap();
            let data = ndarray::arr1(&[1.0_f32, 2.0, 3.0, 4.0]).into_dyn();
            ds.write_array("arr", vec![0], data.view()).await.unwrap();
        }
        s.flush().await.unwrap();

        let s2 = Atlas::open(store, prefix).await.unwrap();
        let ds2 = s2.open_dataset("ds").await.unwrap();
        let result = ds2.read_array::<f32>("arr", vec![], vec![]).await.unwrap().unwrap();
        let expected = ndarray::arr1(&[1.0_f32, 2.0, 3.0, 4.0]).into_dyn();
        assert_eq!(result, expected.into_shared());
    }

    #[tokio::test]
    async fn uncompressed_codec_roundtrip() {
        let (store, prefix) = make_store();
        let config = StoreConfig { codec: Codec::Uncompressed, ..Default::default() };
        let mut s = Atlas::create(store.clone(), prefix.clone(), config).await.unwrap();

        {
            let mut ds = s.create_dataset("ds").await.unwrap();
            ds.define_array::<i32>("arr", vec!["x".into()], vec![3], None, None)
                .await
                .unwrap();
            let data = ndarray::arr1(&[10_i32, 20, 30]).into_dyn();
            ds.write_array("arr", vec![0], data.view()).await.unwrap();
        }
        s.flush().await.unwrap();

        let s2 = Atlas::open(store, prefix).await.unwrap();
        let ds2 = s2.open_dataset("ds").await.unwrap();
        let result = ds2.read_array::<i32>("arr", vec![], vec![]).await.unwrap().unwrap();
        let expected = ndarray::arr1(&[10_i32, 20, 30]).into_dyn();
        assert_eq!(result, expected.into_shared());
    }

    #[tokio::test]
    async fn path_api_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let data = ndarray::arr1(&[1.0_f32, 2.0, 3.0]).into_dyn();

        {
            let mut s = Atlas::create_path(tmp.path(), StoreConfig::default()).await.unwrap();
            {
                let mut ds = s.create_dataset("ds").await.unwrap();
                ds.define_array::<f32>("arr", vec!["x".into()], vec![3], None, None).await.unwrap();
                ds.write_array("arr", vec![0], data.view()).await.unwrap();
            }
            s.flush().await.unwrap();
        }

        let s2 = Atlas::open_path(tmp.path()).await.unwrap();
        let ds2 = s2.open_dataset("ds").await.unwrap();
        let result = ds2.read_array::<f32>("arr", vec![], vec![]).await.unwrap().unwrap();
        assert_eq!(result, data.into_shared());
    }

    #[tokio::test]
    async fn msgpack_meta_format_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let data = ndarray::arr1(&[1.0_f32, 2.0, 3.0]).into_dyn();

        {
            let config = StoreConfig {
                meta_format: MetaFormat::MsgPack,
                ..Default::default()
            };
            let mut s = Atlas::create_path(tmp.path(), config).await.unwrap();
            {
                let mut ds = s.create_dataset("ds").await.unwrap();
                ds.define_array::<f32>("arr", vec!["x".into()], vec![3], None, None).await.unwrap();
                ds.write_array("arr", vec![0], data.view()).await.unwrap();
            }
            s.flush().await.unwrap();
        }

        // On-disk file is atlas.msgpack, not atlas.json.
        assert!(tmp.path().join("atlas.msgpack").exists());
        assert!(!tmp.path().join("atlas.json").exists());

        // Open auto-detects format and reads data back.
        let s2 = Atlas::open_path(tmp.path()).await.unwrap();
        let ds2 = s2.open_dataset("ds").await.unwrap();
        let result = ds2.read_array::<f32>("arr", vec![], vec![]).await.unwrap().unwrap();
        assert_eq!(result, data.into_shared());
    }

    #[tokio::test]
    async fn compressed_meta_roundtrip_through_atlas() {
        let tmp = tempfile::tempdir().unwrap();
        let data = ndarray::arr1(&[1.0_f32, 2.0, 3.0]).into_dyn();

        {
            let config = StoreConfig {
                meta_format: MetaFormat::MsgPack,
                meta_compression: Codec::Zstd,
                ..Default::default()
            };
            let mut s = Atlas::create_path(tmp.path(), config).await.unwrap();
            {
                let mut ds = s.create_dataset("ds").await.unwrap();
                ds.define_array::<f32>("arr", vec!["x".into()], vec![3], None, None).await.unwrap();
                ds.write_array("arr", vec![0], data.view()).await.unwrap();
            }
            s.flush().await.unwrap();
        }

        // On-disk file is the zstd-compressed msgpack variant.
        assert!(tmp.path().join("atlas.msgpack.zst").exists());
        assert!(!tmp.path().join("atlas.json").exists());
        assert!(!tmp.path().join("atlas.msgpack").exists());

        let s2 = Atlas::open_path(tmp.path()).await.unwrap();
        let ds2 = s2.open_dataset("ds").await.unwrap();
        let result = ds2.read_array::<f32>("arr", vec![], vec![]).await.unwrap().unwrap();
        assert_eq!(result, data.into_shared());
    }

    #[tokio::test]
    async fn create_path_creates_missing_directory() {
        let tmp = tempfile::tempdir().unwrap();
        let nested = tmp.path().join("missing").join("nested");
        assert!(!nested.exists());

        let _atlas = Atlas::create_path(&nested, StoreConfig::default()).await.unwrap();

        assert!(nested.exists() && nested.is_dir());
        assert!(nested.join("atlas.json").exists());
    }

    #[tokio::test]
    async fn create_path_succeeds_when_directory_exists() {
        let tmp = tempfile::tempdir().unwrap();
        let _atlas = Atlas::create_path(tmp.path(), StoreConfig::default()).await.unwrap();
        assert!(tmp.path().join("atlas.json").exists());
    }

    /// Reading array `x` from many datasets must not open files for arrays
    /// `y` and `z` that those datasets also reference. This is the load-bearing
    /// regression test for lazy initialization.
    #[tokio::test]
    async fn reading_one_array_leaves_others_uninitialized() {
        let (store, prefix) = make_store();

        // Seed: two datasets, each defining arrays x, y, z.
        let mut s = Atlas::create(store.clone(), prefix.clone(), StoreConfig::default())
            .await
            .unwrap();
        for ds_name in ["ds_a", "ds_b"] {
            let mut ds = s.create_dataset(ds_name).await.unwrap();
            for arr in ["x", "y", "z"] {
                ds.define_array::<f32>(arr, vec!["i".into()], vec![2], None, None)
                    .await
                    .unwrap();
                let data = ndarray::arr1(&[1.0_f32, 2.0]).into_dyn();
                ds.write_array(arr, vec![0], data.view()).await.unwrap();
            }
        }
        s.flush().await.unwrap();
        drop(s);

        // Reopen — fresh cache, nothing initialized.
        let s = Atlas::open(store, prefix).await.unwrap();
        assert!(
            s.cache.files.read().is_empty(),
            "cache should start empty after open"
        );

        // Read only `x` from both datasets.
        let ds_a = s.open_dataset("ds_a").await.unwrap();
        let ds_b = s.open_dataset("ds_b").await.unwrap();
        let _ = ds_a.read_array::<f32>("x", vec![], vec![]).await.unwrap();
        let _ = ds_b.read_array::<f32>("x", vec![], vec![]).await.unwrap();

        let files = s.cache.files.read();
        assert!(
            files.get("x").is_some_and(|a| a.try_get().is_some()),
            "array `x` must be initialized after read"
        );
        assert!(
            files.get("y").map_or(true, |a| a.try_get().is_none()),
            "array `y` must NOT be initialized — was never read"
        );
        assert!(
            files.get("z").map_or(true, |a| a.try_get().is_none()),
            "array `z` must NOT be initialized — was never read"
        );
    }
}
