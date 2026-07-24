use std::sync::Arc;

use array_format::DeltaCache;
use object_store::{ObjectStore, local::LocalFileSystem, path::Path, prefix::PrefixStore};
use parking_lot::Mutex;
use tracing::{debug, info, instrument};

use crate::{
    Error, Result,
    config::{Codec, MetaFormat, StoreConfig, TypeMismatchPolicy},
    dataset::{ArrayCache, DatasetView, GLOBAL_ATTRS_ARRAY, PendingAttrs, open_dataset_view},
    meta::{StoreMeta, load_meta, save_meta},
    pruning::{ColumnKey, ColumnSummary, PruningIndex, StatColumn, StatVal},
};
use std::collections::HashMap;

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
    /// Cached `dataset name → ordinal` map for the on-demand pruning-index
    /// pivot. Rebuilt lazily and invalidated whenever the dataset set changes,
    /// so a query doesn't pay to rebuild it (~O(datasets)) on every call.
    ordinal_map: Arc<Mutex<Option<Arc<HashMap<String, usize>>>>>,
    cache: Arc<ArrayCache>,
    codec: Codec,
    /// How type mismatches across datasets are reported. Per-session, not
    /// persisted to `atlas.json`.
    on_type_mismatch: TypeMismatchPolicy,
    meta_format: MetaFormat,
    meta_compression: Codec,
}

mod bulk_read;
mod durability;

impl Atlas {
    /// Open an existing store at `prefix` within `store`.
    ///
    /// Reads `atlas.json` exactly once. Subsequent mutations only touch the
    /// in-memory meta until [`Atlas::flush`] is called.
    #[instrument(skip(store), fields(prefix = %prefix))]
    pub async fn open(store: Arc<dyn ObjectStore>, prefix: Path) -> Result<Self> {
        Self::open_with_config(store, prefix, StoreConfig::default()).await
    }

    /// Like [`Atlas::open`], but lets you choose the per-session
    /// [`TypeMismatchPolicy`](crate::TypeMismatchPolicy) via
    /// [`StoreConfig::on_type_mismatch`].
    ///
    /// Only `on_type_mismatch` is honoured here: the array codec and the
    /// metadata format/compression are detected from the files on disk, so
    /// those fields of `config` are ignored when opening.
    #[instrument(skip(store, config), fields(prefix = %prefix, on_type_mismatch = ?config.on_type_mismatch))]
    pub async fn open_with_config(
        store: Arc<dyn ObjectStore>,
        prefix: Path,
        config: StoreConfig,
    ) -> Result<Self> {
        let store = prefixed(store, prefix);
        let (meta, meta_format, meta_compression) = load_meta(&store).await?;
        let codec = meta.codec;
        info!(
            datasets = meta.live_count(),
            ?codec,
            ?meta_format,
            ?meta_compression,
            "opened atlas store"
        );
        Ok(Self {
            store,
            meta: Arc::new(Mutex::new(meta)),
            pending_attrs: Arc::new(Mutex::new(PendingAttrs::default())),
            ordinal_map: Arc::new(Mutex::new(None)),
            cache: default_cache(),
            codec,
            on_type_mismatch: config.on_type_mismatch,
            meta_format,
            meta_compression,
        })
    }

    /// Create a new store at `prefix` within `store`.
    #[instrument(skip(store, config), fields(prefix = %prefix, codec = ?config.codec, meta_format = ?config.meta_format, meta_compression = ?config.meta_compression))]
    pub async fn create(store: Arc<dyn ObjectStore>, prefix: Path, config: StoreConfig) -> Result<Self> {
        let store = prefixed(store, prefix);
        let meta = StoreMeta::new(config.codec);
        save_meta(&store, &meta, config.meta_format, config.meta_compression).await?;
        info!("created atlas store");
        Ok(Self {
            store,
            meta: Arc::new(Mutex::new(meta)),
            pending_attrs: Arc::new(Mutex::new(PendingAttrs::default())),
            ordinal_map: Arc::new(Mutex::new(None)),
            cache: default_cache(),
            codec: config.codec,
            on_type_mismatch: config.on_type_mismatch,
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
        Self::open_path_with_config(path, StoreConfig::default()).await
    }

    /// Like [`Atlas::open_path`], but lets you choose the per-session
    /// [`TypeMismatchPolicy`](crate::TypeMismatchPolicy). Only
    /// `config.on_type_mismatch` is honoured; the codec and metadata
    /// format/compression are detected from disk.
    pub async fn open_path_with_config(
        path: impl AsRef<std::path::Path>,
        config: StoreConfig,
    ) -> Result<Self> {
        let store = Arc::new(LocalFileSystem::new_with_prefix(path.as_ref())?);
        Self::open_with_config(store, Path::from(""), config).await
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
            if meta.is_live(name) {
                return Err(Error::DatasetAlreadyExists(name.to_string()));
            }
            // Reuses a previously-tombstoned slot for this name so ordinals stay
            // stable (the pruning index reads current stats, so nothing to reset).
            meta.add_dataset(name);
        }
        *self.ordinal_map.lock() = None; // dataset set changed
        debug!("created dataset");
        Ok(DatasetView::new(
            self.store.clone(),
            self.cache.clone(),
            name.to_string(),
            self.meta.clone(),
            self.pending_attrs.clone(),
            self.codec,
            self.on_type_mismatch,
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
            self.codec,
            self.on_type_mismatch,
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
        *self.ordinal_map.lock() = None; // dataset set changed
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
        meta.live_names()
    }

    /// `true` if a dataset with this name is registered. O(1) hash lookup in
    /// the in-memory store metadata.
    pub fn dataset_exists(&self, name: &str) -> bool {
        let meta = self.meta.lock();
        meta.is_live(name)
    }

    /// This dataset's **row ordinal**: its fixed position in the collection,
    /// and its row in the pruning index. `None` if the dataset doesn't exist.
    ///
    /// Ordinals are assigned on creation and are stable across deletions — a
    /// deleted dataset keeps its slot (masked) so no other dataset's row moves.
    /// [`compact`](Self::compact) is the only operation that renumbers, and it
    /// invalidates any ordinal held from before the call.
    pub fn dataset_row(&self, name: &str) -> Option<usize> {
        self.meta.lock().live_ordinal(name)
    }

    /// Total row slots, live and tombstoned — the pruning index's row count.
    /// Larger than the number of live datasets until the next
    /// [`compact`](Self::compact).
    pub fn row_slots(&self) -> usize {
        self.meta.lock().row_slots()
    }

    /// Assembles the pruning index for **only** the given columns, **on demand**
    /// from the array files' own statistics — there is no persisted index.
    ///
    /// Each array column costs one `StatsFile` read of that array's `.af` file,
    /// pivoted into a flat length-N column by dataset ordinal; attribute columns
    /// cost nothing beyond the in-memory schema. The result is a flat, columnar
    /// table — every column has one row per dataset (matching
    /// [`dataset_row`](Self::dataset_row)) — self-describing via its liveness
    /// mask (so [`PruningIndex::view`] hides deleted rows) and row↔name mapping.
    /// Datasets written but not yet flushed have no committed stats and read
    /// back as absent.
    ///
    /// Columns are built in parallel (each reads an independent array file), and
    /// the `name → ordinal` pivot map is cached across calls, so the per-call
    /// cost is dominated by the requested columns' stats — not the collection.
    pub async fn pruning_index(&self, columns: &[ColumnKey]) -> Result<PruningIndex> {
        let name_to_row = self.ordinal_map();
        let (rows, live, names) = {
            let meta = self.meta.lock();
            (meta.row_slots(), meta.live_mask(), meta.names_by_row())
        };

        // Dedup while preserving request order.
        let mut seen = std::collections::HashSet::new();
        let unique: Vec<ColumnKey> =
            columns.iter().filter(|k| seen.insert((*k).clone())).cloned().collect();

        let mut index = PruningIndex::with_rows(rows);
        index.set_live(live);
        index.set_dataset_names(names);
        for (key, column) in self.build_columns(unique, rows, name_to_row).await? {
            index.insert_column(key, column);
        }
        Ok(index)
    }

    /// The cached `dataset name → ordinal` map, rebuilt from `meta` on first use
    /// after any change to the dataset set (create / delete / compact).
    fn ordinal_map(&self) -> Arc<HashMap<String, usize>> {
        if let Some(map) = self.ordinal_map.lock().as_ref() {
            return map.clone();
        }
        let map = {
            let meta = self.meta.lock();
            let mut m = HashMap::with_capacity(meta.live_count());
            for (ordinal, name, _) in meta.live_datasets() {
                m.insert(name.clone(), ordinal);
            }
            m
        };
        let arc = Arc::new(map);
        *self.ordinal_map.lock() = Some(arc.clone());
        arc
    }

    /// Builds the given columns concurrently — each reads an independent array
    /// file, so they run as separate tasks (bounded to the CPU count so a wide
    /// request can't open too many files at once).
    async fn build_columns(
        &self,
        keys: Vec<ColumnKey>,
        rows: usize,
        name_to_row: Arc<HashMap<String, usize>>,
    ) -> Result<Vec<(ColumnKey, StatColumn)>> {
        let limit = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
        let sem = Arc::new(tokio::sync::Semaphore::new(limit));
        let mut tasks = Vec::with_capacity(keys.len());
        for key in keys {
            let permit = sem
                .clone()
                .acquire_owned()
                .await
                .map_err(|e| Error::Internal(format!("pruning semaphore closed: {e}")))?;
            let store = self.store.clone();
            let cache = self.cache.clone();
            let meta = self.meta.clone();
            let codec = self.codec;
            let n2r = name_to_row.clone();
            tasks.push(tokio::spawn(async move {
                let _permit = permit;
                let column = build_column(&store, &cache, &meta, codec, &key, rows, &n2r).await?;
                Ok::<_, Error>((key, column))
            }));
        }
        let mut out = Vec::with_capacity(tasks.len());
        for task in tasks {
            out.push(
                task.await
                    .map_err(|e| Error::Internal(format!("pruning column task failed: {e}")))??,
            );
        }
        Ok(out)
    }

    /// Every column's collection-wide min/max and present count, folded from the
    /// array files' statistics.
    ///
    /// Unlike [`pruning_index`](Self::pruning_index) this touches *every* column,
    /// so its cost scales with the whole collection — use it to decide which
    /// columns are worth a full read (see [`ColumnSummary::might_match`]), not on
    /// a hot path.
    pub async fn column_summaries(&self) -> Result<Vec<(ColumnKey, ColumnSummary)>> {
        let name_to_row = self.ordinal_map();
        let (keys, live) = {
            let meta = self.meta.lock();
            let mut keys: Vec<ColumnKey> = Vec::new();
            let mut seen = std::collections::HashSet::new();
            let mut push = |k: ColumnKey| {
                if seen.insert(k.clone()) {
                    keys.push(k);
                }
            };
            for (_, _, schema) in meta.live_datasets() {
                for array in schema.arrays.keys() {
                    push(ColumnKey::array(array));
                }
                for a in schema.global_attrs.keys() {
                    push(ColumnKey::global_attr(a));
                }
                for (array, attrs) in &schema.array_attrs {
                    for a in attrs.keys() {
                        push(ColumnKey::array_attr(array, a));
                    }
                }
            }
            (keys, meta.live_mask())
        };

        let rows = live.len();
        let built = self.build_columns(keys, rows, name_to_row).await?;
        Ok(built
            .into_iter()
            .map(|(key, column)| {
                let summary = column.summarize(&live);
                (key, summary)
            })
            .collect())
    }

    /// Distinct array names across all datasets in this store, sorted.
    /// One entry per physical `.af` file — datasets sharing an array name
    /// (the common case) collapse to a single entry here.
    pub fn list_arrays(&self) -> Vec<String> {
        let meta = self.meta.lock();
        let mut arrays: Vec<String> = meta
            .live_schemas()
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
        self.meta.lock().merged_schema()
    }

    /// Returns the dtype of `array` if any dataset in this store declares it.
    /// Used by `read_array_across`'s Python binding to pick the generic
    /// instantiation without round-tripping through a `DatasetView`.
    pub fn array_dtype(&self, array: &str) -> Option<array_format::DType> {
        let meta = self.meta.lock();
        meta.live_schemas()
            .find_map(|d| d.arrays.get(array))
            .map(|schema| schema.dtype.clone())
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

/// Builds one flat pruning column of length `rows` from the array files. A free
/// function (owns only `Arc`s and refs) so it can run as an independent task —
/// see [`Atlas::build_columns`].
async fn build_column(
    store: &Arc<dyn ObjectStore>,
    cache: &Arc<ArrayCache>,
    meta: &Arc<Mutex<StoreMeta>>,
    default_codec: Codec,
    key: &ColumnKey,
    rows: usize,
    name_to_row: &HashMap<String, usize>,
) -> Result<StatColumn> {
    let mut column = StatColumn::new(rows);
    match key {
        // Array data stats: read the array's `.af` StatsFile and scatter each
        // entry into its dataset's ordinal row.
        ColumnKey::Array(array) => {
            let codec = meta.lock().array_file_codec(array).unwrap_or(default_codec);
            let handle = cache.get_or_insert(store, array, &codec);
            if let Some(arc) = handle.get_existing().await? {
                let guard = arc.read().await;
                if let Some(stats) = guard.stats() {
                    for entry in stats.entries() {
                        if let Some(&row) = name_to_row.get(&entry.name) {
                            column.set_stats(row, entry);
                        }
                    }
                }
            }
        }
        // Attribute columns carry one value per dataset: dataset-global
        // attributes live in the reserved `_global` file, per-array ones in the
        // array's own. Scalars become a point range `[value, value]` (so a
        // caller can range-prune on them); list-valued attributes mark presence.
        ColumnKey::GlobalAttr(k) => {
            fill_attr_column(&mut column, store, cache, default_codec, GLOBAL_ATTRS_ARRAY, k, name_to_row)
                .await?;
        }
        ColumnKey::ArrayAttr(array, k) => {
            let codec = meta.lock().array_file_codec(array).unwrap_or(default_codec);
            fill_attr_column(&mut column, store, cache, codec, array, k, name_to_row).await?;
        }
    }
    Ok(column)
}

/// Fills `column` from attribute `key` on the array file `file`, one value per
/// dataset. `attribute_index` returns every dataset's value for the key in one
/// read; scalars become a point range, lists mark presence only.
async fn fill_attr_column(
    column: &mut StatColumn,
    store: &Arc<dyn ObjectStore>,
    cache: &Arc<ArrayCache>,
    codec: Codec,
    file: &str,
    key: &str,
    name_to_row: &HashMap<String, usize>,
) -> Result<()> {
    let handle = cache.get_or_insert(store, file, &codec);
    let Some(arc) = handle.get_existing().await? else {
        return Ok(());
    };
    let guard = arc.read().await;
    for (name, value) in guard.attribute_index(key) {
        let Some(&row) = name_to_row.get(&name) else { continue };
        match value {
            Some(v) => match StatVal::scalar_from_attribute(v) {
                Some(scalar) => column.set_scalar(row, scalar),
                None => column.mark_present(row),
            },
            None => {} // this dataset doesn't carry the attribute
        }
    }
    Ok(())
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
            files.get("y").is_none_or(|a| a.try_get().is_none()),
            "array `y` must NOT be initialized — was never read"
        );
        assert!(
            files.get("z").is_none_or(|a| a.try_get().is_none()),
            "array `z` must NOT be initialized — was never read"
        );
    }
}
