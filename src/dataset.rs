use std::{collections::HashMap, sync::Arc};

use array_format::{ArrayElement, ArrayStats, DeltaCache, FillValue};
use ndarray::{ArcArray, ArrayView, IxDyn};
use object_store::ObjectStore;
use parking_lot::{Mutex, RwLock};
use tracing::{debug, instrument, trace};

use crate::{
    Error, Result,
    array::AtlasArray,
    config::Codec,
    meta::{DatasetMeta, StoreMeta},
    schema::{ArraySchema, Attr},
};

/// Shared lazy-handle map: array name → `Arc<AtlasArray>`. Cloned by reference
/// from `Atlas` into every `DatasetView`, so all views observe the same
/// initialization state. The map lock (`parking_lot::RwLock`) is never held
/// across an `await` point; `AtlasArray` defers its actual I/O via
/// `tokio::sync::OnceCell` so each underlying file opens at most once.
pub(crate) struct ArrayCache {
    pub(crate) files: RwLock<HashMap<String, Arc<AtlasArray>>>,
    pub(crate) delta: Arc<DeltaCache>,
}

impl ArrayCache {
    pub(crate) fn new(delta: Arc<DeltaCache>) -> Self {
        Self {
            files: RwLock::new(HashMap::new()),
            delta,
        }
    }

    /// Returns the lazy handle for `array_name`, registering a new one if
    /// absent. Does **not** open or create the underlying file — that happens
    /// on the first `AtlasArray::get().await`.
    pub(crate) fn get_or_insert(
        &self,
        store: &Arc<dyn ObjectStore>,
        array_name: &str,
        codec: &Codec,
    ) -> Arc<AtlasArray> {
        if let Some(arc) = self.files.read().get(array_name) {
            return arc.clone();
        }
        let mut guard = self.files.write();
        guard
            .entry(array_name.to_string())
            .or_insert_with(|| {
                Arc::new(AtlasArray::new(
                    store.clone(),
                    codec.clone(),
                    array_name.to_string(),
                    self.delta.clone(),
                ))
            })
            .clone()
    }
}

/// A borrowed handle to one dataset within an [`Atlas`](crate::Atlas).
///
/// Carries no independent state — every mutation (`define_array`,
/// `write_array`, `set_attribute`, `delete_array`) updates the parent
/// atlas's shared in-memory metadata. Persistence happens when the
/// parent [`Atlas::flush`](crate::Atlas::flush) is called; `DatasetView`
/// has no `flush` of its own.
///
/// `DatasetView` is `Send` and can be moved across tasks, but holding
/// many concurrent views to the same dataset and mutating from each is
/// not protected — the underlying lock is on the shared metadata, not
/// per-view.
pub struct DatasetView {
    store: Arc<dyn ObjectStore>,
    pub(crate) cache: Arc<ArrayCache>,
    name: String,
    /// Shared handle to the parent `Atlas`'s in-memory `StoreMeta`. All
    /// mutations on this view go through here; persistence happens on
    /// `Atlas::flush()`.
    atlas_meta: Arc<Mutex<StoreMeta>>,
    codec: Codec,
}

impl DatasetView {
    pub(crate) fn new(
        store: Arc<dyn ObjectStore>,
        cache: Arc<ArrayCache>,
        name: String,
        atlas_meta: Arc<Mutex<StoreMeta>>,
        codec: Codec,
    ) -> Self {
        Self {
            store,
            cache,
            name,
            atlas_meta,
            codec,
        }
    }

    /// Returns a clone of the metadata for this dataset.
    pub fn meta(&self) -> DatasetMeta {
        self.atlas_meta
            .lock()
            .datasets
            .get(&self.name)
            .cloned()
            .unwrap_or_default()
    }

    /// The dataset name this view points to.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// All array names declared in this dataset, in insertion order.
    /// Reads from the shared in-memory meta — no disk I/O.
    pub fn list_arrays(&self) -> Vec<String> {
        self.atlas_meta
            .lock()
            .datasets
            .get(&self.name)
            .map(|d| d.arrays.keys().cloned().collect())
            .unwrap_or_default()
    }

    /// Set or overwrite a typed attribute on this dataset. Buffered in the
    /// in-memory meta until the parent [`Atlas::flush`](crate::Atlas::flush).
    pub fn set_attribute(&mut self, key: &str, value: Attr) {
        let mut meta = self.atlas_meta.lock();
        meta.datasets
            .entry(self.name.clone())
            .or_default()
            .attributes
            .insert(key.to_string(), value);
    }

    /// Look up an attribute by key. `None` if the key isn't present (or the
    /// dataset has no attributes at all yet).
    pub fn get_attribute(&self, key: &str) -> Option<Attr> {
        self.atlas_meta
            .lock()
            .datasets
            .get(&self.name)
            .and_then(|d| d.attributes.get(key).cloned())
    }

    /// Returns the cached schema for `array`, or `None` if no array with that
    /// name exists in this dataset.
    pub fn array_meta(&self, array: &str) -> Option<ArraySchema> {
        self.atlas_meta
            .lock()
            .datasets
            .get(&self.name)
            .and_then(|d| d.arrays.get(array).cloned())
    }

    /// Returns aggregate statistics for `array` in this dataset, or `None`
    /// if no such array exists or stats haven't been computed yet (stats are
    /// computed on flush).
    pub async fn array_stats(&self, array: &str) -> Option<ArrayStats> {
        let codec = self.array_codec(array)?;
        let handle = self.cache.get_or_insert(&self.store, array, &codec);
        let arc = handle.get().await.ok()?;
        let guard = arc.read().await;
        guard.array_stats(&self.name).cloned()
    }

    /// Declare a new array in this dataset.
    ///
    /// `dims` are named dimensions (one per axis); `shape` is the logical
    /// size per axis. `chunk_shape = None` means one chunk per axis (a
    /// single block per dataset entry — fastest write for small arrays;
    /// pessimal for slice reads on large arrays). `fill_value` is the
    /// scalar returned for unwritten cells; cells equal to it are tallied
    /// as nulls in `array_stats` after [`Atlas::flush`](crate::Atlas::flush).
    ///
    /// Errors with [`Error::ArrayAlreadyExists`] if this dataset already
    /// declares an array with that name, or [`Error::InvalidName`] if
    /// `array` violates the naming rules.
    #[instrument(skip(self, fill_value), fields(dataset = %self.name, dtype = ?T::DTYPE))]
    pub async fn define_array<T: ArrayElement>(
        &mut self,
        array: &str,
        dims: Vec<String>,
        shape: Vec<usize>,
        chunk_shape: Option<Vec<usize>>,
        fill_value: Option<FillValue>,
    ) -> Result<()> {
        crate::validate_name(array)?;
        {
            let meta = self.atlas_meta.lock();
            if let Some(ds) = meta.datasets.get(&self.name) {
                if ds.arrays.contains_key(array) {
                    return Err(Error::ArrayAlreadyExists(array.to_string()));
                }
            }
        }

        let handle = self.cache.get_or_insert(&self.store, array, &self.codec);
        let arc = handle.get().await?;
        arc.write().await.define_array::<T>(
            &self.name,
            dims.clone(),
            shape.clone(),
            chunk_shape.clone(),
            fill_value,
        )?;

        let actual_chunk = chunk_shape.unwrap_or_else(|| shape.clone());
        debug!(?shape, chunk_shape = ?actual_chunk, "defined array");
        let schema = ArraySchema {
            dtype: T::DTYPE.clone(),
            shape,
            chunk_shape: actual_chunk,
            dimension_names: dims,
            codec: self.codec.clone(),
        };
        let mut meta = self.atlas_meta.lock();
        meta.datasets
            .entry(self.name.clone())
            .or_default()
            .arrays
            .insert(array.to_string(), schema);
        Ok(())
    }

    /// Write a slab of values into an array previously declared via
    /// [`define_array`](Self::define_array).
    ///
    /// `start` is the per-axis offset to begin writing at; `data`'s shape
    /// determines the extent. Out-of-bounds writes truncate at the array's
    /// declared shape. The bytes are buffered in the per-array in-memory
    /// layer; nothing reaches disk until [`Atlas::flush`](crate::Atlas::flush).
    ///
    /// Errors with [`Error::ArrayNotFound`] if no array with this name has
    /// been declared.
    #[instrument(skip(self, data), fields(dataset = %self.name, elems = data.len()))]
    pub async fn write_array<T: ArrayElement>(
        &mut self,
        array: &str,
        start: Vec<usize>,
        data: ArrayView<'_, T, IxDyn>,
    ) -> Result<()> {
        let codec = self
            .array_codec(array)
            .ok_or_else(|| Error::ArrayNotFound(array.to_string()))?;
        let handle = self.cache.get_or_insert(&self.store, array, &codec);
        let arc = handle.get().await?;
        let mut guard = arc.write().await;
        let shape: Vec<usize> = data.shape().to_vec();
        let bytes = data.len() * std::mem::size_of::<T>();
        let start_log = start.clone();
        let t0 = std::time::Instant::now();
        guard.write_array::<T>(&self.name, start, data).await?;
        debug!(
            array,
            start = ?start_log,
            ?shape,
            bytes,
            elapsed_us = t0.elapsed().as_micros() as u64,
            "wrote chunk"
        );
        Ok(())
    }

    /// Read a full or partial array from this dataset.
    ///
    /// Empty `start` + empty `shape` reads the full array. Otherwise both
    /// must have one entry per dimension; only chunks overlapping the
    /// requested region are decompressed.
    ///
    /// Returns `Ok(None)` if this dataset doesn't declare an array with
    /// that name.
    ///
    /// # Examples
    ///
    /// ```
    /// # tokio::runtime::Runtime::new().unwrap().block_on(async {
    /// use atlas::{Atlas, StoreConfig};
    /// use ndarray::Array2;
    /// let tmp = tempfile::tempdir().unwrap();
    /// let mut s = Atlas::create_path(tmp.path(), StoreConfig::default()).await.unwrap();
    /// {
    ///     let mut ds = s.create_dataset("ds").await.unwrap();
    ///     ds.define_array::<f32>("temp", vec!["x".into(), "y".into()],
    ///                            vec![4, 8], None, None).await.unwrap();
    ///     let data = Array2::<f32>::from_elem([4, 8], 9.0).into_dyn();
    ///     ds.write_array("temp", vec![0, 0], data.view()).await.unwrap();
    ///
    ///     // Full read.
    ///     let full = ds.read_array::<f32>("temp", vec![], vec![]).await.unwrap().unwrap();
    ///     assert_eq!(full.shape(), &[4, 8]);
    ///
    ///     // Partial read — a 2×4 sub-region.
    ///     let part = ds.read_array::<f32>("temp", vec![1, 2], vec![2, 4]).await.unwrap().unwrap();
    ///     assert_eq!(part.shape(), &[2, 4]);
    /// }
    /// s.flush().await.unwrap();
    /// # });
    /// ```
    #[instrument(skip(self), fields(dataset = %self.name))]
    pub async fn read_array<T: ArrayElement>(
        &self,
        array: &str,
        start: Vec<usize>,
        shape: Vec<usize>,
    ) -> Result<Option<ArcArray<T, IxDyn>>> {
        let codec = match self.array_codec(array) {
            Some(c) => c,
            None => {
                debug!("array not present in dataset");
                return Ok(None);
            }
        };
        trace!(?start, ?shape, "reading array");
        let handle = self.cache.get_or_insert(&self.store, array, &codec);
        let arc = handle.get().await?;
        let guard = arc.read().await;
        Ok(Some(guard.read_array::<T>(&self.name, start, shape).await?))
    }

    /// Returns the fill value passed to `define_array` for `array`, or `None`
    /// if the array isn't present in this dataset or was defined without one.
    pub async fn array_fill_value(&self, array: &str) -> Result<Option<FillValue>> {
        let codec = match self.array_codec(array) {
            Some(c) => c,
            None => return Ok(None),
        };
        let handle = self.cache.get_or_insert(&self.store, array, &codec);
        let arc = handle.get().await?;
        let guard = arc.read().await;
        Ok(guard.get_array(&self.name)?.fill_value.clone())
    }

    /// Remove an array from this dataset. Tombstones the dataset's entry
    /// inside the shared array file; persistence happens on the next
    /// [`Atlas::flush`](crate::Atlas::flush). Errors with
    /// [`Error::ArrayNotFound`] if no array with that name is declared here.
    #[instrument(skip(self), fields(dataset = %self.name))]
    pub async fn delete_array(&mut self, array: &str) -> Result<()> {
        let codec = self
            .array_codec(array)
            .ok_or_else(|| Error::ArrayNotFound(array.to_string()))?;
        let handle = self.cache.get_or_insert(&self.store, array, &codec);
        let arc = handle.get().await?;
        arc.write().await.delete(&self.name)?;
        let mut meta = self.atlas_meta.lock();
        if let Some(ds_meta) = meta.datasets.get_mut(&self.name) {
            ds_meta.arrays.shift_remove(array);
        }
        debug!("deleted array");
        Ok(())
    }

    /// Looks up the per-array codec from `atlas_meta`. Returns `None` if the
    /// array isn't defined in this dataset.
    fn array_codec(&self, array: &str) -> Option<Codec> {
        self.atlas_meta
            .lock()
            .datasets
            .get(&self.name)
            .and_then(|d| d.arrays.get(array).map(|s| s.codec.clone()))
    }
}

pub(crate) async fn open_dataset_view(
    store: Arc<dyn ObjectStore>,
    cache: Arc<ArrayCache>,
    atlas_meta: Arc<Mutex<StoreMeta>>,
    name: &str,
    codec: Codec,
) -> Result<DatasetView> {
    {
        let meta = atlas_meta.lock();
        if !meta.datasets.contains_key(name) {
            return Err(Error::DatasetNotFound(name.to_string()));
        }
    }
    Ok(DatasetView::new(
        store,
        cache,
        name.to_string(),
        atlas_meta,
        codec,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use object_store::memory::InMemory;

    fn make_store() -> Arc<dyn ObjectStore> {
        Arc::new(InMemory::new())
    }

    fn shared_meta_with(name: &str) -> Arc<Mutex<StoreMeta>> {
        let mut meta = StoreMeta::default();
        meta.datasets
            .insert(name.to_string(), DatasetMeta::default());
        Arc::new(Mutex::new(meta))
    }

    fn test_cache() -> Arc<ArrayCache> {
        Arc::new(ArrayCache::new(Arc::new(DeltaCache::new(
            256 * 1024 * 1024,
            64 * 1024 * 1024,
        ))))
    }

    fn empty_view(store: Arc<dyn ObjectStore>, name: &str) -> DatasetView {
        DatasetView::new(
            store,
            test_cache(),
            name.to_string(),
            shared_meta_with(name),
            Codec::default(),
        )
    }

    // --- attribute tests (synchronous, no I/O) ---

    #[test]
    fn get_attribute_missing_returns_none() {
        let view = empty_view(make_store(), "ds");
        assert!(view.get_attribute("x").is_none());
    }

    #[test]
    fn set_and_get_attribute_roundtrip() {
        let mut view = empty_view(make_store(), "ds");
        view.set_attribute("k", Attr::Int64(42));
        assert_eq!(view.get_attribute("k"), Some(Attr::Int64(42)));
    }

    #[test]
    fn set_attribute_overwrites_previous() {
        let mut view = empty_view(make_store(), "ds");
        view.set_attribute("k", Attr::Int64(1));
        view.set_attribute("k", Attr::Int64(2));
        assert_eq!(view.get_attribute("k"), Some(Attr::Int64(2)));
    }

    #[test]
    fn name_returns_dataset_name() {
        let view = empty_view(make_store(), "my_dataset");
        assert_eq!(view.name(), "my_dataset");
    }

    #[test]
    fn list_arrays_empty_when_no_arrays_defined() {
        let view = empty_view(make_store(), "ds");
        assert!(view.list_arrays().is_empty());
    }

    // --- array lookup without I/O ---

    #[tokio::test]
    async fn read_array_returns_none_for_unknown_array() {
        let view = empty_view(make_store(), "ds");
        let result = view
            .read_array::<f32>("missing", vec![], vec![])
            .await
            .unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn array_meta_returns_none_for_unknown_array() {
        let view = empty_view(make_store(), "ds");
        assert!(view.array_meta("missing").is_none());
    }

    // --- define_array behaviour ---

    #[tokio::test]
    async fn define_array_appears_in_list() {
        let mut view = empty_view(make_store(), "ds");
        view.define_array::<f32>("arr", vec!["x".into()], vec![4], None, None)
            .await
            .unwrap();
        assert_eq!(view.list_arrays(), vec!["arr"]);
    }

    #[tokio::test]
    async fn define_duplicate_array_rejected() {
        let mut view = empty_view(make_store(), "ds");
        view.define_array::<f32>("arr", vec!["x".into()], vec![4], None, None)
            .await
            .unwrap();
        let err = view
            .define_array::<f32>("arr", vec!["x".into()], vec![4], None, None)
            .await
            .unwrap_err();
        assert!(matches!(err, crate::Error::ArrayAlreadyExists(_)));
    }

    #[tokio::test]
    async fn define_array_invalid_name_rejected() {
        let mut view = empty_view(make_store(), "ds");
        let err = view
            .define_array::<f32>("a/b", vec!["x".into()], vec![4], None, None)
            .await
            .unwrap_err();
        assert!(matches!(err, crate::Error::InvalidName(_)));
    }

    // --- write / read roundtrip ---

    #[tokio::test]
    async fn write_then_read_returns_data() {
        use ndarray::ArrayD;
        let mut view = empty_view(make_store(), "ds");
        view.define_array::<f32>("arr", vec!["x".into()], vec![4], None, None)
            .await
            .unwrap();
        let data = ArrayD::<f32>::from_elem(vec![4], 7.0_f32);
        view.write_array("arr", vec![0], data.view()).await.unwrap();
        let result = view
            .read_array::<f32>("arr", vec![], vec![])
            .await
            .unwrap()
            .unwrap();
        assert_eq!(result, data.into_shared());
    }

    // --- delete_array ---

    #[tokio::test]
    async fn delete_array_removes_from_list() {
        let mut view = empty_view(make_store(), "ds");
        view.define_array::<f32>("arr", vec!["x".into()], vec![4], None, None)
            .await
            .unwrap();
        view.delete_array("arr").await.unwrap();
        assert!(view.list_arrays().is_empty());
    }

    #[tokio::test]
    async fn delete_nonexistent_array_errors() {
        let mut view = empty_view(make_store(), "ds");
        let err = view.delete_array("ghost").await.unwrap_err();
        assert!(matches!(err, crate::Error::ArrayNotFound(_)));
    }

    // --- meta ---

    #[tokio::test]
    async fn define_array_records_meta() {
        use array_format::DType;
        let mut view = empty_view(make_store(), "ds");
        view.define_array::<f32>(
            "arr",
            vec!["x".into(), "y".into()],
            vec![4, 8],
            Some(vec![2, 2]),
            None,
        )
        .await
        .unwrap();

        let meta = view.meta();
        let arr_schema = meta.arrays.get("arr").expect("meta entry missing");
        assert_eq!(arr_schema.dtype, DType::Float32);
        assert_eq!(arr_schema.shape, vec![4, 8]);
        assert_eq!(arr_schema.chunk_shape, vec![2, 2]);
        assert_eq!(arr_schema.dimension_names, vec!["x", "y"]);
        assert!(meta.attributes.is_empty());
    }

    #[tokio::test]
    async fn define_array_default_chunk_equals_shape() {
        use array_format::DType;
        let mut view = empty_view(make_store(), "ds");
        view.define_array::<i32>("arr", vec!["t".into()], vec![10], None, None)
            .await
            .unwrap();

        let meta = view.meta();
        let arr_schema = meta.arrays.get("arr").unwrap();
        assert_eq!(arr_schema.dtype, DType::Int32);
        assert_eq!(arr_schema.chunk_shape, vec![10]);
    }

    #[test]
    fn set_attribute_records_value_in_meta() {
        let mut view = empty_view(make_store(), "ds");
        view.set_attribute("count", Attr::Int64(5));
        view.set_attribute("label", Attr::String("x".into()));

        let meta = view.meta();
        assert_eq!(meta.attributes.get("count"), Some(&Attr::Int64(5)));
        assert_eq!(
            meta.attributes.get("label"),
            Some(&Attr::String("x".into()))
        );
    }

    #[tokio::test]
    async fn delete_array_removes_meta_entry() {
        let mut view = empty_view(make_store(), "ds");
        view.define_array::<f64>("arr", vec!["x".into()], vec![4], None, None)
            .await
            .unwrap();
        assert!(view.meta().arrays.contains_key("arr"));
        view.delete_array("arr").await.unwrap();
        assert!(!view.meta().arrays.contains_key("arr"));
    }

    #[tokio::test]
    async fn array_meta_returns_schema_after_define() {
        use array_format::DType;
        let mut view = empty_view(make_store(), "ds");
        view.define_array::<f64>("arr", vec!["t".into()], vec![5], None, None)
            .await
            .unwrap();
        let meta = view.array_meta("arr").unwrap();
        assert_eq!(meta.dtype, DType::Float64);
        assert_eq!(meta.shape, vec![5]);
    }

    // --- array_stats ---

    #[tokio::test]
    async fn array_stats_returns_none_for_unknown_array() {
        let view = empty_view(make_store(), "ds");
        assert!(view.array_stats("ghost").await.is_none());
    }

    #[tokio::test]
    async fn array_stats_none_before_flush() {
        let mut view = empty_view(make_store(), "ds");
        view.define_array::<f32>("arr", vec!["x".into()], vec![4], None, None)
            .await
            .unwrap();
        assert!(view.array_stats("arr").await.is_none());
    }

    /// Flush every initialized array file in the shared cache. Used by tests
    /// that need to persist stats without going through `Atlas::flush`.
    async fn flush_initialized(cache: &Arc<ArrayCache>) {
        let snapshot: Vec<_> = {
            let guard = cache.files.read();
            guard
                .values()
                .filter_map(|a| a.try_get().map(|arc| (a.clone(), arc)))
                .collect()
        };
        for (_handle, arc) in snapshot {
            arc.write().await.flush().await.unwrap();
        }
    }

    #[tokio::test]
    async fn array_stats_populated_after_flush() {
        use array_format::StatValue;
        let store = make_store();
        let mut view = empty_view(store.clone(), "ds");
        view.define_array::<f32>("arr", vec!["x".into()], vec![4], None, None)
            .await
            .unwrap();
        let data = ndarray::arr1(&[1.0_f32, 3.0, 2.0, 4.0]).into_dyn();
        view.write_array("arr", vec![0], data.view()).await.unwrap();
        flush_initialized(&view.cache).await;

        let stats = view.array_stats("arr").await.unwrap();
        assert_eq!(stats.row_count, 4);
        assert_eq!(stats.null_count, 0);
        assert_eq!(stats.min, Some(StatValue::Float(1.0)));
        assert_eq!(stats.max, Some(StatValue::Float(4.0)));
    }

    #[tokio::test]
    async fn array_stats_count_fill_value_as_null() {
        use array_format::{FillValue, StatValue};
        let store = make_store();
        let mut view = empty_view(store.clone(), "ds");
        view.define_array::<i32>(
            "arr",
            vec!["x".into()],
            vec![6],
            None,
            Some(FillValue::Int(-1)),
        )
        .await
        .unwrap();
        // Two cells equal the fill (-1); four are real data.
        let data = ndarray::arr1(&[5_i32, -1, 7, -1, 2, 9]).into_dyn();
        view.write_array("arr", vec![0], data.view()).await.unwrap();
        flush_initialized(&view.cache).await;

        let stats = view.array_stats("arr").await.unwrap();
        assert_eq!(stats.row_count, 6);
        assert_eq!(
            stats.null_count, 2,
            "two fill-equal cells must count as null"
        );
        // min/max exclude fill-valued cells.
        assert_eq!(stats.min, Some(StatValue::Int(2)));
        assert_eq!(stats.max, Some(StatValue::Int(9)));
    }

    #[tokio::test]
    async fn array_stats_without_fill_value_treats_sentinel_as_data() {
        use array_format::StatValue;
        // Baseline: same `-1` values but no fill_value declared — they must
        // not count as null, and must be included in min/max.
        let store = make_store();
        let mut view = empty_view(store.clone(), "ds");
        view.define_array::<i32>("arr", vec!["x".into()], vec![4], None, None)
            .await
            .unwrap();
        let data = ndarray::arr1(&[5_i32, -1, 7, 9]).into_dyn();
        view.write_array("arr", vec![0], data.view()).await.unwrap();
        flush_initialized(&view.cache).await;

        let stats = view.array_stats("arr").await.unwrap();
        assert_eq!(stats.row_count, 4);
        assert_eq!(stats.null_count, 0);
        assert_eq!(stats.min, Some(StatValue::Int(-1)));
        assert_eq!(stats.max, Some(StatValue::Int(9)));
    }

    #[tokio::test]
    async fn array_stats_nan_fill_value_for_float() {
        use array_format::{FillValue, StatValue};
        let store = make_store();
        let mut view = empty_view(store.clone(), "ds");
        view.define_array::<f64>(
            "arr",
            vec!["x".into()],
            vec![4],
            None,
            Some(FillValue::Float(f64::NAN)),
        )
        .await
        .unwrap();
        // NaN cells are matched to the NaN fill (bit-pattern compare in array_format).
        let data = ndarray::arr1(&[1.0_f64, f64::NAN, 3.0, f64::NAN]).into_dyn();
        view.write_array("arr", vec![0], data.view()).await.unwrap();
        flush_initialized(&view.cache).await;

        let stats = view.array_stats("arr").await.unwrap();
        assert_eq!(stats.row_count, 4);
        assert_eq!(stats.null_count, 2);
        assert_eq!(stats.min, Some(StatValue::Float(1.0)));
        assert_eq!(stats.max, Some(StatValue::Float(3.0)));
    }

    // --- cache sharing ---

    #[tokio::test]
    async fn two_views_share_cached_array_file() {
        let store = make_store();
        let cache = test_cache();
        let shared = Arc::new(Mutex::new({
            let mut m = StoreMeta::default();
            m.datasets.insert("ds_a".into(), DatasetMeta::default());
            m.datasets.insert("ds_b".into(), DatasetMeta::default());
            m
        }));

        let mut view_a = DatasetView::new(
            store.clone(),
            cache.clone(),
            "ds_a".to_string(),
            shared.clone(),
            Codec::default(),
        );
        view_a
            .define_array::<f32>("arr", vec!["x".into()], vec![2], None, None)
            .await
            .unwrap();

        let mut view_b = DatasetView::new(
            store.clone(),
            cache.clone(),
            "ds_b".to_string(),
            shared.clone(),
            Codec::default(),
        );
        view_b
            .define_array::<f32>("arr", vec!["x".into()], vec![2], None, None)
            .await
            .unwrap();

        // Both views share the same lazy handle from the global cache.
        let handle_a = view_a.cache.files.read().get("arr").unwrap().clone();
        let handle_b = view_b.cache.files.read().get("arr").unwrap().clone();
        assert!(
            Arc::ptr_eq(&handle_a, &handle_b),
            "expected both views to share the same AtlasArray handle"
        );
    }
}
