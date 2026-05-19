use std::{collections::HashMap, sync::Arc};

use array_format::{ArrayElement, ArrayFile, FileConfig, FillValue, ZstdCodec};
use ndarray::{ArcArray, ArrayView, IxDyn};
use object_store::{ObjectStore, ObjectStoreExt, path::Path as OsPath};
use parking_lot::RwLock;
use tokio::sync::Mutex;

use crate::{
    Error, Result,
    meta::{DatasetMeta, StoreMeta, load_meta, save_meta},
    schema::{Attr, ArraySchema},
};

/// Per-file lock: `tokio::sync::Mutex` is async-aware so the guard may safely
/// be held across await points without risk of deadlock.
pub(crate) type CachedFile = Arc<Mutex<ArrayFile>>;
/// Shared cache: array_name → open ArrayFile.
/// The cache *map* lock is `parking_lot::RwLock` and is never held across an await.
pub(crate) type ArrayCache = RwLock<HashMap<String, CachedFile>>;

pub struct DatasetView {
    store: Arc<dyn ObjectStore>,
    cache: Arc<ArrayCache>,
    name: String,
    arrays: HashMap<String, CachedFile>,
    meta: DatasetMeta,
}

impl DatasetView {
    pub(crate) fn new(
        store: Arc<dyn ObjectStore>,
        cache: Arc<ArrayCache>,
        name: String,
        arrays: HashMap<String, CachedFile>,
        meta: DatasetMeta,
    ) -> Self {
        Self { store, cache, name, arrays, meta }
    }

    /// Returns the metadata for this dataset: array schemas and per-dataset attributes.
    pub fn meta(&self) -> &DatasetMeta {
        &self.meta
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn list_arrays(&self) -> Vec<&str> {
        self.arrays.keys().map(|s| s.as_str()).collect()
    }

    pub fn set_attribute(&mut self, key: &str, value: Attr) {
        self.meta.attributes.insert(key.to_string(), value);
    }

    pub fn get_attribute(&self, key: &str) -> Option<&Attr> {
        self.meta.attributes.get(key)
    }

    /// Returns the cached schema for `array` (dtype, shape, chunk shape, dimension names).
    pub fn array_meta(&self, array: &str) -> Result<ArraySchema> {
        self.meta
            .arrays
            .get(array)
            .cloned()
            .ok_or_else(|| Error::ArrayNotFound(array.to_string()))
    }

    pub async fn define_array<T: ArrayElement>(
        &mut self,
        array: &str,
        dims: Vec<String>,
        shape: Vec<usize>,
        chunk_shape: Option<Vec<usize>>,
        fill_value: Option<FillValue>,
    ) -> Result<()> {
        crate::validate_name(array)?;
        if self.arrays.contains_key(array) {
            return Err(Error::ArrayAlreadyExists(array.to_string()));
        }

        let arc = get_or_open_cached(&self.store, &self.cache, array).await?;
        arc.lock()
            .await
            .define_array::<T>(&self.name, dims.clone(), shape.clone(), chunk_shape.clone(), fill_value)?;
        self.arrays.insert(array.to_string(), arc);

        let actual_chunk = chunk_shape.unwrap_or_else(|| shape.clone());
        self.meta.arrays.insert(array.to_string(), ArraySchema {
            dtype: T::DTYPE.clone(),
            shape,
            chunk_shape: actual_chunk,
            dimension_names: dims,
        });
        Ok(())
    }

    pub async fn write_array<T: ArrayElement>(
        &mut self,
        array: &str,
        start: Vec<usize>,
        data: ArrayView<'_, T, IxDyn>,
    ) -> Result<()> {
        let arc = self
            .arrays
            .get(array)
            .ok_or_else(|| Error::ArrayNotFound(array.to_string()))?
            .clone();
        let mut guard = arc.lock().await;
        guard.write_array::<T>(&self.name, start, data).await?;
        Ok(())
    }

    /// Returns `Ok(None)` if this dataset has no array with that name.
    /// Returns `Err` only for I/O or format errors.
    pub async fn read_array<T: ArrayElement>(
        &self,
        array: &str,
        start: Vec<usize>,
        shape: Vec<usize>,
    ) -> Result<Option<ArcArray<T, IxDyn>>> {
        let arc = match self.arrays.get(array) {
            Some(arc) => arc.clone(),
            None => return Ok(None),
        };
        let guard = arc.lock().await;
        Ok(Some(guard.read_array::<T>(&self.name, start, shape).await?))
    }

    pub async fn delete_array(&mut self, array: &str) -> Result<()> {
        let arc = self
            .arrays
            .get(array)
            .ok_or_else(|| Error::ArrayNotFound(array.to_string()))?
            .clone();
        arc.lock().await.delete(&self.name)?;
        self.arrays.remove(array);
        self.meta.arrays.remove(array);
        Ok(())
    }

    pub async fn flush(&mut self) -> Result<()> {
        for arc in self.arrays.values() {
            arc.lock().await.flush().await?;
        }
        let mut store_meta = load_meta(&self.store).await?;
        store_meta.datasets.insert(self.name.clone(), self.meta.clone());
        save_meta(&self.store, &store_meta).await?;
        Ok(())
    }

    pub async fn compact(&mut self) -> Result<()> {
        for arc in self.arrays.values() {
            arc.lock().await.compact().await?;
        }
        Ok(())
    }
}

pub(crate) async fn open_dataset_view(
    store: Arc<dyn ObjectStore>,
    cache: Arc<ArrayCache>,
    name: &str,
    meta: &StoreMeta,
) -> Result<DatasetView> {
    let dataset_meta = meta
        .datasets
        .get(name)
        .ok_or_else(|| Error::DatasetNotFound(name.to_string()))?;

    let mut arrays = HashMap::new();
    for array_name in dataset_meta.arrays.keys() {
        let arc = get_or_open_cached(&store, &cache, array_name).await?;
        arrays.insert(array_name.clone(), arc);
    }

    Ok(DatasetView::new(store, cache, name.to_string(), arrays, dataset_meta.clone()))
}

/// Returns the cached `ArrayFile` for `array_name`, opening it first if needed.
/// The cache map lock (`parking_lot::RwLock`) is never held across an `await` point.
pub(crate) async fn get_or_open_cached(
    store: &Arc<dyn ObjectStore>,
    cache: &Arc<ArrayCache>,
    array_name: &str,
) -> Result<CachedFile> {
    // Fast path: already cached.
    {
        let guard = cache.read();
        if let Some(arc) = guard.get(array_name) {
            return Ok(arc.clone());
        }
    }

    // Slow path: open (or create) the file without holding the cache lock.
    let path = OsPath::from(format!("{}/data.af", array_name));
    let file = match store.head(&path).await {
        Ok(_) => ArrayFile::open(store.clone(), path, default_config()).await?,
        Err(object_store::Error::NotFound { .. }) => {
            ArrayFile::create(store.clone(), path, default_config()).await?
        }
        Err(e) => return Err(Error::ObjectStore(e)),
    };

    // Insert — use `entry` to avoid overwriting a concurrent insert.
    let arc = Arc::new(Mutex::new(file));
    let mut guard = cache.write();
    Ok(guard.entry(array_name.to_string()).or_insert(arc).clone())
}

pub(crate) fn default_config() -> FileConfig<ZstdCodec> {
    FileConfig {
        codec: ZstdCodec::default(),
        block_target_size: 8 * 1024 * 1024,
        cache_capacity: 256 * 1024 * 1024,
        io_cache_capacity: 64 * 1024 * 1024,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use object_store::memory::InMemory;

    fn make_store() -> Arc<dyn ObjectStore> {
        Arc::new(InMemory::new())
    }

    fn empty_view(store: Arc<dyn ObjectStore>, name: &str) -> DatasetView {
        DatasetView::new(
            store,
            Arc::new(ArrayCache::new(HashMap::new())),
            name.to_string(),
            HashMap::new(),
            DatasetMeta::default(),
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
        view.set_attribute("k", Attr::Int32(42));
        assert_eq!(view.get_attribute("k"), Some(&Attr::Int32(42)));
    }

    #[test]
    fn set_attribute_overwrites_previous() {
        let mut view = empty_view(make_store(), "ds");
        view.set_attribute("k", Attr::Int32(1));
        view.set_attribute("k", Attr::Int32(2));
        assert_eq!(view.get_attribute("k"), Some(&Attr::Int32(2)));
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
        let result = view.read_array::<f32>("missing", vec![], vec![]).await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn array_meta_errors_for_unknown_array() {
        let view = empty_view(make_store(), "ds");
        assert!(matches!(view.array_meta("missing"), Err(crate::Error::ArrayNotFound(_))));
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
        let result = view.read_array::<f32>("arr", vec![], vec![]).await.unwrap().unwrap();
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
        view.define_array::<f32>("arr", vec!["x".into(), "y".into()], vec![4, 8], Some(vec![2, 2]), None)
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

        let arr_schema = view.meta().arrays.get("arr").unwrap();
        assert_eq!(arr_schema.dtype, DType::Int32);
        assert_eq!(arr_schema.chunk_shape, vec![10]);
    }

    #[test]
    fn set_attribute_records_value_in_meta() {
        let mut view = empty_view(make_store(), "ds");
        view.set_attribute("count", Attr::UInt32(5));
        view.set_attribute("label", Attr::String("x".into()));

        let meta = view.meta();
        assert_eq!(meta.attributes.get("count"), Some(&Attr::UInt32(5)));
        assert_eq!(meta.attributes.get("label"), Some(&Attr::String("x".into())));
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

    // --- cache sharing ---

    #[tokio::test]
    async fn two_views_share_cached_array_file() {
        let store = make_store();
        let cache = Arc::new(ArrayCache::new(HashMap::new()));

        let mut view_a = DatasetView::new(
            store.clone(),
            cache.clone(),
            "ds_a".to_string(),
            HashMap::new(),
            DatasetMeta::default(),
        );
        view_a
            .define_array::<f32>("arr", vec!["x".into()], vec![2], None, None)
            .await
            .unwrap();

        let mut view_b = DatasetView::new(
            store.clone(),
            cache.clone(),
            "ds_b".to_string(),
            HashMap::new(),
            DatasetMeta::default(),
        );
        view_b
            .define_array::<f32>("arr", vec!["x".into()], vec![2], None, None)
            .await
            .unwrap();

        let ptr_a = Arc::as_ptr(view_a.arrays.get("arr").unwrap());
        let ptr_b = Arc::as_ptr(view_b.arrays.get("arr").unwrap());
        assert_eq!(ptr_a, ptr_b, "expected both views to share the same CachedFile");
    }
}
