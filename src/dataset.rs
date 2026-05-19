use std::{collections::HashMap, sync::Arc};

use array_format::{
    ArrayElement, ArrayFile, ArrayStats, FileConfig, FillValue, Lz4Codec, NoCompression,
    ZstdCodec,
};
use ndarray::{ArcArray, ArrayView, IxDyn};
use object_store::{ObjectStore, ObjectStoreExt, path::Path as OsPath};
use parking_lot::{Mutex, RwLock};
use tokio::sync::RwLock as AsyncRwLock;

use crate::{
    Error, Result,
    config::Codec,
    meta::{DatasetMeta, StoreMeta},
    schema::{Attr, ArraySchema},
};

/// Per-file lock: readers (`read_array`, `array_stats`) share concurrent access;
/// writers (`write_array`, …) take an exclusive lock.
pub(crate) type CachedFile = Arc<AsyncRwLock<ArrayFile>>;
/// Shared cache map: array_name → open ArrayFile.
/// Uses `parking_lot::RwLock` — never held across an `await` point.
pub(crate) type ArrayCache = RwLock<HashMap<String, CachedFile>>;

pub struct DatasetView {
    store: Arc<dyn ObjectStore>,
    cache: Arc<ArrayCache>,
    name: String,
    arrays: HashMap<String, CachedFile>,
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
        arrays: HashMap<String, CachedFile>,
        atlas_meta: Arc<Mutex<StoreMeta>>,
        codec: Codec,
    ) -> Self {
        Self { store, cache, name, arrays, atlas_meta, codec }
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

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn list_arrays(&self) -> Vec<String> {
        self.arrays.keys().cloned().collect()
    }

    pub fn set_attribute(&mut self, key: &str, value: Attr) {
        let mut meta = self.atlas_meta.lock();
        meta.datasets
            .entry(self.name.clone())
            .or_default()
            .attributes
            .insert(key.to_string(), value);
    }

    pub fn get_attribute(&self, key: &str) -> Option<Attr> {
        self.atlas_meta
            .lock()
            .datasets
            .get(&self.name)
            .and_then(|d| d.attributes.get(key).cloned())
    }

    /// Returns the cached schema for `array`.
    pub fn array_meta(&self, array: &str) -> Result<ArraySchema> {
        self.atlas_meta
            .lock()
            .datasets
            .get(&self.name)
            .and_then(|d| d.arrays.get(array).cloned())
            .ok_or_else(|| Error::ArrayNotFound(array.to_string()))
    }

    /// Returns aggregate statistics for `array` in this dataset, or `Ok(None)`
    /// if no stats exist yet (stats are computed on flush).
    pub async fn array_stats(&self, array: &str) -> Result<Option<ArrayStats>> {
        let arc = match self.arrays.get(array) {
            Some(arc) => arc.clone(),
            None => return Err(Error::ArrayNotFound(array.to_string())),
        };
        Ok(arc.read().await.array_stats(&self.name).cloned())
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

        let arc = get_or_open_cached(&self.store, &self.cache, array, &self.codec).await?;
        arc.write()
            .await
            .define_array::<T>(&self.name, dims.clone(), shape.clone(), chunk_shape.clone(), fill_value)?;
        self.arrays.insert(array.to_string(), arc);

        let actual_chunk = chunk_shape.unwrap_or_else(|| shape.clone());
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
        let mut guard = arc.write().await;
        guard.write_array::<T>(&self.name, start, data).await?;
        Ok(())
    }

    /// Returns `Ok(None)` if this dataset has no array with that name.
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
        let guard = arc.read().await;
        Ok(Some(guard.read_array::<T>(&self.name, start, shape).await?))
    }

    pub async fn delete_array(&mut self, array: &str) -> Result<()> {
        let arc = self
            .arrays
            .get(array)
            .ok_or_else(|| Error::ArrayNotFound(array.to_string()))?
            .clone();
        arc.write().await.delete(&self.name)?;
        self.arrays.remove(array);
        let mut meta = self.atlas_meta.lock();
        if let Some(ds_meta) = meta.datasets.get_mut(&self.name) {
            ds_meta.arrays.shift_remove(array);
        }
        Ok(())
    }
}

pub(crate) async fn open_dataset_view(
    store: Arc<dyn ObjectStore>,
    cache: Arc<ArrayCache>,
    atlas_meta: Arc<Mutex<StoreMeta>>,
    name: &str,
    array_specs: Vec<(String, Codec)>,
    codec: Codec,
) -> Result<DatasetView> {
    let mut arrays = HashMap::new();
    for (array_name, array_codec) in array_specs {
        let arc = get_or_open_cached(&store, &cache, &array_name, &array_codec).await?;
        arrays.insert(array_name, arc);
    }

    Ok(DatasetView::new(store, cache, name.to_string(), arrays, atlas_meta, codec))
}

/// Returns the cached `ArrayFile` for `array_name`, opening it first if needed.
/// The cache map lock (`parking_lot::RwLock`) is never held across an `await` point.
pub(crate) async fn get_or_open_cached(
    store: &Arc<dyn ObjectStore>,
    cache: &Arc<ArrayCache>,
    array_name: &str,
    codec: &Codec,
) -> Result<CachedFile> {
    {
        let guard = cache.read();
        if let Some(arc) = guard.get(array_name) {
            return Ok(arc.clone());
        }
    }

    let path = OsPath::from(format!("{}/data.af", array_name));
    let file = match store.head(&path).await {
        Ok(_) => open_array_file(store.clone(), path, codec).await?,
        Err(object_store::Error::NotFound { .. }) => {
            create_array_file(store.clone(), path, codec).await?
        }
        Err(e) => return Err(Error::ObjectStore(e)),
    };

    let arc = Arc::new(AsyncRwLock::new(file));
    let mut guard = cache.write();
    Ok(guard.entry(array_name.to_string()).or_insert(arc).clone())
}

async fn open_array_file(store: Arc<dyn ObjectStore>, path: OsPath, codec: &Codec) -> Result<ArrayFile> {
    Ok(match codec {
        Codec::Zstd => ArrayFile::open(store, path, file_config(ZstdCodec::default())).await?,
        Codec::Lz4 => ArrayFile::open(store, path, file_config(Lz4Codec)).await?,
        Codec::Uncompressed => ArrayFile::open(store, path, file_config(NoCompression)).await?,
    })
}

async fn create_array_file(store: Arc<dyn ObjectStore>, path: OsPath, codec: &Codec) -> Result<ArrayFile> {
    Ok(match codec {
        Codec::Zstd => ArrayFile::create(store, path, file_config(ZstdCodec::default())).await?,
        Codec::Lz4 => ArrayFile::create(store, path, file_config(Lz4Codec)).await?,
        Codec::Uncompressed => ArrayFile::create(store, path, file_config(NoCompression)).await?,
    })
}

fn file_config<C: array_format::CompressionCodec>(codec: C) -> FileConfig<C> {
    FileConfig {
        codec,
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

    fn shared_meta_with(name: &str) -> Arc<Mutex<StoreMeta>> {
        let mut meta = StoreMeta::default();
        meta.datasets.insert(name.to_string(), DatasetMeta::default());
        Arc::new(Mutex::new(meta))
    }

    fn empty_view(store: Arc<dyn ObjectStore>, name: &str) -> DatasetView {
        DatasetView::new(
            store,
            Arc::new(ArrayCache::new(HashMap::new())),
            name.to_string(),
            HashMap::new(),
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

    // --- array_stats ---

    #[tokio::test]
    async fn array_stats_errors_for_unknown_array() {
        let view = empty_view(make_store(), "ds");
        assert!(matches!(view.array_stats("ghost").await, Err(crate::Error::ArrayNotFound(_))));
    }

    #[tokio::test]
    async fn array_stats_none_before_flush() {
        let mut view = empty_view(make_store(), "ds");
        view.define_array::<f32>("arr", vec!["x".into()], vec![4], None, None)
            .await
            .unwrap();
        let stats = view.array_stats("arr").await.unwrap();
        assert!(stats.is_none());
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
        // Flush via direct ArrayFile flush (view no longer exposes flush).
        for arc in view.arrays.values() {
            arc.write().await.flush().await.unwrap();
        }

        let stats = view.array_stats("arr").await.unwrap().unwrap();
        assert_eq!(stats.row_count, 4);
        assert_eq!(stats.null_count, 0);
        assert_eq!(stats.min, Some(StatValue::Float(1.0)));
        assert_eq!(stats.max, Some(StatValue::Float(4.0)));
    }

    // --- cache sharing ---

    #[tokio::test]
    async fn two_views_share_cached_array_file() {
        let store = make_store();
        let cache = Arc::new(ArrayCache::new(HashMap::new()));
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
            HashMap::new(),
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
            HashMap::new(),
            shared.clone(),
            Codec::default(),
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
