//! [`DatasetView`]: the handle for reading and writing one dataset.

use std::sync::Arc;

use array_format::{ArrayElement, ArrayStats, DType, FillValue};
use indexmap::IndexMap;
use ndarray::{ArcArray, ArrayView, IxDyn};
use object_store::ObjectStore;
use parking_lot::Mutex;
use tracing::{debug, instrument, trace, warn};

use super::{ArrayCache, PendingAttrs};
use crate::{
    Error, Result,
    config::{Codec, TypeMismatchPolicy},
    meta::{DatasetSchema, StoreMeta},
    schema::{ArraySchema, Attr, widen_dtype},
};

/// Check `new` against the type the collection already records.
///
/// If the two widen (numeric ↔ numeric, string ↔ timestamp, element-wise for
/// lists) this is a no-op and the merged schema will widen to cover both. If
/// they don't, the value is still stored under this dataset's own type and the
/// merged schema keeps the first-seen type — [`TypeMismatchPolicy`] only
/// decides whether that is reported as a warning or an error.
///
/// The `existing` type comes from [`StoreMeta`]'s incrementally-maintained type
/// index rather than a scan over every dataset; it folds first-seen-wins to
/// match the collection's merged schema, so a stored mismatch never becomes the
/// reference type that later inserts are checked against.
fn check_type_alignment(
    policy: TypeMismatchPolicy,
    kind: &str,
    name: &str,
    existing: Option<DType>,
    new: &DType,
) -> Result<()> {
    let Some(m) = existing else { return Ok(()) };
    if widen_dtype(&m, new).is_some() {
        return Ok(());
    }
    match policy {
        TypeMismatchPolicy::Error => Err(Error::TypeMismatch {
            name: name.to_string(),
            existing: format!("{m:?}"),
            new: format!("{new:?}"),
        }),
        TypeMismatchPolicy::Warn => {
            warn!(
                kind,
                name,
                existing = ?m,
                new = ?new,
                "type mismatch: stored under this dataset's own type, but the merged \
                 schema keeps the first-seen type"
            );
            Ok(())
        }
    }
}

/// Physical array-file name that holds dataset-level (global) attributes. One
/// empty-shape entry per dataset name lives inside `_global/data.af`, carrying
/// that dataset's global attribute values. The leading `_` guarantees it never
/// collides with a user array (see [`crate::validate_name`]).
pub(crate) const GLOBAL_ATTRS_ARRAY: &str = "_global";

/// A borrowed handle to one dataset within an [`Atlas`](crate::Atlas).
///
/// Carries no independent state — every mutation (`define_array`,
/// `write_array`, `set_attribute`, `delete_array`) updates the parent
/// atlas's shared in-memory state. Persistence happens when the parent
/// [`Atlas::flush`](crate::Atlas::flush) is called; `DatasetView` has no
/// `flush` of its own.
///
/// Attribute **writes** are buffered in memory and flushed into the `.af`
/// files with everything else, so they stay non-blocking. Attribute **reads**
/// (`get_attribute`, `attributes`, `array_attributes`) are `async`: they merge
/// the buffer over the values persisted in the array files.
pub struct DatasetView {
    store: Arc<dyn ObjectStore>,
    pub(crate) cache: Arc<ArrayCache>,
    name: String,
    /// Shared handle to the parent `Atlas`'s in-memory `StoreMeta`. Schema
    /// mutations go through here; persistence happens on `Atlas::flush()`.
    atlas_meta: Arc<Mutex<StoreMeta>>,
    /// Shared buffer of attribute writes not yet flushed to the `.af` files.
    pending_attrs: Arc<Mutex<PendingAttrs>>,
    codec: Codec,
    /// How to report a type that can't merge with the collection's existing type.
    on_type_mismatch: TypeMismatchPolicy,
}

/// Interning hook: once a view goes away the dataset is fully written, so its
/// schema can be deduplicated against the rest of the collection. Datasets that
/// share a layout (an ingest of like-shaped files) then share one allocation.
impl Drop for DatasetView {
    fn drop(&mut self) {
        self.atlas_meta.lock().seal_dataset(&self.name);
    }
}

impl DatasetView {
    pub(crate) fn new(
        store: Arc<dyn ObjectStore>,
        cache: Arc<ArrayCache>,
        name: String,
        atlas_meta: Arc<Mutex<StoreMeta>>,
        pending_attrs: Arc<Mutex<PendingAttrs>>,
        codec: Codec,
        on_type_mismatch: TypeMismatchPolicy,
    ) -> Self {
        Self {
            store,
            cache,
            name,
            atlas_meta,
            pending_attrs,
            codec,
            on_type_mismatch,
        }
    }

    /// Returns a clone of this dataset's schema (array schemas + the
    /// attribute-key namespace). Reads the shared in-memory meta — no disk I/O.
    pub fn schema(&self) -> DatasetSchema {
        self.atlas_meta
            .lock()
            .live_schema(&self.name)
            .map(|s| (**s).clone())
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
            .live_schema(&self.name)
            .map(|d| d.arrays.keys().cloned().collect())
            .unwrap_or_default()
    }

    /// Set or overwrite a dataset-level (global) attribute. Buffered in memory
    /// until the parent [`Atlas::flush`](crate::Atlas::flush), which writes it
    /// into `_global/data.af`. The key and its type are recorded in the
    /// dataset's schema.
    ///
    /// Errors with [`Error::TypeMismatch`] if another dataset already uses this
    /// global key with a type that can't widen to this value's type.
    pub fn set_attribute(&mut self, key: &str, value: Attr) -> Result<()> {
        let ty = value.dtype();
        {
            let mut meta = self.atlas_meta.lock();
            let existing = meta.other_global_attr_dtype(&self.name, key);
            check_type_alignment(self.on_type_mismatch, "attribute", key, existing, &ty)?;
            meta.record_global_attr(&self.name, key, ty);
        }
        self.pending_attrs
            .lock()
            .set(GLOBAL_ATTRS_ARRAY, &self.name, key, value);
        Ok(())
    }

    /// Look up a dataset-level attribute by key. Checks the pending buffer
    /// first, then the value persisted in `_global/data.af`. `None` if unset.
    pub async fn get_attribute(&self, key: &str) -> Result<Option<Attr>> {
        self.read_attr(GLOBAL_ATTRS_ARRAY, key).await
    }

    /// All dataset-level attributes as key → value, in the order the keys were
    /// first set. Merges the pending buffer over persisted values.
    pub async fn attributes(&self) -> Result<IndexMap<String, Attr>> {
        let keys: Vec<String> = self.schema().global_attrs.keys().cloned().collect();
        self.collect_attrs(GLOBAL_ATTRS_ARRAY, &keys).await
    }

    /// Set or overwrite a per-variable attribute on `array` (e.g. `units`).
    /// Buffered until flush, which writes it into `<array>/data.af`. Errors
    /// with [`Error::ArrayNotFound`] if the array isn't declared here.
    pub fn set_array_attribute(&mut self, array: &str, key: &str, value: Attr) -> Result<()> {
        let ty = value.dtype();
        {
            let mut meta = self.atlas_meta.lock();
            let present = meta
                .live_schema(&self.name)
                .is_some_and(|d| d.arrays.contains_key(array));
            if !present {
                return Err(Error::ArrayNotFound(array.to_string()));
            }
            let existing = meta.other_array_attr_dtype(&self.name, array, key);
            check_type_alignment(self.on_type_mismatch, "array attribute", key, existing, &ty)?;
            meta.record_array_attr(&self.name, array, key, ty);
        }
        self.pending_attrs.lock().set(array, &self.name, key, value);
        Ok(())
    }

    /// Look up a per-variable attribute on `array`. Buffer first, then
    /// `<array>/data.af`.
    pub async fn get_array_attribute(&self, array: &str, key: &str) -> Result<Option<Attr>> {
        self.read_attr(array, key).await
    }

    /// All per-variable attributes on `array`, key → value, in first-set order.
    pub async fn array_attributes(&self, array: &str) -> Result<IndexMap<String, Attr>> {
        let keys: Vec<String> = self
            .schema()
            .array_attrs
            .get(array)
            .map(|m| m.keys().cloned().collect())
            .unwrap_or_default();
        self.collect_attrs(array, &keys).await
    }

    /// Returns the cached schema for `array`, or `None` if no array with that
    /// name exists in this dataset. Sync — reads the in-memory schema.
    pub fn array_meta(&self, array: &str) -> Option<ArraySchema> {
        self.atlas_meta
            .lock()
            .live_schema(&self.name)
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
        let new_dtype = T::DTYPE.clone();
        {
            let meta = self.atlas_meta.lock();
            if let Some(ds) = meta.live_schema(&self.name)
                && ds.arrays.contains_key(array) {
                    return Err(Error::ArrayAlreadyExists(array.to_string()));
                }
            // Reject a dtype that can't merge with the same array name in other
            // datasets (e.g. an int32 array here vs a string array elsewhere).
            let existing = meta.other_array_dtype(&self.name, array);
            check_type_alignment(self.on_type_mismatch, "array", array, existing, &new_dtype)?;
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
            codec: self.codec,
        };
        self.atlas_meta
            .lock()
            .record_array(&self.name, array, schema);
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
        // Drop any buffered per-variable attribute writes for this array.
        self.pending_attrs.lock().remove(array, &self.name);
        self.atlas_meta.lock().unrecord_array(&self.name, array);
        debug!("deleted array");
        Ok(())
    }

    /// Reads one attribute (`file` = `_global` or an array name) for this
    /// dataset: pending buffer first, then the persisted `.af` value. Returns
    /// `None` if the file doesn't exist, the dataset has no entry in it, or the
    /// key is unset — never creates a file.
    async fn read_attr(&self, file: &str, key: &str) -> Result<Option<Attr>> {
        if let Some(v) = self.pending_attrs.lock().get(file, &self.name, key) {
            return Ok(Some(v));
        }
        let codec = self.file_codec(file);
        let handle = self.cache.get_or_insert(&self.store, file, &codec);
        let Some(arc) = handle.get_existing().await? else {
            return Ok(None);
        };
        let guard = arc.read().await;
        // A missing dataset entry in the file surfaces as an error from
        // `get_attribute`; treat "not present" as simply unset.
        Ok(match guard.get_attribute(&self.name, key) {
            Ok(Some(v)) => Some(Attr::from(v.clone())),
            Ok(None) | Err(_) => None,
        })
    }

    /// Collects the given keys for `file` into an ordered map, dropping keys
    /// with no value for this dataset.
    async fn collect_attrs(&self, file: &str, keys: &[String]) -> Result<IndexMap<String, Attr>> {
        let mut out = IndexMap::with_capacity(keys.len());
        for key in keys {
            if let Some(v) = self.read_attr(file, key).await? {
                out.insert(key.clone(), v);
            }
        }
        Ok(out)
    }

    /// Codec to open a physical file with: the store default for `_global`,
    /// otherwise the array's recorded codec (falling back to the store codec).
    fn file_codec(&self, file: &str) -> Codec {
        if file == GLOBAL_ATTRS_ARRAY {
            self.codec
        } else {
            self.array_codec(file).unwrap_or(self.codec)
        }
    }

    /// Looks up the per-array codec from `atlas_meta`. Returns `None` if the
    /// array isn't defined in this dataset.
    fn array_codec(&self, array: &str) -> Option<Codec> {
        self.atlas_meta
            .lock()
            .live_schema(&self.name)
            .and_then(|d| d.arrays.get(array).map(|s| s.codec))
    }
}

pub(crate) async fn open_dataset_view(
    store: Arc<dyn ObjectStore>,
    cache: Arc<ArrayCache>,
    atlas_meta: Arc<Mutex<StoreMeta>>,
    pending_attrs: Arc<Mutex<PendingAttrs>>,
    name: &str,
    codec: Codec,
    on_type_mismatch: TypeMismatchPolicy,
) -> Result<DatasetView> {
    {
        let meta = atlas_meta.lock();
        if !meta.is_live(name) {
            return Err(Error::DatasetNotFound(name.to_string()));
        }
    }
    Ok(DatasetView::new(
        store,
        cache,
        name.to_string(),
        atlas_meta,
        pending_attrs,
        codec,
        on_type_mismatch,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use array_format::DeltaCache;
    use object_store::memory::InMemory;

    fn make_store() -> Arc<dyn ObjectStore> {
        Arc::new(InMemory::new())
    }

    fn shared_meta_with(name: &str) -> Arc<Mutex<StoreMeta>> {
        let mut meta = StoreMeta::default();
        meta.add_dataset(name);
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
            Arc::new(Mutex::new(PendingAttrs::default())),
            Codec::default(),
            TypeMismatchPolicy::default(),
        )
    }

    // --- attribute tests ---

    #[tokio::test]
    async fn get_attribute_missing_returns_none() {
        let view = empty_view(make_store(), "ds");
        assert!(view.get_attribute("x").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn set_and_get_attribute_roundtrip() {
        // Reads come straight from the pending buffer — no flush needed.
        let mut view = empty_view(make_store(), "ds");
        view.set_attribute("k", Attr::Int64(42)).unwrap();
        assert_eq!(view.get_attribute("k").await.unwrap(), Some(Attr::Int64(42)));
    }

    #[tokio::test]
    async fn set_attribute_overwrites_previous() {
        let mut view = empty_view(make_store(), "ds");
        view.set_attribute("k", Attr::Int64(1)).unwrap();
        view.set_attribute("k", Attr::Int64(2)).unwrap();
        assert_eq!(view.get_attribute("k").await.unwrap(), Some(Attr::Int64(2)));
    }

    #[tokio::test]
    async fn set_attribute_records_key_in_schema() {
        let mut view = empty_view(make_store(), "ds");
        view.set_attribute("count", Attr::Int64(5)).unwrap();
        view.set_attribute("label", Attr::String("x".into())).unwrap();
        let schema = view.schema();
        let keys: Vec<&String> = schema.global_attrs.keys().collect();
        assert_eq!(keys, vec!["count", "label"]);
        assert_eq!(schema.global_attrs["count"].0, DType::Int64);
    }

    #[tokio::test]
    async fn attributes_returns_all_set() {
        let mut view = empty_view(make_store(), "ds");
        view.set_attribute("count", Attr::Int64(5)).unwrap();
        view.set_attribute("label", Attr::String("x".into())).unwrap();
        let attrs = view.attributes().await.unwrap();
        assert_eq!(attrs.get("count"), Some(&Attr::Int64(5)));
        assert_eq!(attrs.get("label"), Some(&Attr::String("x".into())));
    }

    #[tokio::test]
    async fn set_array_attribute_requires_array() {
        let mut view = empty_view(make_store(), "ds");
        let err = view
            .set_array_attribute("missing", "units", Attr::String("m/s".into()))
            .unwrap_err();
        assert!(matches!(err, crate::Error::ArrayNotFound(_)));
    }

    #[tokio::test]
    async fn per_variable_attribute_roundtrip() {
        let mut view = empty_view(make_store(), "ds");
        view.define_array::<f32>("wind", vec!["x".into()], vec![4], None, None)
            .await
            .unwrap();
        view.set_array_attribute("wind", "units", Attr::String("m/s".into()))
            .unwrap();
        assert_eq!(
            view.get_array_attribute("wind", "units").await.unwrap(),
            Some(Attr::String("m/s".into()))
        );
        let attrs = view.array_attributes("wind").await.unwrap();
        assert_eq!(attrs.get("units"), Some(&Attr::String("m/s".into())));
        // Recorded in the schema namespace with its type.
        let schema = view.schema();
        let wind_keys: Vec<&String> = schema.array_attrs["wind"].keys().collect();
        assert_eq!(wind_keys, vec!["units"]);
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

    // --- schema ---

    #[tokio::test]
    async fn define_array_records_schema() {
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

        let schema = view.schema();
        let arr_schema = schema.arrays.get("arr").expect("schema entry missing");
        assert_eq!(arr_schema.dtype, DType::Float32);
        assert_eq!(arr_schema.shape, vec![4, 8]);
        assert_eq!(arr_schema.chunk_shape, vec![2, 2]);
        assert_eq!(arr_schema.dimension_names, vec!["x", "y"]);
        assert!(schema.global_attrs.is_empty());
    }

    #[tokio::test]
    async fn define_array_default_chunk_equals_shape() {
        use array_format::DType;
        let mut view = empty_view(make_store(), "ds");
        view.define_array::<i32>("arr", vec!["t".into()], vec![10], None, None)
            .await
            .unwrap();

        let schema = view.schema();
        let arr_schema = schema.arrays.get("arr").unwrap();
        assert_eq!(arr_schema.dtype, DType::Int32);
        assert_eq!(arr_schema.chunk_shape, vec![10]);
    }

    #[tokio::test]
    async fn delete_array_removes_schema_entry() {
        let mut view = empty_view(make_store(), "ds");
        view.define_array::<f64>("arr", vec!["x".into()], vec![4], None, None)
            .await
            .unwrap();
        assert!(view.schema().arrays.contains_key("arr"));
        view.delete_array("arr").await.unwrap();
        assert!(!view.schema().arrays.contains_key("arr"));
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

    // --- attribute persistence through the array files ---

    #[tokio::test]
    async fn attributes_persist_through_af_files() {
        // Global + per-variable attributes read back from the .af files after
        // the pending buffer is drained (here via a manual drain + flush).
        let store = make_store();
        let cache = test_cache();
        let meta = shared_meta_with("ds");
        let pending = Arc::new(Mutex::new(PendingAttrs::default()));
        {
            let mut view = DatasetView::new(
                store.clone(),
                cache.clone(),
                "ds".into(),
                meta.clone(),
                pending.clone(),
                Codec::default(),
                TypeMismatchPolicy::default(),
            );
            view.define_array::<f32>("temp", vec!["x".into()], vec![2], None, None)
                .await
                .unwrap();
            view.set_attribute("region", Attr::String("north".into())).unwrap();
            view.set_array_attribute("temp", "units", Attr::String("K".into()))
                .unwrap();

            // Drain the buffer into the .af files the way Atlas::flush does.
            // Bind the drained Vec so the lock is released before the awaits.
            let drained = pending.lock().drain_all();
            for ((file, ds), attrs) in drained {
                let codec = Codec::default();
                let handle = cache.get_or_insert(&store, &file, &codec);
                let arc = handle.get().await.unwrap();
                let mut guard = arc.write().await;
                if guard.get_array(&ds).is_err() {
                    guard
                        .define_array::<u8>(ds.to_string(), vec![], vec![], None, None)
                        .unwrap();
                }
                for (k, v) in attrs {
                    guard.set_attribute(&ds, &k, (*v).clone().into()).unwrap();
                }
            }
            flush_initialized(&cache).await;
        }

        // Fresh view over the same in-memory meta + on-disk files, empty buffer.
        let view = DatasetView::new(
            store,
            cache,
            "ds".into(),
            meta,
            Arc::new(Mutex::new(PendingAttrs::default())),
            Codec::default(),
            TypeMismatchPolicy::default(),
        );
        assert_eq!(
            view.get_attribute("region").await.unwrap(),
            Some(Attr::String("north".into()))
        );
        assert_eq!(
            view.get_array_attribute("temp", "units").await.unwrap(),
            Some(Attr::String("K".into()))
        );
    }

    // --- cache sharing ---

    #[tokio::test]
    async fn two_views_share_cached_array_file() {
        let store = make_store();
        let cache = test_cache();
        let shared = Arc::new(Mutex::new({
            let mut m = StoreMeta::default();
            m.add_dataset("ds_a");
            m.add_dataset("ds_b");
            m
        }));
        let pending = Arc::new(Mutex::new(PendingAttrs::default()));

        let mut view_a = DatasetView::new(
            store.clone(),
            cache.clone(),
            "ds_a".to_string(),
            shared.clone(),
            pending.clone(),
            Codec::default(),
            TypeMismatchPolicy::default(),
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
            pending.clone(),
            Codec::default(),
            TypeMismatchPolicy::default(),
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
