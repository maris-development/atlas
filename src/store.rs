use std::sync::Arc;

use array_format::DeltaCache;
use object_store::{ObjectStore, local::LocalFileSystem, path::Path, prefix::PrefixStore};
use parking_lot::Mutex;
use tracing::{debug, info, instrument};

use crate::{
    Error, Result,
    config::{Codec, MetaFormat, StoreConfig},
    dataset::{ArrayCache, DatasetView, open_dataset_view},
    meta::{StoreMeta, load_meta, save_meta},
};

pub struct Atlas {
    store: Arc<dyn ObjectStore>,
    meta: Arc<Mutex<StoreMeta>>,
    cache: Arc<ArrayCache>,
    codec: Codec,
    meta_format: MetaFormat,
}

impl Atlas {
    /// Open an existing store at `prefix` within `store`.
    ///
    /// Reads `atlas.json` exactly once. Subsequent mutations only touch the
    /// in-memory meta until [`Atlas::flush`] / [`Atlas::close`] is called.
    #[instrument(skip(store), fields(prefix = %prefix))]
    pub async fn open(store: Arc<dyn ObjectStore>, prefix: Path) -> Result<Self> {
        let store = prefixed(store, prefix);
        let (meta, meta_format) = load_meta(&store).await?;
        let codec = meta.codec.clone();
        info!(
            datasets = meta.datasets.len(),
            ?codec,
            ?meta_format,
            "opened atlas store"
        );
        Ok(Self {
            store,
            meta: Arc::new(Mutex::new(meta)),
            cache: default_cache(),
            codec,
            meta_format,
        })
    }

    /// Create a new store at `prefix` within `store`.
    #[instrument(skip(store, config), fields(prefix = %prefix, codec = ?config.codec, meta_format = ?config.meta_format))]
    pub async fn create(store: Arc<dyn ObjectStore>, prefix: Path, config: StoreConfig) -> Result<Self> {
        let store = prefixed(store, prefix);
        let meta = StoreMeta { version: 1, codec: config.codec.clone(), ..Default::default() };
        save_meta(&store, &meta, config.meta_format).await?;
        info!("created atlas store");
        Ok(Self {
            store,
            meta: Arc::new(Mutex::new(meta)),
            cache: default_cache(),
            codec: config.codec,
            meta_format: config.meta_format,
        })
    }

    /// Open an existing store at the given local filesystem path.
    pub async fn open_path(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let store = Arc::new(LocalFileSystem::new_with_prefix(path.as_ref())?);
        Self::open(store, Path::from("")).await
    }

    /// Create a new store at the given local filesystem path. The directory is created
    /// (recursively, like `mkdir -p`) if it does not already exist.
    pub async fn create_path(path: impl AsRef<std::path::Path>, config: StoreConfig) -> Result<Self> {
        let path = path.as_ref();
        std::fs::create_dir_all(path)?;
        let store = Arc::new(LocalFileSystem::new_with_prefix(path)?);
        Self::create(store, Path::from(""), config).await
    }

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
            self.codec.clone(),
        ))
    }

    #[instrument(skip(self))]
    pub async fn open_dataset(&self, name: &str) -> Result<DatasetView> {
        open_dataset_view(
            self.store.clone(),
            self.cache.clone(),
            self.meta.clone(),
            name,
            self.codec.clone(),
        )
        .await
    }

    #[instrument(skip(self))]
    pub async fn delete_dataset(&mut self, name: &str) -> Result<()> {
        let dataset_meta = {
            let mut meta = self.meta.lock();
            meta.datasets
                .shift_remove(name)
                .ok_or_else(|| Error::DatasetNotFound(name.to_string()))?
        };
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
        Ok(())
    }

    pub fn list_datasets(&self) -> Vec<String> {
        let meta = self.meta.lock();
        meta.datasets.keys().cloned().collect()
    }

    pub fn dataset_exists(&self, name: &str) -> bool {
        let meta = self.meta.lock();
        meta.datasets.contains_key(name)
    }

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

    /// Flush every known array file's pending writes AND persist the in-memory
    /// `atlas.json`. This is the single durability boundary for the store.
    ///
    /// Force-initializes every array referenced in meta, even ones never
    /// touched by a `DatasetView` (lazy-init wins are on the read path, not
    /// on flush).
    #[instrument(skip(self))]
    pub async fn flush(&mut self) -> Result<()> {
        let snapshot = self.force_init_all_known_arrays().await?;
        let files = snapshot.len();
        debug!(files, "flushing array files");
        for arc in snapshot {
            arc.write().await.flush().await?;
        }
        let meta_snapshot = self.meta.lock().clone();
        let datasets = meta_snapshot.datasets.len();
        save_meta(&self.store, &meta_snapshot, self.meta_format).await?;
        info!(files, datasets, "flushed atlas store");
        Ok(())
    }

    /// Compact every known array file in place (reclaims tombstoned space).
    /// Force-initializes every array referenced in meta.
    #[instrument(skip(self))]
    pub async fn compact(&mut self) -> Result<()> {
        let snapshot = self.force_init_all_known_arrays().await?;
        let files = snapshot.len();
        debug!(files, "compacting array files");
        for arc in snapshot {
            arc.write().await.compact().await?;
        }
        info!(files, "compacted atlas store");
        Ok(())
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
    async fn create_path_creates_missing_directory() {
        let tmp = tempfile::tempdir().unwrap();
        let nested = tmp.path().join("missing").join("nested");
        assert!(!nested.exists());

        let _atlas = Atlas::create_path(&nested, StoreConfig::default()).await.unwrap();

        assert!(nested.exists() && nested.is_dir());
        assert!(nested.join("atlas.json").exists());
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
