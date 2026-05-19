use std::{collections::HashMap, sync::Arc};

use object_store::{ObjectStore, path::Path, prefix::PrefixStore};

use crate::{
    Error, Result,
    config::{Codec, StoreConfig},
    dataset::{ArrayCache, DatasetView, get_or_open_cached, open_dataset_view},
    meta::{DatasetMeta, StoreMeta, load_meta, save_meta},
};

pub struct Atlas {
    store: Arc<dyn ObjectStore>,
    meta: StoreMeta,
    cache: Arc<ArrayCache>,
    codec: Codec,
}

impl Atlas {
    /// Open an existing store at `prefix` within `store`.
    ///
    /// The codec is read from the store's metadata JSON — no codec argument needed.
    pub async fn open(store: Arc<dyn ObjectStore>, prefix: Path) -> Result<Self> {
        let store = prefixed(store, prefix);
        let meta = load_meta(&store).await?;
        let codec = meta.codec.clone();
        Ok(Self { store, meta, cache: default_cache(), codec })
    }

    /// Create a new store at `prefix` within `store`.
    ///
    /// The codec in `config` is persisted to `atlas.json` and will be used
    /// automatically whenever this store is reopened with [`Atlas::open`].
    pub async fn create(store: Arc<dyn ObjectStore>, prefix: Path, config: StoreConfig) -> Result<Self> {
        let store = prefixed(store, prefix);
        let meta = StoreMeta { version: 1, codec: config.codec.clone(), ..Default::default() };
        save_meta(&store, &meta).await?;
        Ok(Self { store, meta, cache: default_cache(), codec: config.codec })
    }

    pub async fn create_dataset(&mut self, name: &str) -> Result<DatasetView> {
        crate::validate_name(name)?;
        // Reload so arrays from previously flushed DatasetViews are preserved.
        self.meta = load_meta(&self.store).await?;
        if self.meta.datasets.contains_key(name) {
            return Err(Error::DatasetAlreadyExists(name.to_string()));
        }
        self.meta.datasets.insert(name.to_string(), Default::default());
        save_meta(&self.store, &self.meta).await?;
        Ok(DatasetView::new(
            self.store.clone(),
            self.cache.clone(),
            name.to_string(),
            HashMap::new(),
            DatasetMeta::default(),
            self.codec.clone(),
        ))
    }

    pub async fn open_dataset(&self, name: &str) -> Result<DatasetView> {
        let meta = load_meta(&self.store).await?;
        open_dataset_view(self.store.clone(), self.cache.clone(), name, &meta, self.codec.clone()).await
    }

    pub async fn delete_dataset(&mut self, name: &str) -> Result<()> {
        let dataset_meta = self
            .meta
            .datasets
            .remove(name)
            .ok_or_else(|| Error::DatasetNotFound(name.to_string()))?;

        for array_name in dataset_meta.arrays.keys() {
            let arc = get_or_open_cached(&self.store, &self.cache, array_name, &self.codec).await?;
            let mut guard = arc.write().await;
            guard.delete(name)?;
            guard.flush().await?;
        }

        save_meta(&self.store, &self.meta).await?;
        Ok(())
    }

    pub fn list_datasets(&self) -> Vec<&str> {
        self.meta.datasets.keys().map(|s| s.as_str()).collect()
    }

    pub fn dataset_exists(&self, name: &str) -> bool {
        self.meta.datasets.contains_key(name)
    }

    pub fn list_arrays(&self) -> Vec<String> {
        let mut arrays: Vec<String> = self
            .meta
            .datasets
            .values()
            .flat_map(|d| d.arrays.keys().cloned())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        arrays.sort();
        arrays
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
    Arc::new(ArrayCache::new(HashMap::new()))
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
        assert!(s.list_datasets().contains(&"ds"));
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

        let mut ds_a = s.create_dataset("a").await.unwrap();
        ds_a.define_array::<f32>("shared", vec!["x".into()], vec![2], None, None)
            .await
            .unwrap();
        ds_a.define_array::<f32>("only_a", vec!["x".into()], vec![2], None, None)
            .await
            .unwrap();
        ds_a.flush().await.unwrap();

        let mut ds_b = s.create_dataset("b").await.unwrap();
        ds_b.define_array::<f32>("shared", vec!["x".into()], vec![2], None, None)
            .await
            .unwrap();
        ds_b.flush().await.unwrap();

        // Reopen so list_arrays reflects flushed state.
        let s2 = Atlas::open(store, prefix).await.unwrap();
        let arrays = s2.list_arrays();
        assert_eq!(arrays, vec!["only_a", "shared"]);
    }

    #[tokio::test]
    async fn lz4_codec_roundtrip() {
        let (store, prefix) = make_store();
        let config = StoreConfig { codec: Codec::Lz4 };
        let mut s = Atlas::create(store.clone(), prefix.clone(), config).await.unwrap();

        let mut ds = s.create_dataset("ds").await.unwrap();
        ds.define_array::<f32>("arr", vec!["x".into()], vec![4], None, None)
            .await
            .unwrap();
        let data = ndarray::arr1(&[1.0_f32, 2.0, 3.0, 4.0]).into_dyn();
        ds.write_array("arr", vec![0], data.view()).await.unwrap();
        ds.flush().await.unwrap();

        let s2 = Atlas::open(store, prefix).await.unwrap();
        let ds2 = s2.open_dataset("ds").await.unwrap();
        let result = ds2.read_array::<f32>("arr", vec![], vec![]).await.unwrap().unwrap();
        assert_eq!(result, data.into_shared());
    }

    #[tokio::test]
    async fn uncompressed_codec_roundtrip() {
        let (store, prefix) = make_store();
        let config = StoreConfig { codec: Codec::Uncompressed };
        let mut s = Atlas::create(store.clone(), prefix.clone(), config).await.unwrap();

        let mut ds = s.create_dataset("ds").await.unwrap();
        ds.define_array::<i32>("arr", vec!["x".into()], vec![3], None, None)
            .await
            .unwrap();
        let data = ndarray::arr1(&[10_i32, 20, 30]).into_dyn();
        ds.write_array("arr", vec![0], data.view()).await.unwrap();
        ds.flush().await.unwrap();

        let s2 = Atlas::open(store, prefix).await.unwrap();
        let ds2 = s2.open_dataset("ds").await.unwrap();
        let result = ds2.read_array::<i32>("arr", vec![], vec![]).await.unwrap().unwrap();
        assert_eq!(result, data.into_shared());
    }
}
