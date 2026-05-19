use std::collections::HashMap;
use std::sync::Arc;

use object_store::{ObjectStore, ObjectStoreExt, path::Path};
use serde::{Deserialize, Serialize};

use crate::{Error, Result, schema::{ArraySchema, Attr}};

/// Metadata for a single dataset: array schemas and per-dataset attributes.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct DatasetMeta {
    #[serde(default)]
    pub arrays: HashMap<String, ArraySchema>,
    #[serde(default)]
    pub attributes: HashMap<String, Attr>,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub(crate) struct StoreMeta {
    pub version: u32,
    pub datasets: HashMap<String, DatasetMeta>,
}

const META_PATH: &str = "array_store.json";

pub(crate) async fn load_meta(store: &Arc<dyn ObjectStore>) -> Result<StoreMeta> {
    match store.get(&Path::from(META_PATH)).await {
        Ok(result) => {
            let bytes = result.bytes().await.map_err(Error::ObjectStore)?;
            Ok(serde_json::from_slice(&bytes)?)
        }
        Err(object_store::Error::NotFound { .. }) => Ok(StoreMeta::default()),
        Err(e) => Err(Error::ObjectStore(e)),
    }
}

pub(crate) async fn save_meta(store: &Arc<dyn ObjectStore>, meta: &StoreMeta) -> Result<()> {
    let bytes = serde_json::to_vec_pretty(meta)?;
    store.put(&Path::from(META_PATH), bytes.into()).await.map_err(Error::ObjectStore)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use array_format::DType;
    use object_store::memory::InMemory;

    fn make_store() -> Arc<dyn ObjectStore> {
        Arc::new(InMemory::new())
    }

    #[tokio::test]
    async fn load_meta_missing_returns_default() {
        let store = make_store();
        let meta = load_meta(&store).await.unwrap();
        assert_eq!(meta.version, 0);
        assert!(meta.datasets.is_empty());
    }

    #[tokio::test]
    async fn save_and_load_roundtrip() {
        use crate::schema::ArraySchema;
        let store = make_store();
        let mut meta = StoreMeta { version: 1, ..Default::default() };
        meta.datasets.insert(
            "ds1".into(),
            DatasetMeta {
                arrays: HashMap::from([(
                    "temp".into(),
                    ArraySchema {
                        dtype: DType::Float32,
                        shape: vec![4, 8],
                        chunk_shape: vec![2, 4],
                        dimension_names: vec!["lat".into(), "lon".into()],
                    },
                )]),
                attributes: HashMap::from([
                    ("month".into(), Attr::Int32(6)),
                    ("active".into(), Attr::Bool(true)),
                ]),
            },
        );
        save_meta(&store, &meta).await.unwrap();

        let loaded = load_meta(&store).await.unwrap();
        assert_eq!(loaded.version, 1);
        assert!(loaded.datasets.contains_key("ds1"));
        let dm = &loaded.datasets["ds1"];
        assert!(dm.arrays.contains_key("temp"));
        assert_eq!(dm.arrays["temp"].dtype, DType::Float32);
        assert_eq!(dm.arrays["temp"].shape, vec![4, 8]);
        assert!(matches!(dm.attributes["month"], Attr::Int32(6)));
        assert!(matches!(dm.attributes["active"], Attr::Bool(true)));
    }

    #[tokio::test]
    async fn save_overwrites_previous_meta() {
        let store = make_store();
        let meta1 = StoreMeta { version: 1, ..Default::default() };
        save_meta(&store, &meta1).await.unwrap();

        let mut meta2 = StoreMeta { version: 2, ..Default::default() };
        meta2.datasets.insert("new_ds".into(), DatasetMeta::default());
        save_meta(&store, &meta2).await.unwrap();

        let loaded = load_meta(&store).await.unwrap();
        assert_eq!(loaded.version, 2);
        assert!(loaded.datasets.contains_key("new_ds"));
    }

    #[test]
    fn attr_roundtrip_via_serde() {
        let cases = vec![
            Attr::Bool(true),
            Attr::Int8(-1),
            Attr::Int16(-100),
            Attr::Int32(-1_000),
            Attr::Int64(-1_000_000),
            Attr::UInt8(1),
            Attr::UInt16(100),
            Attr::UInt32(1_000),
            Attr::UInt64(1_000_000),
            Attr::Float32(1.5),
            Attr::Float64(2.5),
            Attr::String("hello".into()),
        ];
        for v in cases {
            let json = serde_json::to_string(&v).unwrap();
            let back: Attr = serde_json::from_str(&json).unwrap();
            assert_eq!(v, back);
        }
    }

    #[test]
    fn array_schema_roundtrip_via_serde() {
        use crate::schema::ArraySchema;
        let schema = ArraySchema {
            dtype: DType::Float64,
            shape: vec![10, 20],
            chunk_shape: vec![5, 5],
            dimension_names: vec!["lat".into(), "lon".into()],
        };
        let json = serde_json::to_string(&schema).unwrap();
        let back: ArraySchema = serde_json::from_str(&json).unwrap();
        assert_eq!(schema, back);
    }
}
