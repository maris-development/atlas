use std::sync::Arc;

use indexmap::IndexMap;
use object_store::{ObjectStore, ObjectStoreExt, path::Path};
use serde::{Deserialize, Serialize};

use crate::{Error, Result, config::Codec, schema::{ArraySchema, Attr}};

/// Metadata for a single dataset: array schemas and per-dataset attributes.
/// Both maps preserve insertion order (via [`IndexMap`]) so on-disk layouts
/// and Python-side dict iteration mirror the order arrays/attributes were
/// added.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct DatasetMeta {
    #[serde(default)]
    pub arrays: IndexMap<String, ArraySchema>,
    #[serde(default)]
    pub attributes: IndexMap<String, Attr>,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub(crate) struct StoreMeta {
    pub version: u32,
    /// Codec used when new arrays are defined in this store.
    /// Written by `create`, read by `open`. Defaults to `Zstd` for stores
    /// created before this field existed.
    #[serde(default)]
    pub codec: Codec,
    pub datasets: IndexMap<String, DatasetMeta>,
}

const META_PATH: &str = "atlas.json";

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
    use crate::config::Codec;
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
                arrays: IndexMap::from([(
                    "temp".into(),
                    ArraySchema {
                        dtype: DType::Float32,
                        shape: vec![4, 8],
                        chunk_shape: vec![2, 4],
                        dimension_names: vec!["lat".into(), "lon".into()],
                        codec: Codec::default(),
                    },
                )]),
                attributes: IndexMap::from([
                    ("month".into(), Attr::Int64(6)),
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
        assert!(matches!(dm.attributes["month"], Attr::Int64(6)));
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
            Attr::Int64(-1_000_000),
            Attr::Float64(2.5),
            Attr::String("hello".into()),
            Attr::TimestampNanoseconds(1_700_000_000_000_000_000),
        ];
        for v in cases {
            let json = serde_json::to_string(&v).unwrap();
            let back: Attr = serde_json::from_str(&json).unwrap();
            assert_eq!(v, back);
        }
    }

    #[test]
    fn attr_json_shapes() {
        assert_eq!(serde_json::to_string(&Attr::Bool(true)).unwrap(), "true");
        assert_eq!(serde_json::to_string(&Attr::Int64(42)).unwrap(), "42");
        assert_eq!(serde_json::to_string(&Attr::Float64(1.5)).unwrap(), "1.5");
        assert_eq!(serde_json::to_string(&Attr::String("x".into())).unwrap(), "\"x\"");
        assert_eq!(
            serde_json::to_string(&Attr::TimestampNanoseconds(1_700_000_000_000_000_000)).unwrap(),
            "\"2023-11-14T22:13:20Z\"",
        );

        // Round-tripped non-RFC-3339 string stays as String, not TimestampNanoseconds.
        let back: Attr = serde_json::from_str("\"not-a-date\"").unwrap();
        assert_eq!(back, Attr::String("not-a-date".into()));

        // RFC 3339 string deserializes as TimestampNanoseconds (won the order race).
        let back: Attr = serde_json::from_str("\"2023-11-14T22:13:20Z\"").unwrap();
        assert_eq!(back, Attr::TimestampNanoseconds(1_700_000_000_000_000_000));
    }

    #[test]
    fn array_schema_roundtrip_via_serde() {
        use crate::schema::ArraySchema;
        let schema = ArraySchema {
            dtype: DType::Float64,
            shape: vec![10, 20],
            chunk_shape: vec![5, 5],
            dimension_names: vec!["lat".into(), "lon".into()],
            codec: Codec::default(),
        };
        let json = serde_json::to_string(&schema).unwrap();
        let back: ArraySchema = serde_json::from_str(&json).unwrap();
        assert_eq!(schema, back);
    }
}
