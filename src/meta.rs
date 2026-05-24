use std::sync::Arc;

use indexmap::IndexMap;
use object_store::{ObjectStore, ObjectStoreExt, path::Path};
use serde::{Deserialize, Serialize};
use tracing::warn;

use crate::{
    Error, Result,
    config::{Codec, MetaFormat},
    schema::{ArraySchema, Attr},
};

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

/// Load store metadata, auto-detecting the on-disk format.
///
/// Probes for `atlas.json` first, then `atlas.msgpack`. If both exist
/// (shouldn't happen unless the directory was hand-edited), JSON wins and a
/// warning is logged. If neither exists, returns the default (empty) metadata
/// and [`MetaFormat::Json`] — preserving the prior behavior for new stores.
///
/// The returned `MetaFormat` is what subsequent saves should use so the same
/// file is overwritten instead of leaving stale copies behind.
pub(crate) async fn load_meta(store: &Arc<dyn ObjectStore>) -> Result<(StoreMeta, MetaFormat)> {
    let json = try_load(store, MetaFormat::Json).await?;
    let msgpack = try_load(store, MetaFormat::MsgPack).await?;
    match (json, msgpack) {
        (Some(meta), None) => Ok((meta, MetaFormat::Json)),
        (None, Some(meta)) => Ok((meta, MetaFormat::MsgPack)),
        (Some(meta), Some(_)) => {
            warn!(
                "both atlas.json and atlas.msgpack exist; loading atlas.json and ignoring atlas.msgpack"
            );
            Ok((meta, MetaFormat::Json))
        }
        (None, None) => Ok((StoreMeta::default(), MetaFormat::Json)),
    }
}

async fn try_load(store: &Arc<dyn ObjectStore>, format: MetaFormat) -> Result<Option<StoreMeta>> {
    match store.get(&Path::from(format.filename())).await {
        Ok(result) => {
            let bytes = result.bytes().await.map_err(Error::ObjectStore)?;
            Ok(Some(decode(&bytes, format)?))
        }
        Err(object_store::Error::NotFound { .. }) => Ok(None),
        Err(e) => Err(Error::ObjectStore(e)),
    }
}

fn decode(bytes: &[u8], format: MetaFormat) -> Result<StoreMeta> {
    match format {
        MetaFormat::Json => Ok(serde_json::from_slice(bytes)?),
        MetaFormat::MsgPack => Ok(rmp_serde::from_slice(bytes)?),
    }
}

fn encode(meta: &StoreMeta, format: MetaFormat) -> Result<Vec<u8>> {
    match format {
        MetaFormat::Json => Ok(serde_json::to_vec_pretty(meta)?),
        MetaFormat::MsgPack => Ok(rmp_serde::to_vec_named(meta)?),
    }
}

pub(crate) async fn save_meta(
    store: &Arc<dyn ObjectStore>,
    meta: &StoreMeta,
    format: MetaFormat,
) -> Result<()> {
    let bytes = encode(meta, format)?;
    store
        .put(&Path::from(format.filename()), bytes.into())
        .await
        .map_err(Error::ObjectStore)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Codec;
    use array_format::DType;
    use object_store::memory::InMemory;

    fn make_store() -> Arc<dyn ObjectStore> {
        Arc::new(InMemory::new())
    }

    fn sample_meta() -> StoreMeta {
        use crate::schema::ArraySchema;
        let mut meta = StoreMeta {
            version: 1,
            ..Default::default()
        };
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
        meta
    }

    #[tokio::test]
    async fn load_meta_missing_returns_default_json() {
        let store = make_store();
        let (meta, format) = load_meta(&store).await.unwrap();
        assert_eq!(meta.version, 0);
        assert!(meta.datasets.is_empty());
        assert_eq!(format, MetaFormat::Json);
    }

    #[tokio::test]
    async fn save_and_load_roundtrip_json() {
        let store = make_store();
        let meta = sample_meta();
        save_meta(&store, &meta, MetaFormat::Json).await.unwrap();

        let (loaded, format) = load_meta(&store).await.unwrap();
        assert_eq!(format, MetaFormat::Json);
        assert_eq!(loaded.version, 1);
        let dm = &loaded.datasets["ds1"];
        assert_eq!(dm.arrays["temp"].dtype, DType::Float32);
        assert_eq!(dm.arrays["temp"].shape, vec![4, 8]);
        assert!(matches!(dm.attributes["month"], Attr::Int64(6)));
    }

    #[tokio::test]
    async fn save_and_load_roundtrip_msgpack() {
        let store = make_store();
        let meta = sample_meta();
        save_meta(&store, &meta, MetaFormat::MsgPack).await.unwrap();

        let (loaded, format) = load_meta(&store).await.unwrap();
        assert_eq!(format, MetaFormat::MsgPack);
        assert_eq!(loaded.version, 1);
        let dm = &loaded.datasets["ds1"];
        assert_eq!(dm.arrays["temp"].dtype, DType::Float32);
        assert_eq!(dm.arrays["temp"].shape, vec![4, 8]);
        assert!(matches!(dm.attributes["month"], Attr::Int64(6)));
    }

    #[tokio::test]
    async fn msgpack_is_smaller_than_json() {
        let meta = sample_meta();
        let json = encode(&meta, MetaFormat::Json).unwrap();
        let mp = encode(&meta, MetaFormat::MsgPack).unwrap();
        assert!(
            mp.len() < json.len(),
            "msgpack ({}) should be smaller than JSON ({})",
            mp.len(),
            json.len()
        );
    }

    #[tokio::test]
    async fn load_detects_msgpack_when_only_msgpack_present() {
        let store = make_store();
        save_meta(&store, &sample_meta(), MetaFormat::MsgPack)
            .await
            .unwrap();
        let (_, format) = load_meta(&store).await.unwrap();
        assert_eq!(format, MetaFormat::MsgPack);
    }

    #[tokio::test]
    async fn load_prefers_json_when_both_present() {
        let store = make_store();
        let mut json_meta = sample_meta();
        json_meta.version = 42;
        let mut bin_meta = sample_meta();
        bin_meta.version = 99;
        save_meta(&store, &json_meta, MetaFormat::Json)
            .await
            .unwrap();
        save_meta(&store, &bin_meta, MetaFormat::MsgPack)
            .await
            .unwrap();

        let (loaded, format) = load_meta(&store).await.unwrap();
        assert_eq!(format, MetaFormat::Json);
        assert_eq!(loaded.version, 42);
    }

    #[tokio::test]
    async fn save_overwrites_previous_meta() {
        let store = make_store();
        let meta1 = StoreMeta {
            version: 1,
            ..Default::default()
        };
        save_meta(&store, &meta1, MetaFormat::Json).await.unwrap();

        let mut meta2 = StoreMeta {
            version: 2,
            ..Default::default()
        };
        meta2
            .datasets
            .insert("new_ds".into(), DatasetMeta::default());
        save_meta(&store, &meta2, MetaFormat::Json).await.unwrap();

        let (loaded, _) = load_meta(&store).await.unwrap();
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
        assert_eq!(
            serde_json::to_string(&Attr::String("x".into())).unwrap(),
            "\"x\""
        );
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
