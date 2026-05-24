use std::sync::Arc;

use indexmap::IndexMap;
use object_store::{ObjectStore, ObjectStoreExt, path::Path};
use serde::{Deserialize, Serialize};
use tracing::warn;

use crate::{
    Error, Result,
    config::{Codec, META_VARIANTS, MetaFormat},
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

/// Load store metadata, auto-detecting both the encoding format and the
/// compression from the on-disk filename.
///
/// Uses a single [`ObjectStore::list_with_delimiter`] to enumerate the
/// top-level files and matches them against the six known
/// `atlas.{json,msgpack}{,.zst,.lz4}` filenames. If more than one matches
/// (shouldn't happen unless the directory was hand-edited), the warning
/// names them and the priority order in
/// [`META_VARIANTS`](crate::config::META_VARIANTS) decides — uncompressed
/// before compressed within each format, JSON before MsgPack overall.
///
/// If no metadata file is found, returns the default (empty) metadata with
/// `(Json, Uncompressed)` so a freshly-created store gets the legacy
/// `atlas.json` filename on its first save.
///
/// The returned `(MetaFormat, Codec)` is what subsequent saves should use so
/// the same file is overwritten instead of leaving stale copies behind.
pub(crate) async fn load_meta(
    store: &Arc<dyn ObjectStore>,
) -> Result<(StoreMeta, MetaFormat, Codec)> {
    let listing = store
        .list_with_delimiter(None)
        .await
        .map_err(Error::ObjectStore)?;

    // Collect filenames present at the root.
    let present: std::collections::HashSet<&str> = listing
        .objects
        .iter()
        .filter_map(|o| o.location.filename())
        .collect();

    let matches: Vec<(MetaFormat, Codec)> = META_VARIANTS
        .iter()
        .copied()
        .filter(|&(fmt, comp)| present.contains(fmt.filename(comp)))
        .collect();

    let (format, compression) = match matches.as_slice() {
        [] => return Ok((StoreMeta::default(), MetaFormat::Json, Codec::Uncompressed)),
        [single] => *single,
        many => {
            let names: Vec<&str> = many
                .iter()
                .map(|&(f, c)| f.filename(c))
                .collect();
            let chosen = many[0];
            warn!(
                "multiple metadata files present ({names:?}); loading {} by priority order",
                chosen.0.filename(chosen.1)
            );
            chosen
        }
    };

    let bytes = store
        .get(&Path::from(format.filename(compression)))
        .await
        .map_err(Error::ObjectStore)?
        .bytes()
        .await
        .map_err(Error::ObjectStore)?;
    let raw = decompress(&bytes, compression)?;
    let meta = decode(&raw, format)?;
    Ok((meta, format, compression))
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

fn compress(bytes: Vec<u8>, codec: Codec) -> Result<Vec<u8>> {
    match codec {
        Codec::Uncompressed => Ok(bytes),
        // zstd default level (3) — good ratio at low CPU. Metadata is small,
        // so even level 19 would be sub-millisecond, but the default is fine.
        Codec::Zstd => Ok(zstd::stream::encode_all(bytes.as_slice(), 0)?),
        // lz4_flex compression is infallible; size prefix lets decode know the
        // output length without scanning.
        Codec::Lz4 => Ok(lz4_flex::compress_prepend_size(&bytes)),
    }
}

fn decompress(bytes: &[u8], codec: Codec) -> Result<Vec<u8>> {
    match codec {
        Codec::Uncompressed => Ok(bytes.to_vec()),
        Codec::Zstd => Ok(zstd::stream::decode_all(bytes)?),
        Codec::Lz4 => Ok(lz4_flex::decompress_size_prepended(bytes)?),
    }
}

pub(crate) async fn save_meta(
    store: &Arc<dyn ObjectStore>,
    meta: &StoreMeta,
    format: MetaFormat,
    compression: Codec,
) -> Result<()> {
    let encoded = encode(meta, format)?;
    let bytes = compress(encoded, compression)?;
    store
        .put(&Path::from(format.filename(compression)), bytes.into())
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
    async fn load_meta_missing_returns_default_json_uncompressed() {
        let store = make_store();
        let (meta, format, compression) = load_meta(&store).await.unwrap();
        assert_eq!(meta.version, 0);
        assert!(meta.datasets.is_empty());
        assert_eq!(format, MetaFormat::Json);
        assert_eq!(compression, Codec::Uncompressed);
    }

    /// Roundtrip every (format, compression) pair through save_meta + load_meta.
    /// Asserts the detected pair matches what was written.
    #[tokio::test]
    async fn save_and_load_roundtrip_all_variants() {
        for &(format, compression) in &META_VARIANTS {
            let store = make_store();
            let meta = sample_meta();
            save_meta(&store, &meta, format, compression).await.unwrap();

            let (loaded, detected_fmt, detected_comp) = load_meta(&store).await.unwrap();
            assert_eq!(detected_fmt, format, "format mismatch for {format:?}/{compression:?}");
            assert_eq!(
                detected_comp, compression,
                "compression mismatch for {format:?}/{compression:?}"
            );
            assert_eq!(loaded.version, 1);
            let dm = &loaded.datasets["ds1"];
            assert_eq!(dm.arrays["temp"].dtype, DType::Float32);
            assert_eq!(dm.arrays["temp"].shape, vec![4, 8]);
            assert!(matches!(dm.attributes["month"], Attr::Int64(6)));
        }
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

    /// Compression should shrink the encoded bytes. Uses a workload large
    /// enough to overcome compression framing overhead.
    #[tokio::test]
    async fn compression_shrinks_encoded_bytes() {
        use crate::schema::ArraySchema;
        let mut meta = StoreMeta {
            version: 1,
            ..Default::default()
        };
        for i in 0..30 {
            let mut dm = DatasetMeta::default();
            for j in 0..5 {
                dm.arrays.insert(
                    format!("arr_{j}"),
                    ArraySchema {
                        dtype: DType::Float32,
                        shape: vec![100, 200, 300],
                        chunk_shape: vec![10, 20, 30],
                        dimension_names: vec!["a".into(), "b".into(), "c".into()],
                        codec: Codec::default(),
                    },
                );
            }
            meta.datasets.insert(format!("dataset_{i}"), dm);
        }

        for format in [MetaFormat::Json, MetaFormat::MsgPack] {
            let raw = encode(&meta, format).unwrap();
            let zstd = compress(raw.clone(), Codec::Zstd).unwrap();
            let lz4 = compress(raw.clone(), Codec::Lz4).unwrap();
            assert!(
                zstd.len() < raw.len(),
                "{format:?}: zstd ({}) should be smaller than raw ({})",
                zstd.len(),
                raw.len()
            );
            assert!(
                lz4.len() < raw.len(),
                "{format:?}: lz4 ({}) should be smaller than raw ({})",
                lz4.len(),
                raw.len()
            );
        }
    }

    #[tokio::test]
    async fn load_detects_msgpack_zstd_when_only_that_present() {
        let store = make_store();
        save_meta(&store, &sample_meta(), MetaFormat::MsgPack, Codec::Zstd)
            .await
            .unwrap();
        let (_, format, compression) = load_meta(&store).await.unwrap();
        assert_eq!(format, MetaFormat::MsgPack);
        assert_eq!(compression, Codec::Zstd);
    }

    /// When more than one metadata file is present, priority order picks
    /// uncompressed JSON over everything else.
    #[tokio::test]
    async fn load_priority_order_when_many_present() {
        let store = make_store();
        let mut a = sample_meta();
        a.version = 1;
        let mut b = sample_meta();
        b.version = 2;
        let mut c = sample_meta();
        c.version = 3;
        // Write three different files; uncompressed-JSON should win.
        save_meta(&store, &c, MetaFormat::MsgPack, Codec::Zstd).await.unwrap();
        save_meta(&store, &b, MetaFormat::Json, Codec::Zstd).await.unwrap();
        save_meta(&store, &a, MetaFormat::Json, Codec::Uncompressed).await.unwrap();

        let (loaded, format, compression) = load_meta(&store).await.unwrap();
        assert_eq!(format, MetaFormat::Json);
        assert_eq!(compression, Codec::Uncompressed);
        assert_eq!(loaded.version, 1);
    }

    #[tokio::test]
    async fn save_overwrites_previous_meta() {
        let store = make_store();
        let meta1 = StoreMeta {
            version: 1,
            ..Default::default()
        };
        save_meta(&store, &meta1, MetaFormat::Json, Codec::Uncompressed)
            .await
            .unwrap();

        let mut meta2 = StoreMeta {
            version: 2,
            ..Default::default()
        };
        meta2
            .datasets
            .insert("new_ds".into(), DatasetMeta::default());
        save_meta(&store, &meta2, MetaFormat::Json, Codec::Uncompressed)
            .await
            .unwrap();

        let (loaded, _, _) = load_meta(&store).await.unwrap();
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
