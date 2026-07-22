//! The `atlas.json` / `atlas.msgpack` on-disk format: encoding, decoding, and
//! reading/writing it against an object store.
//!
//! On disk, identical dataset schemas are **interned** — each distinct
//! [`DatasetSchema`] is stored once in `schemas` and referenced by index — so a
//! homogeneous collection stores its schema once no matter how many datasets
//! share it.

use std::sync::Arc;

use indexmap::IndexMap;
use object_store::{ObjectStore, ObjectStoreExt, path::Path};
use serde::{Deserialize, Serialize};
use tracing::warn;

use super::schema::{DatasetSchema, MergedSchema, compute_merged};
use super::{STORE_FORMAT_VERSION, StoreMeta};
use crate::{
    Error, Result,
    config::{Codec, META_VARIANTS, MetaFormat},
};

/// On-disk wire form of [`StoreMeta`].
#[derive(Serialize, Deserialize)]
struct StoreMetaWire {
    version: u32,
    #[serde(default)]
    codec: Codec,
    /// Pool of distinct dataset schemas, in first-seen order.
    #[serde(default)]
    schemas: Vec<DatasetSchema>,
    /// Dataset name → index into `schemas`, insertion-ordered and **including
    /// tombstoned datasets**, whose ordinals must be preserved.
    #[serde(default)]
    datasets: IndexMap<String, usize>,
    /// Ordinals in `datasets` that are tombstoned — the logical delete mask.
    #[serde(default)]
    deleted: Vec<u32>,
    /// See [`StoreMeta::meta_epoch`].
    #[serde(default)]
    meta_epoch: u64,
    /// Collection-wide merged schema. Derived from `schemas`, written for
    /// external tooling, ignored on load (per-dataset schemas are the truth).
    #[serde(default)]
    merged: MergedSchema,
}

/// Minimal probe to read the version before a full decode, so a store written
/// by an older atlas fails clearly rather than with an opaque parse error.
#[derive(Deserialize)]
struct MetaVersion {
    #[serde(default)]
    version: u32,
}

/// Build the interned wire form from in-memory metadata.
///
/// Schemas are already interned in memory, so deduplication is a pointer-keyed
/// hash lookup; a fallback equality scan catches distinct-but-equal schemas
/// (possible only for datasets never sealed).
///
/// **Tombstones are written too** — this is the one enumeration over datasets
/// that must not filter by liveness. A dataset's position is its pruning-index
/// row ordinal, so dropping dead entries would shift every later dataset up one
/// on reload and silently re-point every row after the first deletion.
fn to_wire(meta: &StoreMeta) -> StoreMetaWire {
    let mut schemas: Vec<DatasetSchema> = Vec::new();
    let mut by_ptr: std::collections::HashMap<*const DatasetSchema, usize> =
        std::collections::HashMap::new();
    let mut datasets: IndexMap<String, usize> = IndexMap::with_capacity(meta.entries().len());
    for (name, schema) in meta.entries() {
        let ptr = Arc::as_ptr(schema);
        let idx = *by_ptr.entry(ptr).or_insert_with(|| {
            match schemas.iter().position(|s| s == &**schema) {
                Some(i) => i,
                None => {
                    schemas.push((**schema).clone());
                    schemas.len() - 1
                }
            }
        });
        datasets.insert(name.clone(), idx);
    }
    StoreMetaWire {
        version: meta.version,
        codec: meta.codec,
        schemas,
        datasets,
        deleted: meta.deleted_ordinals(),
        meta_epoch: meta.meta_epoch,
        merged: compute_merged(meta.live_schemas()),
    }
}

/// Expand the interned wire form, sharing the pooled schema for every dataset
/// that references the same index.
fn from_wire(wire: StoreMetaWire) -> Result<StoreMeta> {
    // One allocation per distinct schema, shared by every dataset that uses it.
    let pooled: Vec<Arc<DatasetSchema>> = wire.schemas.into_iter().map(Arc::new).collect();
    let mut datasets: IndexMap<String, Arc<DatasetSchema>> =
        IndexMap::with_capacity(wire.datasets.len());
    for (name, idx) in wire.datasets {
        let schema = pooled.get(idx).cloned().ok_or_else(|| {
            corrupt(format!(
                "dataset '{name}' references schema index {idx} of {}",
                pooled.len()
            ))
        })?;
        datasets.insert(name, schema);
    }

    // Rebuild the liveness mask from the persisted ordinals.
    let mut live = vec![true; datasets.len()];
    for ordinal in &wire.deleted {
        *live.get_mut(*ordinal as usize).ok_or_else(|| {
            corrupt(format!(
                "deleted ordinal {ordinal} of {} datasets",
                datasets.len()
            ))
        })? = false;
    }

    Ok(StoreMeta::from_loaded(
        wire.version,
        wire.codec,
        wire.meta_epoch,
        datasets,
        live,
    ))
}

fn corrupt(detail: String) -> Error {
    Error::CorruptMetadata(detail)
}

/// Load store metadata, auto-detecting the encoding format and compression from
/// the on-disk filename.
///
/// A single [`ObjectStore::list_with_delimiter`] enumerates the root and
/// matches the six known `atlas.{json,msgpack}{,.zst,.lz4}` filenames. If more
/// than one is present (only if hand-edited), [`META_VARIANTS`] priority order
/// decides — uncompressed before compressed, JSON before MsgPack.
///
/// No metadata file found returns the default (empty) metadata with
/// `(Json, Uncompressed)`, so a fresh store gets the legacy `atlas.json` on its
/// first save. The returned `(MetaFormat, Codec)` is what later saves reuse so
/// the same file is overwritten instead of leaving stale copies.
pub(crate) async fn load_meta(
    store: &Arc<dyn ObjectStore>,
) -> Result<(StoreMeta, MetaFormat, Codec)> {
    let listing = store
        .list_with_delimiter(None)
        .await
        .map_err(Error::ObjectStore)?;
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
            let names: Vec<&str> = many.iter().map(|&(f, c)| f.filename(c)).collect();
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
    let meta = decode(&compression.decompress(&bytes)?, format)?;
    Ok((meta, format, compression))
}

/// Encode + compress + atomically replace the on-disk metadata file.
pub(crate) async fn save_meta(
    store: &Arc<dyn ObjectStore>,
    meta: &StoreMeta,
    format: MetaFormat,
    compression: Codec,
) -> Result<()> {
    let bytes = compression.compress(encode(meta, format)?)?;
    store
        .put(&Path::from(format.filename(compression)), bytes.into())
        .await
        .map_err(Error::ObjectStore)?;
    Ok(())
}

fn decode(bytes: &[u8], format: MetaFormat) -> Result<StoreMeta> {
    // Read the version first so an old store fails with a clear message rather
    // than an opaque schema-shape parse error.
    let probe: MetaVersion = deserialize(bytes, format)?;
    if probe.version != STORE_FORMAT_VERSION {
        return Err(Error::UnsupportedVersion {
            found: probe.version,
            expected: STORE_FORMAT_VERSION,
        });
    }
    from_wire(deserialize(bytes, format)?)
}

fn encode(meta: &StoreMeta, format: MetaFormat) -> Result<Vec<u8>> {
    let wire = to_wire(meta);
    Ok(match format {
        MetaFormat::Json => serde_json::to_vec_pretty(&wire)?,
        MetaFormat::MsgPack => rmp_serde::to_vec_named(&wire)?,
    })
}

fn deserialize<T: for<'de> Deserialize<'de>>(bytes: &[u8], format: MetaFormat) -> Result<T> {
    Ok(match format {
        MetaFormat::Json => serde_json::from_slice(bytes)?,
        MetaFormat::MsgPack => rmp_serde::from_slice(bytes)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::ArraySchema;
    use array_format::DType;
    use object_store::memory::InMemory;

    fn make_store() -> Arc<dyn ObjectStore> {
        Arc::new(InMemory::new())
    }

    fn sample_schema() -> DatasetSchema {
        let mut s = DatasetSchema {
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
            ..Default::default()
        };
        s.register_global_attr("month", DType::Int64);
        s.register_global_attr("active", DType::Bool);
        s.register_array_attr("temp", "units", DType::String);
        s
    }

    fn sample_meta() -> StoreMeta {
        let mut meta = StoreMeta::new(Codec::default());
        meta.insert_dataset("ds1", sample_schema());
        meta
    }

    #[tokio::test]
    async fn load_meta_missing_returns_default_json_uncompressed() {
        let store = make_store();
        let (meta, format, compression) = load_meta(&store).await.unwrap();
        assert_eq!(meta.version, 0);
        assert_eq!(meta.row_slots(), 0);
        assert_eq!(format, MetaFormat::Json);
        assert_eq!(compression, Codec::Uncompressed);
    }

    /// Round-trip every (format, compression) pair; the detected pair must
    /// match what was written, and the schema must survive.
    #[tokio::test]
    async fn save_and_load_roundtrip_all_variants() {
        for &(format, compression) in &META_VARIANTS {
            let store = make_store();
            save_meta(&store, &sample_meta(), format, compression).await.unwrap();

            let (loaded, detected_fmt, detected_comp) = load_meta(&store).await.unwrap();
            assert_eq!(detected_fmt, format);
            assert_eq!(detected_comp, compression);
            assert_eq!(loaded.version, STORE_FORMAT_VERSION);

            let ds = loaded.live_schema("ds1").unwrap();
            assert_eq!(ds.arrays["temp"].dtype, DType::Float32);
            assert_eq!(ds.arrays["temp"].shape, vec![4, 8]);
            assert_eq!(ds.global_attrs.keys().collect::<Vec<_>>(), vec!["month", "active"]);
            assert_eq!(ds.global_attrs["month"].0, DType::Int64);
            assert_eq!(ds.array_attrs["temp"]["units"].0, DType::String);
        }
    }

    /// Identical schemas are pooled to a single wire entry and reload equal.
    #[tokio::test]
    async fn identical_schemas_are_interned() {
        let mut meta = StoreMeta::new(Codec::default());
        meta.insert_dataset("a", sample_schema());
        meta.insert_dataset("b", sample_schema());
        meta.insert_dataset("c", sample_schema());
        let mut other = sample_schema();
        other.register_global_attr("extra", DType::Float64);
        meta.insert_dataset("d", other);

        let wire = to_wire(&meta);
        assert_eq!(wire.schemas.len(), 2, "three identical + one distinct");
        assert_eq!(wire.datasets["a"], wire.datasets["b"]);
        assert_eq!(wire.datasets["a"], wire.datasets["c"]);
        assert_ne!(wire.datasets["a"], wire.datasets["d"]);

        let store = make_store();
        save_meta(&store, &meta, MetaFormat::Json, Codec::Uncompressed)
            .await
            .unwrap();
        let (loaded, _, _) = load_meta(&store).await.unwrap();
        assert_eq!(loaded.row_slots(), 4);
        assert_eq!(loaded.live_schema("a"), loaded.live_schema("b"));
        assert_eq!(loaded.live_schema("a").unwrap().arrays["temp"].shape, vec![4, 8]);
        assert!(loaded.live_schema("d").unwrap().global_attrs.contains_key("extra"));
    }

    #[tokio::test]
    async fn msgpack_is_smaller_than_json() {
        let meta = sample_meta();
        let json = encode(&meta, MetaFormat::Json).unwrap();
        let mp = encode(&meta, MetaFormat::MsgPack).unwrap();
        assert!(mp.len() < json.len(), "msgpack {} vs json {}", mp.len(), json.len());
    }

    /// Compression shrinks the encoded bytes on a workload big enough to beat
    /// framing overhead.
    #[tokio::test]
    async fn compression_shrinks_encoded_bytes() {
        let mut meta = StoreMeta::new(Codec::default());
        for i in 0..30 {
            let mut ds = DatasetSchema::default();
            for j in 0..5 {
                ds.arrays.insert(
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
            ds.register_global_attr(&format!("k_{i}"), DType::Int64); // defeat interning
            meta.insert_dataset(&format!("dataset_{i}"), ds);
        }

        for format in [MetaFormat::Json, MetaFormat::MsgPack] {
            let raw = encode(&meta, format).unwrap();
            assert!(Codec::Zstd.compress(raw.clone()).unwrap().len() < raw.len(), "{format:?} zstd");
            assert!(Codec::Lz4.compress(raw.clone()).unwrap().len() < raw.len(), "{format:?} lz4");
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

    /// With several metadata files present, uncompressed JSON wins.
    #[tokio::test]
    async fn load_priority_order_when_many_present() {
        let store = make_store();
        let mut a = sample_meta();
        a.insert_dataset("only_a", DatasetSchema::default());
        save_meta(&store, &sample_meta(), MetaFormat::MsgPack, Codec::Zstd).await.unwrap();
        save_meta(&store, &sample_meta(), MetaFormat::Json, Codec::Zstd).await.unwrap();
        save_meta(&store, &a, MetaFormat::Json, Codec::Uncompressed).await.unwrap();

        let (loaded, format, compression) = load_meta(&store).await.unwrap();
        assert_eq!(format, MetaFormat::Json);
        assert_eq!(compression, Codec::Uncompressed);
        assert!(loaded.is_live("only_a"));
    }

    #[tokio::test]
    async fn save_overwrites_previous_meta() {
        let store = make_store();
        save_meta(&store, &StoreMeta::new(Codec::default()), MetaFormat::Json, Codec::Uncompressed)
            .await
            .unwrap();

        let mut meta2 = StoreMeta::new(Codec::default());
        meta2.add_dataset("new_ds");
        save_meta(&store, &meta2, MetaFormat::Json, Codec::Uncompressed)
            .await
            .unwrap();

        let (loaded, _, _) = load_meta(&store).await.unwrap();
        assert!(loaded.is_live("new_ds"));
    }

    /// A store from an older atlas (version != 3) is rejected clearly rather
    /// than misparsed.
    #[tokio::test]
    async fn legacy_version_rejected() {
        let store = make_store();
        let mut legacy = StoreMeta::new(Codec::default());
        legacy.version = 1;
        save_meta(&store, &legacy, MetaFormat::Json, Codec::Uncompressed)
            .await
            .unwrap();
        let err = load_meta(&store).await.unwrap_err();
        assert!(
            matches!(err, Error::UnsupportedVersion { found: 1, expected: 3 }),
            "got {err:?}"
        );
    }

    #[test]
    fn merged_schema_serialized_in_atlas_json() {
        let mut meta = StoreMeta::new(Codec::default());
        let mut schema = DatasetSchema::default();
        schema.arrays.insert(
            "temp".into(),
            ArraySchema {
                dtype: DType::Int32,
                shape: vec![2],
                chunk_shape: vec![2],
                dimension_names: vec!["x".into()],
                codec: Codec::default(),
            },
        );
        meta.insert_dataset("a", schema);
        let json = String::from_utf8(encode(&meta, MetaFormat::Json).unwrap()).unwrap();
        assert!(json.contains("\"merged\""), "atlas.json must include merged schema:\n{json}");
    }
}
