/// Compression codec applied when writing new array blocks.
///
/// The codec is stored per-variable in `atlas.json` so that each array
/// can be reopened with the correct codec regardless of the store-level default.
/// Existing blocks are always decompressed using whatever codec they were
/// originally written with, so the choice only affects the write path.
///
/// Also reused for metadata compression via
/// [`StoreConfig::meta_compression`] — the same three options apply.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Codec {
    /// Zstd compression (default). Best compression ratio at moderate speed.
    #[default]
    Zstd,
    /// LZ4 compression. Faster than Zstd, larger files.
    Lz4,
    /// No compression. Fastest write path, no size reduction.
    Uncompressed,
}

/// On-disk encoding for the store's metadata file.
///
/// The format choice lives in the filename (`atlas.json` vs `atlas.msgpack`,
/// with optional `.zst` / `.lz4` suffix) rather than inside the file, so
/// [`crate::Atlas::open`] can detect it without a caller-supplied hint.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum MetaFormat {
    /// Pretty-printed JSON (`atlas.json`). Human-readable, default for
    /// backwards compatibility with stores created before this option existed.
    #[default]
    Json,
    /// MessagePack (`atlas.msgpack`). Compact binary encoding — typically
    /// 30–50% smaller than JSON and faster to parse, but not human-readable.
    MsgPack,
}

impl MetaFormat {
    /// Filename used for this format / compression pair. All six combinations
    /// resolve to a `&'static str` so the result can be passed straight to
    /// [`object_store::path::Path::from`].
    pub(crate) const fn filename(self, compression: Codec) -> &'static str {
        match (self, compression) {
            (MetaFormat::Json, Codec::Uncompressed) => "atlas.json",
            (MetaFormat::Json, Codec::Zstd) => "atlas.json.zst",
            (MetaFormat::Json, Codec::Lz4) => "atlas.json.lz4",
            (MetaFormat::MsgPack, Codec::Uncompressed) => "atlas.msgpack",
            (MetaFormat::MsgPack, Codec::Zstd) => "atlas.msgpack.zst",
            (MetaFormat::MsgPack, Codec::Lz4) => "atlas.msgpack.lz4",
        }
    }
}

/// The six (format, compression) pairs in priority order — used by `open`
/// to disambiguate when multiple metadata files happen to coexist.
/// Uncompressed comes before compressed within each format, and JSON comes
/// before MsgPack overall (preserving the existing "JSON wins if both
/// exist" precedent from the format-detection logic).
pub(crate) const META_VARIANTS: [(MetaFormat, Codec); 6] = [
    (MetaFormat::Json, Codec::Uncompressed),
    (MetaFormat::Json, Codec::Zstd),
    (MetaFormat::Json, Codec::Lz4),
    (MetaFormat::MsgPack, Codec::Uncompressed),
    (MetaFormat::MsgPack, Codec::Zstd),
    (MetaFormat::MsgPack, Codec::Lz4),
];

/// What to do when a dataset declares a type that cannot merge with the type
/// the collection already records for that array name / attribute key.
///
/// Types merge when they widen — within numeric types, or between string and
/// timestamp (see the merged schema in
/// [`Atlas::merged_schema`](crate::Atlas::merged_schema)). Anything else (say
/// an `int32` array in one dataset and a `string` array under the same name in
/// another) is a mismatch. Either way the dataset is **still stored** with its
/// own type, and the merged schema keeps the **first-seen** type; this policy
/// only decides whether the mismatch is reported as a warning or an error.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum TypeMismatchPolicy {
    /// Log a `tracing` warning and carry on. The default — an ingest of many
    /// heterogeneous files shouldn't abort because one of them disagrees.
    #[default]
    Warn,
    /// Reject the insert with [`Error::TypeMismatch`](crate::Error::TypeMismatch).
    Error,
}

/// Configuration for opening or creating an [`Atlas`](crate::Atlas).
#[derive(Debug, Clone)]
pub struct StoreConfig {
    /// Compression codec used when writing array blocks. Defaults to [`Codec::Zstd`].
    pub codec: Codec,
    /// How to report a type that can't merge with the collection's existing
    /// type for an array/attribute. Defaults to [`TypeMismatchPolicy::Warn`].
    ///
    /// Unlike the other fields, this is a per-session policy rather than an
    /// on-disk property: it is honoured by `create` and by the
    /// `*_with_config` open variants, and is not persisted to `atlas.json`.
    pub on_type_mismatch: TypeMismatchPolicy,
    /// On-disk encoding for the metadata file. Defaults to [`MetaFormat::Json`].
    /// Only consulted by `create`; `open` detects the format from the filename
    /// present on disk.
    pub meta_format: MetaFormat,
    /// Compression applied to the encoded metadata bytes. Defaults to
    /// [`Codec::Uncompressed`] so the on-disk filename matches the format
    /// (`atlas.json` / `atlas.msgpack`). Only consulted by `create`; `open`
    /// detects compression from the filename suffix on disk.
    pub meta_compression: Codec,
}

// Manual Default — `meta_compression` defaults to `Uncompressed`, not
// `Codec::default()` (which is `Zstd`), so new stores keep the legacy
// `atlas.json` / `atlas.msgpack` filenames unless the caller opts in.
impl Default for StoreConfig {
    fn default() -> Self {
        Self {
            codec: Codec::default(),
            on_type_mismatch: TypeMismatchPolicy::default(),
            meta_format: MetaFormat::default(),
            meta_compression: Codec::Uncompressed,
        }
    }
}
