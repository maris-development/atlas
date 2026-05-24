/// Compression codec applied when writing new array blocks.
///
/// The codec is stored per-variable in `atlas.json` so that each array
/// can be reopened with the correct codec regardless of the store-level default.
/// Existing blocks are always decompressed using whatever codec they were
/// originally written with, so the choice only affects the write path.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
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
/// The format choice lives in the filename (`atlas.json` vs `atlas.msgpack`)
/// rather than inside the file, so [`crate::Atlas::open`] can detect it
/// without a caller-supplied hint.
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
    pub(crate) const fn filename(self) -> &'static str {
        match self {
            MetaFormat::Json => "atlas.json",
            MetaFormat::MsgPack => "atlas.msgpack",
        }
    }
}

/// Configuration for opening or creating an [`Atlas`](crate::Atlas).
#[derive(Debug, Clone, Default)]
pub struct StoreConfig {
    /// Compression codec used when writing array blocks. Defaults to [`Codec::Zstd`].
    pub codec: Codec,
    /// On-disk encoding for the metadata file. Defaults to [`MetaFormat::Json`].
    /// Only consulted by `create`; `open` detects the format from the filename
    /// present on disk.
    pub meta_format: MetaFormat,
}
