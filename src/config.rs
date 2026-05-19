/// Compression codec applied when writing new array blocks.
///
/// The codec is stored per-variable in `array_store.json` so that each array
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

/// Configuration for opening or creating an [`ArrayStore`](crate::ArrayStore).
#[derive(Debug, Clone, Default)]
pub struct StoreConfig {
    /// Compression codec used when writing array blocks. Defaults to [`Codec::Zstd`].
    pub codec: Codec,
}
