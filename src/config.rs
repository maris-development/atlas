//! Write-time configuration. Reading needs none: every block records the codec
//! that produced it, and the container framing is fixed.

/// Compression codec applied to array blocks as the collection is written.
///
/// The choice affects the write path only. Each block stores its own codec, so
/// a reader decodes whatever it finds without being told.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize,
)]
pub enum Codec {
    /// Zstd. The default: best ratio at moderate speed.
    #[default]
    Zstd,
    /// LZ4. Faster than Zstd, larger output.
    Lz4,
    /// No compression. Fastest write path, no size reduction.
    Uncompressed,
}

/// Block size `array-format` targets when it packs chunks. Chunks smaller than
/// this share a block; a larger chunk gets its own.
pub(crate) const DEFAULT_BLOCK_TARGET_SIZE: usize = 8 * 1024 * 1024;

/// Decompressed-block cache shared by every segment a reader opens.
pub(crate) const DEFAULT_CACHE_CAPACITY: u64 = 256 * 1024 * 1024;

/// Raw I/O cache shared by every segment a reader opens.
pub(crate) const DEFAULT_IO_CACHE_CAPACITY: u64 = 64 * 1024 * 1024;

/// Settings for building a collection.
#[derive(Debug, Clone)]
pub struct WriterConfig {
    /// Codec for array blocks. Defaults to [`Codec::Zstd`].
    pub codec: Codec,
    /// Target size of a compressed block, in bytes. Defaults to 8 MiB.
    pub block_target_size: usize,
}

impl Default for WriterConfig {
    fn default() -> Self {
        Self {
            codec: Codec::default(),
            block_target_size: DEFAULT_BLOCK_TARGET_SIZE,
        }
    }
}
