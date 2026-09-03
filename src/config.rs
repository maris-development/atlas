//! Write-time configuration. Reading needs none. Every block records its own
//! codec, and the container framing is fixed.

/// Compression codec applied to array blocks as the collection is written.
///
/// The choice affects the write path only. Each block stores its own codec.
/// A reader decodes what it finds, and needs no argument.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize,
)]
pub enum Codec {
    /// Zstd. The default. It gives the best ratio at moderate speed.
    #[default]
    Zstd,
    /// LZ4. Faster than Zstd, larger output.
    Lz4,
    /// No compression. The fastest write path. It makes the file no smaller.
    Uncompressed,
}

/// Block size `array-format` aims at when it packs chunks. Chunks below this
/// size share a block. A larger chunk gets a block of its own.
pub(crate) const DEFAULT_BLOCK_TARGET_SIZE: usize = 8 * 1024 * 1024;

/// Bytes one variable may hold in memory before its staging file flushes.
///
/// `array-format` keeps a pending write in memory until `flush`, and each
/// flush seals a sidecar layer that `compact` must later merge. A small budget
/// costs layers, and a large one costs memory. Every dataset stages before any
/// of it is written out, so without a budget the whole collection would sit in
/// memory at once.
pub(crate) const STAGING_FLUSH_BUDGET: usize = 64 * 1024 * 1024;

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
