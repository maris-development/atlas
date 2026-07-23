use thiserror::Error;

/// Every error returned by this crate. Each variant carries enough context
/// to identify what failed; `Display` (via [`thiserror`]) renders the same
/// message shown in the `///` line above each variant.
#[derive(Debug, Error)]
pub enum Error {
    /// The named dataset doesn't exist in this store.
    #[error("dataset not found: {0}")]
    DatasetNotFound(String),
    /// A dataset with this name was already created in this store.
    #[error("dataset already exists: {0}")]
    DatasetAlreadyExists(String),
    /// The named array isn't defined in the relevant dataset.
    #[error("array not found: {0}")]
    ArrayNotFound(String),
    /// An array with this name was already defined in the dataset.
    #[error("array already exists: {0}")]
    ArrayAlreadyExists(String),
    /// Dataset or array name failed validation. Names must be non-empty,
    /// cannot contain `/`, `..`, or `.`, and cannot start with `_`.
    #[error("invalid name '{0}': must be non-empty, no '/', no '..', no leading '_'")]
    InvalidName(String),
    /// Two datasets declare the same array name (or attribute key) with
    /// incompatible types. Widening is only allowed within numeric types or
    /// between string and timestamp; anything else collides.
    #[error(
        "type mismatch for {name}: existing type {existing} cannot merge with {new} \
         (widening is only allowed within numeric types or between string and timestamp)"
    )]
    TypeMismatch {
        /// Array name or attribute key that collided.
        name: String,
        /// The already-recorded (merged) type.
        existing: String,
        /// The incompatible new type being inserted.
        new: String,
    },
    /// The store's on-disk metadata (`atlas.json` / `atlas.msgpack`) is
    /// internally inconsistent — e.g. a dataset references a schema index that
    /// doesn't exist, or a tombstone ordinal is out of range. Not raised for
    /// mere parse failures (those surface as [`Error::Meta`] / [`Error::MetaDecode`]).
    #[error("corrupt store metadata: {0}")]
    CorruptMetadata(String),
    /// The `pruning.idx` file is malformed, or stale relative to the metadata
    /// (its epoch doesn't match). A stale index is recoverable — flush to
    /// rebuild it.
    #[error("corrupt or stale pruning index: {0}")]
    CorruptIndex(String),
    /// An internal invariant was violated — a spawned read task failed, or a
    /// buffer had an unexpected shape. Indicates a bug in atlas rather than bad
    /// input, and shouldn't occur in normal use.
    #[error("internal error: {0}")]
    Internal(String),
    /// Underlying `array-format` failure — see the wrapped error for the
    /// specific block/codec/storage problem.
    #[error("array format error: {0}")]
    ArrayFormat(#[from] array_format::Error),
    /// Local filesystem I/O failure (used by `create_path` / `open_path`).
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// The store's metadata declares a format version this build of atlas
    /// cannot read. Stores written by an older atlas (which inlined
    /// per-dataset attributes and duplicated schemas) must be re-exported.
    #[error(
        "unsupported store format version {found}; this atlas reads version {expected} \
         (store written by an older atlas — re-export it to upgrade)"
    )]
    UnsupportedVersion {
        /// Version found in the on-disk metadata.
        found: u32,
        /// Version this build expects.
        expected: u32,
    },
    /// Failed to parse the JSON form of the store metadata.
    #[error("metadata error: {0}")]
    Meta(#[from] serde_json::Error),
    /// Failed to encode store metadata to MessagePack (`atlas.msgpack` /
    /// `atlas.msgpack.zst` / `atlas.msgpack.lz4`).
    #[error("metadata encode error: {0}")]
    MetaEncode(#[from] rmp_serde::encode::Error),
    /// Failed to decode the MessagePack form of the store metadata.
    #[error("metadata decode error: {0}")]
    MetaDecode(#[from] rmp_serde::decode::Error),
    /// Failed to LZ4-decompress the on-disk metadata file
    /// (`atlas.json.lz4` / `atlas.msgpack.lz4`).
    #[error("metadata lz4 decompress error: {0}")]
    MetaLz4Decompress(#[from] lz4_flex::block::DecompressError),
    /// Underlying `object_store` failure — bubbled up from the backend
    /// (local FS, S3, GCS, Azure, in-memory).
    #[error("object store error: {0}")]
    ObjectStore(#[from] object_store::Error),
}

/// Convenience alias for `Result<T, atlas::Error>` returned by every
/// fallible operation in the crate.
pub type Result<T> = std::result::Result<T, Error>;
