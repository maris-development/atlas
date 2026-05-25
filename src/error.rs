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
    /// `Atlas::open` / `open_path` was called against a location with no
    /// metadata file (neither `atlas.json` nor any of the `atlas.msgpack*`
    /// variants).
    #[error("store not found at path")]
    StoreNotFound,
    /// Dataset or array name failed validation. Names must be non-empty,
    /// cannot contain `/`, `..`, or `.`, and cannot start with `_`.
    #[error("invalid name '{0}': must be non-empty, no '/', no '..', no leading '_'")]
    InvalidName(String),
    /// Underlying `array-format` failure — see the wrapped error for the
    /// specific block/codec/storage problem.
    #[error("array format error: {0}")]
    ArrayFormat(#[from] array_format::Error),
    /// Local filesystem I/O failure (used by `create_path` / `open_path`).
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
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
