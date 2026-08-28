use thiserror::Error;

/// Every error this crate returns.
#[derive(Debug, Error)]
pub enum Error {
    /// The named dataset is not in this collection, or the deletion mask hides
    /// it.
    #[error("dataset not found: {0}")]
    DatasetNotFound(String),
    /// A dataset of this name was already written to the collection being
    /// built. Names are unique within a collection.
    #[error("dataset already exists: {0}")]
    DatasetAlreadyExists(String),
    /// The named array is not defined in this dataset.
    #[error("array not found: {0}")]
    ArrayNotFound(String),
    /// An array of this name was already defined in the dataset being written.
    #[error("array already exists: {0}")]
    ArrayAlreadyExists(String),
    /// A dataset or array name failed validation. Names must be non-empty, and
    /// must not contain `/`, equal `.` or `..`, or start with `_`.
    #[error("invalid name '{0}': must be non-empty, no '/', no '..', no leading '_'")]
    InvalidName(String),
    /// The object at `data.atlas` is not an atlas container. `hint` says what
    /// gave it away.
    #[error("not an atlas collection: {hint}")]
    NotAnAtlasCollection {
        /// Why the file was rejected.
        hint: String,
    },
    /// The container declares a format version this build cannot read.
    #[error(
        "unsupported collection format version {found}; this atlas reads version {expected} \
         (rewrite the collection with a matching atlas to upgrade)"
    )]
    UnsupportedVersion {
        /// Version found on disk.
        found: u32,
        /// Version this build expects.
        expected: u32,
    },
    /// The container framing is intact but its footer does not describe a
    /// usable collection, for example a dataset referencing a schema that is
    /// not in the pool.
    #[error("corrupt collection: {0}")]
    CorruptCollection(String),
    /// The object at `deleted.mask` is not a deletion mask. Ordinals that name
    /// no dataset are ignored rather than reported here.
    #[error("corrupt deletion mask: {0}")]
    CorruptMask(String),
    /// A writer method was called after `finish`.
    #[error("this writer has already finished; create a new collection to write more")]
    WriterFinished,
    /// An internal invariant was violated. A bug in atlas rather than bad
    /// input.
    #[error("internal error: {0}")]
    Internal(String),
    /// Failure inside `array-format`, which encodes the segments.
    #[error("array format error: {0}")]
    ArrayFormat(#[from] array_format::Error),
    /// Local filesystem failure, from the scratch area or from `*_path`.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// Failed to encode the container footer.
    #[error("footer encode error: {0}")]
    MetaEncode(#[from] rmp_serde::encode::Error),
    /// Failed to decode the container footer.
    #[error("footer decode error: {0}")]
    MetaDecode(#[from] rmp_serde::decode::Error),
    /// Failure from the backing object store: local filesystem, S3, GCS,
    /// Azure, or in-memory.
    #[error("object store error: {0}")]
    ObjectStore(#[from] object_store::Error),
}

/// Convenience alias for `Result<T, atlas::Error>`.
pub type Result<T> = std::result::Result<T, Error>;
