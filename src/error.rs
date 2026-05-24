use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("dataset not found: {0}")]
    DatasetNotFound(String),
    #[error("dataset already exists: {0}")]
    DatasetAlreadyExists(String),
    #[error("array not found: {0}")]
    ArrayNotFound(String),
    #[error("array already exists: {0}")]
    ArrayAlreadyExists(String),
    #[error("store not found at path")]
    StoreNotFound,
    #[error("invalid name '{0}': must be non-empty, no '/', no '..', no leading '_'")]
    InvalidName(String),
    #[error("array format error: {0}")]
    ArrayFormat(#[from] array_format::Error),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("metadata error: {0}")]
    Meta(#[from] serde_json::Error),
    #[error("metadata encode error: {0}")]
    MetaEncode(#[from] rmp_serde::encode::Error),
    #[error("metadata decode error: {0}")]
    MetaDecode(#[from] rmp_serde::decode::Error),
    #[error("object store error: {0}")]
    ObjectStore(#[from] object_store::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
