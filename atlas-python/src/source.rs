//! Where a collection lives, and how it compresses. The reader bindings and
//! the writer bindings share this.

use std::path::PathBuf;

use atlas::Codec;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_object_store::AnyObjectStore;

/// Either a local filesystem path or an obstore-constructed store handle.
/// `Atlas.open` and `AtlasWriter.create` accept both.
///
/// PyO3 tries the `ObjectStore` variant first, through the `FromPyObject` of
/// `AnyObjectStore`. That accepts a native pyo3-object_store instance, and an
/// outside handle such as `obstore.store.S3Store(...)`. A string and an
/// `os.PathLike` fall through to the `Path` arm.
#[derive(FromPyObject)]
pub enum AtlasSource {
    ObjectStore(AnyObjectStore),
    Path(PathBuf),
}

pub fn parse_codec(s: &str) -> PyResult<Codec> {
    Ok(match s.to_ascii_lowercase().as_str() {
        "zstd" => Codec::Zstd,
        "lz4" => Codec::Lz4,
        "none" | "uncompressed" => Codec::Uncompressed,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown codec: {other:?} (expected 'zstd', 'lz4', or 'none')"
            )));
        }
    })
}
