//! Where a collection lives, and how it is compressed. Shared by the reader
//! and the writer bindings.

use std::path::PathBuf;

use atlas::Codec;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_object_store::AnyObjectStore;

/// Either a local filesystem path or an obstore-constructed store handle.
/// `Atlas.open` and `AtlasWriter.create` accept both.
///
/// PyO3 tries the `ObjectStore` variant first, via `AnyObjectStore`'s own
/// `FromPyObject`, which accepts native pyo3-object_store instances and
/// externally-constructed handles such as `obstore.store.S3Store(...)`. Strings
/// and `os.PathLike` fall through to the `Path` arm.
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
