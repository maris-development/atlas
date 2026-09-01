use pyo3::exceptions::{PyFileExistsError, PyKeyError, PyOSError, PyRuntimeError, PyValueError};
use pyo3::PyErr;

pub fn to_py_err(err: atlas::Error) -> PyErr {
    match err {
        atlas::Error::DatasetNotFound(name) => {
            PyKeyError::new_err(format!("dataset not found: {name}"))
        }
        atlas::Error::ArrayNotFound(name) => {
            PyKeyError::new_err(format!("array not found: {name}"))
        }
        atlas::Error::DatasetAlreadyExists(name) => {
            PyFileExistsError::new_err(format!("dataset already exists: {name}"))
        }
        atlas::Error::ArrayAlreadyExists(name) => {
            PyFileExistsError::new_err(format!("array already exists: {name}"))
        }
        atlas::Error::InvalidName(name) => PyValueError::new_err(format!("invalid name: {name}")),
        // Bad input, not a broken store. The caller opened the wrong path,
        // or a collection this build is too old to read.
        e @ (atlas::Error::NotAnAtlasCollection { .. }
        | atlas::Error::UnsupportedVersion { .. }
        | atlas::Error::WriterFinished) => PyValueError::new_err(e.to_string()),
        atlas::Error::Io(e) => PyOSError::new_err(e.to_string()),
        // On-disk damage, or a broken atlas invariant. The caller cannot fix
        // this with a different argument.
        e @ (atlas::Error::CorruptCollection(_)
        | atlas::Error::CorruptMask(_)
        | atlas::Error::Internal(_)
        | atlas::Error::ObjectStore(_)
        | atlas::Error::ArrayFormat(_)
        | atlas::Error::MetaEncode(_)
        | atlas::Error::MetaDecode(_)) => PyRuntimeError::new_err(e.to_string()),
    }
}
