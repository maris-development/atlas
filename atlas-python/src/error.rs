use pyo3::exceptions::{
    PyFileExistsError, PyKeyError, PyOSError, PyRuntimeError, PyValueError,
};
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
        atlas::Error::InvalidName(name) => {
            PyValueError::new_err(format!("invalid name: {name}"))
        }
        e @ (atlas::Error::UnsupportedVersion { .. } | atlas::Error::TypeMismatch { .. }) => {
            PyValueError::new_err(e.to_string())
        }
        atlas::Error::Io(e) => PyOSError::new_err(e.to_string()),
        // On-disk inconsistency or an atlas-internal invariant violation: a
        // runtime failure the caller can't fix by changing arguments.
        e @ (atlas::Error::CorruptMetadata(_)
        | atlas::Error::CorruptIndex(_)
        | atlas::Error::Internal(_)
        | atlas::Error::ObjectStore(_)
        | atlas::Error::ArrayFormat(_)
        | atlas::Error::Meta(_)
        | atlas::Error::MetaEncode(_)
        | atlas::Error::MetaDecode(_)
        | atlas::Error::MetaLz4Decompress(_)) => PyRuntimeError::new_err(e.to_string()),
    }
}
