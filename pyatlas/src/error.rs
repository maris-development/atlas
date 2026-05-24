use pyo3::exceptions::{
    PyFileExistsError, PyFileNotFoundError, PyKeyError, PyOSError, PyRuntimeError, PyValueError,
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
        atlas::Error::StoreNotFound => PyFileNotFoundError::new_err("store not found at path"),
        atlas::Error::Io(e) => PyOSError::new_err(e.to_string()),
        e @ (atlas::Error::ObjectStore(_)
        | atlas::Error::ArrayFormat(_)
        | atlas::Error::Meta(_)
        | atlas::Error::MetaEncode(_)
        | atlas::Error::MetaDecode(_)) => PyRuntimeError::new_err(e.to_string()),
    }
}
