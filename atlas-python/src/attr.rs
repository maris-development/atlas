use atlas::Attr;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyFloat, PyInt, PyString};

pub fn py_to_attr(value: &Bound<'_, PyAny>, dtype_hint: Option<&str>) -> PyResult<Attr> {
    if let Some(hint) = dtype_hint {
        return py_to_attr_typed(value, hint);
    }

    if let Ok(b) = value.downcast::<PyBool>() {
        return Ok(Attr::Bool(b.is_true()));
    }
    if value.downcast::<PyString>().is_ok() {
        return Ok(Attr::String(value.extract::<String>()?));
    }
    if value.downcast::<PyInt>().is_ok() {
        return Ok(Attr::Int64(value.extract::<i64>()?));
    }
    if value.downcast::<PyFloat>().is_ok() {
        return Ok(Attr::Float64(value.extract::<f64>()?));
    }
    Err(PyValueError::new_err(format!(
        "unsupported attribute type: {:?}",
        value.get_type().name()?
    )))
}

fn py_to_attr_typed(value: &Bound<'_, PyAny>, dtype: &str) -> PyResult<Attr> {
    Ok(match dtype.to_ascii_lowercase().as_str() {
        "bool" => Attr::Bool(value.extract()?),
        // All integer hints land in Int64. Python `int.extract::<i64>()`
        // raises OverflowError on overflow, which surfaces as a PyErr.
        "i8" | "int8" | "i16" | "int16" | "i32" | "int32" | "i64" | "int64"
        | "u8" | "uint8" | "u16" | "uint16" | "u32" | "uint32" | "u64" | "uint64" => {
            Attr::Int64(value.extract()?)
        }
        "f32" | "float32" | "f64" | "float64" => Attr::Float64(value.extract()?),
        "string" | "str" => Attr::String(value.extract()?),
        "timestamp_ns" | "timestamp_nanoseconds" | "datetime64[ns]" => {
            Attr::TimestampNanoseconds(value.extract()?)
        }
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown attribute dtype: {other:?}"
            )))
        }
    })
}

pub fn attr_to_py(py: Python<'_>, attr: &Attr) -> PyResult<Py<PyAny>> {
    use pyo3::IntoPyObjectExt;
    match attr {
        Attr::Bool(v) => (*v).into_py_any(py),
        Attr::Int64(v) => (*v).into_py_any(py),
        Attr::Float64(v) => (*v).into_py_any(py),
        Attr::String(v) => v.clone().into_py_any(py),
        Attr::TimestampNanoseconds(v) => (*v).into_py_any(py),
    }
}
