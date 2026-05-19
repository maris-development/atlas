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
        "i8" | "int8" => Attr::Int8(value.extract()?),
        "i16" | "int16" => Attr::Int16(value.extract()?),
        "i32" | "int32" => Attr::Int32(value.extract()?),
        "i64" | "int64" => Attr::Int64(value.extract()?),
        "u8" | "uint8" => Attr::UInt8(value.extract()?),
        "u16" | "uint16" => Attr::UInt16(value.extract()?),
        "u32" | "uint32" => Attr::UInt32(value.extract()?),
        "u64" | "uint64" => Attr::UInt64(value.extract()?),
        "f32" | "float32" => Attr::Float32(value.extract()?),
        "f64" | "float64" => Attr::Float64(value.extract()?),
        "string" | "str" => Attr::String(value.extract()?),
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown attribute dtype: {other:?}"
            )))
        }
    })
}

pub fn attr_to_py(py: Python<'_>, attr: &Attr) -> PyObject {
    match attr {
        Attr::Bool(v) => (*v).into_py(py),
        Attr::Int8(v) => (*v).into_py(py),
        Attr::Int16(v) => (*v).into_py(py),
        Attr::Int32(v) => (*v).into_py(py),
        Attr::Int64(v) => (*v).into_py(py),
        Attr::UInt8(v) => (*v).into_py(py),
        Attr::UInt16(v) => (*v).into_py(py),
        Attr::UInt32(v) => (*v).into_py(py),
        Attr::UInt64(v) => (*v).into_py(py),
        Attr::Float32(v) => (*v).into_py(py),
        Attr::Float64(v) => (*v).into_py(py),
        Attr::String(v) => v.clone().into_py(py),
    }
}
