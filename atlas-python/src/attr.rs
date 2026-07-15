use atlas::Attr;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyFloat, PyInt, PyList, PyString};

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
    if value.downcast::<PyBytes>().is_ok() {
        return Ok(Attr::Binary(value.extract::<Vec<u8>>()?));
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
        // Width-precise integer hints so the on-disk attribute keeps its type.
        // Python `int.extract::<T>()` raises OverflowError on overflow, which
        // surfaces as a PyErr.
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
        "binary" | "bytes" => Attr::Binary(value.extract()?),
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
        Attr::Int8(v) => (*v).into_py_any(py),
        Attr::Int16(v) => (*v).into_py_any(py),
        Attr::Int32(v) => (*v).into_py_any(py),
        Attr::Int64(v) => (*v).into_py_any(py),
        Attr::UInt8(v) => (*v).into_py_any(py),
        Attr::UInt16(v) => (*v).into_py_any(py),
        Attr::UInt32(v) => (*v).into_py_any(py),
        Attr::UInt64(v) => (*v).into_py_any(py),
        Attr::Float32(v) => (*v).into_py_any(py),
        Attr::Float64(v) => (*v).into_py_any(py),
        Attr::String(v) => v.clone().into_py_any(py),
        Attr::Binary(v) => PyBytes::new(py, v).into_py_any(py),
        Attr::TimestampNanoseconds(v) => (*v).into_py_any(py),
        Attr::BoolList(v) => PyList::new(py, v)?.into_py_any(py),
        Attr::Int8List(v) => PyList::new(py, v)?.into_py_any(py),
        Attr::Int16List(v) => PyList::new(py, v)?.into_py_any(py),
        Attr::Int32List(v) => PyList::new(py, v)?.into_py_any(py),
        Attr::Int64List(v) => PyList::new(py, v)?.into_py_any(py),
        Attr::UInt8List(v) => PyList::new(py, v)?.into_py_any(py),
        Attr::UInt16List(v) => PyList::new(py, v)?.into_py_any(py),
        Attr::UInt32List(v) => PyList::new(py, v)?.into_py_any(py),
        Attr::UInt64List(v) => PyList::new(py, v)?.into_py_any(py),
        Attr::Float32List(v) => PyList::new(py, v)?.into_py_any(py),
        Attr::Float64List(v) => PyList::new(py, v)?.into_py_any(py),
        Attr::StringList(v) => PyList::new(py, v)?.into_py_any(py),
        Attr::BinaryList(v) => {
            let items: Vec<Bound<'_, PyBytes>> = v.iter().map(|b| PyBytes::new(py, b)).collect();
            PyList::new(py, items)?.into_py_any(py)
        }
    }
}
