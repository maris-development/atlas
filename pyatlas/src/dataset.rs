use atlas::{DType, DatasetView};
use numpy::{IntoPyArray, PyArrayDyn, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::{PyNotImplementedError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::attr::{attr_to_py, py_to_attr};
use crate::dtype::{dtype_to_string, parse_dtype};
use crate::error::to_py_err;
use crate::runtime::runtime;

#[pyclass(name = "DatasetView", module = "pyatlas._pyatlas")]
pub struct PyDatasetView {
    inner: DatasetView,
}

impl PyDatasetView {
    pub(crate) fn new(inner: DatasetView) -> Self {
        Self { inner }
    }
}

/// Expand a body for each numeric-array dtype that the underlying array_format
/// crate supports (notably excludes Bool and the recursive list dtypes).
macro_rules! numeric_dispatch {
    ($dtype:expr, $body:ident) => {
        match $dtype {
            DType::Int8 => $body!(i8),
            DType::Int16 => $body!(i16),
            DType::Int32 => $body!(i32),
            DType::Int64 => $body!(i64),
            DType::UInt8 => $body!(u8),
            DType::UInt16 => $body!(u16),
            DType::UInt32 => $body!(u32),
            DType::UInt64 => $body!(u64),
            DType::Float32 => $body!(f32),
            DType::Float64 => $body!(f64),
            DType::Bool => return Err(PyNotImplementedError::new_err(
                "Bool arrays are not supported by the underlying array-format crate",
            )),
            DType::String => return Err(PyNotImplementedError::new_err(
                "String arrays are not yet exposed in the Python bindings",
            )),
            DType::Binary => return Err(PyNotImplementedError::new_err(
                "Binary arrays are not yet exposed in the Python bindings",
            )),
            DType::List { .. } | DType::FixedSizeList { .. } => {
                return Err(PyNotImplementedError::new_err(
                    "List / FixedSizeList arrays are not yet exposed in the Python bindings",
                ))
            }
        }
    };
}

#[pymethods]
impl PyDatasetView {
    #[getter]
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn list_arrays(&self) -> Vec<String> {
        self.inner.list_arrays().into_iter().map(String::from).collect()
    }

    /// Returns a dict of attribute name -> Python value.
    fn attributes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new_bound(py);
        for (k, v) in &self.inner.meta().attributes {
            dict.set_item(k, attr_to_py(py, v))?;
        }
        Ok(dict)
    }

    #[pyo3(signature = (key, value, dtype=None))]
    fn set_attribute(
        &mut self,
        key: &str,
        value: &Bound<'_, PyAny>,
        dtype: Option<&str>,
    ) -> PyResult<()> {
        let attr = py_to_attr(value, dtype)?;
        self.inner.set_attribute(key, attr);
        Ok(())
    }

    fn get_attribute(&self, py: Python<'_>, key: &str) -> Option<PyObject> {
        self.inner.get_attribute(key).map(|attr| attr_to_py(py, attr))
    }

    /// Returns `{"dtype", "shape", "chunk_shape", "dimension_names"}` for `array`.
    fn array_meta<'py>(&self, py: Python<'py>, array: &str) -> PyResult<Bound<'py, PyDict>> {
        let schema = self.inner.array_meta(array).map_err(to_py_err)?;
        let dict = PyDict::new_bound(py);
        dict.set_item("dtype", dtype_to_string(&schema.dtype))?;
        dict.set_item("shape", schema.shape)?;
        dict.set_item("chunk_shape", schema.chunk_shape)?;
        dict.set_item("dimension_names", schema.dimension_names)?;
        Ok(dict)
    }

    /// Returns `{"row_count", "null_count", "min", "max"}` or `None` if stats
    /// have not been computed yet (call `flush()` first).
    fn array_stats<'py>(
        &self,
        py: Python<'py>,
        array: &str,
    ) -> PyResult<Option<Bound<'py, PyDict>>> {
        let stats = py
            .allow_threads(|| runtime().block_on(self.inner.array_stats(array)))
            .map_err(to_py_err)?;
        let Some(stats) = stats else { return Ok(None) };
        let dict = PyDict::new_bound(py);
        dict.set_item("row_count", stats.row_count)?;
        dict.set_item("null_count", stats.null_count)?;
        dict.set_item("min", stat_value_to_py(py, &stats.min))?;
        dict.set_item("max", stat_value_to_py(py, &stats.max))?;
        Ok(Some(dict))
    }

    #[pyo3(signature = (name, dtype, dims, shape, chunk_shape=None))]
    fn define_array(
        &mut self,
        py: Python<'_>,
        name: &str,
        dtype: &str,
        dims: Vec<String>,
        shape: Vec<usize>,
        chunk_shape: Option<Vec<usize>>,
    ) -> PyResult<()> {
        let dtype = parse_dtype(dtype)?;

        macro_rules! define_typed {
            ($t:ty) => {{
                py.allow_threads(|| {
                    runtime().block_on(self.inner.define_array::<$t>(
                        name,
                        dims.clone(),
                        shape.clone(),
                        chunk_shape.clone(),
                        None,
                    ))
                })
                .map_err(to_py_err)?
            }};
        }
        numeric_dispatch!(&dtype, define_typed);
        Ok(())
    }

    #[pyo3(signature = (name, start, data))]
    fn write_array(
        &mut self,
        py: Python<'_>,
        name: &str,
        start: Vec<usize>,
        data: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let stored = self.inner.array_meta(name).map_err(to_py_err)?.dtype;

        macro_rules! write_typed {
            ($t:ty) => {{
                let arr = data.downcast::<PyArrayDyn<$t>>().map_err(|_| {
                    PyTypeError::new_err(format!(
                        "expected numpy ndarray with dtype {} for array {:?}",
                        dtype_to_string(&stored),
                        name
                    ))
                })?;
                if !arr.is_c_contiguous() {
                    return Err(PyValueError::new_err(
                        "input numpy array must be C-contiguous",
                    ));
                }
                let view = unsafe { arr.as_array() };
                py.allow_threads(|| {
                    runtime().block_on(self.inner.write_array::<$t>(name, start, view))
                })
                .map_err(to_py_err)?
            }};
        }
        numeric_dispatch!(&stored, write_typed);
        Ok(())
    }

    /// Read an array. If `start` and `shape` are omitted, reads the full array.
    /// Returns `None` if the array doesn't exist in this dataset.
    #[pyo3(signature = (name, start=None, shape=None))]
    fn read_array<'py>(
        &self,
        py: Python<'py>,
        name: &str,
        start: Option<Vec<usize>>,
        shape: Option<Vec<usize>>,
    ) -> PyResult<Option<Bound<'py, PyAny>>> {
        let start = start.unwrap_or_default();
        let shape = shape.unwrap_or_default();
        let stored = self.inner.array_meta(name).map_err(to_py_err)?.dtype;

        macro_rules! read_typed {
            ($t:ty) => {{
                let result = py
                    .allow_threads(|| {
                        runtime().block_on(self.inner.read_array::<$t>(
                            name,
                            start.clone(),
                            shape.clone(),
                        ))
                    })
                    .map_err(to_py_err)?;
                return Ok(match result {
                    Some(arc) => Some(arc.to_owned().into_pyarray_bound(py).into_any()),
                    None => None,
                });
            }};
        }
        numeric_dispatch!(&stored, read_typed);
    }

    fn delete_array(&mut self, py: Python<'_>, name: &str) -> PyResult<()> {
        py.allow_threads(|| runtime().block_on(self.inner.delete_array(name)))
            .map_err(to_py_err)
    }

    fn flush(&mut self, py: Python<'_>) -> PyResult<()> {
        py.allow_threads(|| runtime().block_on(self.inner.flush()))
            .map_err(to_py_err)
    }

    fn compact(&mut self, py: Python<'_>) -> PyResult<()> {
        py.allow_threads(|| runtime().block_on(self.inner.compact()))
            .map_err(to_py_err)
    }

    fn __repr__(&self) -> String {
        format!(
            "<DatasetView name={:?} arrays={}>",
            self.inner.name(),
            self.inner.list_arrays().len()
        )
    }
}

fn stat_value_to_py(py: Python<'_>, val: &Option<atlas::StatValue>) -> PyObject {
    use atlas::StatValue;
    match val {
        None => py.None(),
        Some(StatValue::Float(f)) => (*f).into_py(py),
        Some(StatValue::Int(i)) => (*i).into_py(py),
        Some(StatValue::UInt(u)) => (*u).into_py(py),
        Some(StatValue::Bytes(b)) => PyList::new_bound(py, b).into_py(py),
    }
}
