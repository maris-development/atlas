use atlas::{DType, DatasetView, FillValue, TimestampNs};
use numpy::{IntoPyArray, PyArray, PyArrayDyn, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::{PyNotImplementedError, PyOverflowError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString};

use crate::attr::{attr_to_py, py_to_attr};
use crate::dtype::{dtype_to_string, parse_dtype};
use crate::error::to_py_err;
use crate::runtime::runtime;

#[pyclass(name = "DatasetView", module = "atlas._atlas")]
pub struct PyDatasetView {
    pub(crate) inner: DatasetView,
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
            DType::TimestampNs => unreachable!("TimestampNs is handled before numeric_dispatch!",),
            DType::Bool => {
                return Err(PyNotImplementedError::new_err(
                    "Bool arrays are not supported by the underlying array-format crate",
                ))
            }
            DType::String => unreachable!("String is handled before numeric_dispatch!"),
            DType::Binary => {
                return Err(PyNotImplementedError::new_err(
                    "Binary arrays are not yet exposed in the Python bindings",
                ))
            }
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
        self.inner
            .list_arrays()
            .into_iter()
            .map(String::from)
            .collect()
    }

    /// Returns a dict of attribute name -> Python value.
    fn attributes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for (k, v) in &self.inner.meta().attributes {
            dict.set_item(k, attr_to_py(py, v)?)?;
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

    fn get_attribute(&self, py: Python<'_>, key: &str) -> PyResult<Option<Py<PyAny>>> {
        self.inner
            .get_attribute(key)
            .map(|attr| attr_to_py(py, &attr))
            .transpose()
    }

    /// Returns `{"dtype", "shape", "chunk_shape", "dimension_names"}` for
    /// `array`, or `None` if the array doesn't exist in this dataset.
    fn array_meta<'py>(
        &self,
        py: Python<'py>,
        array: &str,
    ) -> PyResult<Option<Bound<'py, PyDict>>> {
        let Some(schema) = self.inner.array_meta(array) else {
            return Ok(None);
        };
        let dict = PyDict::new(py);
        dict.set_item("dtype", dtype_to_string(&schema.dtype))?;
        dict.set_item("shape", schema.shape)?;
        dict.set_item("chunk_shape", schema.chunk_shape)?;
        dict.set_item("dimension_names", schema.dimension_names)?;
        Ok(Some(dict))
    }

    /// Returns `{"row_count", "null_count", "min", "max"}`, or `None` if the
    /// array doesn't exist in this dataset or stats haven't been computed yet
    /// (call `flush()` first).
    fn array_stats<'py>(
        &self,
        py: Python<'py>,
        array: &str,
    ) -> PyResult<Option<Bound<'py, PyDict>>> {
        let stats = py.detach(|| runtime().block_on(self.inner.array_stats(array)));
        let Some(stats) = stats else { return Ok(None) };
        let dict = PyDict::new(py);
        dict.set_item("row_count", stats.row_count)?;
        dict.set_item("null_count", stats.null_count)?;
        dict.set_item("min", stat_value_to_py(py, &stats.min)?)?;
        dict.set_item("max", stat_value_to_py(py, &stats.max)?)?;
        Ok(Some(dict))
    }

    /// Returns the fill value for `array`, or `None` if the array doesn't
    /// exist in this dataset or was defined without one.
    fn array_fill_value(&self, py: Python<'_>, array: &str) -> PyResult<Py<PyAny>> {
        let fv = py
            .detach(|| runtime().block_on(self.inner.array_fill_value(array)))
            .map_err(to_py_err)?;
        fill_value_to_py(py, fv.as_ref())
    }

    #[pyo3(signature = (name, dtype, dims, shape, chunk_shape=None, fill_value=None))]
    fn define_array(
        &mut self,
        py: Python<'_>,
        name: &str,
        dtype: &str,
        dims: Vec<String>,
        shape: Vec<usize>,
        chunk_shape: Option<Vec<usize>>,
        fill_value: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let dtype = parse_dtype(dtype)?;
        let fill = fill_value
            .map(|v| py_to_fill_value(v, &dtype))
            .transpose()?;

        if matches!(&dtype, DType::TimestampNs) {
            py.detach(|| {
                runtime().block_on(self.inner.define_array::<TimestampNs>(
                    name,
                    dims.clone(),
                    shape.clone(),
                    chunk_shape.clone(),
                    fill.clone(),
                ))
            })
            .map_err(to_py_err)?;
            return Ok(());
        }

        if matches!(&dtype, DType::String) {
            py.detach(|| {
                runtime().block_on(self.inner.define_array::<String>(
                    name,
                    dims.clone(),
                    shape.clone(),
                    chunk_shape.clone(),
                    fill.clone(),
                ))
            })
            .map_err(to_py_err)?;
            return Ok(());
        }

        macro_rules! define_typed {
            ($t:ty) => {{
                py.detach(|| {
                    runtime().block_on(self.inner.define_array::<$t>(
                        name,
                        dims.clone(),
                        shape.clone(),
                        chunk_shape.clone(),
                        fill.clone(),
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
        let stored = self
            .inner
            .array_meta(name)
            .ok_or_else(|| to_py_err(atlas::Error::ArrayNotFound(name.to_string())))?
            .dtype;

        if matches!(&stored, DType::String) {
            // Normalize |S<n>, |U<n>, and object inputs to object dtype so they
            // flow through one extraction path. astype('object') is a no-op for
            // already-object arrays.
            let obj = data.call_method1("astype", ("object",))?;
            let arr = obj.downcast::<PyArrayDyn<Py<PyAny>>>().map_err(|_| {
                PyTypeError::new_err(format!(
                    "expected numpy ndarray of object/bytes/unicode strings for array {:?}",
                    name
                ))
            })?;
            if !arr.is_c_contiguous() {
                return Err(PyValueError::new_err(
                    "input numpy array must be C-contiguous",
                ));
            }
            let view = unsafe { arr.as_array() };
            let elem_shape: Vec<usize> = view.shape().to_vec();
            let mut owned: Vec<String> = Vec::with_capacity(view.len());
            for obj in view.iter() {
                let bound = obj.bind(py);
                let s = if let Ok(s) = bound.extract::<String>() {
                    s
                } else if let Ok(b) = bound.extract::<Vec<u8>>() {
                    String::from_utf8_lossy(&b).into_owned()
                } else {
                    return Err(PyTypeError::new_err(format!(
                        "string array element must be str or bytes, got {:?}",
                        bound.get_type().name()?
                    )));
                };
                owned.push(s);
            }
            let nd = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&elem_shape), owned)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;

            py.detach(|| {
                runtime().block_on(self.inner.write_array::<String>(name, start, nd.view()))
            })
            .map_err(to_py_err)?;
            return Ok(());
        }

        if matches!(&stored, DType::TimestampNs) {
            // Accept np.int64 input. For datetime64[ns] callers should pass
            // arr.view(np.int64) -- pyo3-numpy distinguishes the dtype kinds.
            let arr = data.downcast::<PyArrayDyn<i64>>().map_err(|_| {
                PyTypeError::new_err(format!(
                    "expected numpy ndarray with dtype int64 (use arr.view(np.int64) \
                     for datetime64[ns]) for array {:?}",
                    name
                ))
            })?;
            if !arr.is_c_contiguous() {
                return Err(PyValueError::new_err(
                    "input numpy array must be C-contiguous",
                ));
            }
            let view_i64 = unsafe { arr.as_array() };
            // SAFETY: TimestampNs is #[repr(transparent)] over i64, so the in-memory
            // layout of ArrayViewD<i64> and ArrayViewD<TimestampNs> is identical
            // (the type parameter only affects the pointee-type and a zero-sized
            // PhantomData in ViewRepr).
            let view_ts: ndarray::ArrayViewD<TimestampNs> = unsafe {
                std::mem::transmute::<ndarray::ArrayViewD<i64>, ndarray::ArrayViewD<TimestampNs>>(
                    view_i64,
                )
            };
            py.detach(|| {
                runtime().block_on(self.inner.write_array::<TimestampNs>(name, start, view_ts))
            })
            .map_err(to_py_err)?;
            return Ok(());
        }

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
                py.detach(|| runtime().block_on(self.inner.write_array::<$t>(name, start, view)))
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
        let Some(meta) = self.inner.array_meta(name) else {
            return Ok(None);
        };
        let stored = meta.dtype;

        if matches!(&stored, DType::String) {
            let result = py
                .detach(|| {
                    runtime().block_on(self.inner.read_array::<String>(
                        name,
                        start.clone(),
                        shape.clone(),
                    ))
                })
                .map_err(to_py_err)?;
            return Ok(match result {
                Some(arc) => {
                    let owned: ndarray::ArrayD<String> = arc.to_owned();
                    let out_shape: Vec<usize> = owned.shape().to_vec();
                    let py_objs: Vec<Py<PyAny>> = {
                        use pyo3::IntoPyObjectExt;
                        owned
                            .iter()
                            .map(|s| s.as_str().into_py_any(py))
                            .collect::<PyResult<Vec<_>>>()?
                    };
                    let nd: ndarray::ArrayD<Py<PyAny>> =
                        ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&out_shape), py_objs)
                            .map_err(|e| PyValueError::new_err(e.to_string()))?;
                    Some(PyArray::from_owned_object_array(py, nd).into_any())
                }
                None => None,
            });
        }

        if matches!(&stored, DType::TimestampNs) {
            let result = py
                .detach(|| {
                    runtime().block_on(self.inner.read_array::<TimestampNs>(
                        name,
                        start.clone(),
                        shape.clone(),
                    ))
                })
                .map_err(to_py_err)?;
            return Ok(match result {
                Some(arc) => {
                    let owned: ndarray::ArrayD<TimestampNs> = arc.to_owned();
                    // SAFETY: TimestampNs is #[repr(transparent)] over i64, so
                    // ArrayD<TimestampNs> and ArrayD<i64> share an identical
                    // in-memory layout.
                    let as_i64: ndarray::ArrayD<i64> = unsafe {
                        std::mem::transmute::<ndarray::ArrayD<TimestampNs>, ndarray::ArrayD<i64>>(
                            owned,
                        )
                    };
                    let py_arr = as_i64.into_pyarray(py);
                    let np = py.import("numpy")?;
                    let dt_dtype = np.getattr("dtype")?.call1(("datetime64[ns]",))?;
                    Some(py_arr.call_method1("view", (dt_dtype,))?.into_any())
                }
                None => None,
            });
        }

        macro_rules! read_typed {
            ($t:ty) => {{
                let result = py
                    .detach(|| {
                        runtime().block_on(self.inner.read_array::<$t>(
                            name,
                            start.clone(),
                            shape.clone(),
                        ))
                    })
                    .map_err(to_py_err)?;
                return Ok(match result {
                    Some(arc) => Some(arc.to_owned().into_pyarray(py).into_any()),
                    None => None,
                });
            }};
        }
        numeric_dispatch!(&stored, read_typed);
    }

    /// Bulk-read multiple arrays from this dataset in one PyO3 call.
    /// Returns `{name: ndarray | None}` — `None` for arrays not in this
    /// dataset. Same `start` / `shape` apply to every array.
    ///
    /// Fast path for "give me these N variables, optionally sliced" — skips
    /// the Python-side `xr.Dataset` construction and dask graph build that
    /// [`to_xarray`] pays per dataset, while still doing one Rust round-trip
    /// per variable. Use this from dask workers (or any per-dataset loop)
    /// where the natural xarray API's overhead dominates over the actual
    /// I/O cost — the gridded benchmark goes from ~7.8s to <3s by switching
    /// the dask branch to call this instead of `to_xarray(name).isel(...).load()`.
    #[pyo3(signature = (names, start=None, shape=None))]
    fn read_arrays<'py>(
        &self,
        py: Python<'py>,
        names: Vec<String>,
        start: Option<Vec<usize>>,
        shape: Option<Vec<usize>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let start = start.unwrap_or_default();
        let shape = shape.unwrap_or_default();
        let dict = PyDict::new(py);
        for name in &names {
            // Reuse the per-dtype dispatch in `read_array` for each variable.
            // The win isn't fewer Rust calls — it's one PyO3 method invocation
            // instead of N, no per-call Python dispatch overhead, and (most
            // importantly) the caller skips `to_xarray`'s xr.Dataset + dask
            // graph construction entirely.
            let arr = self.read_array(py, name, Some(start.clone()), Some(shape.clone()))?;
            match arr {
                Some(arr) => dict.set_item(name, arr)?,
                None => dict.set_item(name, py.None())?,
            }
        }
        Ok(dict)
    }

    fn delete_array(&mut self, py: Python<'_>, name: &str) -> PyResult<()> {
        py.detach(|| runtime().block_on(self.inner.delete_array(name)))
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

/// Build a `FillValue` from a Python scalar, type-checked against the target dtype.
///
/// Rejects mismatches with a clear `TypeError` (or `OverflowError` for out-of-range
/// integers) rather than silently casting:
///   - int dtypes refuse floats, bools, strings, and out-of-range ints
///   - uint dtypes additionally refuse negative values
///   - float dtypes accept ints (coerced) but refuse bools and strings
///   - bool dtype requires an actual `bool` (rejects `0`/`1`)
///   - string dtype requires a `str`
fn py_to_fill_value(value: &Bound<'_, PyAny>, dtype: &DType) -> PyResult<FillValue> {
    // PyBool must be checked before PyInt — `isinstance(True, int)` is True in Python.
    let is_bool = value.downcast::<PyBool>().is_ok();
    let is_int = !is_bool && value.downcast::<PyInt>().is_ok();
    let is_float = value.downcast::<PyFloat>().is_ok();
    let is_str = value.downcast::<PyString>().is_ok();

    let type_err = |expected: &str| -> PyErr {
        PyTypeError::new_err(format!(
            "fill_value for {} array must be {}, got {}",
            dtype_to_string(dtype),
            expected,
            value
                .get_type()
                .name()
                .map(|n| n.to_string())
                .unwrap_or_else(|_| "?".into()),
        ))
    };

    match dtype {
        DType::Bool => {
            if !is_bool {
                return Err(type_err("a bool"));
            }
            Ok(FillValue::Bool(value.extract::<bool>()?))
        }
        DType::Int8 | DType::Int16 | DType::Int32 | DType::Int64 | DType::TimestampNs => {
            if !is_int {
                return Err(type_err("an int"));
            }
            let v: i64 = value.extract()?;
            let (lo, hi) = match dtype {
                DType::Int8 => (i8::MIN as i64, i8::MAX as i64),
                DType::Int16 => (i16::MIN as i64, i16::MAX as i64),
                DType::Int32 => (i32::MIN as i64, i32::MAX as i64),
                DType::Int64 | DType::TimestampNs => (i64::MIN, i64::MAX),
                _ => unreachable!(),
            };
            if v < lo || v > hi {
                return Err(PyOverflowError::new_err(format!(
                    "fill_value {} is out of range for {}",
                    v,
                    dtype_to_string(dtype),
                )));
            }
            Ok(FillValue::Int(v))
        }
        DType::UInt8 | DType::UInt16 | DType::UInt32 | DType::UInt64 => {
            if !is_int {
                return Err(type_err("a non-negative int"));
            }
            let v: i128 = value.extract()?;
            if v < 0 {
                return Err(PyOverflowError::new_err(format!(
                    "fill_value {} is negative; {} is unsigned",
                    v,
                    dtype_to_string(dtype),
                )));
            }
            let hi: u128 = match dtype {
                DType::UInt8 => u8::MAX as u128,
                DType::UInt16 => u16::MAX as u128,
                DType::UInt32 => u32::MAX as u128,
                DType::UInt64 => u64::MAX as u128,
                _ => unreachable!(),
            };
            if (v as u128) > hi {
                return Err(PyOverflowError::new_err(format!(
                    "fill_value {} is out of range for {}",
                    v,
                    dtype_to_string(dtype),
                )));
            }
            Ok(FillValue::UInt(v as u64))
        }
        DType::Float32 | DType::Float64 => {
            if is_bool || is_str {
                return Err(type_err("a float or int"));
            }
            if !is_float && !is_int {
                return Err(type_err("a float or int"));
            }
            Ok(FillValue::Float(value.extract::<f64>()?))
        }
        DType::String => {
            if !is_str {
                return Err(type_err("a str"));
            }
            Ok(FillValue::String(value.extract::<String>()?))
        }
        DType::Binary => Err(PyNotImplementedError::new_err(
            "fill_value for binary arrays is not yet supported",
        )),
        DType::List { .. } | DType::FixedSizeList { .. } => Err(PyNotImplementedError::new_err(
            "fill_value for list / fixed_size_list arrays is not supported",
        )),
    }
}

/// Convert an atlas `FillValue` back to a Python scalar for `array_meta()`.
fn fill_value_to_py(py: Python<'_>, val: Option<&FillValue>) -> PyResult<Py<PyAny>> {
    use pyo3::IntoPyObjectExt;
    let Some(val) = val else { return Ok(py.None()) };
    match val {
        FillValue::Bool(b) => b.into_py_any(py),
        FillValue::Int(i) => i.into_py_any(py),
        FillValue::UInt(u) => u.into_py_any(py),
        FillValue::Float(f) => f.into_py_any(py),
        FillValue::String(s) => s.into_py_any(py),
        FillValue::TimestampNs(t) => t.into_py_any(py),
    }
}

fn stat_value_to_py(py: Python<'_>, val: &Option<atlas::StatValue>) -> PyResult<Py<PyAny>> {
    use atlas::StatValue;
    use pyo3::IntoPyObjectExt;
    match val {
        None => Ok(py.None()),
        Some(StatValue::Float(f)) => (*f).into_py_any(py),
        Some(StatValue::Int(i)) => (*i).into_py_any(py),
        Some(StatValue::UInt(u)) => (*u).into_py_any(py),
        Some(StatValue::Bytes(b)) => PyList::new(py, b)?.into_py_any(py),
        Some(StatValue::TimestampNs(t)) => (*t).into_py_any(py),
    }
}
