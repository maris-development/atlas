//! Building a collection from Python.
//!
//! Creation is the only thing Python can do to a collection's data. Reading it
//! back is metadata only; see [`crate::reader`].

use std::sync::Arc;

use atlas::{AtlasWriter, DType, DatasetWriter, FillValue, TimestampNs, WriterConfig};
use ndarray::{ArrayD, IxDyn};
use numpy::{PyArrayDyn, PyArrayMethods, PyUntypedArrayMethods};
use object_store::path::Path as ObjStorePath;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyFloat, PyInt, PyString};
use pyo3::exceptions::{PyNotImplementedError, PyOverflowError, PyTypeError, PyValueError};
use tokio::sync::Mutex;

use crate::attr::py_to_attr;
use crate::dtype::{dtype_to_string, parse_dtype};
use crate::error::to_py_err;
use crate::runtime::runtime;
use crate::source::{AtlasSource, parse_codec};

/// Expand a body for each numeric array dtype `array-format` supports.
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
            DType::TimestampNs => unreachable!("TimestampNs is handled before numeric_dispatch!"),
            DType::String => unreachable!("String is handled before numeric_dispatch!"),
            DType::Bool => {
                return Err(PyNotImplementedError::new_err(
                    "Bool arrays are not supported by the underlying array-format crate",
                ));
            }
            DType::Binary => {
                return Err(PyNotImplementedError::new_err(
                    "Binary arrays are not yet exposed in the Python bindings",
                ));
            }
            DType::List { .. } | DType::FixedSizeList { .. } => {
                return Err(PyNotImplementedError::new_err(
                    "List / FixedSizeList arrays are not yet exposed in the Python bindings",
                ));
            }
        }
    };
}

#[pyclass(name = "AtlasWriter", module = "atlas._atlas")]
pub struct PyAtlasWriter {
    // Option so `finish` can consume the writer; a second call raises rather
    // than corrupting the container.
    inner: Arc<Mutex<Option<AtlasWriter>>>,
}

#[pymethods]
impl PyAtlasWriter {
    /// Start writing a collection.
    ///
    /// `source` is a local filesystem path (`str` / `os.PathLike`) or an
    /// [obstore](https://github.com/developmentseed/obstore) store handle
    /// (`S3Store`, `GCSStore`, `AzureStore`, `MemoryStore`, ...). Credentials
    /// and endpoint configuration are obstore's business; atlas receives an
    /// opaque store and writes through it.
    #[staticmethod]
    #[pyo3(signature = (source, codec="zstd", block_target_size=None))]
    fn create(
        py: Python<'_>,
        source: AtlasSource,
        codec: &str,
        block_target_size: Option<usize>,
    ) -> PyResult<Self> {
        let mut config = WriterConfig {
            codec: parse_codec(codec)?,
            ..Default::default()
        };
        if let Some(size) = block_target_size {
            if size == 0 {
                return Err(PyValueError::new_err("block_target_size must be positive"));
            }
            config.block_target_size = size;
        }
        let inner = match source {
            AtlasSource::Path(path) => py
                .detach(|| runtime().block_on(AtlasWriter::create_path(path, config)))
                .map_err(to_py_err)?,
            AtlasSource::ObjectStore(store) => {
                let store = store.into_dyn();
                py.detach(|| {
                    runtime().block_on(AtlasWriter::create(
                        store,
                        ObjStorePath::from(""),
                        config,
                    ))
                })
                .map_err(to_py_err)?
            }
        };
        Ok(Self {
            inner: Arc::new(Mutex::new(Some(inner))),
        })
    }

    /// Begin a dataset. Call `finish()` on the result to commit it.
    fn add_dataset(&self, py: Python<'_>, name: &str) -> PyResult<PyDatasetWriter> {
        let inner = Arc::clone(&self.inner);
        let name = name.to_string();
        let ds = py
            .detach(|| {
                runtime().block_on(async move {
                    let guard = inner.lock().await;
                    let writer = guard.as_ref().ok_or(atlas::Error::WriterFinished)?;
                    writer.add_dataset(&name).await
                })
            })
            .map_err(to_py_err)?;
        Ok(PyDatasetWriter {
            inner: Some(ds),
            arrays: Vec::new(),
        })
    }

    /// Number of datasets committed so far.
    fn dataset_count(&self, py: Python<'_>) -> PyResult<usize> {
        let inner = Arc::clone(&self.inner);
        py.detach(|| {
            runtime().block_on(async move {
                match inner.lock().await.as_ref() {
                    Some(w) => Ok(w.dataset_count().await),
                    None => Err(to_py_err(atlas::Error::WriterFinished)),
                }
            })
        })
    }

    /// Write the footer and close the collection. Nothing is readable before
    /// this returns.
    fn finish(&self, py: Python<'_>) -> PyResult<()> {
        let inner = Arc::clone(&self.inner);
        py.detach(|| {
            runtime().block_on(async move {
                let writer = inner.lock().await.take();
                match writer {
                    Some(w) => w.finish().await.map_err(to_py_err),
                    None => Err(to_py_err(atlas::Error::WriterFinished)),
                }
            })
        })
    }

    /// Whether `finish()` has run.
    #[getter]
    fn closed(&self, py: Python<'_>) -> bool {
        let inner = Arc::clone(&self.inner);
        py.detach(|| runtime().block_on(async move { inner.lock().await.is_none() }))
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// Finishes on a clean exit. On an exception the writer is dropped instead,
    /// so no readable collection is left behind and the exception propagates.
    #[pyo3(signature = (exc_type=None, exc_value=None, traceback=None))]
    fn __exit__(
        &self,
        py: Python<'_>,
        exc_type: Option<&Bound<'_, PyAny>>,
        exc_value: Option<&Bound<'_, PyAny>>,
        traceback: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<bool> {
        let _ = (exc_value, traceback);
        if exc_type.is_some() {
            let inner = Arc::clone(&self.inner);
            py.detach(|| {
                runtime().block_on(async move {
                    inner.lock().await.take();
                })
            });
            return Ok(false);
        }
        self.finish(py)?;
        Ok(false)
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        if self.closed(py) {
            "<AtlasWriter finished>".to_string()
        } else {
            format!(
                "<AtlasWriter datasets={}>",
                self.dataset_count(py).unwrap_or(0)
            )
        }
    }
}

#[pyclass(name = "DatasetWriter", module = "atlas._atlas")]
pub struct PyDatasetWriter {
    // Option so `finish` can consume it.
    inner: Option<DatasetWriter>,
    /// Declared dtypes, so `write_array` knows how to read the numpy input
    /// without going back to the Rust writer for each block.
    arrays: Vec<(String, DType)>,
}

impl PyDatasetWriter {
    fn get(&mut self) -> PyResult<&mut DatasetWriter> {
        self.inner
            .as_mut()
            .ok_or_else(|| to_py_err(atlas::Error::WriterFinished))
    }

    fn dtype_of(&self, array: &str) -> PyResult<DType> {
        self.arrays
            .iter()
            .find(|(name, _)| name == array)
            .map(|(_, dtype)| dtype.clone())
            .ok_or_else(|| to_py_err(atlas::Error::ArrayNotFound(array.to_string())))
    }
}

#[pymethods]
impl PyDatasetWriter {
    #[getter]
    fn name(&self) -> PyResult<String> {
        Ok(self
            .inner
            .as_ref()
            .ok_or_else(|| to_py_err(atlas::Error::WriterFinished))?
            .name()
            .to_string())
    }

    fn list_arrays(&self) -> Vec<String> {
        self.arrays.iter().map(|(name, _)| name.clone()).collect()
    }

    /// Declare an array. `chunk_shape` defaults to `shape`, storing the array
    /// as a single chunk.
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
        let writer = self.get()?;

        macro_rules! define_typed {
            ($t:ty) => {{
                py.detach(|| {
                    runtime().block_on(writer.define_array::<$t>(
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

        match &dtype {
            DType::TimestampNs => define_typed!(TimestampNs),
            DType::String => define_typed!(String),
            other => numeric_dispatch!(other, define_typed),
        }
        self.arrays.push((name.to_string(), dtype));
        Ok(())
    }

    /// Write `data` into `array` with its origin at `start`. The region may
    /// span chunks and need not be chunk-aligned.
    #[pyo3(signature = (name, start, data))]
    fn write_array(
        &mut self,
        py: Python<'_>,
        name: &str,
        start: Vec<usize>,
        data: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let stored = self.dtype_of(name)?;

        if matches!(&stored, DType::String) {
            // Normalize |S<n>, |U<n>, and object inputs to object dtype so they
            // flow through one extraction path. astype('object') is a no-op for
            // arrays that are already object.
            let obj = data.call_method1("astype", ("object",))?;
            let arr = obj.cast::<PyArrayDyn<Py<PyAny>>>().map_err(|_| {
                PyTypeError::new_err(format!(
                    "expected numpy ndarray of object/bytes/unicode strings for array {name:?}"
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
            let nd = ArrayD::from_shape_vec(IxDyn(&elem_shape), owned)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let writer = self.get()?;
            py.detach(|| runtime().block_on(writer.write_array::<String>(name, start, nd.view())))
                .map_err(to_py_err)?;
            return Ok(());
        }

        if matches!(&stored, DType::TimestampNs) {
            // Accept int64 input. For datetime64[ns] the caller passes
            // arr.view(np.int64); numpy distinguishes the dtype kinds.
            let arr = data.cast::<PyArrayDyn<i64>>().map_err(|_| {
                PyTypeError::new_err(format!(
                    "expected numpy ndarray with dtype int64 (use arr.view(np.int64) \
                     for datetime64[ns]) for array {name:?}"
                ))
            })?;
            if !arr.is_c_contiguous() {
                return Err(PyValueError::new_err(
                    "input numpy array must be C-contiguous",
                ));
            }
            let view_i64 = unsafe { arr.as_array() };
            // SAFETY: TimestampNs is #[repr(transparent)] over i64, so
            // ArrayViewD<i64> and ArrayViewD<TimestampNs> have identical
            // layout: the type parameter only affects the pointee type and a
            // zero-sized PhantomData in ViewRepr.
            let view_ts: ndarray::ArrayViewD<TimestampNs> = unsafe {
                std::mem::transmute::<ndarray::ArrayViewD<i64>, ndarray::ArrayViewD<TimestampNs>>(
                    view_i64,
                )
            };
            let writer = self.get()?;
            py.detach(|| {
                runtime().block_on(writer.write_array::<TimestampNs>(name, start, view_ts))
            })
            .map_err(to_py_err)?;
            return Ok(());
        }

        macro_rules! write_typed {
            ($t:ty) => {{
                let arr = data.cast::<PyArrayDyn<$t>>().map_err(|_| {
                    PyTypeError::new_err(format!(
                        "expected numpy ndarray with dtype {} for array {name:?}",
                        dtype_to_string(&stored),
                    ))
                })?;
                if !arr.is_c_contiguous() {
                    return Err(PyValueError::new_err(
                        "input numpy array must be C-contiguous",
                    ));
                }
                let view = unsafe { arr.as_array() };
                let writer = self.get()?;
                py.detach(|| runtime().block_on(writer.write_array::<$t>(name, start, view)))
                    .map_err(to_py_err)?
            }};
        }
        numeric_dispatch!(&stored, write_typed);
        Ok(())
    }

    /// Attach a dataset-level attribute.
    #[pyo3(signature = (key, value, dtype=None))]
    fn set_attribute(
        &mut self,
        py: Python<'_>,
        key: &str,
        value: &Bound<'_, PyAny>,
        dtype: Option<&str>,
    ) -> PyResult<()> {
        let attr = py_to_attr(value, dtype)?;
        let writer = self.get()?;
        // Release the GIL: an xarray ingest sets on the order of a hundred
        // attributes per file, so this call count is not small.
        py.detach(|| writer.set_attribute(key, attr));
        Ok(())
    }

    /// Attach an attribute to one array, which must already be defined.
    #[pyo3(signature = (array, key, value, dtype=None))]
    fn set_array_attribute(
        &mut self,
        py: Python<'_>,
        array: &str,
        key: &str,
        value: &Bound<'_, PyAny>,
        dtype: Option<&str>,
    ) -> PyResult<()> {
        let attr = py_to_attr(value, dtype)?;
        let writer = self.get()?;
        py.detach(|| writer.set_array_attribute(array, key, attr))
            .map_err(to_py_err)
    }

    /// Commit the dataset into the collection.
    fn finish(&mut self, py: Python<'_>) -> PyResult<()> {
        let writer = self
            .inner
            .take()
            .ok_or_else(|| to_py_err(atlas::Error::WriterFinished))?;
        py.detach(|| runtime().block_on(writer.finish()))
            .map_err(to_py_err)
    }

    /// Discard the dataset. It never enters the collection.
    fn abort(&mut self) {
        self.inner = None;
        self.arrays.clear();
    }

    fn __enter__(slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf
    }

    /// Commits on a clean exit, discards on an exception.
    #[pyo3(signature = (exc_type=None, exc_value=None, traceback=None))]
    fn __exit__(
        &mut self,
        py: Python<'_>,
        exc_type: Option<&Bound<'_, PyAny>>,
        exc_value: Option<&Bound<'_, PyAny>>,
        traceback: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<bool> {
        let _ = (exc_value, traceback);
        if exc_type.is_some() {
            self.abort();
            return Ok(false);
        }
        self.finish(py)?;
        Ok(false)
    }

    fn __repr__(&self) -> String {
        match self.inner.as_ref() {
            Some(w) => format!(
                "<DatasetWriter name={:?} arrays={}>",
                w.name(),
                self.arrays.len()
            ),
            None => "<DatasetWriter finished>".to_string(),
        }
    }
}

/// Build a `FillValue` from a Python scalar, type-checked against the target
/// dtype.
///
/// Mismatches are rejected with a clear `TypeError` (or `OverflowError` for an
/// out-of-range integer) rather than silently cast:
///   - int dtypes refuse floats, bools, strings, and out-of-range ints
///   - uint dtypes additionally refuse negative values
///   - float dtypes accept ints (coerced) but refuse bools and strings
///   - bool requires an actual `bool`, so `0` / `1` are rejected
///   - string requires a `str`
fn py_to_fill_value(value: &Bound<'_, PyAny>, dtype: &DType) -> PyResult<FillValue> {
    // PyBool before PyInt: `isinstance(True, int)` is True in Python.
    let is_bool = value.cast::<PyBool>().is_ok();
    let is_int = !is_bool && value.cast::<PyInt>().is_ok();
    let is_float = value.cast::<PyFloat>().is_ok();
    let is_str = value.cast::<PyString>().is_ok();

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
                    "fill_value {v} is out of range for {}",
                    dtype_to_string(dtype),
                )));
            }
            Ok(match dtype {
                DType::TimestampNs => FillValue::TimestampNs(v),
                _ => FillValue::Int(v),
            })
        }
        DType::UInt8 | DType::UInt16 | DType::UInt32 | DType::UInt64 => {
            if !is_int {
                return Err(type_err("a non-negative int"));
            }
            let v: i128 = value.extract()?;
            if v < 0 {
                return Err(PyOverflowError::new_err(format!(
                    "fill_value {v} is negative; {} is unsigned",
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
                    "fill_value {v} is out of range for {}",
                    dtype_to_string(dtype),
                )));
            }
            Ok(FillValue::UInt(v as u64))
        }
        DType::Float32 | DType::Float64 => {
            if is_bool || is_str || (!is_float && !is_int) {
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
