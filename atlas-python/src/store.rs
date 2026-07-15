use std::path::PathBuf;

use atlas::{Atlas, Codec, DType, MetaFormat, StoreConfig, TimestampNs};
use numpy::{IntoPyArray, PyArray};
use object_store::path::Path as ObjStorePath;
use pyo3::exceptions::{PyKeyError, PyNotImplementedError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3_object_store::AnyObjectStore;

use crate::dataset::PyDatasetView;
use crate::error::to_py_err;
use crate::runtime::runtime;

/// Either a local filesystem path or an obstore-constructed object store
/// handle. The Python-facing `Atlas.create` / `Atlas.open` accept either.
///
/// PyO3 tries the `ObjectStore` variant first via `AnyObjectStore`'s own
/// `FromPyObject` impl, which accepts both native pyo3-object_store
/// instances and externally-constructed handles (e.g.
/// `obstore.store.S3Store(...)`); strings and `os.PathLike` fall through
/// to the `Path` arm.
#[derive(FromPyObject)]
pub enum AtlasSource {
    ObjectStore(AnyObjectStore),
    Path(PathBuf),
}

fn parse_codec(s: &str) -> PyResult<Codec> {
    Ok(match s.to_ascii_lowercase().as_str() {
        "zstd" => Codec::Zstd,
        "lz4" => Codec::Lz4,
        "none" | "uncompressed" => Codec::Uncompressed,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown codec: {other:?} (expected 'zstd', 'lz4', or 'none')"
            )))
        }
    })
}

fn parse_meta_format(s: &str) -> PyResult<MetaFormat> {
    Ok(match s.to_ascii_lowercase().as_str() {
        "json" => MetaFormat::Json,
        "msgpack" | "mp" => MetaFormat::MsgPack,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown meta_format: {other:?} (expected 'json' or 'msgpack')"
            )))
        }
    })
}

#[pyclass(name = "Atlas", module = "atlas._atlas")]
pub struct PyAtlas {
    pub(crate) inner: Atlas,
}

#[pymethods]
impl PyAtlas {
    /// Create a new atlas store.
    ///
    /// `source` is either a local filesystem path (`str` / `os.PathLike`)
    /// or an [obstore](https://github.com/developmentseed/obstore)-
    /// constructed store handle (`obstore.store.S3Store`,
    /// `obstore.store.GCSStore`, `obstore.store.AzureStore`,
    /// `obstore.store.MemoryStore`, etc.). Cloud-store credentials and
    /// configuration are entirely obstore's responsibility — atlas
    /// receives an opaque `Arc<dyn ObjectStore>` and writes through it.
    #[staticmethod]
    #[pyo3(signature = (source, codec="zstd", meta_format="json", meta_compression="none"))]
    fn create(
        py: Python<'_>,
        source: AtlasSource,
        codec: &str,
        meta_format: &str,
        meta_compression: &str,
    ) -> PyResult<Self> {
        let codec = parse_codec(codec)?;
        let meta_format = parse_meta_format(meta_format)?;
        let meta_compression = parse_codec(meta_compression)?;
        let config = StoreConfig {
            codec,
            meta_format,
            meta_compression,
        };
        let inner = match source {
            AtlasSource::Path(path) => py
                .detach(|| runtime().block_on(Atlas::create_path(path, config)))
                .map_err(to_py_err)?,
            AtlasSource::ObjectStore(store) => {
                let store = store.into_dyn();
                py.detach(|| {
                    runtime().block_on(Atlas::create(store, ObjStorePath::from(""), config))
                })
                .map_err(to_py_err)?
            }
        };
        Ok(Self { inner })
    }

    /// Open an existing atlas store.
    ///
    /// `source` accepts the same shapes as [`Atlas::create`]: a local
    /// filesystem path or an obstore-constructed store handle. Codec,
    /// metadata format and metadata compression are auto-detected from
    /// the on-disk files in both cases.
    #[staticmethod]
    fn open(py: Python<'_>, source: AtlasSource) -> PyResult<Self> {
        let inner = match source {
            AtlasSource::Path(path) => py
                .detach(|| runtime().block_on(Atlas::open_path(path)))
                .map_err(to_py_err)?,
            AtlasSource::ObjectStore(store) => {
                let store = store.into_dyn();
                py.detach(|| runtime().block_on(Atlas::open(store, ObjStorePath::from(""))))
                    .map_err(to_py_err)?
            }
        };
        Ok(Self { inner })
    }

    fn create_dataset(&mut self, py: Python<'_>, name: &str) -> PyResult<PyDatasetView> {
        let view = py
            .detach(|| runtime().block_on(self.inner.create_dataset(name)))
            .map_err(to_py_err)?;
        Ok(PyDatasetView::new(view))
    }

    fn open_dataset(&self, py: Python<'_>, name: &str) -> PyResult<PyDatasetView> {
        let view = py
            .detach(|| runtime().block_on(self.inner.open_dataset(name)))
            .map_err(to_py_err)?;
        Ok(PyDatasetView::new(view))
    }

    fn delete_dataset(&mut self, py: Python<'_>, name: &str) -> PyResult<()> {
        py.detach(|| runtime().block_on(self.inner.delete_dataset(name)))
            .map_err(to_py_err)
    }

    fn list_datasets(&self) -> Vec<String> {
        self.inner.list_datasets().into_iter().map(String::from).collect()
    }

    fn list_arrays(&self) -> Vec<String> {
        self.inner.list_arrays()
    }

    fn dataset_exists(&self, name: &str) -> bool {
        self.inner.dataset_exists(name)
    }

    /// The collection-wide merged schema: every unique array (widened dtype +
    /// dims + per-variable attribute types) and every global attribute type.
    /// Returns `{"arrays": {name: {"dtype", "dimension_names", "attributes"}},
    /// "global_attributes": {key: dtype}}`.
    fn merged_schema<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        use crate::dtype::dtype_to_string;
        let merged = self.inner.merged_schema();
        let out = PyDict::new(py);
        let arrays = PyDict::new(py);
        for (name, arr) in &merged.arrays {
            let entry = PyDict::new(py);
            entry.set_item("dtype", dtype_to_string(&arr.dtype.0))?;
            entry.set_item("dimension_names", arr.dimension_names.clone())?;
            let attrs = PyDict::new(py);
            for (k, ty) in &arr.attributes {
                attrs.set_item(k, dtype_to_string(&ty.0))?;
            }
            entry.set_item("attributes", attrs)?;
            arrays.set_item(name, entry)?;
        }
        out.set_item("arrays", arrays)?;
        let globals = PyDict::new(py);
        for (k, ty) in &merged.global_attributes {
            globals.set_item(k, dtype_to_string(&ty.0))?;
        }
        out.set_item("global_attributes", globals)?;
        Ok(out)
    }

    fn __repr__(&self) -> String {
        format!("<Atlas datasets={}>", self.inner.list_datasets().len())
    }

    /// Persist the in-memory atlas.json + every cached array file. This is
    /// the single durability boundary; until this is called nothing reaches
    /// disk.
    fn flush(&mut self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| runtime().block_on(self.inner.flush()))
            .map_err(to_py_err)
    }

    /// Final flush; alias for `flush()`. Mirrors the context-manager exit.
    fn close(&mut self, py: Python<'_>) -> PyResult<()> {
        self.flush(py)
    }

    /// Compact every cached array file in place (reclaims tombstoned space).
    fn compact(&mut self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| runtime().block_on(self.inner.compact()))
            .map_err(to_py_err)
    }

    fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    fn __exit__(
        &mut self,
        py: Python<'_>,
        _exc_type: Py<PyAny>,
        _exc_val: Py<PyAny>,
        _exc_tb: Py<PyAny>,
    ) -> PyResult<()> {
        self.close(py)
    }

    /// Append an atlas dataset populated from an `xarray.Dataset`.
    ///
    /// Dask-backed variables are streamed one chunk at a time, with the dask
    /// chunk shape becoming the atlas chunk shape unless overridden via `chunks`.
    ///
    /// `fill_value` overrides the per-array fill value: a bare scalar applies to
    /// numeric arrays, a `{var: scalar}` dict targets named variables (`None`
    /// disables the default for that variable). When omitted, arrays default to a
    /// sentinel fill so mask_and_scale'd missing cells are recorded as null: `NaN`
    /// for floats, `NaT` for `datetime64[ns]`, and `""` for strings (integers have
    /// none).
    #[pyo3(signature = (ds, name, chunks=None, fill_value=None))]
    fn add_xarray_dataset(
        slf: Py<Self>,
        py: Python<'_>,
        ds: Py<PyAny>,
        name: &str,
        chunks: Option<Py<PyAny>>,
        fill_value: Option<Py<PyAny>>,
    ) -> PyResult<()> {
        let helper = py
            .import("atlas.xarray")?
            .getattr("_write_xarray_new_dataset")?;
        let chunks_arg: Py<PyAny> = chunks.unwrap_or_else(|| py.None());
        let fill_arg: Py<PyAny> = fill_value.unwrap_or_else(|| py.None());
        helper.call1((slf, ds, name, chunks_arg, fill_arg))?;
        Ok(())
    }

    /// Open `name` and return it as an `xarray.Dataset` (eager read).
    fn open_as_xarray_dataset(&self, py: Python<'_>, name: &str) -> PyResult<Py<PyAny>> {
        let view = py
            .detach(|| runtime().block_on(self.inner.open_dataset(name)))
            .map_err(to_py_err)?;
        let py_view = Py::new(py, PyDatasetView::new(view))?;
        let helper = py
            .import("atlas.xarray")?
            .getattr("_view_to_xarray")?;
        Ok(helper.call1((py_view,))?.unbind())
    }

    /// Open many datasets and return them stacked along `concat_dim` as a
    /// single lazy `xarray.Dataset`. atlas-native equivalent of
    /// `xr.open_mfdataset(...)`. See [`atlas.xarray._atlas_to_xarray_many`]
    /// for the actual builder.
    #[pyo3(signature = (names, concat_dim="dataset", parallel=true))]
    fn open_as_many_xarray_dataset(
        slf: Py<Self>,
        py: Python<'_>,
        names: Vec<String>,
        concat_dim: &str,
        parallel: bool,
    ) -> PyResult<Py<PyAny>> {
        let helper = py
            .import("atlas.xarray")?
            .getattr("_atlas_to_xarray_many")?;
        Ok(helper.call1((slf, names, concat_dim, parallel))?.unbind())
    }

    /// Bulk-read the same slice of `array` across many datasets in a single
    /// Rust call. Returns a Python `list[np.ndarray | None]` of length
    /// `len(dataset_names)` — `None` for datasets that don't declare the
    /// array.
    ///
    /// Sister API to [`Atlas::read_array_across`] in the atlas crate. One
    /// `RwLock::read` guard on the shared physical file; per-dataset reads
    /// dispatched concurrently on the tokio runtime via
    /// `futures::future::try_join_all`. Lets `open_as_many_xarray_dataset` skip the
    /// per-dataset Python ↔ Rust round-trip entirely.
    #[pyo3(signature = (array, dataset_names, start=None, shape=None))]
    fn read_array_across<'py>(
        &self,
        py: Python<'py>,
        array: &str,
        dataset_names: Vec<String>,
        start: Option<Vec<usize>>,
        shape: Option<Vec<usize>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if dataset_names.is_empty() {
            return Err(PyValueError::new_err("dataset_names is empty"));
        }
        let start = start.unwrap_or_default();
        let shape = shape.unwrap_or_default();

        let dtype = self
            .inner
            .array_dtype(array)
            .ok_or_else(|| PyKeyError::new_err(format!("array not found: {array}")))?;

        // String not exposed here — open_as_many_xarray_dataset's schema check rejects
        // strings before they reach this fast path. Bool / Binary likewise
        // aren't supported through array-format's read path.
        if matches!(dtype, DType::String | DType::Bool | DType::Binary) {
            return Err(PyNotImplementedError::new_err(format!(
                "read_array_across does not support dtype {dtype:?}"
            )));
        }

        if matches!(dtype, DType::TimestampNs) {
            let results: Vec<Option<ndarray::ArcArray<TimestampNs, ndarray::IxDyn>>> = py
                .detach(|| {
                    runtime().block_on(self.inner.read_array_across::<TimestampNs>(
                        array,
                        &dataset_names,
                        start.clone(),
                        shape.clone(),
                    ))
                })
                .map_err(to_py_err)?;
            let list = PyList::empty(py);
            let np = py.import("numpy")?;
            let dt_dtype = np.getattr("dtype")?.call1(("datetime64[ns]",))?;
            for opt in results {
                match opt {
                    Some(arc) => {
                        let owned: ndarray::ArrayD<TimestampNs> = arc.into_owned();
                        // SAFETY: TimestampNs is #[repr(transparent)] over i64.
                        let as_i64: ndarray::ArrayD<i64> = unsafe {
                            std::mem::transmute::<
                                ndarray::ArrayD<TimestampNs>,
                                ndarray::ArrayD<i64>,
                            >(owned)
                        };
                        let py_arr = as_i64.into_pyarray(py);
                        let viewed = py_arr.call_method1("view", (dt_dtype.clone(),))?;
                        list.append(viewed.into_any())?;
                    }
                    None => list.append(py.None())?,
                }
            }
            return Ok(list.into_any());
        }

        macro_rules! read_many_typed {
            ($t:ty) => {{
                let results: Vec<Option<ndarray::ArcArray<$t, ndarray::IxDyn>>> = py
                    .detach(|| {
                        runtime().block_on(self.inner.read_array_across::<$t>(
                            array,
                            &dataset_names,
                            start.clone(),
                            shape.clone(),
                        ))
                    })
                    .map_err(to_py_err)?;
                let list = PyList::empty(py);
                for opt in results {
                    match opt {
                        Some(arc) => list
                            .append(arc.into_owned().into_pyarray(py).into_any())?,
                        None => list.append(py.None())?,
                    }
                }
                return Ok(list.into_any());
            }};
        }

        match dtype {
            DType::Int8 => read_many_typed!(i8),
            DType::Int16 => read_many_typed!(i16),
            DType::Int32 => read_many_typed!(i32),
            DType::Int64 => read_many_typed!(i64),
            DType::UInt8 => read_many_typed!(u8),
            DType::UInt16 => read_many_typed!(u16),
            DType::UInt32 => read_many_typed!(u32),
            DType::UInt64 => read_many_typed!(u64),
            DType::Float32 => read_many_typed!(f32),
            DType::Float64 => read_many_typed!(f64),
            _ => unreachable!("dtype filtered above"),
        }
    }

    /// Stacked variant of [`PyAtlas::read_array_across`]: returns a single
    /// numpy `ndarray` of shape `(len(dataset_names), *per_dataset_shape)`
    /// instead of a Python list of N arrays. Skips the Python-side `np.stack`
    /// step that `open_as_many_xarray_dataset` would otherwise pay.
    ///
    /// Errors if any listed dataset doesn't declare `array`.
    #[pyo3(signature = (array, dataset_names, start=None, shape=None))]
    fn read_array_across_stacked<'py>(
        &self,
        py: Python<'py>,
        array: &str,
        dataset_names: Vec<String>,
        start: Option<Vec<usize>>,
        shape: Option<Vec<usize>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if dataset_names.is_empty() {
            return Err(PyValueError::new_err("dataset_names is empty"));
        }
        let start = start.unwrap_or_default();
        let shape = shape.unwrap_or_default();

        let dtype = self
            .inner
            .array_dtype(array)
            .ok_or_else(|| PyKeyError::new_err(format!("array not found: {array}")))?;

        if matches!(dtype, DType::String | DType::Bool | DType::Binary) {
            return Err(PyNotImplementedError::new_err(format!(
                "read_array_across_stacked does not support dtype {dtype:?}"
            )));
        }

        if matches!(dtype, DType::TimestampNs) {
            let arr: ndarray::Array<TimestampNs, ndarray::IxDyn> = py
                .detach(|| {
                    runtime().block_on(self.inner.read_array_across_stacked::<TimestampNs>(
                        array,
                        &dataset_names,
                        start.clone(),
                        shape.clone(),
                    ))
                })
                .map_err(to_py_err)?;
            // SAFETY: TimestampNs is #[repr(transparent)] over i64.
            let as_i64: ndarray::ArrayD<i64> = unsafe {
                std::mem::transmute::<
                    ndarray::ArrayD<TimestampNs>,
                    ndarray::ArrayD<i64>,
                >(arr)
            };
            let py_arr = as_i64.into_pyarray(py);
            let np = py.import("numpy")?;
            let dt_dtype = np.getattr("dtype")?.call1(("datetime64[ns]",))?;
            return Ok(py_arr.call_method1("view", (dt_dtype,))?.into_any());
        }

        macro_rules! stacked_typed {
            ($t:ty) => {{
                let arr: ndarray::Array<$t, ndarray::IxDyn> = py
                    .detach(|| {
                        runtime().block_on(self.inner.read_array_across_stacked::<$t>(
                            array,
                            &dataset_names,
                            start.clone(),
                            shape.clone(),
                        ))
                    })
                    .map_err(to_py_err)?;
                return Ok(arr.into_pyarray(py).into_any());
            }};
        }

        match dtype {
            DType::Int8 => stacked_typed!(i8),
            DType::Int16 => stacked_typed!(i16),
            DType::Int32 => stacked_typed!(i32),
            DType::Int64 => stacked_typed!(i64),
            DType::UInt8 => stacked_typed!(u8),
            DType::UInt16 => stacked_typed!(u16),
            DType::UInt32 => stacked_typed!(u32),
            DType::UInt64 => stacked_typed!(u64),
            DType::Float32 => stacked_typed!(f32),
            DType::Float64 => stacked_typed!(f64),
            _ => unreachable!("dtype filtered above"),
        }
    }
}
