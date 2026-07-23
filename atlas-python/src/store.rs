use std::path::PathBuf;

use atlas::{
    Atlas, Codec, ColumnKey, DType, MetaFormat, StatVal, StoreConfig, TimestampNs,
    TypeMismatchPolicy,
};
use numpy::IntoPyArray;
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

fn parse_type_mismatch_policy(s: &str) -> PyResult<TypeMismatchPolicy> {
    Ok(match s.to_ascii_lowercase().as_str() {
        "warn" | "warning" => TypeMismatchPolicy::Warn,
        "error" | "raise" | "strict" => TypeMismatchPolicy::Error,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown on_type_mismatch: {other:?} (expected 'warn' or 'error')"
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
    #[pyo3(signature = (source, codec="zstd", meta_format="json", meta_compression="none", on_type_mismatch="warn"))]
    fn create(
        py: Python<'_>,
        source: AtlasSource,
        codec: &str,
        meta_format: &str,
        meta_compression: &str,
        on_type_mismatch: &str,
    ) -> PyResult<Self> {
        let codec = parse_codec(codec)?;
        let meta_format = parse_meta_format(meta_format)?;
        let meta_compression = parse_codec(meta_compression)?;
        let config = StoreConfig {
            codec,
            on_type_mismatch: parse_type_mismatch_policy(on_type_mismatch)?,
            meta_format,
            meta_compression,
            ..StoreConfig::default()
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
    ///
    /// `on_type_mismatch` ("warn" | "error") sets the per-session policy for a
    /// dataset whose type can't merge with the collection's existing type for
    /// that array/attribute.
    #[staticmethod]
    #[pyo3(signature = (source, on_type_mismatch="warn"))]
    fn open(py: Python<'_>, source: AtlasSource, on_type_mismatch: &str) -> PyResult<Self> {
        let config = StoreConfig {
            on_type_mismatch: parse_type_mismatch_policy(on_type_mismatch)?,
            ..Default::default()
        };
        let inner = match source {
            AtlasSource::Path(path) => py
                .detach(|| runtime().block_on(Atlas::open_path_with_config(path, config)))
                .map_err(to_py_err)?,
            AtlasSource::ObjectStore(store) => {
                let store = store.into_dyn();
                py.detach(|| {
                    runtime().block_on(Atlas::open_with_config(
                        store,
                        ObjStorePath::from(""),
                        config,
                    ))
                })
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
        self.inner.list_datasets()
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

    /// Reads the pruning index for **only** the requested columns.
    ///
    /// `arrays` / `global_attrs` are lists of names; `array_attrs` is a list of
    /// `(array, key)` pairs. Returns
    /// `{"rows", "datasets", "live", "columns": {label: {...}}}`, where each
    /// column carries numpy arrays over the full row space, with `present`
    /// marking which datasets actually declare it and `row_count` 0 for those
    /// that don't. Statistics keep their source type; for a column's declared
    /// type use `merged_schema()`.
    ///
    /// Only the named columns are fetched from storage — a store with hundreds
    /// of columns costs the same here as one with two.
    #[pyo3(signature = (arrays=None, global_attrs=None, array_attrs=None))]
    fn pruning_index<'py>(
        &self,
        py: Python<'py>,
        arrays: Option<Vec<String>>,
        global_attrs: Option<Vec<String>>,
        array_attrs: Option<Vec<(String, String)>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let mut keys: Vec<ColumnKey> = Vec::new();
        for name in arrays.unwrap_or_default() {
            keys.push(ColumnKey::Array(name));
        }
        for key in global_attrs.unwrap_or_default() {
            keys.push(ColumnKey::GlobalAttr(key));
        }
        for (array, key) in array_attrs.unwrap_or_default() {
            keys.push(ColumnKey::ArrayAttr(array, key));
        }

        // The index is self-describing: it carries the liveness mask and the
        // row↔name mapping, so no separate store calls are needed.
        let index = py
            .detach(|| runtime().block_on(self.inner.pruning_index(&keys)))
            .map_err(to_py_err)?;

        let out = PyDict::new(py);
        out.set_item("rows", index.rows())?;
        out.set_item("datasets", index.dataset_names().to_vec())?;
        out.set_item("live", index.live().to_vec().into_pyarray(py))?;

        let columns = PyDict::new(py);
        for key in index.column_keys() {
            let Some(column) = index.column(key) else {
                continue;
            };
            let entry = PyDict::new(py);
            entry.set_item("present", column.present_mask().into_pyarray(py))?;
            entry.set_item("stats_valid", column.stats_valid_mask().into_pyarray(py))?;
            entry.set_item("row_count", column.row_count.clone().into_pyarray(py))?;
            entry.set_item("null_count", column.null_count.clone().into_pyarray(py))?;
            entry.set_item("min", stat_vec_to_py(py, &column.min)?)?;
            entry.set_item("max", stat_vec_to_py(py, &column.max)?)?;
            columns.set_item(column_label(key), entry)?;
        }
        out.set_item("columns", columns)?;
        Ok(out)
    }

    /// Every column's dtype and collection-wide min/max, read from the index
    /// footer alone — no column data is fetched.
    ///
    /// Use it to rule a column out before asking for it: if its global range
    /// can't satisfy a predicate, no dataset in it can either.
    fn column_summaries<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let summaries = py
            .detach(|| runtime().block_on(self.inner.column_summaries()))
            .map_err(to_py_err)?;
        let out = PyDict::new(py);
        for (key, summary) in summaries {
            let entry = PyDict::new(py);
            entry.set_item("present_count", summary.present_count)?;
            entry.set_item("min", summary.min.map(|v| stat_to_py(py, &v)).transpose()?)?;
            entry.set_item("max", summary.max.map(|v| stat_to_py(py, &v)).transpose()?)?;
            out.set_item(column_label(&key), entry)?;
        }
        Ok(out)
    }

    /// This dataset's fixed row ordinal in the pruning index, or `None`.
    fn dataset_row(&self, name: &str) -> Option<usize> {
        self.inner.dataset_row(name)
    }

    /// Total row slots including tombstoned ones — the pruning index's height.
    fn row_slots(&self) -> usize {
        self.inner.row_slots()
    }
}

/// Dict key for a column: the array/attribute name, or `"array:key"` for a
/// per-variable attribute.
fn column_label(key: &ColumnKey) -> String {
    match key {
        ColumnKey::Array(name) => name.clone(),
        ColumnKey::GlobalAttr(key) => key.clone(),
        ColumnKey::ArrayAttr(array, key) => format!("{array}:{key}"),
    }
}

fn stat_to_py(py: Python<'_>, value: &StatVal) -> PyResult<Py<PyAny>> {
    Ok(match value {
        StatVal::Int(i) => i.into_pyobject(py)?.into_any().unbind(),
        StatVal::UInt(u) => u.into_pyobject(py)?.into_any().unbind(),
        StatVal::Float(f) => f.into_pyobject(py)?.into_any().unbind(),
        StatVal::TimestampNs(t) => t.into_pyobject(py)?.into_any().unbind(),
        StatVal::Bytes(b) => pyo3::types::PyBytes::new(py, b).into_any().unbind(),
    })
}

/// Expands a column's per-row values into a dense array.
///
/// Statistics keep their source type, so the output type is chosen from the
/// values actually present: any byte value makes it a list of `bytes | None`,
/// any float promotes the whole column to `float64`, otherwise it is an
/// integer array. Rows without a value read as 0 (or NaN for floats) — consult
/// `present` / `stats_valid` rather than the value itself.
fn stat_vec_to_py(py: Python<'_>, values: &[Option<StatVal>]) -> PyResult<Py<PyAny>> {
    let mut has_bytes = false;
    let mut has_float = false;
    let mut has_unsigned = false;
    for value in values.iter().flatten() {
        match value {
            StatVal::Bytes(_) => has_bytes = true,
            StatVal::Float(_) => has_float = true,
            StatVal::UInt(_) => has_unsigned = true,
            _ => {}
        }
    }

    if has_bytes {
        let list = PyList::empty(py);
        for value in values {
            match value {
                Some(StatVal::Bytes(b)) => list.append(pyo3::types::PyBytes::new(py, b))?,
                _ => list.append(py.None())?,
            }
        }
        return Ok(list.into_any().unbind());
    }
    if has_float {
        let out: Vec<f64> = values
            .iter()
            .map(|v| match v {
                Some(StatVal::Float(f)) => *f,
                Some(StatVal::Int(i)) => *i as f64,
                Some(StatVal::UInt(u)) => *u as f64,
                Some(StatVal::TimestampNs(t)) => *t as f64,
                _ => f64::NAN,
            })
            .collect();
        return Ok(out.into_pyarray(py).into_any().unbind());
    }
    if has_unsigned {
        let out: Vec<u64> = values
            .iter()
            .map(|v| match v {
                Some(StatVal::UInt(u)) => *u,
                Some(StatVal::Int(i)) => (*i).max(0) as u64,
                _ => 0,
            })
            .collect();
        return Ok(out.into_pyarray(py).into_any().unbind());
    }
    let out: Vec<i64> = values
        .iter()
        .map(|v| match v {
            Some(StatVal::Int(i)) => *i,
            Some(StatVal::TimestampNs(t)) => *t,
            _ => 0,
        })
        .collect();
    Ok(out.into_pyarray(py).into_any().unbind())
}
