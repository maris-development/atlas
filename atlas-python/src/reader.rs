//! Reading a collection from Python: metadata only.
//!
//! Python can list datasets, inspect schemas, read attributes, and delete
//! datasets. It cannot read array data — that is what the Rust API is for. The
//! split is deliberate: a collection is written from Python and then served,
//! and serving does not need to pull array bytes through the GIL.

use atlas::{Atlas, DatasetView, FillValue};
use object_store::path::Path as ObjStorePath;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::attr::attr_to_py;
use crate::dtype::dtype_to_string;
use crate::error::to_py_err;
use crate::runtime::runtime;
use crate::source::AtlasSource;

#[pyclass(name = "Atlas", module = "atlas._atlas")]
pub struct PyAtlas {
    inner: Atlas,
}

#[pymethods]
impl PyAtlas {
    /// Open a collection.
    ///
    /// `source` is a local filesystem path (`str` / `os.PathLike`) or an
    /// obstore store handle. Opening reads the container footer and the
    /// deletion mask, and nothing else.
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

    /// Names of the live datasets, in write order.
    fn list_datasets(&self) -> Vec<String> {
        self.inner.list_datasets()
    }

    /// Every distinct array name across the live datasets, sorted.
    fn list_arrays(&self) -> Vec<String> {
        self.inner.list_arrays()
    }

    /// Whether a live dataset of this name exists.
    fn dataset_exists(&self, name: &str) -> bool {
        self.inner.dataset_exists(name)
    }

    /// How many datasets are live.
    fn dataset_count(&self) -> usize {
        self.inner.dataset_count()
    }

    /// When the collection was written, in milliseconds since the Unix epoch.
    #[getter]
    fn created_unix_ms(&self) -> i64 {
        self.inner.created_unix_ms()
    }

    /// A metadata view of one dataset. Raises `KeyError` if it is absent or
    /// deleted.
    fn dataset(&self, name: &str) -> PyResult<PyDatasetView> {
        let inner = self.inner.dataset(name).map_err(to_py_err)?;
        Ok(PyDatasetView { inner })
    }

    /// Hide a dataset by adding it to the deletion mask.
    ///
    /// The container is not touched, so this reclaims no space and shifts no
    /// ordinals. Rewrite the collection to reclaim the bytes.
    fn delete_dataset(&self, py: Python<'_>, name: &str) -> PyResult<()> {
        py.detach(|| runtime().block_on(self.inner.delete_dataset(name)))
            .map_err(to_py_err)
    }

    fn __contains__(&self, name: &str) -> bool {
        self.inner.dataset_exists(name)
    }

    fn __len__(&self) -> usize {
        self.inner.dataset_count()
    }

    fn __iter__(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let names = PyList::new(py, slf.inner.list_datasets())?;
        Ok(names.try_iter()?.into())
    }

    fn __repr__(&self) -> String {
        format!("<Atlas datasets={}>", self.inner.dataset_count())
    }
}

#[pyclass(name = "DatasetView", module = "atlas._atlas")]
pub struct PyDatasetView {
    inner: DatasetView,
}

#[pymethods]
impl PyDatasetView {
    #[getter]
    fn name(&self) -> &str {
        self.inner.name()
    }

    /// The dataset's position in the collection. Stable for the life of the
    /// container.
    #[getter]
    fn ordinal(&self) -> u32 {
        self.inner.ordinal()
    }

    /// `(start, end)` byte offsets of this dataset's segment in `data.atlas`.
    /// Those bytes are a complete array-format file.
    #[getter]
    fn segment_range(&self) -> (u64, u64) {
        let r = self.inner.segment_range();
        (r.start, r.end)
    }

    /// Array names, in definition order.
    fn list_arrays(&self) -> Vec<String> {
        self.inner.list_arrays()
    }

    /// `{"dtype", "shape", "chunk_shape", "dimension_names", "fill_value"}` for
    /// `array`, or `None` if this dataset does not declare it.
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
        dict.set_item("shape", schema.shape.clone())?;
        dict.set_item("chunk_shape", schema.chunk_shape.clone())?;
        dict.set_item("dimension_names", schema.dimension_names.clone())?;
        dict.set_item(
            "fill_value",
            fill_value_to_py(py, self.inner.array_fill_value(array).as_ref())?,
        )?;
        Ok(Some(dict))
    }

    /// The value a read returns for elements that were never written, or
    /// `None` if the array has no fill value.
    fn array_fill_value(&self, py: Python<'_>, array: &str) -> PyResult<Py<PyAny>> {
        fill_value_to_py(py, self.inner.array_fill_value(array).as_ref())
    }

    /// Dataset-level attributes, in the order they were set.
    fn attributes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for (k, v) in &self.inner.attributes() {
            dict.set_item(k, attr_to_py(py, v)?)?;
        }
        Ok(dict)
    }

    /// One dataset-level attribute, or `None`.
    fn get_attribute(&self, py: Python<'_>, key: &str) -> PyResult<Option<Py<PyAny>>> {
        self.inner
            .get_attribute(key)
            .map(|attr| attr_to_py(py, &attr))
            .transpose()
    }

    /// The attributes of one array, in the order they were set.
    fn array_attributes<'py>(&self, py: Python<'py>, array: &str) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for (k, v) in &self.inner.array_attributes(array) {
            dict.set_item(k, attr_to_py(py, v)?)?;
        }
        Ok(dict)
    }

    /// One attribute of one array, or `None`.
    fn get_array_attribute(
        &self,
        py: Python<'_>,
        array: &str,
        key: &str,
    ) -> PyResult<Option<Py<PyAny>>> {
        self.inner
            .get_array_attribute(array, key)
            .map(|attr| attr_to_py(py, &attr))
            .transpose()
    }

    fn __contains__(&self, array: &str) -> bool {
        self.inner.array_meta(array).is_some()
    }

    fn __len__(&self) -> usize {
        self.inner.list_arrays().len()
    }

    fn __repr__(&self) -> String {
        format!(
            "<DatasetView name={:?} arrays={}>",
            self.inner.name(),
            self.inner.list_arrays().len()
        )
    }
}

/// Convert a fill value to a Python scalar.
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
