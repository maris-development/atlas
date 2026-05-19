use std::path::PathBuf;

use atlas::{Atlas, Codec, StoreConfig};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::dataset::PyDatasetView;
use crate::error::to_py_err;
use crate::runtime::runtime;

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

#[pyclass(name = "Atlas", module = "pyatlas._pyatlas")]
pub struct PyAtlas {
    pub(crate) inner: Atlas,
}

#[pymethods]
impl PyAtlas {
    /// Create a new store at the given local filesystem path.
    #[staticmethod]
    #[pyo3(signature = (path, codec="zstd"))]
    fn create(py: Python<'_>, path: PathBuf, codec: &str) -> PyResult<Self> {
        let codec = parse_codec(codec)?;
        let config = StoreConfig { codec };
        let inner = py
            .allow_threads(|| runtime().block_on(Atlas::create_path(path, config)))
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    /// Open an existing store at the given local filesystem path.
    #[staticmethod]
    fn open(py: Python<'_>, path: PathBuf) -> PyResult<Self> {
        let inner = py
            .allow_threads(|| runtime().block_on(Atlas::open_path(path)))
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    fn create_dataset(&mut self, py: Python<'_>, name: &str) -> PyResult<PyDatasetView> {
        let view = py
            .allow_threads(|| runtime().block_on(self.inner.create_dataset(name)))
            .map_err(to_py_err)?;
        Ok(PyDatasetView::new(view))
    }

    fn open_dataset(&self, py: Python<'_>, name: &str) -> PyResult<PyDatasetView> {
        let view = py
            .allow_threads(|| runtime().block_on(self.inner.open_dataset(name)))
            .map_err(to_py_err)?;
        Ok(PyDatasetView::new(view))
    }

    fn delete_dataset(&mut self, py: Python<'_>, name: &str) -> PyResult<()> {
        py.allow_threads(|| runtime().block_on(self.inner.delete_dataset(name)))
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

    fn __repr__(&self) -> String {
        format!("<Atlas datasets={}>", self.inner.list_datasets().len())
    }

    /// Append an atlas dataset populated from an `xarray.Dataset`.
    ///
    /// Dask-backed variables are streamed one chunk at a time, with the dask
    /// chunk shape becoming the atlas chunk shape unless overridden via `chunks`.
    #[pyo3(signature = (ds, name, chunks=None))]
    fn add_xr_dataset(
        slf: Py<Self>,
        py: Python<'_>,
        ds: PyObject,
        name: &str,
        chunks: Option<PyObject>,
    ) -> PyResult<()> {
        let helper = py
            .import_bound("pyatlas.xarray")?
            .getattr("_write_xarray_new_dataset")?;
        let chunks_arg: PyObject = chunks.unwrap_or_else(|| py.None());
        helper.call1((slf, ds, name, chunks_arg))?;
        Ok(())
    }

    /// Open `name` and return it as an `xarray.Dataset` (eager read).
    fn to_xarray(&self, py: Python<'_>, name: &str) -> PyResult<PyObject> {
        let view = py
            .allow_threads(|| runtime().block_on(self.inner.open_dataset(name)))
            .map_err(to_py_err)?;
        let py_view = Py::new(py, PyDatasetView::new(view))?;
        let helper = py
            .import_bound("pyatlas.xarray")?
            .getattr("_view_to_xarray")?;
        Ok(helper.call1((py_view,))?.unbind())
    }
}
