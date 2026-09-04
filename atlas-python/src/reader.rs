//! How Python reads a collection. Metadata only.
//!
//! Python lists datasets, inspects schemas, reads attributes, and deletes
//! datasets. It does not read array data. The Rust API does that. The split is
//! deliberate. Python writes a collection and then serves it, and to serve it
//! needs no array bytes through the GIL.

use atlas::{ArrayStats, Atlas, DatasetView, FillValue, StatValue};
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
    /// `source` is a local filesystem path (`str` / `os.PathLike`), or an
    /// obstore store handle. The open reads the container footer and the
    /// deletion mask. Nothing else.
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

    /// `{"min", "max", "null_count", "row_count"}` for `array`, over every
    /// live dataset that holds it. `None` if no live dataset does.
    ///
    /// The counts add up. Each bound takes the wider of the two.
    ///
    /// `array-format` keeps the statistics beside the data, so this reads that
    /// variable's segment. One open covers every dataset.
    ///
    /// Use `DatasetView.array_stats` for one dataset on its own.
    fn array_stats<'py>(
        &self,
        py: Python<'py>,
        array: &str,
    ) -> PyResult<Option<Bound<'py, PyDict>>> {
        py.detach(|| runtime().block_on(self.inner.array_stats(array)))
            .map_err(to_py_err)?
            .map(|stats| stats_to_py(py, &stats))
            .transpose()
    }

    /// `{dataset: {"min", "max", "null_count", "row_count"}}` for `array`,
    /// over every live dataset that holds statistics for it, in write order.
    ///
    /// The deletion mask applies, so a hidden dataset never appears. A dataset
    /// that does not declare the array does not appear either. An array no
    /// dataset declares gives an empty dict.
    ///
    /// The keys join to `list_datasets` and to `attributes_by_dataset`.
    fn array_stats_by_dataset<'py>(
        &self,
        py: Python<'py>,
        array: &str,
    ) -> PyResult<Bound<'py, PyDict>> {
        let per_dataset = py
            .detach(|| runtime().block_on(self.inner.array_stats_by_dataset(array)))
            .map_err(to_py_err)?;
        let dict = PyDict::new(py);
        for stats in per_dataset {
            // Every row names its own dataset.
            dict.set_item(stats.name.clone(), stats_to_py(py, &stats)?)?;
        }
        Ok(dict)
    }

    /// `{dataset: value}` for one attribute, over every live dataset that
    /// carries it.
    ///
    /// `array` names the array the attribute annotates. Leave it out for a
    /// dataset-level attribute.
    ///
    /// A dataset that does not carry the key has no entry, so the dict is
    /// empty when nobody carries it. It holds no placeholder, and is shorter
    /// than `list_datasets` whenever some dataset lacks the key. Walk
    /// `list_datasets` for the rows, and look each name up here.
    ///
    /// The keys keep write order: the datasets that carry the key, in the
    /// order `list_datasets` gives them. The deletion mask applies.
    ///
    /// One open covers the whole collection, whatever the dataset count. Call
    /// it once per key to build a table of what every dataset holds. The keys
    /// join to `list_datasets` and to `array_stats_by_dataset`.
    #[pyo3(signature = (attr, array=None))]
    fn attributes_by_dataset<'py>(
        &self,
        py: Python<'py>,
        attr: &str,
        array: Option<&str>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let attrs = py
            .detach(|| runtime().block_on(self.inner.attributes_by_dataset(array, attr)))
            .map_err(to_py_err)?;
        let dict = PyDict::new(py);
        for (name, value) in &attrs {
            dict.set_item(name, attr_to_py(py, value)?)?;
        }
        Ok(dict)
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

    /// Everything the collection knows about itself. The format version, the
    /// creation time, the codec, the container size, the dataset counts, and
    /// how many distinct schemas its datasets share.
    fn summary<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("format_version", self.inner.format_version())?;
        dict.set_item("created_unix_ms", self.inner.created_unix_ms())?;
        dict.set_item("codec", codec_name(self.inner.codec()))?;
        dict.set_item("container_bytes", self.inner.container_bytes())?;
        dict.set_item("total_datasets", self.inner.total_datasets())?;
        dict.set_item("interned_schemas", self.inner.interned_schemas())?;
        Ok(dict)
    }

    /// A metadata view of one dataset. Raises `KeyError` when the dataset is
    /// absent or deleted.
    fn dataset(&self, name: &str) -> PyResult<PyDatasetView> {
        let inner = self.inner.dataset(name).map_err(to_py_err)?;
        Ok(PyDatasetView { inner })
    }

    /// Hide a dataset by adding it to the deletion mask.
    ///
    /// The container does not change. This reclaims no space, and moves no
    /// ordinal. Rewrite the collection to reclaim the bytes.
    fn delete_dataset(&self, py: Python<'_>, name: &str) -> PyResult<()> {
        py.detach(|| runtime().block_on(self.inner.delete_dataset(name)))
            .map_err(to_py_err)
    }

    /// Hide many datasets in one pass. Returns how many the mask gained.
    ///
    /// The cost is two requests, whatever the number of names: one read of the
    /// mask, and one write of it.
    ///
    /// A repeated name counts once. Every name must be live, so an absent or
    /// already deleted one raises `KeyError` and writes nothing.
    fn delete_datasets(&self, py: Python<'_>, names: Vec<String>) -> PyResult<usize> {
        py.detach(|| runtime().block_on(self.inner.delete_datasets(&names)))
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

    /// The dataset's position in the collection. It is stable for the life of
    /// the container.
    #[getter]
    fn ordinal(&self) -> u32 {
        self.inner.ordinal()
    }

    /// Array names, in definition order.
    fn list_arrays(&self) -> Vec<String> {
        self.inner.list_arrays()
    }

    /// `{"dtype", "shape", "chunk_shape", "dimension_names", "fill_value"}`
    /// for `array`. `None` if this dataset does not declare it.
    ///
    /// The name and the dtype come from the footer. The rest lives in the
    /// array's segment, so this opens it. One segment covers that array for
    /// every dataset, so the open happens once per variable.
    fn array_meta<'py>(
        &self,
        py: Python<'py>,
        array: &str,
    ) -> PyResult<Option<Bound<'py, PyDict>>> {
        let Some(meta) = self.inner.array_meta(array) else {
            return Ok(None);
        };
        let dtype = dtype_to_string(meta.dtype());
        let layout = py
            .detach(|| runtime().block_on(self.inner.array_layout(array)))
            .map_err(to_py_err)?;
        let dict = PyDict::new(py);
        dict.set_item("dtype", dtype)?;
        dict.set_item("shape", layout.shape().to_vec())?;
        dict.set_item("chunk_shape", layout.chunk_shape().to_vec())?;
        dict.set_item("dimension_names", layout.dimension_names())?;
        dict.set_item("fill_value", fill_value_to_py(py, layout.fill_value())?)?;
        Ok(Some(dict))
    }

    /// The value a read returns for every element nobody wrote. `None` when
    /// the array has no fill value.
    ///
    /// This reads the array's segment, as [`array_meta`] does.
    fn array_fill_value(&self, py: Python<'_>, array: &str) -> PyResult<Py<PyAny>> {
        let layout = py
            .detach(|| runtime().block_on(self.inner.array_layout(array)))
            .map_err(to_py_err)?;
        fill_value_to_py(py, layout.fill_value())
    }

    /// Dataset-level attributes, in the order somebody set them.
    ///
    /// The values live in the reserved `_datasets` segment, so this reads it.
    /// A dataset that declares no attribute costs no I/O.
    fn attributes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let attrs = py
            .detach(|| runtime().block_on(self.inner.attributes()))
            .map_err(to_py_err)?;
        let dict = PyDict::new(py);
        for (k, v) in &attrs {
            dict.set_item(k, attr_to_py(py, v)?)?;
        }
        Ok(dict)
    }

    /// One dataset-level attribute, or `None`.
    fn get_attribute(&self, py: Python<'_>, key: &str) -> PyResult<Option<Py<PyAny>>> {
        py.detach(|| runtime().block_on(self.inner.get_attribute(key)))
            .map_err(to_py_err)?
            .map(|attr| attr_to_py(py, &attr))
            .transpose()
    }

    /// The attributes of one array, in the order somebody set them.
    ///
    /// The values live on this dataset's array in that variable's segment, so
    /// this reads it.
    fn array_attributes<'py>(&self, py: Python<'py>, array: &str) -> PyResult<Bound<'py, PyDict>> {
        let attrs = py
            .detach(|| runtime().block_on(self.inner.array_attributes(array)))
            .map_err(to_py_err)?;
        let dict = PyDict::new(py);
        for (k, v) in &attrs {
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
        py.detach(|| runtime().block_on(self.inner.get_array_attribute(array, key)))
            .map_err(to_py_err)?
            .map(|attr| attr_to_py(py, &attr))
            .transpose()
    }

    /// `{"min", "max", "null_count", "row_count"}` for `array`, as the write
    /// recorded them. `None` if this dataset does not declare the array.
    ///
    /// `null_count` counts the elements equal to the fill value. That is how
    /// the format stores a cell nobody wrote. `min` and `max` are `None` for a
    /// dtype with no order, and raw `bytes` for a string.
    fn array_stats<'py>(
        &self,
        py: Python<'py>,
        array: &str,
    ) -> PyResult<Option<Bound<'py, PyDict>>> {
        py.detach(|| runtime().block_on(self.inner.array_stats(array)))
            .map_err(to_py_err)?
            .map(|stats| stats_to_py(py, &stats))
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

fn codec_name(codec: atlas::Codec) -> &'static str {
    match codec {
        atlas::Codec::Zstd => "zstd",
        atlas::Codec::Lz4 => "lz4",
        atlas::Codec::Uncompressed => "none",
    }
}

/// Convert one set of statistics to a Python dictionary.
fn stats_to_py<'py>(py: Python<'py>, stats: &ArrayStats) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("min", stat_value_to_py(py, stats.min.as_ref())?)?;
    dict.set_item("max", stat_value_to_py(py, stats.max.as_ref())?)?;
    dict.set_item("null_count", stats.null_count)?;
    dict.set_item("row_count", stats.row_count)?;
    Ok(dict)
}

/// Converts a statistic to a Python scalar. A string and a binary value come
/// back as `bytes`, not as a list of integers. A bound is a value to compare.
fn stat_value_to_py(py: Python<'_>, val: Option<&StatValue>) -> PyResult<Py<PyAny>> {
    use pyo3::IntoPyObjectExt;
    let Some(val) = val else { return Ok(py.None()) };
    match val {
        StatValue::Int(v) => v.into_py_any(py),
        StatValue::UInt(v) => v.into_py_any(py),
        StatValue::Float(v) => v.into_py_any(py),
        StatValue::Bytes(v) => pyo3::types::PyBytes::new(py, v).into_py_any(py),
        StatValue::TimestampNs(v) => v.into_py_any(py),
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
