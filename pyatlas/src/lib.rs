use pyo3::prelude::*;

mod attr;
mod dataset;
mod dtype;
mod error;
mod runtime;
mod store;

#[pymodule]
fn _pyatlas(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<store::PyAtlas>()?;
    m.add_class::<dataset::PyDatasetView>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
