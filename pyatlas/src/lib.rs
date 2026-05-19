use pyo3::prelude::*;

mod attr;
mod dataset;
mod dtype;
mod error;
mod logging;
mod runtime;
mod store;

#[pymodule]
fn _pyatlas(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Auto-install a tracing subscriber if ATLAS_LOG or RUST_LOG is set.
    // No-op if neither is set, or if a global subscriber already exists.
    logging::try_init_from_env();

    m.add_class::<store::PyAtlas>()?;
    m.add_class::<dataset::PyDatasetView>()?;
    m.add_function(wrap_pyfunction!(logging::init_tracing, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
