use pyo3::prelude::*;

mod attr;
mod dtype;
mod error;
mod logging;
mod reader;
mod runtime;
mod source;
mod writer;

#[pymodule]
fn _atlas(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Auto-install a tracing subscriber if ATLAS_LOG or RUST_LOG is set.
    // No-op if neither is set, or if a global subscriber already exists.
    logging::try_init_from_env();

    m.add_class::<writer::PyAtlasWriter>()?;
    m.add_class::<writer::PyDatasetWriter>()?;
    m.add_class::<reader::PyAtlas>()?;
    m.add_class::<reader::PyDatasetView>()?;
    m.add_function(wrap_pyfunction!(logging::init_tracing, m)?)?;
    m.add_function(wrap_pyfunction!(logging::log_chunk_event, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
