use pyo3::prelude::*;
use tracing_subscriber::EnvFilter;

/// Installs a tracing subscriber that filters on environment variables.
///
/// The filter order is `ATLAS_LOG`, then `RUST_LOG`, then silence.
/// Examples:
///   ATLAS_LOG=debug                 # everything from atlas/atlas_python at debug
///   ATLAS_LOG=atlas=debug,atlas_python=info
///   RUST_LOG=atlas::store=trace
///
/// A second call does nothing. The first call that succeeds sets the global
/// subscriber.
pub(crate) fn try_init_from_env() {
    let filter = EnvFilter::try_from_env("ATLAS_LOG")
        .or_else(|_| EnvFilter::try_from_default_env())
        .ok();

    if let Some(filter) = filter {
        let _ = tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_writer(std::io::stderr)
            .try_init();
    }
}

/// Emits a debug event from Python. The xarray write loop in
/// `atlas-python/python/atlas/xarray.py` sends its per-chunk timing here. That
/// timing then reaches the same `tracing` subscriber as the Rust side. This
/// does nothing unless the `atlas_python::xarray` target runs at debug level,
/// such as under `ATLAS_LOG=atlas_python=debug`.
#[pyfunction]
#[pyo3(signature = (event, var, elapsed_us, chunks=None, bytes=None))]
pub(crate) fn log_chunk_event(
    event: &str,
    var: &str,
    elapsed_us: u64,
    chunks: Option<u64>,
    bytes: Option<u64>,
) {
    tracing::debug!(
        target: "atlas_python::xarray",
        event,
        var,
        elapsed_us,
        chunks,
        bytes,
        "chunk event"
    );
}

/// The Python entry point. `atlas.init_tracing("debug")` forces one filter
/// directive, and ignores the environment variables. `None` reads them again.
#[pyfunction]
#[pyo3(signature = (filter=None))]
pub(crate) fn init_tracing(filter: Option<&str>) -> PyResult<()> {
    let env_filter = match filter {
        Some(directive) => EnvFilter::try_new(directive)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("invalid filter: {e}")))?,
        None => EnvFilter::try_from_env("ATLAS_LOG")
            .or_else(|_| EnvFilter::try_from_default_env())
            .unwrap_or_else(|_| EnvFilter::new("info")),
    };
    let _ = tracing_subscriber::fmt()
        .with_env_filter(env_filter)
        .with_writer(std::io::stderr)
        .try_init();
    Ok(())
}
