use pyo3::prelude::*;
use tracing_subscriber::EnvFilter;

/// Initialize a tracing subscriber that filters via env vars.
///
/// Filter precedence: `ATLAS_LOG` → `RUST_LOG` → silent (off).
/// Examples:
///   ATLAS_LOG=debug                 # everything from atlas/pyatlas at debug
///   ATLAS_LOG=atlas=debug,pyatlas=info
///   RUST_LOG=atlas::store=trace
///
/// Calling this more than once is a no-op (the global subscriber is set on
/// the first successful call).
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

/// Python-callable variant: `pyatlas.init_tracing("debug")` forces a filter
/// directive regardless of env vars. Passing `None` re-reads env vars.
#[pyfunction]
#[pyo3(signature = (filter=None))]
pub(crate) fn init_tracing(filter: Option<&str>) -> PyResult<()> {
    let env_filter = match filter {
        Some(directive) => EnvFilter::try_new(directive).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("invalid filter: {e}"))
        })?,
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
