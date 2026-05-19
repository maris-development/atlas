//! `array-store` is a directory-based store for thousands of named datasets.
//!
//! Each dataset is a virtual collection of named N-dimensional arrays with per-dataset and
//! per-array attributes, backed by the `array-format` crate. Datasets sharing an array name
//! are co-located in the same physical file, keyed by dataset name.
//!
//! # Layout
//!
//! ```text
//! my_store/
//! ├── array_store.json    <- dataset registry + per-dataset attributes
//! ├── temperature/
//! │   └── data.af         <- ArrayFile: one named array per dataset
//! └── latitude/
//!     └── data.af
//! ```
//!
//! # Thread safety
//!
//! `ArrayStore` and `DatasetView` are `Send + Sync`. Each physical array file
//! is guarded by a `tokio::sync::RwLock`: concurrent reads (`read_array`,
//! `array_stats`) proceed in parallel without contention, while writes
//! (`write_array`, `define_array`, `flush`, `compact`, …) take an exclusive
//! lock. The cache map uses a `parking_lot::RwLock` that is never held across
//! an `await` point.

mod dataset;
mod error;
mod meta;
mod schema;
mod store;

pub use dataset::DatasetView;
pub use error::{Error, Result};
pub use meta::DatasetMeta;
pub use store::ArrayStore;

pub use array_format::{ArrayElement, ArrayStats, DType, FillValue, MergedArrayMeta, StatValue};
pub use schema::{ArraySchema, Attr};

pub(crate) fn validate_name(name: &str) -> Result<()> {
    if name.is_empty()
        || name.starts_with('_')
        || name.contains('/')
        || name == ".."
        || name == "."
    {
        return Err(Error::InvalidName(name.to_string()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_names_pass() {
        for name in ["temperature", "my-array", "x1", "lat.lon", "a"] {
            assert!(validate_name(name).is_ok(), "expected '{name}' to be valid");
        }
    }

    #[test]
    fn empty_name_rejected() {
        assert!(matches!(validate_name(""), Err(Error::InvalidName(_))));
    }

    #[test]
    fn leading_underscore_rejected() {
        assert!(matches!(validate_name("_hidden"), Err(Error::InvalidName(_))));
        assert!(matches!(validate_name("_"), Err(Error::InvalidName(_))));
    }

    #[test]
    fn slash_in_name_rejected() {
        assert!(matches!(validate_name("a/b"), Err(Error::InvalidName(_))));
        assert!(matches!(validate_name("/abs"), Err(Error::InvalidName(_))));
    }

    #[test]
    fn dotdot_rejected() {
        assert!(matches!(validate_name(".."), Err(Error::InvalidName(_))));
    }

    #[test]
    fn single_dot_rejected() {
        assert!(matches!(validate_name("."), Err(Error::InvalidName(_))));
    }
}

#[cfg(test)]
mod send_check {
    use super::*;
    fn _assert_send<T: Send>() {}
    fn _assert_sync<T: Sync>() {}
    #[test] fn store_send() { _assert_send::<ArrayStore>(); }
    #[test] fn view_send() { _assert_send::<DatasetView>(); }
    #[test] fn store_sync() { _assert_sync::<ArrayStore>(); }
    #[test] fn view_sync() { _assert_sync::<DatasetView>(); }
}
