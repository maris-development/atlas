#![warn(missing_docs)]

//! ATLAS (Aggregated Tensor Large Array Store) is a directory-based store for thousands of named datasets.
//!
//! Each dataset is a virtual collection of named N-dimensional arrays with per-dataset and
//! per-array attributes, backed by the `array-format` crate. Datasets sharing an array name
//! are co-located in the same physical file, keyed by dataset name.
//!
//! # Layout
//!
//! ```text
//! my_store/
//! ├── atlas.json          <- interned per-dataset schema + attribute-key namespace
//! ├── _global/
//! │   └── data.af         <- dataset-level (global) attribute VALUES, one entry per dataset
//! ├── temperature/
//! │   └── data.af         <- ArrayFile: one entry per dataset + per-variable attribute VALUES
//! └── latitude/
//!     └── data.af
//! ```
//!
//! `atlas.json` stores only the *schema* of the collection — each distinct
//! per-dataset schema once (interned), the attribute-key namespace, and a
//! collection-wide **merged schema** (every unique array/attribute with its
//! type widened across datasets; see [`Atlas::merged_schema`]). All attribute
//! *values* live in the `.af` files: per-variable attributes on the variable's
//! own file, and dataset-level attributes in the reserved `_global` file.
//!
//! Declaring the same array name (or attribute key) in two datasets with
//! incompatible types is rejected — types may only widen within numeric types
//! or between string and timestamp.
//!
//! # Quick start
//!
//! ```
//! use atlas::{Atlas, Attr, StoreConfig};
//! use ndarray::Array2;
//!
//! # tokio::runtime::Runtime::new().unwrap().block_on(async {
//! let tmp = tempfile::tempdir().unwrap();
//!
//! // Create — codec persists in atlas.json so `open_path` doesn't need it.
//! let mut s = Atlas::create_path(tmp.path(), StoreConfig::default()).await.unwrap();
//! {
//!     let mut ds = s.create_dataset("jan_2024").await.unwrap();
//!     ds.define_array::<f32>(
//!         "temperature",
//!         vec!["lat".into(), "lon".into()],
//!         vec![4, 8],
//!         None,        // chunk_shape — defaults to full shape (one chunk)
//!         None,        // fill_value
//!     ).await.unwrap();
//!     let data = Array2::<f32>::from_elem([4, 8], 20.0).into_dyn();
//!     ds.write_array("temperature", vec![0, 0], data.view()).await.unwrap();
//!     ds.set_attribute("month", Attr::Int64(1)).unwrap();
//! }
//! s.flush().await.unwrap();   // single durability boundary
//!
//! // Reopen — no config needed.
//! let s2 = Atlas::open_path(tmp.path()).await.unwrap();
//! let ds2 = s2.open_dataset("jan_2024").await.unwrap();
//! let temp = ds2.read_array::<f32>("temperature", vec![], vec![]).await.unwrap().unwrap();
//! assert_eq!(temp.shape(), &[4, 8]);
//! assert_eq!(temp[[0, 0]], 20.0);
//! # });
//! ```
//!
//! # Thread safety
//!
//! `Atlas` and `DatasetView` are `Send + Sync`. Each physical array file
//! is guarded by a `tokio::sync::RwLock`: concurrent reads (`read_array`,
//! `array_stats`) proceed in parallel without contention, while writes
//! (`write_array`, `define_array`, `flush`, `compact`, …) take an exclusive
//! lock. The cache map uses a `parking_lot::RwLock` that is never held across
//! an `await` point.
//!
//! # Durability
//!
//! `atlas.json` is loaded **once** when the store is opened or created; from
//! then on every schema mutation (`create_dataset`, `define_array`, …) only
//! touches the in-memory `StoreMeta`. Attribute writes (`set_attribute`,
//! `set_array_attribute`) are buffered in memory too. The store does **not**
//! persist until [`Atlas::flush`] is called, which drains buffered attributes
//! into the `.af` files and rewrites `atlas.json`. Dropping an `Atlas` without
//! flushing abandons every pending in-memory write.

mod array;
mod config;
mod dataset;
mod error;
mod meta;
mod schema;
mod store;

pub use config::{Codec, MetaFormat, StoreConfig, TypeMismatchPolicy};
pub use dataset::DatasetView;
pub use error::{Error, Result};
pub use meta::{DatasetSchema, MergedArray, MergedSchema};
pub use store::Atlas;

pub use array_format::{
    ArrayElement, ArrayStats, DType, DeltaCache, FillValue, MergedArrayMeta, StatValue, TimestampNs,
};
pub use schema::{ArraySchema, Attr, DTypeS};

pub(crate) fn validate_name(name: &str) -> Result<()> {
    if name.is_empty() || name.starts_with('_') || name.contains('/') || name == ".." || name == "."
    {
        return Err(Error::InvalidName(name.to_string()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn create_and_read_dataset() {
        let tmp = tempfile::tempdir().unwrap();

        {
            let mut atlas = Atlas::create_path(tmp.path(), StoreConfig::default())
                .await
                .unwrap();
            {
                let mut view = atlas.create_dataset("ds").await.unwrap();
                view.define_array::<f32>("temp", vec!["x".into()], vec![4], None, None)
                    .await
                    .unwrap();
            }
            atlas.flush().await.unwrap();
        }

        let atlas = Atlas::open_path(tmp.path()).await.unwrap();
        let view = atlas.open_dataset("ds").await.unwrap();
        assert_eq!(view.list_arrays(), vec!["temp".to_string()]);
    }

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
        assert!(matches!(
            validate_name("_hidden"),
            Err(Error::InvalidName(_))
        ));
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
    #[test]
    fn store_send() {
        _assert_send::<Atlas>();
    }
    #[test]
    fn view_send() {
        _assert_send::<DatasetView>();
    }
    #[test]
    fn store_sync() {
        _assert_sync::<Atlas>();
    }
    #[test]
    fn view_sync() {
        _assert_sync::<DatasetView>();
    }
}
