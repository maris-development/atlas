#![warn(missing_docs)]

//! ATLAS (Aggregated Tensor Large Array Store) keeps thousands of named
//! datasets in a single immutable file.
//!
//! A dataset is a set of named N-dimensional arrays with attributes, the shape
//! an xarray `Dataset` or a NetCDF file has. A collection holds many of them.
//!
//! # The format in one paragraph
//!
//! A collection is one write-once file, `data.atlas`. Each dataset occupies a
//! contiguous segment; a footer at the end records every dataset's name,
//! schema, attributes, and segment byte range. Opening reads the footer, so
//! every metadata question is answered by one range read however large the
//! collection is. Array data is fetched chunk by chunk, only when asked for.
//!
//! ```text
//! my_collection/
//! ├── data.atlas      ATLS | segment | segment | ... | footer | trailer
//! └── deleted.mask    optional: ordinals of deleted datasets
//! ```
//!
//! # Immutability
//!
//! A collection cannot be changed once written. There is no append, no
//! in-place update, and no compaction: to change a dataset you rewrite the
//! whole collection. The single exception is [`Atlas::delete_dataset`], which
//! adds an ordinal to a small mask sidecar and leaves the container alone.
//!
//! That constraint is what makes the format simple. There are no delta layers
//! to resolve, no tombstones inside the data, and no ordinal that shifts under
//! a reader. A segment is a complete, self-describing `array-format` file: you
//! can cut one out of the container with `dd` and open it on its own.
//!
//! # Writing
//!
//! ```
//! use atlas::{Atlas, AtlasWriter, Attr, WriterConfig};
//! use ndarray::Array2;
//!
//! # tokio::runtime::Runtime::new().unwrap().block_on(async {
//! let tmp = tempfile::tempdir().unwrap();
//!
//! let w = AtlasWriter::create_path(tmp.path(), WriterConfig::default())
//!     .await
//!     .unwrap();
//! {
//!     let mut ds = w.add_dataset("jan_2024").await.unwrap();
//!     ds.define_array::<f32>(
//!         "temperature",
//!         vec!["lat".into(), "lon".into()],
//!         vec![4, 8],
//!         None, // chunk_shape: defaults to the full shape, one chunk
//!         None, // fill_value
//!     )
//!     .await
//!     .unwrap();
//!     let data = Array2::<f32>::from_elem([4, 8], 20.0).into_dyn();
//!     ds.write_array("temperature", vec![0, 0], data.view()).await.unwrap();
//!     ds.set_attribute("month", Attr::Int64(1));
//!     ds.finish().await.unwrap();
//! }
//! w.finish().await.unwrap();
//!
//! // Reading. Opening touches only the footer.
//! let atlas = Atlas::open_path(tmp.path()).await.unwrap();
//! assert_eq!(atlas.list_datasets(), vec!["jan_2024".to_string()]);
//!
//! let ds = atlas.dataset("jan_2024").unwrap();
//! assert_eq!(ds.array_meta("temperature").unwrap().shape, vec![4, 8]);
//! assert_eq!(ds.get_attribute("month"), Some(Attr::Int64(1)));
//!
//! // Only this line fetches array bytes.
//! let temp = ds.read_array::<f32>("temperature", vec![], vec![]).await.unwrap();
//! assert_eq!(temp[[0, 0]], 20.0);
//! # });
//! ```
//!
//! # Thread safety
//!
//! [`Atlas`] and [`DatasetView`] are `Send + Sync`, and reads never block one
//! another: the data is immutable, so there is nothing to lock. Segment handles
//! open once through a `OnceCell` and are shared, as is the block cache.
//!
//! [`AtlasWriter`] is `Send + Sync` too, and several [`DatasetWriter`]s may be
//! staged at once. A dataset touches the shared output only in
//! [`DatasetWriter::finish`], which holds one lock for the whole append, so
//! concurrent datasets land in finish order and never interleave.

mod config;
mod error;
mod format;
mod reader;
mod schema;
mod writer;

pub use config::{Codec, WriterConfig};
pub use error::{Error, Result};
pub use reader::{Atlas, DatasetView};
pub use schema::{ArraySchema, Attr, DatasetSchema, FillValueS};
pub use writer::{AtlasWriter, DatasetWriter};

pub use array_format::{ArrayElement, DType, DeltaCache, FillValue, TimestampNs};

/// Rejects names that would be ambiguous or unsafe as path components.
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
    fn atlas_is_send_and_sync() {
        _assert_send::<Atlas>();
        _assert_sync::<Atlas>();
    }
    #[test]
    fn view_is_send_and_sync() {
        _assert_send::<DatasetView>();
        _assert_sync::<DatasetView>();
    }
    #[test]
    fn writers_are_send_and_sync() {
        _assert_send::<AtlasWriter>();
        _assert_sync::<AtlasWriter>();
        _assert_send::<DatasetWriter>();
        _assert_sync::<DatasetWriter>();
    }
}
