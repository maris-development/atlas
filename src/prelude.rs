//! Every public item of this crate, for one glob import.
//!
//! ```
//! use atlas::prelude::*;
//! ```
//!
//! The names match the crate root, with one exception. [`crate::Result`] takes
//! one type parameter, so a glob import of that name would hide
//! `std::result::Result` and break every two-parameter use in the importing
//! module. The prelude therefore calls it [`AtlasResult`]. Import
//! `atlas::Result` directly to get the short name.
//!
//! A new public item belongs here as well as at the crate root.

pub use crate::{
    ArrayElement, ArrayFile, ArrayLayout, ArrayMeta, ArrayStats, Atlas, AtlasWriter, Attr,
    AttrKeys, Codec, CollectionFooter, CollectionSchema, DType, DTypeSet, DatasetSchema,
    DatasetView, DatasetWriter, Error, FORMAT_VERSION, FillValue, INLINE_ARRAYS, INLINE_ATTRS,
    InternedSchema, Interner, SchemaView, StatValue, TimestampNs, VariableEntry, WriterConfig,
};

/// The whole `array-format` crate, so `array_format::` resolves after the
/// glob import. [`Atlas::segment`](crate::Atlas::segment) hands out its
/// [`ArrayFile`].
pub use crate::array_format;

/// [`crate::Result`], renamed so a glob import leaves `std::result::Result`
/// alone.
pub use crate::Result as AtlasResult;
