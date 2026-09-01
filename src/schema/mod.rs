//! What a collection holds. These types say nothing about where the bytes are.
//!
//! - [`attr`]: [`Attr`], the public attribute value
//! - [`array`]: [`ArraySchema`] and [`FillValueS`]
//! - [`dtype`]: serde support for `array_format`'s [`DType`](array_format::DType)

mod array;
mod attr;
mod dtype;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

pub use array::{ArraySchema, FillValueS};
pub use attr::Attr;

/// What one dataset holds: its named arrays, in definition order.
///
/// Attribute values are not here. They sit beside the schema in the container
/// footer. Two datasets with the same arrays therefore share one interned
/// schema, whatever their attributes.
#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
pub struct DatasetSchema {
    /// Array name to schema, in definition order.
    pub arrays: IndexMap<String, ArraySchema>,
}
