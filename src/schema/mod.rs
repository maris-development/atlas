//! What a collection holds. These types say nothing about where the bytes are.
//!
//! - [`attr`]: [`Attr`], the public attribute value
//! - [`view`]: [`SchemaView`] and [`ArrayMeta`], what a dataset declares
//! - [`layout`]: [`ArrayLayout`], shape and chunking, read from a segment
//! - [`dtype`]: serde support for `array_format`'s [`DType`](array_format::DType)

mod attr;
pub(crate) mod dtype;
mod layout;
mod view;

use array_format::DType;
use indexmap::IndexMap;

pub use attr::Attr;
pub use layout::ArrayLayout;
pub use view::{ArrayMeta, SchemaView};

/// What one dataset declares, owned.
///
/// This is the owned form of [`SchemaView`], which borrows from the footer and
/// copies no name. Call [`SchemaView::to_owned_schema`] for this one.
///
/// It names arrays and attributes, and gives the type of each. It holds no
/// shape and no chunking. Those live in the segment, and
/// [`ArrayLayout`] carries them.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct DatasetSchema {
    /// Array name to element type, in definition order.
    pub arrays: IndexMap<String, DType>,
    /// Dataset-level attribute key to value type, in the order somebody set
    /// them.
    pub attributes: IndexMap<String, DType>,
    /// Per-array attribute keys with their value types. An array nobody
    /// annotated has no entry.
    pub array_attributes: IndexMap<String, IndexMap<String, DType>>,
}
