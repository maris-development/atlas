//! The types describing what a store holds: attribute values, array schemas,
//! and the type-widening rules that reconcile them across datasets.
//!
//! - [`attr`] — [`Attr`], atlas's public attribute value, and its mapping to
//!   the `array_format` storage type
//! - [`dtype`] — [`widen_dtype`] and the [`DTypeS`] serde wrapper
//! - [`array`] — [`ArraySchema`]

mod array;
mod attr;
mod dtype;

pub use array::ArraySchema;
pub use attr::Attr;
pub use dtype::DTypeS;

pub(crate) use dtype::widen_dtype;
