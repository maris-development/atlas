//! One dataset within a store: the read/write [`DatasetView`], the attribute
//! write buffer, and the shared array-file cache.

mod cache;
mod pending_attrs;
mod view;

pub(crate) use cache::ArrayCache;
pub(crate) use pending_attrs::PendingAttrs;
pub use view::DatasetView;
pub(crate) use view::{GLOBAL_ATTRS_ARRAY, open_dataset_view};
