//! Store metadata: what a store holds and how it persists.
//!
//! - [`schema`] — [`DatasetSchema`] and the collection-wide [`MergedSchema`]
//! - [`type_index`] — the incremental "what type does this key hold?" index
//! - [`store_meta`] — [`StoreMeta`], the in-memory metadata and its mutators
//! - [`persist`] — the `atlas.json` / `atlas.msgpack` on-disk format

mod persist;
mod schema;
mod store_meta;
mod type_index;

/// Current on-disk store-format version. A store written by an older atlas
/// (which inlined per-dataset attributes, or predated the tombstone mask) is
/// rejected rather than misread — see `persist::decode`.
pub(crate) const STORE_FORMAT_VERSION: u32 = 3;

pub use schema::{DatasetSchema, MergedArray, MergedSchema};

pub(crate) use persist::{load_meta, save_meta};
pub(crate) use store_meta::StoreMeta;
