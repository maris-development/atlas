//! [`ArrayCache`]: shared lazy handles to the store's array files.

use std::{collections::HashMap, sync::Arc};

use array_format::DeltaCache;
use object_store::ObjectStore;
use parking_lot::RwLock;

use crate::{array::AtlasArray, config::Codec};

/// Shared lazy-handle map: array name → `Arc<AtlasArray>`. Cloned by reference
/// from `Atlas` into every `DatasetView`, so all views observe the same
/// initialization state. The map lock (`parking_lot::RwLock`) is never held
/// across an `await` point; `AtlasArray` defers its actual I/O via
/// `tokio::sync::OnceCell` so each underlying file opens at most once.
pub(crate) struct ArrayCache {
    pub(crate) files: RwLock<HashMap<String, Arc<AtlasArray>>>,
    pub(crate) delta: Arc<DeltaCache>,
}

impl ArrayCache {
    pub(crate) fn new(delta: Arc<DeltaCache>) -> Self {
        Self {
            files: RwLock::new(HashMap::new()),
            delta,
        }
    }

    /// Returns the lazy handle for `array_name`, registering a new one if
    /// absent. Does **not** open or create the underlying file — that happens
    /// on the first `AtlasArray::get().await`.
    pub(crate) fn get_or_insert(
        &self,
        store: &Arc<dyn ObjectStore>,
        array_name: &str,
        codec: &Codec,
    ) -> Arc<AtlasArray> {
        if let Some(arc) = self.files.read().get(array_name) {
            return arc.clone();
        }
        let mut guard = self.files.write();
        guard
            .entry(array_name.to_string())
            .or_insert_with(|| {
                Arc::new(AtlasArray::new(
                    store.clone(),
                    *codec,
                    array_name.to_string(),
                    self.delta.clone(),
                ))
            })
            .clone()
    }
}
