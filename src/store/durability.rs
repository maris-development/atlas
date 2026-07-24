//! The durability boundary: flush and compact, plus the pruning-index
//! maintenance and array-file bookkeeping they drive.

use std::sync::Arc;

use tracing::{debug, info, instrument};

use super::Atlas;
use crate::{
    Result,
    config::Codec,
    dataset::GLOBAL_ATTRS_ARRAY,
    meta::save_meta,
};

impl Atlas {
    /// Flush every known array file's pending writes AND persist the in-memory
    /// `atlas.json`. This is the single durability boundary for the store.
    ///
    /// Force-initializes every array referenced in meta, even ones never
    /// touched by a `DatasetView` (lazy-init wins are on the read path, not
    /// on flush).
    #[instrument(skip(self))]
    pub async fn flush(&mut self) -> Result<()> {
        // Apply buffered attribute writes into the array files (creating the
        // `_global` file on demand), then commit every touched file.
        self.drain_pending_attrs().await?;
        self.force_init_all_known_arrays().await?;
        let snapshot = self.all_initialized_files();
        let files = snapshot.len();
        debug!(files, "flushing array files");
        for arc in snapshot {
            arc.write().await.flush().await?;
        }
        // The array files now hold up-to-date per-dataset statistics. There is
        // no separate pruning index to build or persist — it is assembled on
        // demand from those stats (see `Atlas::pruning_index`).
        let meta_snapshot = {
            let mut meta = self.meta.lock();
            meta.meta_epoch += 1;
            meta.clone()
        };
        let datasets = meta_snapshot.live_count();
        save_meta(&self.store, &meta_snapshot, self.meta_format, self.meta_compression).await?;
        info!(files, datasets, "flushed atlas store");
        Ok(())
    }

    /// Compact every known array file in place (reclaims tombstoned space).
    /// Drains buffered attributes and commits them first, then force-initializes
    /// every array referenced in meta.
    #[instrument(skip(self))]
    pub async fn compact(&mut self) -> Result<()> {
        self.drain_pending_attrs().await?;
        self.force_init_all_known_arrays().await?;
        let snapshot = self.all_initialized_files();
        let files = snapshot.len();
        debug!(files, "compacting array files");
        for arc in snapshot {
            let mut guard = arc.write().await;
            // Commit any pending (incl. just-drained attributes) so the compact
            // merges them into the new base.
            guard.flush().await?;
            guard.compact().await?;
        }
        // Tombstoned datasets have now been dropped from every array file, so
        // their row slots can go too. This is the only point at which dataset
        // ordinals change, which is why it invalidates any cached row number.
        self.meta.lock().drop_tombstones();
        *self.ordinal_map.lock() = None; // ordinals renumbered
        // Ordinals just changed. Nothing to rebuild — the pruning index is
        // assembled on demand from the (now renumbered) array statistics.
        let meta_snapshot = {
            let mut meta = self.meta.lock();
            meta.meta_epoch += 1;
            meta.clone()
        };
        save_meta(
            &self.store,
            &meta_snapshot,
            self.meta_format,
            self.meta_compression,
        )
        .await?;
        info!(files, "compacted atlas store");
        Ok(())
    }

    /// Drain the buffered attribute writes into their `.af` files.
    ///
    /// Writes land in each file's in-memory pending layer and are committed by
    /// the subsequent flush/compact, so this does not touch disk on its own.
    /// Grouped by file so each file's write lock is acquired once, not once per
    /// dataset.
    async fn drain_pending_attrs(&self) -> Result<()> {
        let drained = self.pending_attrs.lock().drain_all();
        if drained.is_empty() {
            return Ok(());
        }

        // Regroup `(file, dataset) → attrs` into `file → [(dataset, attrs)]` so
        // one file lock covers every dataset's entry in it.
        let mut by_file: std::collections::HashMap<Arc<str>, Vec<_>> =
            std::collections::HashMap::new();
        for ((file, dataset), attrs) in drained {
            by_file.entry(file).or_default().push((dataset, attrs));
        }

        for (file, datasets) in by_file {
            let codec = self.file_codec(&file);
            let handle = self.cache.get_or_insert(&self.store, &file, &codec);
            let arc = handle.get().await?;
            let mut guard = arc.write().await;
            for (dataset, attrs) in datasets {
                // The `_global` file has no data arrays defined up front —
                // create a scalar placeholder entry for this dataset if missing.
                if guard.get_array(&dataset).is_err() {
                    guard.define_array::<u8>(dataset.to_string(), vec![], vec![], None, None)?;
                }
                for (key, value) in attrs {
                    guard.set_attribute(&dataset, &key, (*value).clone().into())?;
                }
            }
        }
        Ok(())
    }

    /// Codec to open a physical file with: the store default for the global
    /// attributes file, otherwise the array's recorded codec (any dataset that
    /// declares it), falling back to the store codec.
    pub(super) fn file_codec(&self, file: &str) -> Codec {
        if file == GLOBAL_ATTRS_ARRAY {
            return self.codec;
        }
        self.meta
            .lock()
            .array_file_codec(file)
            .unwrap_or(self.codec)
    }

    /// Every array file currently initialized in the cache (including
    /// `_global` and any read-opened file), deduped by handle.
    fn all_initialized_files(&self) -> Vec<Arc<tokio::sync::RwLock<array_format::ArrayFile>>> {
        self.cache
            .files
            .read()
            .values()
            .filter_map(|a| a.try_get())
            .collect()
    }

    /// Ensures every array referenced by any dataset in meta has an
    /// initialized `ArrayFile` in the cache, and returns the inner locks
    /// (deduped by array name).
    async fn force_init_all_known_arrays(
        &self,
    ) -> Result<Vec<Arc<tokio::sync::RwLock<array_format::ArrayFile>>>> {
        let specs: Vec<(String, Codec)> = {
            let meta = self.meta.lock();
            let mut seen = std::collections::HashSet::new();
            let mut out = Vec::new();
            for ds in meta.live_schemas() {
                for (name, schema) in &ds.arrays {
                    if seen.insert(name.clone()) {
                        out.push((name.clone(), schema.codec));
                    }
                }
            }
            out
        };
        let mut result = Vec::with_capacity(specs.len());
        for (name, codec) in specs {
            let handle = self.cache.get_or_insert(&self.store, &name, &codec);
            result.push(handle.get().await?);
        }
        Ok(result)
    }
}
