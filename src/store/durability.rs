//! The durability boundary: flush and compact, plus the pruning-index
//! maintenance and array-file bookkeeping they drive.

use std::sync::Arc;

use tracing::{debug, info, instrument};

use super::Atlas;
use crate::{
    Result,
    config::Codec,
    dataset::GLOBAL_ATTRS_ARRAY,
    meta::{StoreMeta, save_meta},
    pruning::{ColumnKey, PruningIndex, PruningStore},
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
        // Stats now exist for everything written above, so the index can be
        // filled in and persisted alongside the metadata.
        self.refresh_pruning_index().await?;

        let meta_snapshot = {
            let mut meta = self.meta.lock();
            meta.meta_epoch += 1;
            meta.clone()
        };
        let datasets = meta_snapshot.live_count();
        self.write_pruning_index(&meta_snapshot).await?;
        save_meta(&self.store, &meta_snapshot, self.meta_format, self.meta_compression).await?;
        info!(files, datasets, "flushed atlas store");
        Ok(())
    }

    /// Fills the in-memory index from the freshly flushed array statistics.
    ///
    /// Reads each array file's whole `StatsFile` in one go rather than looking
    /// entries up per dataset — the latter is O(datasets²), since each lookup
    /// scans the table.
    async fn refresh_pruning_index(&self) -> Result<()> {
        self.ensure_pruning_loaded().await?;

        // Snapshot what each dataset declares, and the merged dtype of every
        // column, before touching the files.
        let (array_keys, rows_by_name, attr_cells) = {
            let meta = self.meta.lock();
            let mut array_keys: Vec<ColumnKey> = Vec::new();
            let mut seen = std::collections::HashSet::new();
            let mut attr_cells: Vec<(ColumnKey, usize)> = Vec::new();
            let mut rows_by_name = std::collections::HashMap::new();
            for (ordinal, name, schema) in meta.live_datasets() {
                rows_by_name.insert(name.clone(), ordinal);
                for array in schema.arrays.keys() {
                    if seen.insert(array.clone()) {
                        array_keys.push(ColumnKey::array(array));
                    }
                }
                for key in schema.global_attrs.keys() {
                    attr_cells.push((ColumnKey::global_attr(key), ordinal));
                }
                for (array, attrs) in &schema.array_attrs {
                    for key in attrs.keys() {
                        attr_cells.push((ColumnKey::array_attr(array, key), ordinal));
                    }
                }
            }
            (array_keys, rows_by_name, attr_cells)
        };

        // Attribute columns: presence is known from the schema. Their values
        // live in the `.af` files as real attributes, so the column records
        // that the dataset carries the key; min/max for attribute *values* are
        // filled from the same statistics path as arrays where available.
        {
            let mut guard = self.pruning.lock();
            let Some(index) = guard.as_mut() else {
                return Ok(());
            };
            for (key, ordinal) in &attr_cells {
                index.ensure_column(key);
                index.set_present(key, *ordinal);
            }
            for key in &array_keys {
                index.ensure_column(key);
            }
        }

        // One StatsFile read per array file, then one pass over its entries.
        for key in &array_keys {
            let ColumnKey::Array(array) = key else {
                continue;
            };
            let codec = self
                .meta
                .lock()
                .array_file_codec(array)
                .unwrap_or(self.codec);
            let handle = self.cache.get_or_insert(&self.store, array, &codec);
            let Some(arc) = handle.get_existing().await? else {
                continue;
            };
            let guard = arc.read().await;
            let Some(stats) = guard.stats() else { continue };
            let mut index_guard = self.pruning.lock();
            let Some(index) = index_guard.as_mut() else {
                continue;
            };
            for entry in stats.entries() {
                if let Some(row) = rows_by_name.get(&entry.name) {
                    index.set_stats(key, *row, entry);
                }
            }
        }
        Ok(())
    }

    /// Writes `pruning.idx`, tagged with the metadata epoch it was built
    /// against so a torn write is detectable rather than silently mis-read.
    async fn write_pruning_index(&self, meta: &StoreMeta) -> Result<()> {
        let index = {
            let mut guard = self.pruning.lock();
            let Some(index) = guard.as_mut() else {
                return Ok(());
            };
            index.set_meta_epoch(meta.meta_epoch);
            index.clone()
        };
        PruningStore::new(self.store.clone())
            .save(&index, meta.meta_epoch, self.pruning_compression, &meta.live_mask())
            .await
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
        // Ordinals just changed, so the index is rebuilt from scratch against
        // the new numbering rather than patched.
        *self.pruning.lock() = Some(PruningIndex::new());
        self.refresh_pruning_index().await?;

        let meta_snapshot = {
            let mut meta = self.meta.lock();
            meta.meta_epoch += 1;
            meta.clone()
        };
        self.write_pruning_index(&meta_snapshot).await?;
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
    fn file_codec(&self, file: &str) -> Codec {
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
