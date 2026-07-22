//! Reading and writing `pruning.idx` against an object store.
//!
//! Split from [`Atlas`](crate::Atlas) so the store isn't also an index codec.
//! The important behaviour lives here: a query fetches the footer plus the
//! byte ranges of exactly the columns asked for, never the whole file.

use std::sync::Arc;

use object_store::{ObjectStore, ObjectStoreExt, path::Path};

use super::{ColumnKey, ColumnSummary, PruningIndex, format};
use crate::{Error, Result, config::Codec};

/// Object name of the collection-wide pruning index at the store root.
pub(crate) const PRUNING_INDEX_FILE: &str = "pruning.idx";

/// How many trailing bytes to fetch when looking for the footer.
///
/// Sized to cover the footer of a large collection in one round trip; a bigger
/// footer costs one extra ranged read, never a full download.
const FOOTER_SUFFIX_HINT: u64 = 256 * 1024;

/// The index file, as an object in a store.
pub(crate) struct PruningStore {
    store: Arc<dyn ObjectStore>,
    path: Path,
}

impl PruningStore {
    pub(crate) fn new(store: Arc<dyn ObjectStore>) -> Self {
        Self {
            store,
            path: Path::from(PRUNING_INDEX_FILE),
        }
    }

    /// Loads every column.
    ///
    /// The write path needs this — a flush rewrites the file, so it must hold
    /// what it isn't changing. Queries should use
    /// [`read_columns`](Self::read_columns) instead.
    ///
    /// Returns an empty index when the file doesn't exist yet.
    pub(crate) async fn load_all(&self) -> Result<PruningIndex> {
        let Ok(result) = self.store.get(&self.path).await else {
            return Ok(PruningIndex::new());
        };
        let bytes = result.bytes().await.map_err(Error::ObjectStore)?;
        PruningIndex::decode(&bytes)
    }

    /// Writes the whole file, tagged with the epoch it was built against.
    pub(crate) async fn save(
        &self,
        index: &PruningIndex,
        meta_epoch: u64,
        codec: Codec,
        live: &[bool],
    ) -> Result<()> {
        let bytes = index.encode(meta_epoch, codec, live)?;
        self.store
            .put(&self.path, bytes.into())
            .await
            .map_err(Error::ObjectStore)?;
        Ok(())
    }

    /// Reads only `columns`.
    ///
    /// Two round trips regardless of collection size: one ranged read of the
    /// tail for the footer, then one batched ranged read covering just those
    /// columns' blocks.
    ///
    /// The returned index carries the full row space, so row ordinals still
    /// line up with the dataset list even though most columns are absent.
    pub(crate) async fn read_columns(
        &self,
        columns: &[ColumnKey],
        expected_epoch: u64,
        live: Vec<bool>,
    ) -> Result<PruningIndex> {
        let Some(footer) = self.read_footer(expected_epoch).await? else {
            return Ok(PruningIndex::new());
        };
        let mut index = PruningIndex::with_rows(footer.row_count);
        index.set_meta_epoch(footer.meta_epoch);
        index.set_live(live);

        let ranges = footer.ranges_for(columns);
        if ranges.is_empty() {
            return Ok(index);
        }
        let blocks = self
            .store
            .get_ranges(
                &self.path,
                &ranges.iter().map(|(_, r)| r.clone()).collect::<Vec<_>>(),
            )
            .await
            .map_err(Error::ObjectStore)?;
        for ((key, _), bytes) in ranges.iter().zip(blocks) {
            let column = footer.decode_column(key, &bytes)?;
            index.insert_column(key.clone(), column);
        }
        Ok(index)
    }

    /// Every column's key and collection-wide summary, from the footer alone.
    pub(crate) async fn summaries(
        &self,
        expected_epoch: u64,
    ) -> Result<Vec<(ColumnKey, ColumnSummary)>> {
        Ok(self
            .read_footer(expected_epoch)
            .await?
            .map(|f| f.summaries())
            .unwrap_or_default())
    }

    /// Fetches and validates the footer with one ranged read of the tail,
    /// re-reading only if the footer is larger than the initial suffix.
    ///
    /// `Ok(None)` means the index has never been written.
    async fn read_footer(&self, expected_epoch: u64) -> Result<Option<format::IndexFooter>> {
        let Ok(meta) = self.store.head(&self.path).await else {
            return Ok(None);
        };
        let len = meta.size;
        let mut suffix_len = FOOTER_SUFFIX_HINT.min(len);
        // At most two attempts: the hint, then the exact size it reports.
        for _ in 0..2 {
            let bytes = self
                .store
                .get_range(&self.path, (len - suffix_len)..len)
                .await
                .map_err(Error::ObjectStore)?;
            match format::footer_from_suffix(&bytes, len)? {
                format::FooterRead::Footer(footer) => {
                    if footer.meta_epoch != expected_epoch {
                        return Err(super::invalid(format!(
                            "pruning index is stale: epoch {} but metadata is at \
                             {expected_epoch}; flush to rebuild it",
                            footer.meta_epoch
                        )));
                    }
                    return Ok(Some(*footer));
                }
                format::FooterRead::NeedMore(needed) => suffix_len = (needed as u64).min(len),
            }
        }
        Err(super::invalid("pruning index footer could not be read"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pruning::{ColumnKey, StatVal};
    use array_format::{ArrayStats, StatValue};
    use object_store::memory::InMemory;

    fn store() -> Arc<dyn ObjectStore> {
        Arc::new(InMemory::new())
    }

    fn int_stats(min: i64, max: i64, rows: u64) -> ArrayStats {
        ArrayStats {
            name: "x".into(),
            min: Some(StatValue::Int(min)),
            max: Some(StatValue::Int(max)),
            row_count: rows,
            null_count: 0,
        }
    }

    /// Build a 3-row index with one array column populated on rows 0 and 2.
    fn sample_index(epoch: u64) -> PruningIndex {
        let mut index = PruningIndex::new();
        index.grow_to(3);
        let key = ColumnKey::array("temp");
        index.ensure_column(&key);
        index.set_stats(&key, 0, &int_stats(1, 5, 4));
        index.set_stats(&key, 2, &int_stats(-2, 2, 4));
        index.set_meta_epoch(epoch);
        index
    }

    #[tokio::test]
    async fn save_then_load_all_round_trips() {
        let ps = PruningStore::new(store());
        let index = sample_index(7);
        ps.save(&index, 7, Codec::Zstd, &[true, true, true]).await.unwrap();

        let back = ps.load_all().await.unwrap();
        assert_eq!(back.rows(), 3);
        assert_eq!(back.meta_epoch(), 7);
        let key = ColumnKey::array("temp");
        assert_eq!(back.column(&key), index.column(&key));
    }

    #[tokio::test]
    async fn load_all_on_missing_file_is_empty() {
        let back = PruningStore::new(store()).load_all().await.unwrap();
        assert_eq!(back.rows(), 0);
        assert!(back.column_keys().is_empty());
    }

    #[tokio::test]
    async fn read_columns_fetches_only_requested() {
        let ps = PruningStore::new(store());
        let mut index = PruningIndex::new();
        index.grow_to(2);
        for name in ["a", "b", "c"] {
            let key = ColumnKey::array(name);
            index.ensure_column(&key);
            index.set_stats(&key, 0, &int_stats(0, 1, 1));
        }
        index.set_meta_epoch(1);
        ps.save(&index, 1, Codec::Uncompressed, &[true, true]).await.unwrap();

        let got = ps
            .read_columns(&[ColumnKey::array("b")], 1, vec![true, true])
            .await
            .unwrap();
        assert_eq!(got.rows(), 2, "full row space even for a partial read");
        assert_eq!(got.column_keys(), &[ColumnKey::array("b")]);
        assert!(got.column(&ColumnKey::array("a")).is_none());
    }

    /// The epoch guard is what keeps a positional index from being read against
    /// the wrong dataset list. A mismatch must be a hard error, not silent.
    #[tokio::test]
    async fn stale_epoch_is_rejected() {
        let ps = PruningStore::new(store());
        let index = sample_index(5);
        ps.save(&index, 5, Codec::Zstd, &[true, true, true]).await.unwrap();

        // Reading with the epoch that was written succeeds...
        assert!(ps.read_columns(&[ColumnKey::array("temp")], 5, vec![true; 3]).await.is_ok());
        assert!(ps.summaries(5).await.is_ok());

        // ...but a newer metadata epoch means the index is stale.
        let err = ps
            .read_columns(&[ColumnKey::array("temp")], 6, vec![true; 3])
            .await
            .unwrap_err();
        assert!(format!("{err}").contains("stale"), "got {err}");
        assert!(ps.summaries(6).await.is_err(), "summaries must reject a stale epoch too");
    }

    #[tokio::test]
    async fn summaries_read_from_footer_reflect_live_rows() {
        let ps = PruningStore::new(store());
        let index = sample_index(2);
        // Row 2 (min -2) is masked out at write time.
        ps.save(&index, 2, Codec::Zstd, &[true, true, false]).await.unwrap();

        let summaries = ps.summaries(2).await.unwrap();
        let (_, summary) = summaries
            .iter()
            .find(|(k, _)| k == &ColumnKey::array("temp"))
            .unwrap();
        assert_eq!(summary.present_count, 1, "only row 0 is live");
        assert_eq!(summary.min, Some(StatVal::Int(1)), "row 2's -2 must be masked");
    }
}
