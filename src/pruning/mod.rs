//! The collection-wide **pruning index**: a flattened, columnar view of every
//! array's and attribute's statistics across all datasets.
//!
//! [`DatasetView::array_stats`](crate::DatasetView::array_stats) answers for one
//! array in one dataset. That is the wrong shape for "which datasets could
//! possibly match this predicate?", which wants a *column*: one value per
//! dataset in a single buffer a caller can compare vectorized, or scan through
//! [`ColumnView::candidates`].
//!
//! # Row space
//!
//! Row `i` is the dataset at ordinal `i` in [`StoreMeta`](crate::meta::StoreMeta)
//! — positional, with no names stored here. Deleted datasets keep their slot and
//! are hidden by the liveness mask, so ordinals never shift and a persisted
//! index stays valid. Only [`Atlas::compact`](crate::Atlas::compact) renumbers.
//!
//! Because rows are positional, an index that drifts out of step with the
//! dataset list does not fail loudly — every row silently means a different
//! dataset. The footer's `meta_epoch` guards against that.
//!
//! # Sparsity
//!
//! Most datasets don't declare most arrays (~26% density on a real
//! 10 000-dataset collection), so a column is a `present` bitmap plus values.
//! Values are dense in memory for O(1) updates and **compacted to the present
//! rows on disk**, where the holes would otherwise dominate.
//!
//! # Module layout
//!
//! - [`bitmap`] — the packed row masks
//! - [`value`] — [`StatVal`] and its ordering
//! - [`column`] — [`StatColumn`] and the masked [`ColumnView`]
//! - `format` — the `pruning.idx` byte layout
//! - `io` — reading and writing it against an object store

mod bitmap;
mod column;
mod format;
mod io;
mod value;

use std::collections::HashMap;

use array_format::ArrayStats;
use serde::{Deserialize, Serialize};

pub use column::{ColumnSummary, ColumnView, StatColumn};
pub use value::StatVal;

pub(crate) use bitmap::Bitmap;
pub(crate) use io::PruningStore;

/// Which array or attribute a column describes.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ColumnKey {
    /// An array's data statistics.
    Array(String),
    /// A dataset-level attribute.
    GlobalAttr(String),
    /// A per-variable attribute, as `(array, key)`.
    ArrayAttr(String, String),
}

impl ColumnKey {
    /// An array column.
    pub fn array(name: impl Into<String>) -> Self {
        Self::Array(name.into())
    }
    /// A dataset-level attribute column.
    pub fn global_attr(key: impl Into<String>) -> Self {
        Self::GlobalAttr(key.into())
    }
    /// A per-variable attribute column.
    pub fn array_attr(array: impl Into<String>, key: impl Into<String>) -> Self {
        Self::ArrayAttr(array.into(), key.into())
    }
}

/// Wraps a message as a pruning-index error (malformed or stale on disk).
pub(crate) fn invalid(msg: impl Into<String>) -> crate::Error {
    crate::Error::CorruptIndex(msg.into())
}

/// The pruning index for a collection.
///
/// Two shapes, same type:
///
/// - **Write side** — held in memory beside the store metadata and kept aligned
///   continuously. A null row is appended the instant a dataset is created, so
///   `rows() == StoreMeta::row_slots()` holds at every point, not only after a
///   flush.
/// - **Read side** — returned by
///   [`Atlas::pruning_index`](crate::Atlas::pruning_index) carrying only the
///   columns asked for, plus the liveness mask. Reach for
///   [`view`](Self::view) rather than [`column`](Self::column): it applies the
///   masks so deleted and absent rows can't leak into a result.
#[derive(Debug, Clone, Default)]
pub struct PruningIndex {
    rows: usize,
    columns: HashMap<ColumnKey, StatColumn>,
    /// Insertion order, so the encoded file is deterministic.
    order: Vec<ColumnKey>,
    meta_epoch: u64,
    /// Liveness per row slot. Empty on the write side, where the mask lives in
    /// the store metadata; populated on the read side.
    live: Vec<bool>,
    /// Dataset name per row slot, `None` for tombstones. Empty on the write
    /// side; populated on the read side so the index is self-describing —
    /// row ↔ name without a second call to the store.
    dataset_names: Vec<Option<String>>,
}

impl PruningIndex {
    /// An empty index covering no rows.
    pub fn new() -> Self {
        Self::default()
    }

    pub(crate) fn with_rows(rows: usize) -> Self {
        Self {
            rows,
            ..Default::default()
        }
    }

    /// Row slots covered, tombstones included.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// The metadata epoch this index was built against.
    pub fn meta_epoch(&self) -> u64 {
        self.meta_epoch
    }

    /// Column keys present in this index, in encode order.
    ///
    /// On the read side this is only what was requested, not everything the
    /// file holds — see
    /// [`Atlas::column_summaries`](crate::Atlas::column_summaries) for that.
    pub fn column_keys(&self) -> &[ColumnKey] {
        &self.order
    }

    /// The liveness mask over row slots: `false` where the dataset was deleted.
    /// Already applied by [`view`](Self::view); needed only for vectorized work
    /// against the raw [`column`](Self::column) buffers.
    pub fn live(&self) -> &[bool] {
        &self.live
    }

    /// Dataset name at each row slot, `None` where the slot is a tombstone.
    ///
    /// The join key between a row and the collection: `dataset_names()[i]` names
    /// row `i`, matching [`Atlas::dataset_row`](crate::Atlas::dataset_row).
    pub fn dataset_names(&self) -> &[Option<String>] {
        &self.dataset_names
    }

    /// The dataset name at `row`, or `None` if the row is out of range or a
    /// tombstone.
    pub fn dataset_name(&self, row: usize) -> Option<&str> {
        self.dataset_names.get(row)?.as_deref()
    }

    /// A column with the liveness mask applied — the interface to prefer.
    ///
    /// ```no_run
    /// # use atlas::{Atlas, ColumnKey, StatVal};
    /// # async fn f(store: &Atlas) -> atlas::Result<()> {
    /// let key = ColumnKey::array("temperature");
    /// let index = store.pruning_index(std::slice::from_ref(&key)).await?;
    /// if let Some(view) = index.view(&key) {
    ///     // Datasets that could hold a value above 25 — deleted and absent
    ///     // rows are already excluded.
    ///     let rows = view.candidates(|_, hi| hi > &StatVal::Float(25.0));
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn view(&self, key: &ColumnKey) -> Option<ColumnView<'_>> {
        self.columns
            .get(key)
            .map(|column| ColumnView::new(column, &self.live))
    }

    /// The raw column, masks not applied. Prefer [`view`](Self::view) unless
    /// you're doing vectorized work and will apply [`live`](Self::live)
    /// yourself.
    pub fn column(&self, key: &ColumnKey) -> Option<&StatColumn> {
        self.columns.get(key)
    }

    // ── Write side ──────────────────────────────────────────────────────

    pub(crate) fn set_meta_epoch(&mut self, epoch: u64) {
        self.meta_epoch = epoch;
    }

    pub(crate) fn set_live(&mut self, live: Vec<bool>) {
        self.live = live;
    }

    /// Attach dataset names, padded/truncated to exactly `rows` so
    /// `dataset_names().len() == rows()` always holds.
    pub(crate) fn set_dataset_names(&mut self, mut names: Vec<Option<String>>) {
        names.resize(self.rows, None);
        self.dataset_names = names;
    }

    pub(crate) fn iter_columns(&self) -> impl Iterator<Item = (&ColumnKey, &StatColumn)> {
        self.order
            .iter()
            .filter_map(|key| self.columns.get(key).map(|column| (key, column)))
    }

    /// Adds an already-decoded column, for assembling a partial index.
    pub(crate) fn insert_column(&mut self, key: ColumnKey, column: StatColumn) {
        if !self.columns.contains_key(&key) {
            self.order.push(key.clone());
        }
        self.columns.insert(key, column);
    }

    /// Appends a null row for a newly created dataset.
    ///
    /// Every column gains an absent cell, so the index is never shorter than
    /// the dataset list.
    pub(crate) fn push_row(&mut self) -> usize {
        let row = self.rows;
        self.rows += 1;
        for column in self.columns.values_mut() {
            column.resize(self.rows);
        }
        row
    }

    /// Grows to `rows` slots, appending null rows.
    pub(crate) fn grow_to(&mut self, rows: usize) {
        while self.rows < rows {
            self.push_row();
        }
    }

    /// Clears a row, for a slot being reused by a new dataset.
    pub(crate) fn reset_row(&mut self, row: usize) {
        for column in self.columns.values_mut() {
            column.clear_row(row);
        }
    }

    /// Ensures a column exists, back-filled as absent for every existing row.
    ///
    /// A column introduced at dataset 9 999 still spans all 10 000 rows — the
    /// column-wise counterpart of appending a null row.
    pub(crate) fn ensure_column(&mut self, key: &ColumnKey) {
        match self.columns.get_mut(key) {
            Some(column) => column.resize(self.rows),
            None => {
                self.columns.insert(key.clone(), StatColumn::new(self.rows));
                self.order.push(key.clone());
            }
        }
    }

    /// Records that the dataset at `row` declares this array/attribute.
    pub(crate) fn set_present(&mut self, key: &ColumnKey, row: usize) {
        if let Some(column) = self.columns.get_mut(key) {
            column.resize(self.rows);
            column.mark_present(row);
        }
    }

    /// Writes one cell's statistics.
    pub(crate) fn set_stats(&mut self, key: &ColumnKey, row: usize, stats: &ArrayStats) {
        if let Some(column) = self.columns.get_mut(key) {
            column.resize(self.rows);
            column.set_stats(row, stats);
        }
    }

    /// Encodes the whole index, stamped with `meta_epoch`.
    ///
    /// `live` selects the rows that feed the footer summaries, so a deleted
    /// dataset can't widen a column's advertised range.
    pub(crate) fn encode(
        &self,
        meta_epoch: u64,
        codec: crate::config::Codec,
        live: &[bool],
    ) -> crate::Result<Vec<u8>> {
        format::write(self.iter_columns(), self.rows, meta_epoch, codec, live)
    }

    /// Decodes a whole index, every column materialized.
    pub(crate) fn decode(bytes: &[u8]) -> crate::Result<Self> {
        let footer = format::read_footer(bytes)?;
        let mut index = PruningIndex::with_rows(footer.row_count);
        index.set_meta_epoch(footer.meta_epoch);
        for key in footer.keys() {
            let column = footer.decode_column_at(bytes, &key)?;
            index.insert_column(key, column);
        }
        Ok(index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Codec;
    use array_format::StatValue;

    fn stats(min: i64, max: i64, rows: u64, nulls: u64) -> ArrayStats {
        ArrayStats {
            name: "x".into(),
            min: Some(StatValue::Int(min)),
            max: Some(StatValue::Int(max)),
            row_count: rows,
            null_count: nulls,
        }
    }

    fn index_with_rows(n: usize) -> PruningIndex {
        let mut index = PruningIndex::new();
        index.grow_to(n);
        index
    }

    #[test]
    fn rows_are_null_until_filled() {
        let mut index = index_with_rows(3);
        let key = ColumnKey::array("temp");
        index.ensure_column(&key);

        let column = index.column(&key).unwrap();
        assert_eq!(column.present.len(), 3);
        assert_eq!(column.present.count_set(), 0, "nothing declared yet");
    }

    #[test]
    fn late_column_is_back_filled() {
        let mut index = index_with_rows(100);
        let key = ColumnKey::array("rare");
        index.ensure_column(&key);
        index.set_stats(&key, 99, &stats(1, 2, 10, 0));

        let column = index.column(&key).unwrap();
        assert_eq!(column.present.len(), 100);
        assert_eq!(column.present.count_set(), 1);
        assert!(column.present.get(99) && !column.present.get(0));
    }

    #[test]
    fn push_row_extends_every_existing_column() {
        let mut index = index_with_rows(2);
        let a = ColumnKey::array("a");
        let b = ColumnKey::global_attr("b");
        index.ensure_column(&a);
        index.ensure_column(&b);
        index.push_row();

        assert_eq!(index.rows(), 3);
        for key in [&a, &b] {
            assert_eq!(index.column(key).unwrap().present.len(), 3);
        }
    }

    #[test]
    fn reset_row_wipes_the_previous_occupant() {
        let mut index = index_with_rows(2);
        let key = ColumnKey::array("temp");
        index.ensure_column(&key);
        index.set_stats(&key, 0, &stats(5, 9, 4, 1));

        index.reset_row(0);
        let column = index.column(&key).unwrap();
        assert!(!column.present.get(0));
        assert_eq!(column.row_count[0], 0);
    }

    #[test]
    fn values_keep_their_source_type() {
        let mut index = index_with_rows(2);
        let key = ColumnKey::array("mixed");
        index.ensure_column(&key);
        index.set_stats(&key, 0, &stats(-3, 7, 10, 0));
        index.set_stats(
            &key,
            1,
            &ArrayStats {
                name: "mixed".into(),
                min: Some(StatValue::Float(0.5)),
                max: Some(StatValue::Float(9.25)),
                row_count: 10,
                null_count: 0,
            },
        );

        let column = index.column(&key).unwrap();
        assert_eq!(column.min[0], Some(StatVal::Int(-3)), "int stays int");
        assert_eq!(column.min[1], Some(StatVal::Float(0.5)), "float stays float");
    }

    /// The view is what most callers should touch, so it must be reachable
    /// straight off a decoded index.
    #[test]
    fn view_applies_the_live_mask() {
        let mut index = index_with_rows(3);
        let key = ColumnKey::array("t");
        index.ensure_column(&key);
        index.set_stats(&key, 0, &stats(1, 5, 2, 0));
        index.set_stats(&key, 1, &stats(100, 200, 2, 0));
        index.set_live(vec![true, false, true]);

        let view = index.view(&key).unwrap();
        assert_eq!(view.present_rows(), vec![0]);
        assert_eq!(view.candidates(|_, hi| hi > &StatVal::Int(50)), Vec::<usize>::new());
    }

    #[test]
    fn encode_decode_roundtrip() {
        let mut index = index_with_rows(5);
        let a = ColumnKey::array("temp");
        let b = ColumnKey::global_attr("cruise");
        index.ensure_column(&a);
        index.ensure_column(&b);
        index.set_stats(&a, 0, &stats(1, 9, 4, 1));
        index.set_stats(&a, 3, &stats(-2, 2, 4, 0));
        index.set_stats(
            &b,
            3,
            &ArrayStats {
                name: "cruise".into(),
                min: Some(StatValue::Bytes(b"CS6151".to_vec())),
                max: Some(StatValue::Bytes(b"CS6151".to_vec())),
                row_count: 1,
                null_count: 0,
            },
        );
        index.set_meta_epoch(7);

        let live = vec![true; 5];
        for codec in [Codec::Uncompressed, Codec::Zstd, Codec::Lz4] {
            let bytes = index.encode(7, codec, &live).unwrap();
            let back = PruningIndex::decode(&bytes).unwrap();

            assert_eq!(back.rows(), 5);
            assert_eq!(back.meta_epoch(), 7);
            assert_eq!(back.column(&a).unwrap(), index.column(&a).unwrap());
            assert_eq!(back.column(&b).unwrap(), index.column(&b).unwrap());
        }
    }

    #[test]
    fn decode_rejects_corrupt_input() {
        assert!(PruningIndex::decode(b"tiny").is_err());
        let mut bytes = index_with_rows(1).encode(0, Codec::Uncompressed, &[true]).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0xff; // break the trailing magic
        assert!(PruningIndex::decode(&bytes).is_err());
    }
}
