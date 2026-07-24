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
mod value;

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

pub use column::{ColumnSummary, ColumnView, StatColumn};
pub use value::StatVal;

pub(crate) use bitmap::Bitmap;

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

/// The pruning index for a collection: a flat, columnar view of every array's
/// and attribute's statistics across all datasets, **built on demand** for the
/// requested columns from the array files' own statistics — there is no
/// persisted index. See [`Atlas::pruning_index`](crate::Atlas::pruning_index).
///
/// Reach for [`view`](Self::view) rather than [`column`](Self::column): it
/// applies the liveness mask so deleted and absent rows can't leak into a
/// result.
#[derive(Debug, Clone, Default)]
pub struct PruningIndex {
    rows: usize,
    columns: HashMap<ColumnKey, StatColumn>,
    /// Insertion order, so column iteration is deterministic.
    order: Vec<ColumnKey>,
    /// Liveness per row slot: `false` where the dataset was deleted.
    live: Vec<bool>,
    /// Dataset name per row slot, `None` for tombstones — so the index is
    /// self-describing: row ↔ name without a second call to the store.
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

    // ── Read-time assembly ──────────────────────────────────────────────

    pub(crate) fn set_live(&mut self, live: Vec<bool>) {
        self.live = live;
    }

    /// Attach dataset names, padded/truncated to exactly `rows` so
    /// `dataset_names().len() == rows()` always holds.
    pub(crate) fn set_dataset_names(&mut self, mut names: Vec<Option<String>>) {
        names.resize(self.rows, None);
        self.dataset_names = names;
    }

    /// Adds a built column to the index. Columns are built on demand from the
    /// array files' statistics (see [`Atlas::pruning_index`]), so this is how
    /// the read-time builder assembles the flat table.
    pub(crate) fn insert_column(&mut self, key: ColumnKey, column: StatColumn) {
        if !self.columns.contains_key(&key) {
            self.order.push(key.clone());
        }
        self.columns.insert(key, column);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use array_format::{ArrayStats, StatValue};

    fn stats(min: i64, max: i64, rows: u64, nulls: u64) -> ArrayStats {
        ArrayStats {
            name: "x".into(),
            min: Some(StatValue::Int(min)),
            max: Some(StatValue::Int(max)),
            row_count: rows,
            null_count: nulls,
        }
    }

    /// Build a column the way the on-demand read path does: a fresh `StatColumn`
    /// scattered by ordinal, then `insert_column`.
    fn index_with_temp(rows: usize, cells: &[(usize, ArrayStats)]) -> PruningIndex {
        let mut column = StatColumn::new(rows);
        for (row, s) in cells {
            column.set_stats(*row, s);
        }
        let mut index = PruningIndex::with_rows(rows);
        index.insert_column(ColumnKey::array("temp"), column);
        index
    }

    #[test]
    fn assembled_column_scatters_by_ordinal() {
        let index = index_with_temp(100, &[(99, stats(1, 2, 10, 0))]);
        let column = index.column(&ColumnKey::array("temp")).unwrap();
        assert_eq!(column.present_mask().iter().filter(|b| **b).count(), 1);
        assert!(column.present.get(99) && !column.present.get(0));
    }

    #[test]
    fn values_keep_their_source_type() {
        let index = index_with_temp(
            2,
            &[
                (0, stats(-3, 7, 10, 0)),
                (
                    1,
                    ArrayStats {
                        name: "temp".into(),
                        min: Some(StatValue::Float(0.5)),
                        max: Some(StatValue::Float(9.25)),
                        row_count: 10,
                        null_count: 0,
                    },
                ),
            ],
        );
        let column = index.column(&ColumnKey::array("temp")).unwrap();
        assert_eq!(column.min[0], Some(StatVal::Int(-3)), "int stays int");
        assert_eq!(column.min[1], Some(StatVal::Float(0.5)), "float stays float");
    }

    /// The view folds in the liveness mask, hiding deleted and absent rows.
    #[test]
    fn view_applies_the_live_mask() {
        let mut index = index_with_temp(3, &[(0, stats(1, 5, 2, 0)), (1, stats(100, 200, 2, 0))]);
        index.set_live(vec![true, false, true]);

        let view = index.view(&ColumnKey::array("temp")).unwrap();
        assert_eq!(view.present_rows(), vec![0]);
        assert_eq!(view.candidates(|_, hi| hi > &StatVal::Int(50)), Vec::<usize>::new());
    }
}
