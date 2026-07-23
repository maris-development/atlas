//! One column of the flattened index, and the safe view over it.

use array_format::ArrayStats;
use serde::{Deserialize, Serialize};

use super::{Bitmap, StatVal};

/// One array's or attribute's statistics across the whole collection.
///
/// Every vector is dense — indexed directly by row ordinal — so a write is
/// O(1). The on-disk form compacts them against the bitmaps, which is where the
/// ~74% of absent cells in a typical collection stop costing anything.
///
/// The column stores no dtype: `min`/`max` keep the type the source statistic
/// used. For a column's collection-wide declared type see
/// [`Atlas::merged_schema`](crate::Atlas::merged_schema).
///
/// The fields are public so a caller can hand the buffers straight to numpy or
/// a vectorized comparison. When reading row by row, prefer
/// [`ColumnView`](ColumnView), which applies the masks for you.
#[derive(Debug, Clone, PartialEq)]
pub struct StatColumn {
    /// Whether the dataset at this row declares the array/attribute. Read as a
    /// dense `Vec<bool>` via [`present_mask`](Self::present_mask).
    pub(crate) present: Bitmap,
    /// Whether `min`/`max` are meaningful here. Unset for rows not yet flushed
    /// and for dtypes array-format computes no statistics for (`List`,
    /// `FixedSizeList`). Read via [`stats_valid_mask`](Self::stats_valid_mask).
    pub(crate) stats_valid: Bitmap,
    /// Per-row minimum; `None` wherever `stats_valid` is unset.
    pub min: Vec<Option<StatVal>>,
    /// Per-row maximum; `None` wherever `stats_valid` is unset.
    pub max: Vec<Option<StatVal>>,
    /// Per-row element count. **Zero where the dataset doesn't declare it** —
    /// it contributes no rows.
    pub row_count: Vec<u64>,
    /// Per-row null count. Zero where the dataset doesn't declare it.
    pub null_count: Vec<u64>,
}

impl StatColumn {
    /// The `present` mask as a dense `Vec<bool>`, one entry per row — the
    /// natural form for a vectorized consumer (e.g. a numpy boolean array).
    pub fn present_mask(&self) -> Vec<bool> {
        (0..self.rows()).map(|i| self.present.get(i)).collect()
    }

    /// The `stats_valid` mask as a dense `Vec<bool>`, one entry per row.
    pub fn stats_valid_mask(&self) -> Vec<bool> {
        (0..self.rows()).map(|i| self.stats_valid.get(i)).collect()
    }
}

impl StatColumn {
    pub(crate) fn new(rows: usize) -> Self {
        Self {
            present: Bitmap::zeros(rows),
            stats_valid: Bitmap::zeros(rows),
            min: vec![None; rows],
            max: vec![None; rows],
            row_count: vec![0; rows],
            null_count: vec![0; rows],
        }
    }

    /// Row slots covered.
    pub fn rows(&self) -> usize {
        self.min.len()
    }

    /// Records that the dataset at `row` declares this array/attribute.
    pub(crate) fn mark_present(&mut self, row: usize) {
        self.present.set(row, true);
    }

    /// Writes one cell from a freshly computed statistic.
    ///
    /// `min`/`max` are stored as-is; `stats_valid` stays unset when the source
    /// reports no range, which is what array-format does for `List` dtypes.
    pub(crate) fn set_stats(&mut self, row: usize, stats: &ArrayStats) {
        if row >= self.rows() {
            return;
        }
        self.present.set(row, true);
        self.row_count[row] = stats.row_count;
        self.null_count[row] = stats.null_count;

        match (
            stats.min.as_ref().map(StatVal::from),
            stats.max.as_ref().map(StatVal::from),
        ) {
            (Some(lo), Some(hi)) => {
                self.min[row] = Some(lo);
                self.max[row] = Some(hi);
                self.stats_valid.set(row, true);
            }
            // No range reported — the counts still stand.
            _ => {
                self.min[row] = None;
                self.max[row] = None;
                self.stats_valid.set(row, false);
            }
        }
    }

    /// Collection-wide summary over the rows selected by `live`.
    pub(crate) fn summarize(&self, live: &[bool]) -> ColumnSummary {
        let mut min: Option<StatVal> = None;
        let mut max: Option<StatVal> = None;
        let mut present_count = 0u64;
        for row in 0..self.rows() {
            if !live.get(row).copied().unwrap_or(false) || !self.present.get(row) {
                continue;
            }
            present_count += 1;
            if !self.stats_valid.get(row) {
                continue;
            }
            if let Some(v) = &self.min[row] {
                min = Some(match min {
                    None => v.clone(),
                    Some(cur) => cur.min_with(v.clone()),
                });
            }
            if let Some(v) = &self.max[row] {
                max = Some(match max {
                    None => v.clone(),
                    Some(cur) => cur.max_with(v.clone()),
                });
            }
        }
        ColumnSummary {
            min,
            max,
            present_count,
        }
    }
}

/// A column's collection-wide range, held in the index footer.
///
/// Reading it costs no column data, so a predicate outside this range rules the
/// whole column out before a single block is fetched.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct ColumnSummary {
    /// Smallest `min` across live rows, if any.
    pub min: Option<StatVal>,
    /// Largest `max` across live rows, if any.
    pub max: Option<StatVal>,
    /// How many live rows declare this array/attribute.
    pub present_count: u64,
}

impl ColumnSummary {
    /// `true` if some row *could* satisfy `predicate`, judged from the
    /// collection-wide range alone.
    ///
    /// A `false` here means no dataset can match, so the column need not be
    /// read at all. A `true` is not a match — it means "worth looking".
    pub fn might_match(&self, predicate: impl Fn(&StatVal, &StatVal) -> bool) -> bool {
        match (&self.min, &self.max) {
            (Some(lo), Some(hi)) => predicate(lo, hi),
            // No range recorded: can't rule it out.
            _ => self.present_count > 0,
        }
    }
}

/// A column with the collection's liveness mask already applied.
///
/// Reading a column correctly means combining three masks — `present`,
/// `stats_valid`, and the store's `live` — and getting that wrong silently
/// yields deleted datasets or garbage ranges. This view folds all three in, so
/// every accessor answers only for rows that genuinely have data.
///
/// For vectorized work, [`raw`](Self::raw) and [`live`](Self::live) hand back
/// the underlying buffers.
#[derive(Debug, Clone, Copy)]
pub struct ColumnView<'a> {
    column: &'a StatColumn,
    live: &'a [bool],
}

impl<'a> ColumnView<'a> {
    pub(crate) fn new(column: &'a StatColumn, live: &'a [bool]) -> Self {
        Self { column, live }
    }

    /// Row slots covered, tombstones included.
    pub fn rows(&self) -> usize {
        self.column.rows()
    }

    /// The dataset at `row` exists and declares this array/attribute.
    pub fn is_present(&self, row: usize) -> bool {
        self.live.get(row).copied().unwrap_or(false) && self.column.present.get(row)
    }

    /// As [`is_present`](Self::is_present), and `min`/`max` are usable.
    pub fn has_stats(&self, row: usize) -> bool {
        self.is_present(row) && self.column.stats_valid.get(row)
    }

    /// Minimum at `row`, or `None` if the row has no usable statistic.
    pub fn min(&self, row: usize) -> Option<&'a StatVal> {
        self.has_stats(row).then(|| self.column.min[row].as_ref())?
    }

    /// Maximum at `row`, or `None` if the row has no usable statistic.
    pub fn max(&self, row: usize) -> Option<&'a StatVal> {
        self.has_stats(row).then(|| self.column.max[row].as_ref())?
    }

    /// Element count at `row`; 0 where the dataset doesn't declare it.
    pub fn row_count(&self, row: usize) -> u64 {
        if self.is_present(row) {
            self.column.row_count[row]
        } else {
            0
        }
    }

    /// Null count at `row`; 0 where the dataset doesn't declare it.
    pub fn null_count(&self, row: usize) -> u64 {
        if self.is_present(row) {
            self.column.null_count[row]
        } else {
            0
        }
    }

    /// Rows whose `(min, max)` range could satisfy `predicate` — the pruning
    /// primitive.
    ///
    /// Rows that are deleted, absent, or without a range are excluded, so the
    /// result is exactly the datasets worth opening. The predicate sees the
    /// row's range, not individual values: `|lo, hi| hi >= &target` asks "could
    /// this dataset contain something at least this large?".
    pub fn candidates(&self, predicate: impl Fn(&StatVal, &StatVal) -> bool) -> Vec<usize> {
        (0..self.rows())
            .filter(|row| match (self.min(*row), self.max(*row)) {
                (Some(lo), Some(hi)) => predicate(lo, hi),
                _ => false,
            })
            .collect()
    }

    /// Every row that exists and declares this array/attribute.
    pub fn present_rows(&self) -> Vec<usize> {
        (0..self.rows()).filter(|row| self.is_present(*row)).collect()
    }

    /// The underlying buffers, for vectorized use. Masks are **not** applied —
    /// combine with [`live`](Self::live) yourself.
    pub fn raw(&self) -> &'a StatColumn {
        self.column
    }

    /// The store's liveness mask over row slots.
    pub fn live(&self) -> &'a [bool] {
        self.live
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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

    fn column_of(rows: usize) -> StatColumn {
        StatColumn::new(rows)
    }

    #[test]
    fn absent_rows_report_zero_counts() {
        let mut column = column_of(3);
        column.set_stats(1, &stats(1, 2, 42, 7));
        assert_eq!(column.row_count, vec![0, 42, 0]);
        assert_eq!(column.null_count, vec![0, 7, 0]);
    }

    #[test]
    fn missing_range_keeps_counts() {
        let mut column = column_of(1);
        column.set_stats(
            0,
            &ArrayStats {
                name: "listy".into(),
                min: None,
                max: None,
                row_count: 6,
                null_count: 1,
            },
        );
        assert!(column.present.get(0), "the dataset does declare it");
        assert!(!column.stats_valid.get(0), "but there is no range");
        assert_eq!(column.row_count[0], 6);
    }

    #[test]
    fn summary_ignores_masked_rows() {
        let mut column = column_of(3);
        column.set_stats(0, &stats(10, 20, 1, 0));
        column.set_stats(1, &stats(-999, 999, 1, 0)); // to be masked
        column.set_stats(2, &stats(15, 25, 1, 0));

        let all = column.summarize(&[true, true, true]);
        assert_eq!(all.min, Some(StatVal::Int(-999)));
        assert_eq!(all.present_count, 3);

        let masked = column.summarize(&[true, false, true]);
        assert_eq!(
            masked.min,
            Some(StatVal::Int(10)),
            "a deleted dataset must not widen the global range"
        );
        assert_eq!(masked.max, Some(StatVal::Int(25)));
        assert_eq!(masked.present_count, 2);
    }

    /// The view is the whole point of the abstraction: a deleted row must be
    /// invisible without the caller doing anything.
    #[test]
    fn view_hides_deleted_and_absent_rows() {
        let mut column = column_of(3);
        column.set_stats(0, &stats(1, 5, 10, 0));
        column.set_stats(1, &stats(100, 200, 10, 0)); // deleted below
        // row 2 left absent

        let live = [true, false, true];
        let view = ColumnView::new(&column, &live);

        assert!(view.is_present(0));
        assert!(!view.is_present(1), "deleted");
        assert!(!view.is_present(2), "never declared");

        assert_eq!(view.min(0), Some(&StatVal::Int(1)));
        assert_eq!(view.min(1), None, "deleted rows expose no statistics");
        assert_eq!(view.row_count(1), 0, "and no counts");
        assert_eq!(view.present_rows(), vec![0]);
    }

    #[test]
    fn candidates_applies_all_three_masks() {
        let mut column = column_of(4);
        column.set_stats(0, &stats(1, 5, 10, 0));
        column.set_stats(1, &stats(1, 500, 10, 0)); // deleted
        column.set_stats(2, &stats(1, 50, 10, 0));
        // row 3 absent

        let live = [true, false, true, true];
        let view = ColumnView::new(&column, &live);

        let hits = view.candidates(|_, hi| hi > &StatVal::Int(10));
        assert_eq!(hits, vec![2], "row 1 is deleted, row 3 has no data");
    }

    #[test]
    fn summary_rules_out_a_whole_column() {
        let mut column = column_of(2);
        column.set_stats(0, &stats(1, 9, 5, 0));
        column.set_stats(1, &stats(2, 8, 5, 0));
        let summary = column.summarize(&[true, true]);

        assert!(!summary.might_match(|_, hi| hi > &StatVal::Int(100)), "nothing can match");
        assert!(summary.might_match(|_, hi| hi > &StatVal::Int(5)), "worth reading");
    }
}
