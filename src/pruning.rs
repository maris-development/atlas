//! The collection-wide **pruning index**: a flattened, columnar view of every
//! array's and attribute's statistics across all datasets.
//!
//! `DatasetView::array_stats` answers for one array in one dataset. That is the
//! wrong shape for "which datasets could possibly match this predicate?", which
//! wants a *column*: one value per dataset, in a single typed buffer a caller
//! can compare vectorized. This module provides that view.
//!
//! # Row space
//!
//! Row `i` is the dataset at ordinal `i` in
//! [`StoreMeta`](crate::meta::StoreMeta) — positional, with no names stored
//! here. Deleted datasets keep their slot and are hidden by the store's
//! liveness mask at read time, so ordinals never shift and a persisted index
//! stays valid. Only [`Atlas::compact`](crate::Atlas::compact) renumbers.
//!
//! Because rows are positional, an index that drifts out of step with the
//! dataset list does not fail loudly — every row silently means a different
//! dataset. [`IndexFooter::meta_epoch`] guards against that.
//!
//! # Sparsity
//!
//! Most datasets don't declare most arrays (measured at ~26% density on a real
//! 10 000-dataset collection), so a column is a `present` bitmap plus values.
//! Values are held densely in memory for O(1) updates and **compacted to the
//! present rows on disk**, where the 74% of holes would otherwise dominate.

use std::collections::HashMap;

use array_format::{ArrayStats, StatValue};
use serde::{Deserialize, Serialize};

use crate::{
    Error, Result,
    config::Codec,
    meta::{compress, decompress},
};

/// Magic at both ends of the file, so a truncated or foreign file is rejected
/// rather than misread.
const MAGIC: [u8; 4] = *b"APRN";
/// `[footer_len: u64 LE][MAGIC]`
const TRAILER_SIZE: usize = 12;
/// Current layout version of `pruning.idx`.
pub const PRUNING_FORMAT_VERSION: u32 = 1;

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
    /// Convenience constructor for an array column.
    pub fn array(name: impl Into<String>) -> Self {
        Self::Array(name.into())
    }
    /// Convenience constructor for a dataset-level attribute column.
    pub fn global_attr(key: impl Into<String>) -> Self {
        Self::GlobalAttr(key.into())
    }
    /// Convenience constructor for a per-variable attribute column.
    pub fn array_attr(array: impl Into<String>, key: impl Into<String>) -> Self {
        Self::ArrayAttr(array.into(), key.into())
    }
}

/// Serde-compatible mirror of [`array_format::StatValue`].
///
/// `StatValue` serializes via rkyv, not serde, so the index carries its own
/// representation rather than dragging rkyv into the metadata path.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum StatVal {
    /// Signed integer.
    Int(i64),
    /// Unsigned integer (also carries `Bool`, as 0/1).
    UInt(u64),
    /// Floating point.
    Float(f64),
    /// String or binary, raw bytes in lexicographic order.
    Bytes(Vec<u8>),
    /// Nanoseconds since the Unix epoch.
    TimestampNs(i64),
}

impl From<&StatValue> for StatVal {
    fn from(v: &StatValue) -> Self {
        match v {
            StatValue::Int(i) => StatVal::Int(*i),
            StatValue::UInt(u) => StatVal::UInt(*u),
            StatValue::Float(f) => StatVal::Float(*f),
            StatValue::Bytes(b) => StatVal::Bytes(b.clone()),
            StatValue::TimestampNs(t) => StatVal::TimestampNs(*t),
        }
    }
}

/// A packed bit per row.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Bitmap {
    bits: Vec<u8>,
    len: usize,
}

impl Bitmap {
    /// A bitmap of `len` zeroes.
    pub fn zeros(len: usize) -> Self {
        Self {
            bits: vec![0u8; len.div_ceil(8)],
            len,
        }
    }

    /// Number of bits.
    pub fn len(&self) -> usize {
        self.len
    }

    /// `true` if the bitmap holds no bits.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Appends one bit.
    pub fn push(&mut self, value: bool) {
        if self.len.is_multiple_of(8) {
            self.bits.push(0);
        }
        let index = self.len;
        self.len += 1;
        self.set(index, value);
    }

    /// Reads bit `index`; `false` when out of range.
    pub fn get(&self, index: usize) -> bool {
        if index >= self.len {
            return false;
        }
        self.bits[index / 8] & (1u8 << (index % 8)) != 0
    }

    /// Writes bit `index`. Out-of-range writes are ignored.
    pub fn set(&mut self, index: usize, value: bool) {
        if index >= self.len {
            return;
        }
        let mask = 1u8 << (index % 8);
        if value {
            self.bits[index / 8] |= mask;
        } else {
            self.bits[index / 8] &= !mask;
        }
    }

    /// How many bits are set.
    pub fn count_set(&self) -> usize {
        (0..self.len).filter(|i| self.get(*i)).count()
    }

    /// Grows to `len` bits, filling with `false`.
    fn resize(&mut self, len: usize) {
        while self.len < len {
            self.push(false);
        }
    }
}

/// One column's statistics over the whole collection.
///
/// All four value vectors are dense — indexed directly by row ordinal — so
/// updates are O(1). The on-disk form compacts them against the bitmaps.
///
/// The column carries no dtype of its own: `min`/`max` keep whatever type the
/// source statistic had. Where a caller wants the collection-wide type, that is
/// what [`Atlas::merged_schema`](crate::Atlas::merged_schema) is for.
#[derive(Debug, Clone, PartialEq)]
pub struct StatColumn {
    /// Whether the dataset at this row declares the array/attribute.
    pub present: Bitmap,
    /// Whether `min`/`max` are meaningful here. `false` for rows not yet
    /// flushed and for dtypes array-format computes no statistics for
    /// (`List`, `FixedSizeList`).
    pub stats_valid: Bitmap,
    /// Per-row minimum, in the source statistic's own type; `None` where
    /// `stats_valid` is unset.
    pub min: Vec<Option<StatVal>>,
    /// Per-row maximum, in the source statistic's own type; `None` where
    /// `stats_valid` is unset.
    pub max: Vec<Option<StatVal>>,
    /// Per-row element count. **Zero for a dataset that doesn't declare this
    /// array/attribute** — it contributes no rows.
    pub row_count: Vec<u64>,
    /// Per-row null count. Zero where the dataset doesn't declare it.
    pub null_count: Vec<u64>,
}

impl StatColumn {
    fn new(rows: usize) -> Self {
        Self {
            present: Bitmap::zeros(rows),
            stats_valid: Bitmap::zeros(rows),
            min: vec![None; rows],
            max: vec![None; rows],
            row_count: vec![0; rows],
            null_count: vec![0; rows],
        }
    }

    fn rows(&self) -> usize {
        self.min.len()
    }

    fn resize(&mut self, rows: usize) {
        self.present.resize(rows);
        self.stats_valid.resize(rows);
        self.min.resize(rows, None);
        self.max.resize(rows, None);
        self.row_count.resize(rows, 0);
        self.null_count.resize(rows, 0);
    }

    /// Clears every trace of one row — used when a slot is revived by a new
    /// dataset, so it can never surface the previous occupant's statistics.
    fn clear_row(&mut self, row: usize) {
        if row >= self.rows() {
            return;
        }
        self.present.set(row, false);
        self.stats_valid.set(row, false);
        self.min[row] = None;
        self.max[row] = None;
        self.row_count[row] = 0;
        self.null_count[row] = 0;
    }

    /// Collection-wide summary over the rows selected by `live`, for the footer.
    fn summary(&self, live: &[bool]) -> ColumnSummary {
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
                    Some(cur) => min_of(cur, v.clone()),
                });
            }
            if let Some(v) = &self.max[row] {
                max = Some(match max {
                    None => v.clone(),
                    Some(cur) => max_of(cur, v.clone()),
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

/// Orders two same-variant values; mismatched variants keep the first.
fn cmp_vals(a: &StatVal, b: &StatVal) -> Option<std::cmp::Ordering> {
    match (a, b) {
        (StatVal::Int(x), StatVal::Int(y)) => Some(x.cmp(y)),
        (StatVal::UInt(x), StatVal::UInt(y)) => Some(x.cmp(y)),
        (StatVal::Float(x), StatVal::Float(y)) => Some(x.total_cmp(y)),
        (StatVal::TimestampNs(x), StatVal::TimestampNs(y)) => Some(x.cmp(y)),
        (StatVal::Bytes(x), StatVal::Bytes(y)) => Some(x.cmp(y)),
        _ => None,
    }
}

fn min_of(a: StatVal, b: StatVal) -> StatVal {
    match cmp_vals(&a, &b) {
        Some(std::cmp::Ordering::Greater) => b,
        _ => a,
    }
}

fn max_of(a: StatVal, b: StatVal) -> StatVal {
    match cmp_vals(&a, &b) {
        Some(std::cmp::Ordering::Less) => b,
        _ => a,
    }
}

/// Collection-wide summary of one column, held in the footer.
///
/// Lets a caller skip fetching a column's block entirely when its global range
/// cannot satisfy a predicate — coarse pruning above the per-row pruning.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct ColumnSummary {
    /// Smallest `min` across live rows, if any.
    pub min: Option<StatVal>,
    /// Largest `max` across live rows, if any.
    pub max: Option<StatVal>,
    /// How many live rows declare this array/attribute.
    pub present_count: u64,
}

/// On-disk compacted form of a column: values only for the rows that have them.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ColumnBlock {
    present: Bitmap,
    stats_valid: Bitmap,
    /// One entry per set bit of `stats_valid`, in row order.
    min: Vec<StatVal>,
    /// One entry per set bit of `stats_valid`, in row order.
    max: Vec<StatVal>,
    /// One entry per set bit of `present`, in row order.
    row_count: Vec<u64>,
    /// One entry per set bit of `present`, in row order.
    null_count: Vec<u64>,
}

impl ColumnBlock {
    fn compact(column: &StatColumn) -> Self {
        let rows = column.rows();
        let mut block = ColumnBlock {
            present: column.present.clone(),
            stats_valid: column.stats_valid.clone(),
            min: Vec::new(),
            max: Vec::new(),
            row_count: Vec::new(),
            null_count: Vec::new(),
        };
        for row in 0..rows {
            if column.present.get(row) {
                block.row_count.push(column.row_count[row]);
                block.null_count.push(column.null_count[row]);
            }
            if column.stats_valid.get(row) {
                // `stats_valid` is only ever set alongside a value.
                block.min.push(column.min[row].clone().unwrap_or(StatVal::Int(0)));
                block.max.push(column.max[row].clone().unwrap_or(StatVal::Int(0)));
            }
        }
        block
    }

    fn expand(self) -> StatColumn {
        let rows = self.present.len().max(self.stats_valid.len());
        let mut column = StatColumn::new(rows);
        column.present = self.present;
        column.stats_valid = self.stats_valid;
        let mut vi = 0usize;
        let mut ci = 0usize;
        for row in 0..rows {
            if column.present.get(row) {
                column.row_count[row] = self.row_count.get(ci).copied().unwrap_or(0);
                column.null_count[row] = self.null_count.get(ci).copied().unwrap_or(0);
                ci += 1;
            }
            if column.stats_valid.get(row) {
                column.min[row] = self.min.get(vi).cloned();
                column.max[row] = self.max.get(vi).cloned();
                vi += 1;
            }
        }
        column
    }
}

/// One column's entry in the footer directory: where its block is and what it
/// contains, without reading the block.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ColumnDirEntry {
    key: ColumnKey,
    offset: u64,
    compressed_len: u64,
    summary: ColumnSummary,
}

/// The footer: everything needed to locate and pre-filter columns.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct IndexFooter {
    version: u32,
    /// Must match the store metadata's epoch, or the index is numbered for a
    /// different dataset list and every row means something else.
    pub(crate) meta_epoch: u64,
    /// Row slots covered, tombstones included.
    pub(crate) row_count: usize,
    /// Codec each block was compressed with.
    pub(crate) codec: Codec,
    columns: Vec<ColumnDirEntry>,
}

impl IndexFooter {
    /// Every column's key and collection-wide summary — enough to decide what
    /// to fetch, read from the footer alone.
    pub(crate) fn summaries(&self) -> Vec<(ColumnKey, ColumnSummary)> {
        self.columns
            .iter()
            .map(|c| (c.key.clone(), c.summary.clone()))
            .collect()
    }

    /// Byte range of each requested column's block, skipping any that aren't in
    /// the file. This is what turns "read 3 of 102 columns" into three ranged
    /// GETs instead of downloading everything.
    pub(crate) fn ranges_for(&self, wanted: &[ColumnKey]) -> Vec<(ColumnKey, std::ops::Range<u64>)> {
        wanted
            .iter()
            .filter_map(|key| {
                let entry = self.columns.iter().find(|c| &c.key == key)?;
                Some((
                    key.clone(),
                    entry.offset..entry.offset + entry.compressed_len,
                ))
            })
            .collect()
    }

    /// Decodes one column from exactly its own block bytes.
    pub(crate) fn decode_column(&self, key: &ColumnKey, block: &[u8]) -> Result<StatColumn> {
        if !self.columns.iter().any(|c| &c.key == key) {
            return Err(invalid(format!("no such pruning column: {key:?}")));
        }
        let raw = decompress(block, self.codec)?;
        let block: ColumnBlock = rmp_serde::from_slice(&raw)
            .map_err(|e| invalid(format!("pruning index column: {e}")))?;
        Ok(block.expand())
    }
}

/// Parses the footer from the file's trailing bytes.
///
/// `suffix` is the last `suffix.len()` bytes of a file of `file_len` bytes — a
/// single ranged read of the tail. If the footer turns out to be longer than
/// the suffix fetched, returns the number of bytes actually needed so the
/// caller can re-read, rather than silently truncating.
pub(crate) fn footer_from_suffix(suffix: &[u8], file_len: u64) -> Result<FooterRead> {
    if suffix.len() < TRAILER_SIZE {
        return Err(invalid("pruning index suffix shorter than its trailer"));
    }
    if suffix[suffix.len() - 4..] != MAGIC {
        return Err(invalid("pruning index magic mismatch"));
    }
    let len_start = suffix.len() - TRAILER_SIZE;
    let footer_len =
        u64::from_le_bytes(suffix[len_start..len_start + 8].try_into().unwrap()) as usize;
    let needed = footer_len + TRAILER_SIZE;
    if needed > suffix.len() {
        if needed as u64 > file_len {
            return Err(invalid("pruning index footer length exceeds file"));
        }
        return Ok(FooterRead::NeedMore(needed));
    }
    let start = suffix.len() - needed;
    let footer: IndexFooter = rmp_serde::from_slice(&suffix[start..len_start])
        .map_err(|e| invalid(format!("pruning index footer: {e}")))?;
    if footer.version != PRUNING_FORMAT_VERSION {
        return Err(invalid(format!(
            "unsupported pruning index version {} (expected {PRUNING_FORMAT_VERSION})",
            footer.version
        )));
    }
    Ok(FooterRead::Footer(Box::new(footer)))
}

/// Outcome of parsing a file tail: either the footer, or how many trailing
/// bytes are actually needed.
pub(crate) enum FooterRead {
    /// The footer, fully parsed.
    Footer(Box<IndexFooter>),
    /// The suffix was too small; re-read this many trailing bytes.
    NeedMore(usize),
}

/// The pruning index for a whole collection.
///
/// Held in memory beside the store metadata and kept aligned continuously: a
/// null row is appended the instant a dataset is created, so
/// `rows == StoreMeta::row_slots()` holds at every point, not just after a
/// flush.
#[derive(Debug, Clone, Default)]
pub struct PruningIndex {
    rows: usize,
    columns: HashMap<ColumnKey, StatColumn>,
    /// Insertion order, so the encoded file is deterministic.
    order: Vec<ColumnKey>,
    meta_epoch: u64,
}

impl PruningIndex {
    /// An empty index covering no rows.
    pub fn new() -> Self {
        Self::default()
    }

    /// Row slots covered, tombstones included.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// The metadata epoch this index was built against.
    pub fn meta_epoch(&self) -> u64 {
        self.meta_epoch
    }

    /// Sets the epoch this index corresponds to.
    pub fn set_meta_epoch(&mut self, epoch: u64) {
        self.meta_epoch = epoch;
    }

    /// Column keys, in encode order.
    pub fn column_keys(&self) -> &[ColumnKey] {
        &self.order
    }

    /// A column, if it exists.
    pub fn column(&self, key: &ColumnKey) -> Option<&StatColumn> {
        self.columns.get(key)
    }

    /// Inserts an already-decoded column, used when assembling a partial index
    /// from selectively fetched blocks.
    pub fn insert_column(&mut self, key: ColumnKey, column: StatColumn) {
        if !self.columns.contains_key(&key) {
            self.order.push(key.clone());
        }
        self.columns.insert(key, column);
    }

    /// Appends a null row for a newly created dataset.
    ///
    /// Every column gains an absent cell, so the index stays exactly as long as
    /// the dataset list from the moment the dataset exists — there is never a
    /// window where a row is missing.
    pub fn push_row(&mut self) -> usize {
        let row = self.rows;
        self.rows += 1;
        for column in self.columns.values_mut() {
            column.resize(self.rows);
        }
        row
    }

    /// Clears a row, for a slot being reused by a new dataset.
    pub fn reset_row(&mut self, row: usize) {
        for column in self.columns.values_mut() {
            column.clear_row(row);
        }
    }

    /// Ensures a column exists, back-filled as absent for every existing row.
    ///
    /// A column introduced at dataset 9 999 still spans all 10 000 rows, with
    /// the earlier ones absent — the column-wise counterpart of appending a
    /// null row.
    pub fn ensure_column(&mut self, key: &ColumnKey) {
        match self.columns.get_mut(key) {
            Some(column) => column.resize(self.rows),
            None => {
                self.columns.insert(key.clone(), StatColumn::new(self.rows));
                self.order.push(key.clone());
            }
        }
    }

    /// Marks that the dataset at `row` declares this array/attribute.
    pub fn set_present(&mut self, key: &ColumnKey, row: usize) {
        if let Some(column) = self.columns.get_mut(key) {
            column.resize(self.rows);
            column.present.set(row, true);
        }
    }

    /// Writes one cell's statistics.
    ///
    /// `min`/`max` are stored in whatever type the source statistic used; no
    /// conversion happens here. `stats_valid` stays unset when the source has
    /// no range at all, which is what array-format reports for `List` dtypes.
    pub fn set_stats(&mut self, key: &ColumnKey, row: usize, stats: &ArrayStats) {
        let Some(column) = self.columns.get_mut(key) else {
            return;
        };
        column.resize(self.rows);
        if row >= column.rows() {
            return;
        }
        column.present.set(row, true);
        column.row_count[row] = stats.row_count;
        column.null_count[row] = stats.null_count;

        let min = stats.min.as_ref().map(StatVal::from);
        let max = stats.max.as_ref().map(StatVal::from);
        match (min, max) {
            (Some(lo), Some(hi)) => {
                column.min[row] = Some(lo);
                column.max[row] = Some(hi);
                column.stats_valid.set(row, true);
            }
            // No range reported — the counts still stand.
            _ => {
                column.min[row] = None;
                column.max[row] = None;
                column.stats_valid.set(row, false);
            }
        }
    }

    /// Encodes the whole index. `live` selects the rows that feed the footer
    /// summaries, so a deleted dataset can't widen a column's global range.
    pub fn encode(&self, codec: Codec, live: &[bool]) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        out.extend_from_slice(&MAGIC);
        out.extend_from_slice(&PRUNING_FORMAT_VERSION.to_le_bytes());

        let mut entries = Vec::with_capacity(self.order.len());
        for key in &self.order {
            let Some(column) = self.columns.get(key) else {
                continue;
            };
            let block = ColumnBlock::compact(column);
            let encoded = rmp_serde::to_vec_named(&block)?;
            let compressed = compress(encoded, codec)?;
            let offset = out.len() as u64;
            out.extend_from_slice(&compressed);
            entries.push(ColumnDirEntry {
                key: key.clone(),
                offset,
                compressed_len: compressed.len() as u64,
                summary: column.summary(live),
            });
        }

        let footer = IndexFooter {
            version: PRUNING_FORMAT_VERSION,
            meta_epoch: self.meta_epoch,
            row_count: self.rows,
            codec,
            columns: entries,
        };
        let footer_bytes = rmp_serde::to_vec_named(&footer)?;
        out.extend_from_slice(&footer_bytes);
        out.extend_from_slice(&(footer_bytes.len() as u64).to_le_bytes());
        out.extend_from_slice(&MAGIC);
        Ok(out)
    }

    /// Decodes a whole index from `bytes` (every column materialized).
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let footer = read_footer(bytes)?;
        let mut index = PruningIndex {
            rows: footer.row_count,
            meta_epoch: footer.meta_epoch,
            ..Default::default()
        };
        for entry in &footer.columns {
            let column = decode_block(bytes, entry, footer.codec)?;
            index.columns.insert(entry.key.clone(), column);
            index.order.push(entry.key.clone());
        }
        Ok(index)
    }
}

fn invalid(msg: impl Into<String>) -> Error {
    Error::ArrayFormat(array_format::Error::Storage(msg.into()))
}

/// Reads the footer out of a full-file buffer.
fn read_footer(bytes: &[u8]) -> Result<IndexFooter> {
    if bytes.len() < TRAILER_SIZE + 8 {
        return Err(invalid("pruning index too short"));
    }
    if bytes[..4] != MAGIC || bytes[bytes.len() - 4..] != MAGIC {
        return Err(invalid("pruning index magic mismatch"));
    }
    let len_start = bytes.len() - TRAILER_SIZE;
    let footer_len =
        u64::from_le_bytes(bytes[len_start..len_start + 8].try_into().unwrap()) as usize;
    if footer_len > len_start {
        return Err(invalid("pruning index footer length out of range"));
    }
    let footer: IndexFooter = rmp_serde::from_slice(&bytes[len_start - footer_len..len_start])
        .map_err(|e| invalid(format!("pruning index footer: {e}")))?;
    if footer.version != PRUNING_FORMAT_VERSION {
        return Err(invalid(format!(
            "unsupported pruning index version {} (expected {PRUNING_FORMAT_VERSION})",
            footer.version
        )));
    }
    Ok(footer)
}

/// Decodes one column's block out of a buffer that covers its byte range.
fn decode_block(bytes: &[u8], entry: &ColumnDirEntry, codec: Codec) -> Result<StatColumn> {
    let start = entry.offset as usize;
    let end = start + entry.compressed_len as usize;
    if end > bytes.len() {
        return Err(invalid("pruning index column range out of bounds"));
    }
    let raw = decompress(&bytes[start..end], codec)?;
    let block: ColumnBlock =
        rmp_serde::from_slice(&raw).map_err(|e| invalid(format!("pruning index column: {e}")))?;
    Ok(block.expand())
}

#[cfg(test)]
mod tests {
    use super::*;

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
        for _ in 0..n {
            index.push_row();
        }
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
        assert_eq!(column.stats_valid.count_set(), 0);
    }

    /// A column introduced late still spans every row, with the earlier ones
    /// absent — the column-wise counterpart of appending a null row.
    #[test]
    fn late_column_is_back_filled() {
        let mut index = index_with_rows(100);
        let key = ColumnKey::array("rare");
        index.ensure_column(&key);
        index.set_stats(&key, 99, &stats(1, 2, 10, 0));

        let column = index.column(&key).unwrap();
        assert_eq!(column.present.len(), 100);
        assert_eq!(column.present.count_set(), 1);
        assert!(column.present.get(99));
        assert!(!column.present.get(0));

        // ...and the stored form holds one value, not 100.
        let block = ColumnBlock::compact(column);
        assert_eq!(block.min.len(), 1);
        assert_eq!(block.row_count.len(), 1);
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
        assert!(index.column(&key).unwrap().present.get(0));

        index.reset_row(0);
        let column = index.column(&key).unwrap();
        assert!(!column.present.get(0));
        assert!(!column.stats_valid.get(0));
        assert_eq!(column.min[0], None);
        assert_eq!(column.row_count[0], 0);
    }

    /// Statistics keep the type they were computed with — no conversion.
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

    /// A dataset that doesn't declare the array contributes a zero row count,
    /// not an undefined one.
    #[test]
    fn absent_rows_have_zero_counts() {
        let mut index = index_with_rows(3);
        let key = ColumnKey::array("temp");
        index.ensure_column(&key);
        index.set_stats(&key, 1, &stats(1, 2, 42, 7));

        let column = index.column(&key).unwrap();
        assert_eq!(column.row_count, vec![0, 42, 0]);
        assert_eq!(column.null_count, vec![0, 7, 0]);
        assert!(!column.present.get(0) && !column.present.get(2));
    }

    /// A statistic with no range at all (what array-format reports for `List`
    /// dtypes) still records its counts.
    #[test]
    fn missing_range_keeps_counts() {
        let mut index = index_with_rows(1);
        let key = ColumnKey::array("listy");
        index.ensure_column(&key);
        index.set_stats(
            &key,
            0,
            &ArrayStats {
                name: "listy".into(),
                min: None,
                max: None,
                row_count: 6,
                null_count: 1,
            },
        );

        let column = index.column(&key).unwrap();
        assert!(column.present.get(0), "the dataset does declare it");
        assert!(!column.stats_valid.get(0), "but there is no range");
        assert_eq!(column.row_count[0], 6);
        assert_eq!(column.null_count[0], 1);
    }

    #[test]
    fn summary_ignores_masked_rows() {
        let mut index = index_with_rows(3);
        let key = ColumnKey::array("t");
        index.ensure_column(&key);
        index.set_stats(&key, 0, &stats(10, 20, 1, 0));
        index.set_stats(&key, 1, &stats(-999, 999, 1, 0)); // to be masked
        index.set_stats(&key, 2, &stats(15, 25, 1, 0));

        let column = index.column(&key).unwrap();
        let all = column.summary(&[true, true, true]);
        assert_eq!(all.min, Some(StatVal::Int(-999)));
        assert_eq!(all.max, Some(StatVal::Int(999)));
        assert_eq!(all.present_count, 3);

        let masked = column.summary(&[true, false, true]);
        assert_eq!(
            masked.min,
            Some(StatVal::Int(10)),
            "a deleted dataset must not widen the global range"
        );
        assert_eq!(masked.max, Some(StatVal::Int(25)));
        assert_eq!(masked.present_count, 2);
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
            let bytes = index.encode(codec, &live).unwrap();
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
        let mut bytes = index_with_rows(1).encode(Codec::Uncompressed, &[true]).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0xff; // break the trailing magic
        assert!(PruningIndex::decode(&bytes).is_err());
    }

    #[test]
    fn bitmap_basics() {
        let mut bits = Bitmap::zeros(0);
        for i in 0..20 {
            bits.push(i % 3 == 0);
        }
        assert_eq!(bits.len(), 20);
        assert_eq!(bits.count_set(), 7);
        assert!(bits.get(0) && bits.get(3) && !bits.get(1));
        bits.set(1, true);
        assert!(bits.get(1));
        bits.set(0, false);
        assert!(!bits.get(0));
        assert!(!bits.get(100), "out of range reads as unset");
    }
}
