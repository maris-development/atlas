//! The `pruning.idx` on-disk layout.
//!
//! ```text
//! [MAGIC][version: u32 LE]
//! [column block 0]      each block independently codec-compressed
//! [column block 1]
//! ...
//! [footer: column directory + summaries]
//! [footer len: u64 LE][MAGIC]
//! ```
//!
//! The directory at the tail is what makes a single column readable on its own:
//! read the trailer, parse the footer, then fetch exactly the byte ranges you
//! want. The trailer convention mirrors array-format's own footer, so the two
//! files are read the same way.

use serde::{Deserialize, Serialize};

use super::{Bitmap, ColumnKey, ColumnSummary, StatColumn, StatVal, invalid};
use crate::{Result, config::Codec};

/// Marks both ends of the file, so a truncated or foreign object is rejected
/// rather than misread.
pub(crate) const MAGIC: [u8; 4] = *b"APRN";
/// `[footer_len: u64 LE][MAGIC]`
pub(crate) const TRAILER_SIZE: usize = 12;
/// Current layout version of `pruning.idx`.
pub const PRUNING_FORMAT_VERSION: u32 = 1;

/// A column's stored form: values only for the rows that have them.
///
/// The bitmaps carry the positions, so a column present in 20 of 10 000 rows
/// stores 20 values, not 10 000 slots.
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
        let mut block = ColumnBlock {
            present: column.present.clone(),
            stats_valid: column.stats_valid.clone(),
            min: Vec::new(),
            max: Vec::new(),
            row_count: Vec::new(),
            null_count: Vec::new(),
        };
        for row in 0..column.rows() {
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
        let mut value_idx = 0usize;
        let mut count_idx = 0usize;
        for row in 0..rows {
            if column.present.get(row) {
                column.row_count[row] = self.row_count.get(count_idx).copied().unwrap_or(0);
                column.null_count[row] = self.null_count.get(count_idx).copied().unwrap_or(0);
                count_idx += 1;
            }
            if column.stats_valid.get(row) {
                column.min[row] = self.min.get(value_idx).cloned();
                column.max[row] = self.max.get(value_idx).cloned();
                value_idx += 1;
            }
        }
        column
    }
}

/// Where one column's block lives, and what it holds — without reading it.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ColumnDirEntry {
    key: ColumnKey,
    offset: u64,
    compressed_len: u64,
    summary: ColumnSummary,
}

/// The directory at the tail of the file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct IndexFooter {
    version: u32,
    /// Must match the store metadata's epoch. Rows are positional, so an index
    /// built against a different dataset list doesn't fail — every row silently
    /// means a different dataset. This is what catches that.
    pub(crate) meta_epoch: u64,
    /// Row slots covered, tombstones included.
    pub(crate) row_count: usize,
    /// Codec each block was compressed with, so a reader adapts without being
    /// told what the writer chose.
    codec: Codec,
    columns: Vec<ColumnDirEntry>,
}

impl IndexFooter {
    /// Every column's key and collection-wide summary.
    pub(crate) fn summaries(&self) -> Vec<(ColumnKey, ColumnSummary)> {
        self.columns
            .iter()
            .map(|c| (c.key.clone(), c.summary.clone()))
            .collect()
    }

    /// Byte range of each requested column, skipping any not in the file.
    ///
    /// This is what turns "read 3 of 102 columns" into three ranged GETs.
    pub(crate) fn ranges_for(&self, wanted: &[ColumnKey]) -> Vec<(ColumnKey, std::ops::Range<u64>)> {
        wanted
            .iter()
            .filter_map(|key| {
                let entry = self.columns.iter().find(|c| &c.key == key)?;
                Some((key.clone(), entry.offset..entry.offset + entry.compressed_len))
            })
            .collect()
    }

    /// Decodes one column from exactly its own block bytes.
    pub(crate) fn decode_column(&self, key: &ColumnKey, block: &[u8]) -> Result<StatColumn> {
        if !self.columns.iter().any(|c| &c.key == key) {
            return Err(invalid(format!("no such pruning column: {key:?}")));
        }
        decode_block(block, self.codec)
    }

    /// Decodes a column from a buffer holding the whole file.
    pub(crate) fn decode_column_at(&self, bytes: &[u8], key: &ColumnKey) -> Result<StatColumn> {
        let entry = self
            .columns
            .iter()
            .find(|c| &c.key == key)
            .ok_or_else(|| invalid(format!("no such pruning column: {key:?}")))?;
        let start = entry.offset as usize;
        let end = start + entry.compressed_len as usize;
        if end > bytes.len() {
            return Err(invalid("pruning index column range out of bounds"));
        }
        decode_block(&bytes[start..end], self.codec)
    }

    /// Column keys in stored order.
    pub(crate) fn keys(&self) -> Vec<ColumnKey> {
        self.columns.iter().map(|c| c.key.clone()).collect()
    }
}

fn decode_block(block: &[u8], codec: Codec) -> Result<StatColumn> {
    let raw = codec.decompress(block)?;
    let block: ColumnBlock =
        rmp_serde::from_slice(&raw).map_err(|e| invalid(format!("pruning index column: {e}")))?;
    Ok(block.expand())
}

/// Writes a complete `pruning.idx` image.
///
/// `live` selects the rows that feed the footer summaries, so a deleted dataset
/// can't widen a column's advertised range.
pub(crate) fn write<'a>(
    columns: impl Iterator<Item = (&'a ColumnKey, &'a StatColumn)>,
    rows: usize,
    meta_epoch: u64,
    codec: Codec,
    live: &[bool],
) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&PRUNING_FORMAT_VERSION.to_le_bytes());

    let mut entries = Vec::new();
    for (key, column) in columns {
        let block = ColumnBlock::compact(column);
        let encoded = rmp_serde::to_vec_named(&block)?;
        let compressed = codec.compress(encoded)?;
        let offset = out.len() as u64;
        out.extend_from_slice(&compressed);
        entries.push(ColumnDirEntry {
            key: key.clone(),
            offset,
            compressed_len: compressed.len() as u64,
            summary: column.summarize(live),
        });
    }

    let footer = IndexFooter {
        version: PRUNING_FORMAT_VERSION,
        meta_epoch,
        row_count: rows,
        codec,
        columns: entries,
    };
    let footer_bytes = rmp_serde::to_vec_named(&footer)?;
    out.extend_from_slice(&footer_bytes);
    out.extend_from_slice(&(footer_bytes.len() as u64).to_le_bytes());
    out.extend_from_slice(&MAGIC);
    Ok(out)
}

/// Parses the footer from a whole-file buffer.
pub(crate) fn read_footer(bytes: &[u8]) -> Result<IndexFooter> {
    match footer_from_suffix(bytes, bytes.len() as u64)? {
        FooterRead::Footer(footer) => Ok(*footer),
        FooterRead::NeedMore(_) => Err(invalid("pruning index footer exceeds the file")),
    }
}

/// Parses the footer from the file's trailing bytes.
///
/// `suffix` is the last `suffix.len()` bytes of a file of `file_len` bytes — a
/// single ranged read of the tail. When the footer is longer than the suffix
/// fetched, reports how many bytes are needed rather than truncating.
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

/// Outcome of parsing a file tail.
pub(crate) enum FooterRead {
    /// The footer, fully parsed.
    Footer(Box<IndexFooter>),
    /// The suffix was too small; re-read this many trailing bytes.
    NeedMore(usize),
}
