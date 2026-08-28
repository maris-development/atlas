//! The deletion mask: the one thing about a collection that can change.
//!
//! The container is write-once, so deleting a dataset cannot touch it. Instead
//! a small sidecar records the ordinals of deleted datasets, and the reader
//! hides them.
//!
//! ```text
//! b"ATLM"          4 B   magic
//! version u32 LE   4 B   = 1
//! count   u32 LE   4 B
//! count x u32 LE         ordinals, strictly increasing
//! ```
//!
//! # Concurrency
//!
//! Deleting reads the mask, adds one ordinal, and writes the whole file back.
//! Two deletions racing on the same collection are last-writer-wins: one of the
//! two deletions can be lost. Serialize deletes if that matters.
//!
//! # Tolerance
//!
//! An absent mask means nothing is deleted; the writer never creates one. An
//! ordinal past the end of the footer is ignored with a warning, so a mask left
//! over from a different container cannot stop a collection from opening. Only
//! a wrong magic is an error, because a foreign file at that path is a mistake
//! worth reporting.

use std::collections::BTreeSet;

use crate::{Error, Result};

const MASK_MAGIC: [u8; 4] = *b"ATLM";
const MASK_VERSION: u32 = 1;
const MASK_HEADER_SIZE: usize = 12;

/// Encodes `ordinals` into the mask sidecar layout.
pub(crate) fn encode(ordinals: &BTreeSet<u32>) -> Vec<u8> {
    let mut out = Vec::with_capacity(MASK_HEADER_SIZE + 4 * ordinals.len());
    out.extend_from_slice(&MASK_MAGIC);
    out.extend_from_slice(&MASK_VERSION.to_le_bytes());
    out.extend_from_slice(&(ordinals.len() as u32).to_le_bytes());
    for &o in ordinals {
        out.extend_from_slice(&o.to_le_bytes());
    }
    out
}

/// Decodes a mask sidecar, dropping ordinals that do not name a dataset.
///
/// `dataset_count` bounds the valid range. Returns an error only when the file
/// is not a mask at all.
pub(crate) fn decode(bytes: &[u8], dataset_count: usize) -> Result<BTreeSet<u32>> {
    if bytes.is_empty() {
        return Ok(BTreeSet::new());
    }
    if bytes.len() < MASK_HEADER_SIZE || bytes[..4] != MASK_MAGIC {
        return Err(Error::CorruptMask(
            "file does not start with the ATLM magic".to_string(),
        ));
    }
    let version = u32::from_le_bytes(bytes[4..8].try_into().expect("4 bytes"));
    if version != MASK_VERSION {
        return Err(Error::CorruptMask(format!(
            "mask version {version} is not supported; this atlas writes version {MASK_VERSION}"
        )));
    }
    let count = u32::from_le_bytes(bytes[8..12].try_into().expect("4 bytes")) as usize;
    let body = &bytes[MASK_HEADER_SIZE..];
    let available = body.len() / 4;
    if available < count {
        tracing::warn!(
            declared = count,
            available,
            "deletion mask is truncated; using the entries that are present"
        );
    }

    let mut out = BTreeSet::new();
    for chunk in body.chunks_exact(4).take(count) {
        let ordinal = u32::from_le_bytes(chunk.try_into().expect("4 bytes"));
        if (ordinal as usize) < dataset_count {
            out.insert(ordinal);
        } else {
            tracing::warn!(
                ordinal,
                dataset_count,
                "deletion mask names a dataset this collection does not have; ignoring it"
            );
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn set(items: &[u32]) -> BTreeSet<u32> {
        items.iter().copied().collect()
    }

    #[test]
    fn mask_roundtrips() {
        let m = set(&[0, 3, 7]);
        assert_eq!(decode(&encode(&m), 8).unwrap(), m);
    }

    #[test]
    fn ordinals_are_written_in_order() {
        // The set is ordered, so insertion order cannot leak into the file.
        let a = encode(&set(&[7, 0, 3]));
        let b = encode(&set(&[0, 3, 7]));
        assert_eq!(a, b);
    }

    #[test]
    fn empty_mask_roundtrips() {
        assert!(decode(&encode(&set(&[])), 4).unwrap().is_empty());
    }

    #[test]
    fn empty_file_means_nothing_deleted() {
        assert!(decode(&[], 4).unwrap().is_empty());
    }

    #[test]
    fn ordinals_past_the_end_are_ignored() {
        let m = set(&[0, 99]);
        assert_eq!(decode(&encode(&m), 4).unwrap(), set(&[0]));
    }

    #[test]
    fn a_truncated_body_keeps_what_is_there() {
        let mut bytes = encode(&set(&[1, 2, 3]));
        bytes.truncate(bytes.len() - 4);
        assert_eq!(decode(&bytes, 8).unwrap(), set(&[1, 2]));
    }

    #[test]
    fn foreign_file_is_an_error() {
        assert!(matches!(
            decode(b"not a mask at all", 4),
            Err(Error::CorruptMask(_))
        ));
    }

    #[test]
    fn unknown_version_is_an_error() {
        let mut bytes = encode(&set(&[1]));
        bytes[4..8].copy_from_slice(&2u32.to_le_bytes());
        assert!(matches!(decode(&bytes, 4), Err(Error::CorruptMask(_))));
    }
}
