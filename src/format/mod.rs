//! On-disk framing for an atlas collection.
//!
//! A collection is a store prefix that holds two objects:
//!
//! ```text
//! my_collection/
//! ├── data.atlas      <- write-once container: segments + footer + trailer
//! └── deleted.mask    <- optional: ordinals of deleted datasets
//! ```
//!
//! # Container layout
//!
//! ```text
//! offset 0    b"ATLS"                     4 B   leading magic
//! offset 4    format_version u32 LE = 1    4 B
//! offset 8    segment[0]                         one array-format file per dataset
//!             segment[1] ...                     back to back, no padding
//!             footer_bytes                       zstd(msgpack(CollectionFooter))
//! end - 16    footer_size u64 LE           8 B  ┐
//! end - 8     format_version u32 LE = 1    4 B  ├ trailer
//! end - 4     b"ATLS"                      4 B  ┘
//! ```
//!
//! Each segment is a complete `array-format` file that describes itself. Cut
//! one out with `dd`, and it opens on its own. The container footer records the
//! byte range of each segment. A reader therefore finds a dataset with no scan.
//!
//! The container never changes after [`AtlasWriter::finish`](crate::AtlasWriter::finish).
//! A delete writes the mask sidecar instead. See [`mask`].

pub(crate) mod footer;
pub(crate) mod mask;
pub(crate) mod segment_store;

/// Magic at the first four bytes of the container, and at its last four.
pub(crate) const MAGIC: [u8; 4] = *b"ATLS";

/// Version of the container framing and of the footer schema. The two move
/// together. A change to one footer field is a format change.
pub(crate) const FORMAT_VERSION: u32 = 1;

/// Bytes before the first segment: magic + version.
pub(crate) const HEADER_SIZE: u64 = 8;

/// Bytes after the footer: footer size + version + magic.
pub(crate) const TRAILER_SIZE: u64 = 16;

/// `array-format` footer version inside every segment. The container footer
/// records it, so a future reader can dispatch on it.
pub(crate) const SEGMENT_FORMAT: u32 = 5;

/// Container object name under the collection prefix.
pub(crate) const DATA_FILE: &str = "data.atlas";

/// Deletion-mask object name under the collection prefix.
pub(crate) const MASK_FILE: &str = "deleted.mask";

/// How many trailing bytes an open reads on speculation. The size suits a
/// small or medium collection. Its footer then arrives with the trailer, in
/// one request.
pub(crate) const TAIL_PROBE_SIZE: u64 = 64 * 1024;

/// Joins a collection prefix and an object name. An empty prefix gives the bare
/// name. A local-directory store uses an empty prefix.
pub(crate) fn child(prefix: &object_store::path::Path, name: &str) -> object_store::path::Path {
    if prefix.as_ref().is_empty() {
        object_store::path::Path::from(name)
    } else {
        prefix.clone().join(name)
    }
}

/// Encodes the 8-byte container header.
pub(crate) fn encode_header() -> [u8; 8] {
    let mut out = [0u8; 8];
    out[..4].copy_from_slice(&MAGIC);
    out[4..].copy_from_slice(&FORMAT_VERSION.to_le_bytes());
    out
}

/// Encodes the 16-byte container trailer for a footer of `footer_size` bytes.
pub(crate) fn encode_trailer(footer_size: u64) -> [u8; 16] {
    let mut out = [0u8; 16];
    out[..8].copy_from_slice(&footer_size.to_le_bytes());
    out[8..12].copy_from_slice(&FORMAT_VERSION.to_le_bytes());
    out[12..].copy_from_slice(&MAGIC);
    out
}

/// Reads the trailer from the last [`TRAILER_SIZE`] bytes of a container.
///
/// Returns the footer size. The magic rejects a foreign file. The version
/// rejects a container from another format version.
pub(crate) fn decode_trailer(bytes: &[u8]) -> crate::Result<u64> {
    if bytes.len() < TRAILER_SIZE as usize {
        return Err(crate::Error::NotAnAtlasCollection {
            hint: format!(
                "file is {} bytes, shorter than the {TRAILER_SIZE}-byte trailer",
                bytes.len()
            ),
        });
    }
    let t = &bytes[bytes.len() - TRAILER_SIZE as usize..];
    if t[12..] != MAGIC {
        return Err(crate::Error::NotAnAtlasCollection {
            hint: "the file does not end with the ATLS magic".to_string(),
        });
    }
    let version = u32::from_le_bytes(t[8..12].try_into().expect("4 bytes"));
    if version != FORMAT_VERSION {
        return Err(crate::Error::UnsupportedVersion {
            found: version,
            expected: FORMAT_VERSION,
        });
    }
    Ok(u64::from_le_bytes(t[..8].try_into().expect("8 bytes")))
}

/// Checks the 8-byte container header. One open calls this once. The bytes are
/// already in hand when the whole container fits in the tail probe. Otherwise
/// they cost one more small read.
pub(crate) fn check_header(bytes: &[u8]) -> crate::Result<()> {
    if bytes.len() < HEADER_SIZE as usize || bytes[..4] != MAGIC {
        return Err(crate::Error::NotAnAtlasCollection {
            hint: "the file does not start with the ATLS magic".to_string(),
        });
    }
    let version = u32::from_le_bytes(bytes[4..8].try_into().expect("4 bytes"));
    if version != FORMAT_VERSION {
        return Err(crate::Error::UnsupportedVersion {
            found: version,
            expected: FORMAT_VERSION,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Error;

    #[test]
    fn header_roundtrip() {
        let h = encode_header();
        assert_eq!(&h[..4], b"ATLS");
        assert!(check_header(&h).is_ok());
    }

    #[test]
    fn trailer_roundtrip() {
        let t = encode_trailer(1234);
        assert_eq!(t.len(), TRAILER_SIZE as usize);
        assert_eq!(&t[12..], b"ATLS");
        assert_eq!(decode_trailer(&t).unwrap(), 1234);
    }

    #[test]
    fn trailer_is_found_at_the_end_of_a_larger_tail() {
        let mut buf = vec![0xAB; 500];
        buf.extend_from_slice(&encode_trailer(7));
        assert_eq!(decode_trailer(&buf).unwrap(), 7);
    }

    #[test]
    fn foreign_file_is_rejected_on_magic() {
        let buf = vec![0u8; 64];
        assert!(matches!(
            decode_trailer(&buf),
            Err(Error::NotAnAtlasCollection { .. })
        ));
        assert!(matches!(
            check_header(b"PK\x03\x04....."),
            Err(Error::NotAnAtlasCollection { .. })
        ));
    }

    #[test]
    fn truncated_file_is_rejected() {
        assert!(matches!(
            decode_trailer(&[0u8; 4]),
            Err(Error::NotAnAtlasCollection { .. })
        ));
        assert!(matches!(
            check_header(b"AT"),
            Err(Error::NotAnAtlasCollection { .. })
        ));
    }

    #[test]
    fn wrong_version_is_reported_as_unsupported() {
        let mut t = encode_trailer(0);
        t[8..12].copy_from_slice(&99u32.to_le_bytes());
        assert!(matches!(
            decode_trailer(&t),
            Err(Error::UnsupportedVersion {
                found: 99,
                expected: 1
            })
        ));

        let mut h = encode_header();
        h[4..8].copy_from_slice(&99u32.to_le_bytes());
        assert!(matches!(
            check_header(&h),
            Err(Error::UnsupportedVersion { found: 99, .. })
        ));
    }
}
