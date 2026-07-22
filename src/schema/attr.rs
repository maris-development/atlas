//! [`Attr`], atlas's public attribute value, and its mapping to storage.

use array_format::{AttributeValue, DType};

/// A typed attribute value.
///
/// Attribute **values** live in the per-array `.af` files (as
/// [`array_format::AttributeValue`]); this enum is atlas's public mirror of
/// that type plus a dedicated nanosecond-precision timestamp variant. Values
/// are never serialized into `atlas.json` — only their *keys* are (as part of
/// the schema namespace).
///
/// [`Attr`] mirrors every [`AttributeValue`] case and adds
/// [`TimestampNanoseconds`](Attr::TimestampNanoseconds). Because
/// `AttributeValue` has no timestamp case, timestamps round-trip through the
/// `.af` file as an RFC 3339 [`String`](AttributeValue::String); on read, a
/// string that parses as RFC 3339 is restored to a timestamp (matching the
/// historical `atlas.json` behaviour), and any other string stays a string.
#[derive(Debug, Clone, PartialEq)]
pub enum Attr {
    /// Boolean.
    Bool(bool),
    /// Signed 8-bit integer.
    Int8(i8),
    /// Signed 16-bit integer.
    Int16(i16),
    /// Signed 32-bit integer.
    Int32(i32),
    /// Signed 64-bit integer.
    Int64(i64),
    /// Unsigned 8-bit integer.
    UInt8(u8),
    /// Unsigned 16-bit integer.
    UInt16(u16),
    /// Unsigned 32-bit integer.
    UInt32(u32),
    /// Unsigned 64-bit integer.
    UInt64(u64),
    /// 32-bit float.
    Float32(f32),
    /// 64-bit float.
    Float64(f64),
    /// UTF-8 string.
    String(String),
    /// Variable-length binary.
    Binary(Vec<u8>),
    /// Nanosecond-precision UTC timestamp. Stored in the `.af` file as an
    /// RFC 3339 string (`AttributeValue` has no timestamp case).
    TimestampNanoseconds(i64),
    /// List of booleans.
    BoolList(Vec<bool>),
    /// List of signed 8-bit integers.
    Int8List(Vec<i8>),
    /// List of signed 16-bit integers.
    Int16List(Vec<i16>),
    /// List of signed 32-bit integers.
    Int32List(Vec<i32>),
    /// List of signed 64-bit integers.
    Int64List(Vec<i64>),
    /// List of unsigned 8-bit integers.
    UInt8List(Vec<u8>),
    /// List of unsigned 16-bit integers.
    UInt16List(Vec<u16>),
    /// List of unsigned 32-bit integers.
    UInt32List(Vec<u32>),
    /// List of unsigned 64-bit integers.
    UInt64List(Vec<u64>),
    /// List of 32-bit floats.
    Float32List(Vec<f32>),
    /// List of 64-bit floats.
    Float64List(Vec<f64>),
    /// List of UTF-8 strings.
    StringList(Vec<String>),
    /// List of variable-length binary values.
    BinaryList(Vec<Vec<u8>>),
}

impl Attr {
    /// The [`DType`] this value occupies.
    ///
    /// Scalars map to the matching scalar dtype (timestamps to
    /// [`DType::TimestampNs`]); lists map to a [`DType::List`] over the element
    /// dtype. Used to type attributes in the merged schema and to check type
    /// alignment across datasets.
    pub(crate) fn dtype(&self) -> DType {
        fn list(child: DType) -> DType {
            DType::List {
                child: Box::new(child),
            }
        }
        match self {
            Attr::Bool(_) => DType::Bool,
            Attr::Int8(_) => DType::Int8,
            Attr::Int16(_) => DType::Int16,
            Attr::Int32(_) => DType::Int32,
            Attr::Int64(_) => DType::Int64,
            Attr::UInt8(_) => DType::UInt8,
            Attr::UInt16(_) => DType::UInt16,
            Attr::UInt32(_) => DType::UInt32,
            Attr::UInt64(_) => DType::UInt64,
            Attr::Float32(_) => DType::Float32,
            Attr::Float64(_) => DType::Float64,
            Attr::String(_) => DType::String,
            Attr::Binary(_) => DType::Binary,
            Attr::TimestampNanoseconds(_) => DType::TimestampNs,
            Attr::BoolList(_) => list(DType::Bool),
            Attr::Int8List(_) => list(DType::Int8),
            Attr::Int16List(_) => list(DType::Int16),
            Attr::Int32List(_) => list(DType::Int32),
            Attr::Int64List(_) => list(DType::Int64),
            Attr::UInt8List(_) => list(DType::UInt8),
            Attr::UInt16List(_) => list(DType::UInt16),
            Attr::UInt32List(_) => list(DType::UInt32),
            Attr::UInt64List(_) => list(DType::UInt64),
            Attr::Float32List(_) => list(DType::Float32),
            Attr::Float64List(_) => list(DType::Float64),
            Attr::StringList(_) => list(DType::String),
            Attr::BinaryList(_) => list(DType::Binary),
        }
    }
}

/// Format a nanosecond UTC timestamp as an RFC 3339 string, shortest faithful
/// representation (drops trailing-zero subsecond digits).
pub(crate) fn timestamp_ns_to_rfc3339(nanos: i64) -> String {
    use chrono::{DateTime, SecondsFormat, Utc};
    DateTime::<Utc>::from_timestamp_nanos(nanos).to_rfc3339_opts(SecondsFormat::AutoSi, true)
}

/// Parse an RFC 3339 string into nanoseconds since the Unix epoch, or `None`
/// if it is not a valid RFC 3339 timestamp within the nanosecond range.
pub(crate) fn rfc3339_to_timestamp_ns(s: &str) -> Option<i64> {
    use chrono::{DateTime, Utc};
    DateTime::parse_from_rfc3339(s)
        .ok()?
        .with_timezone(&Utc)
        .timestamp_nanos_opt()
}

impl From<Attr> for AttributeValue {
    fn from(a: Attr) -> Self {
        match a {
            Attr::Bool(v) => AttributeValue::Bool(v),
            Attr::Int8(v) => AttributeValue::Int8(v),
            Attr::Int16(v) => AttributeValue::Int16(v),
            Attr::Int32(v) => AttributeValue::Int32(v),
            Attr::Int64(v) => AttributeValue::Int64(v),
            Attr::UInt8(v) => AttributeValue::UInt8(v),
            Attr::UInt16(v) => AttributeValue::UInt16(v),
            Attr::UInt32(v) => AttributeValue::UInt32(v),
            Attr::UInt64(v) => AttributeValue::UInt64(v),
            Attr::Float32(v) => AttributeValue::Float32(v),
            Attr::Float64(v) => AttributeValue::Float64(v),
            Attr::String(v) => AttributeValue::String(v),
            Attr::Binary(v) => AttributeValue::Binary(v),
            // No native timestamp case: encode as an RFC 3339 string.
            Attr::TimestampNanoseconds(v) => AttributeValue::String(timestamp_ns_to_rfc3339(v)),
            Attr::BoolList(v) => AttributeValue::BoolList(v),
            Attr::Int8List(v) => AttributeValue::Int8List(v),
            Attr::Int16List(v) => AttributeValue::Int16List(v),
            Attr::Int32List(v) => AttributeValue::Int32List(v),
            Attr::Int64List(v) => AttributeValue::Int64List(v),
            Attr::UInt8List(v) => AttributeValue::UInt8List(v),
            Attr::UInt16List(v) => AttributeValue::UInt16List(v),
            Attr::UInt32List(v) => AttributeValue::UInt32List(v),
            Attr::UInt64List(v) => AttributeValue::UInt64List(v),
            Attr::Float32List(v) => AttributeValue::Float32List(v),
            Attr::Float64List(v) => AttributeValue::Float64List(v),
            Attr::StringList(v) => AttributeValue::StringList(v),
            Attr::BinaryList(v) => AttributeValue::BinaryList(v),
        }
    }
}

impl From<AttributeValue> for Attr {
    fn from(v: AttributeValue) -> Self {
        match v {
            AttributeValue::Bool(v) => Attr::Bool(v),
            AttributeValue::Int8(v) => Attr::Int8(v),
            AttributeValue::Int16(v) => Attr::Int16(v),
            AttributeValue::Int32(v) => Attr::Int32(v),
            AttributeValue::Int64(v) => Attr::Int64(v),
            AttributeValue::UInt8(v) => Attr::UInt8(v),
            AttributeValue::UInt16(v) => Attr::UInt16(v),
            AttributeValue::UInt32(v) => Attr::UInt32(v),
            AttributeValue::UInt64(v) => Attr::UInt64(v),
            AttributeValue::Float32(v) => Attr::Float32(v),
            AttributeValue::Float64(v) => Attr::Float64(v),
            // A string that parses as RFC 3339 is restored to a timestamp,
            // matching the historical untagged `atlas.json` behaviour.
            AttributeValue::String(s) => match rfc3339_to_timestamp_ns(&s) {
                Some(ns) => Attr::TimestampNanoseconds(ns),
                None => Attr::String(s),
            },
            AttributeValue::Binary(v) => Attr::Binary(v),
            AttributeValue::BoolList(v) => Attr::BoolList(v),
            AttributeValue::Int8List(v) => Attr::Int8List(v),
            AttributeValue::Int16List(v) => Attr::Int16List(v),
            AttributeValue::Int32List(v) => Attr::Int32List(v),
            AttributeValue::Int64List(v) => Attr::Int64List(v),
            AttributeValue::UInt8List(v) => Attr::UInt8List(v),
            AttributeValue::UInt16List(v) => Attr::UInt16List(v),
            AttributeValue::UInt32List(v) => Attr::UInt32List(v),
            AttributeValue::UInt64List(v) => Attr::UInt64List(v),
            AttributeValue::Float32List(v) => Attr::Float32List(v),
            AttributeValue::Float64List(v) => Attr::Float64List(v),
            AttributeValue::StringList(v) => Attr::StringList(v),
            AttributeValue::BinaryList(v) => Attr::BinaryList(v),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attr_attributevalue_roundtrip() {
        let cases = vec![
            Attr::Bool(true),
            Attr::Int8(-3),
            Attr::Int64(-1_000_000),
            Attr::UInt32(42),
            Attr::Float32(2.5),
            Attr::Float64(2.5),
            Attr::String("hello".into()),
            Attr::Binary(vec![0xde, 0xad]),
            Attr::Int32List(vec![1, 2, 3]),
            Attr::Float64List(vec![0.0, 1.5]),
            Attr::StringList(vec!["a".into(), "b".into()]),
        ];
        for v in cases {
            let av: AttributeValue = v.clone().into();
            let back: Attr = av.into();
            assert_eq!(v, back);
        }
    }

    #[test]
    fn timestamp_attr_roundtrips_through_rfc3339_string() {
        let ts = Attr::TimestampNanoseconds(1_700_000_000_000_000_000);
        let av: AttributeValue = ts.clone().into();
        // Stored as an RFC 3339 string in the .af file.
        assert_eq!(av, AttributeValue::String("2023-11-14T22:13:20Z".into()));
        // A string that parses as RFC 3339 comes back as a timestamp...
        let back: Attr = av.into();
        assert_eq!(back, ts);
        // ...while a non-timestamp string stays a string.
        let plain: Attr = AttributeValue::String("not-a-date".into()).into();
        assert_eq!(plain, Attr::String("not-a-date".into()));
    }
}
