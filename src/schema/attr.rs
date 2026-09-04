//! [`Attr`], atlas's public attribute value.

use array_format::{AttributeValue, DType, EcoString};

/// A typed attribute value.
///
/// An attribute annotates a dataset or one of its arrays. Every variant
/// carries its own tag to disk and back, so a value says what it is and the
/// schema settles nothing. A write and a read give the same variant.
///
/// There is no timestamp variant, because `array-format` has no timestamp
/// attribute: one would have to store as an `i64` and could not come back.
/// Store the number as [`Int64`](Self::Int64) and name the unit in a second
/// attribute. An array element type still has
/// [`DType::TimestampNs`](array_format::DType::TimestampNs); this is about
/// attributes alone. A string that looks like a date stays a string.
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
    /// The [`DType`] this value occupies. A scalar maps to the matching scalar
    /// dtype. A list maps to a [`DType::List`] over the element dtype.
    pub fn dtype(&self) -> DType {
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

    /// The form a segment stores.
    ///
    /// One variant of `AttributeValue` per variant here, so the mapping is
    /// total and loses nothing. [`from_stored`](Self::from_stored) inverts
    /// it.
    pub(crate) fn into_stored(self) -> AttributeValue {
        match self {
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
            Attr::String(v) => AttributeValue::String(v.into()),
            Attr::Binary(v) => AttributeValue::Binary(v.into()),
            Attr::BoolList(v) => AttributeValue::BoolList(v.into()),
            Attr::Int8List(v) => AttributeValue::Int8List(v.into()),
            Attr::Int16List(v) => AttributeValue::Int16List(v.into()),
            Attr::Int32List(v) => AttributeValue::Int32List(v.into()),
            Attr::Int64List(v) => AttributeValue::Int64List(v.into()),
            Attr::UInt8List(v) => AttributeValue::UInt8List(v.into()),
            Attr::UInt16List(v) => AttributeValue::UInt16List(v.into()),
            Attr::UInt32List(v) => AttributeValue::UInt32List(v.into()),
            Attr::UInt64List(v) => AttributeValue::UInt64List(v.into()),
            Attr::Float32List(v) => AttributeValue::Float32List(v.into()),
            Attr::Float64List(v) => AttributeValue::Float64List(v.into()),
            Attr::StringList(v) => {
                AttributeValue::StringList(v.into_iter().map(EcoString::from).collect())
            }
            Attr::BinaryList(v) => {
                AttributeValue::BinaryList(v.into_iter().map(Box::from).collect())
            }
        }
    }

    /// Rebuilds a value a segment stored.
    ///
    /// The stored value carries its own tag, so this reads no schema. It is
    /// the exact inverse of [`into_stored`](Self::into_stored).
    pub(crate) fn from_stored(value: &AttributeValue) -> Self {
        match value {
            AttributeValue::Bool(v) => Attr::Bool(*v),
            AttributeValue::Int8(v) => Attr::Int8(*v),
            AttributeValue::Int16(v) => Attr::Int16(*v),
            AttributeValue::Int32(v) => Attr::Int32(*v),
            AttributeValue::Int64(v) => Attr::Int64(*v),
            AttributeValue::UInt8(v) => Attr::UInt8(*v),
            AttributeValue::UInt16(v) => Attr::UInt16(*v),
            AttributeValue::UInt32(v) => Attr::UInt32(*v),
            AttributeValue::UInt64(v) => Attr::UInt64(*v),
            AttributeValue::Float32(v) => Attr::Float32(*v),
            AttributeValue::Float64(v) => Attr::Float64(*v),
            AttributeValue::String(v) => Attr::String(v.to_string()),
            AttributeValue::Binary(v) => Attr::Binary(v.to_vec()),
            AttributeValue::BoolList(v) => Attr::BoolList(v.to_vec()),
            AttributeValue::Int8List(v) => Attr::Int8List(v.to_vec()),
            AttributeValue::Int16List(v) => Attr::Int16List(v.to_vec()),
            AttributeValue::Int32List(v) => Attr::Int32List(v.to_vec()),
            AttributeValue::Int64List(v) => Attr::Int64List(v.to_vec()),
            AttributeValue::UInt8List(v) => Attr::UInt8List(v.to_vec()),
            AttributeValue::UInt16List(v) => Attr::UInt16List(v.to_vec()),
            AttributeValue::UInt32List(v) => Attr::UInt32List(v.to_vec()),
            AttributeValue::UInt64List(v) => Attr::UInt64List(v.to_vec()),
            AttributeValue::Float32List(v) => Attr::Float32List(v.to_vec()),
            AttributeValue::Float64List(v) => Attr::Float64List(v.to_vec()),
            AttributeValue::StringList(v) => {
                Attr::StringList(v.iter().map(|s| s.to_string()).collect())
            }
            AttributeValue::BinaryList(v) => {
                Attr::BinaryList(v.iter().map(|b| b.to_vec()).collect())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_value_survives_a_round_trip_through_a_segment() {
        // Every variant. The list is exhaustive on purpose: a new variant
        // that `into_stored` cannot invert must fail here.
        let cases = vec![
            Attr::Bool(true),
            Attr::Int8(-3),
            Attr::Int16(-300),
            Attr::Int32(-70_000),
            Attr::Int64(-5_000_000_000),
            Attr::UInt8(200),
            Attr::UInt16(60_000),
            Attr::UInt32(4_000_000_000),
            Attr::UInt64(18_000_000_000_000_000_000),
            Attr::Float32(2.5),
            Attr::Float64(-0.125),
            Attr::String("hello".into()),
            Attr::Binary(vec![0xde, 0xad]),
            Attr::BoolList(vec![true, false]),
            Attr::Int8List(vec![1, -1]),
            Attr::Int16List(vec![2]),
            Attr::Int32List(vec![1, 2, 3]),
            Attr::Int64List(vec![4]),
            Attr::UInt8List(vec![5]),
            Attr::UInt16List(vec![6]),
            Attr::UInt32List(vec![7]),
            Attr::UInt64List(vec![8]),
            Attr::Float32List(vec![1.5]),
            Attr::Float64List(vec![0.0, 1.5]),
            Attr::StringList(vec!["a".into(), "b".into()]),
            Attr::BinaryList(vec![vec![1], vec![2]]),
        ];
        for v in cases {
            let stored = v.clone().into_stored();
            assert_eq!(Attr::from_stored(&stored), v, "{v:?}");
        }
    }

    #[test]
    fn a_date_shaped_string_stays_a_string() {
        let v = Attr::String("2023-11-14T22:13:20Z".into());
        assert_eq!(Attr::from_stored(&v.clone().into_stored()), v);
    }

    #[test]
    fn scalars_and_lists_report_their_dtype() {
        assert_eq!(Attr::Int32(1).dtype(), DType::Int32);
        assert_eq!(Attr::Int64(0).dtype(), DType::Int64);
        assert_eq!(
            Attr::StringList(vec![]).dtype(),
            DType::List {
                child: Box::new(DType::String)
            }
        );
    }
}
