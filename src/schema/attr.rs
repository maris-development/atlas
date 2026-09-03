//! [`Attr`], atlas's public attribute value.

use array_format::{AttributeValue, DType};

/// A typed attribute value.
///
/// An attribute annotates a dataset or one of its arrays. Values live in the
/// container footer. One range read therefore lists the datasets and gives
/// their attributes.
///
/// A timestamp has its own variant and its own wire tag. A string that looks
/// like a date stays a string.
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
    /// Nanosecond-precision UTC timestamp.
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

    /// The form a segment stores.
    ///
    /// `array-format` has no timestamp attribute, so a timestamp goes in as
    /// its `i64`. [`from_stored`](Self::from_stored) reads the tag back from
    /// the schema, which records the declared type of every key.
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
            Attr::String(v) => AttributeValue::String(v),
            Attr::Binary(v) => AttributeValue::Binary(v),
            // The one lossy step. The schema carries the tag.
            Attr::TimestampNanoseconds(v) => AttributeValue::Int64(v),
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

    /// Rebuilds a value a segment stored, given the type the schema declared
    /// for that key.
    ///
    /// `declared` settles the one case the segment cannot: an `i64` is a
    /// timestamp when the schema says [`DType::TimestampNs`], and a plain
    /// integer otherwise. Every other variant carries its own tag.
    pub(crate) fn from_stored(value: &AttributeValue, declared: &DType) -> Self {
        match value {
            AttributeValue::Bool(v) => Attr::Bool(*v),
            AttributeValue::Int8(v) => Attr::Int8(*v),
            AttributeValue::Int16(v) => Attr::Int16(*v),
            AttributeValue::Int32(v) => Attr::Int32(*v),
            AttributeValue::Int64(v) if *declared == DType::TimestampNs => {
                Attr::TimestampNanoseconds(*v)
            }
            AttributeValue::Int64(v) => Attr::Int64(*v),
            AttributeValue::UInt8(v) => Attr::UInt8(*v),
            AttributeValue::UInt16(v) => Attr::UInt16(*v),
            AttributeValue::UInt32(v) => Attr::UInt32(*v),
            AttributeValue::UInt64(v) => Attr::UInt64(*v),
            AttributeValue::Float32(v) => Attr::Float32(*v),
            AttributeValue::Float64(v) => Attr::Float64(*v),
            AttributeValue::String(v) => Attr::String(v.clone()),
            AttributeValue::Binary(v) => Attr::Binary(v.clone()),
            AttributeValue::BoolList(v) => Attr::BoolList(v.clone()),
            AttributeValue::Int8List(v) => Attr::Int8List(v.clone()),
            AttributeValue::Int16List(v) => Attr::Int16List(v.clone()),
            AttributeValue::Int32List(v) => Attr::Int32List(v.clone()),
            AttributeValue::Int64List(v) => Attr::Int64List(v.clone()),
            AttributeValue::UInt8List(v) => Attr::UInt8List(v.clone()),
            AttributeValue::UInt16List(v) => Attr::UInt16List(v.clone()),
            AttributeValue::UInt32List(v) => Attr::UInt32List(v.clone()),
            AttributeValue::UInt64List(v) => Attr::UInt64List(v.clone()),
            AttributeValue::Float32List(v) => Attr::Float32List(v.clone()),
            AttributeValue::Float64List(v) => Attr::Float64List(v.clone()),
            AttributeValue::StringList(v) => Attr::StringList(v.clone()),
            AttributeValue::BinaryList(v) => Attr::BinaryList(v.clone()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_value_survives_a_round_trip_through_a_segment() {
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
            Attr::TimestampNanoseconds(1_700_000_000_000_000_000),
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
            let declared = v.dtype();
            let stored = v.clone().into_stored();
            assert_eq!(Attr::from_stored(&stored, &declared), v, "{v:?}");
        }
    }

    #[test]
    fn a_timestamp_and_an_integer_share_a_stored_form_and_stay_distinct() {
        // `array-format` has no timestamp attribute, so both store as i64.
        // Only the declared type tells them apart.
        let when = Attr::TimestampNanoseconds(1_700_000_000_000_000_000);
        let count = Attr::Int64(1_700_000_000_000_000_000);
        assert_eq!(when.clone().into_stored(), count.clone().into_stored());

        assert_eq!(
            Attr::from_stored(&when.clone().into_stored(), &DType::TimestampNs),
            when
        );
        assert_eq!(
            Attr::from_stored(&count.clone().into_stored(), &DType::Int64),
            count
        );
    }

    #[test]
    fn a_date_shaped_string_stays_a_string() {
        let v = Attr::String("2023-11-14T22:13:20Z".into());
        let declared = v.dtype();
        assert_eq!(Attr::from_stored(&v.clone().into_stored(), &declared), v);
    }

    #[test]
    fn scalars_and_lists_report_their_dtype() {
        assert_eq!(Attr::Int32(1).dtype(), DType::Int32);
        assert_eq!(Attr::TimestampNanoseconds(0).dtype(), DType::TimestampNs);
        assert_eq!(
            Attr::StringList(vec![]).dtype(),
            DType::List {
                child: Box::new(DType::String)
            }
        );
    }
}
