//! [`Attr`], atlas's public attribute value.

use array_format::DType;

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
}

#[cfg(test)]
mod tests {
    use super::*;

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
