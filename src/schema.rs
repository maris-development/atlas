use array_format::{AttributeValue, DType};
use serde::{Deserialize, Serialize};

use crate::config::Codec;

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
            Attr::TimestampNanoseconds(v) => {
                AttributeValue::String(timestamp_ns_to_rfc3339(v))
            }
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

/// The [`DType`] an [`Attr`] value occupies. Scalars map to the matching
/// scalar dtype (timestamps to [`DType::TimestampNs`]); lists map to a
/// [`DType::List`] over the element dtype. Used to type attributes in the
/// merged schema and to check type alignment across datasets.
pub(crate) fn attr_dtype(attr: &Attr) -> DType {
    match attr {
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

fn list(child: DType) -> DType {
    DType::List {
        child: Box::new(child),
    }
}

/// Merge two dtypes into one that describes both, or `None` if they don't
/// align. Widening is allowed only:
/// - within numeric types (int/uint/float) — promotes to a type that holds
///   both (mixed sign widens to a larger signed type; any float or ≥32-bit
///   integer with a float promotes to `Float64`);
/// - between `String` and `TimestampNs` — merges to `String` (timestamps are
///   representable as RFC 3339 strings);
/// - element-wise between two `List`s.
///
/// Anything else (e.g. an integer array and a string array under the same
/// name) returns `None`, signalling a collision the caller must reject.
pub(crate) fn widen_dtype(a: &DType, b: &DType) -> Option<DType> {
    if a == b {
        return Some(a.clone());
    }
    if let (Some(na), Some(nb)) = (as_numeric(a), as_numeric(b)) {
        return Some(from_numeric(widen_numeric(na, nb)));
    }
    match (a, b) {
        (DType::String, DType::TimestampNs) | (DType::TimestampNs, DType::String) => {
            Some(DType::String)
        }
        (DType::List { child: ca }, DType::List { child: cb }) => {
            widen_dtype(ca, cb).map(|c| DType::List { child: Box::new(c) })
        }
        _ => None,
    }
}

/// Numeric-type classification for widening: kind + bit width.
#[derive(Clone, Copy)]
enum Numeric {
    UInt(u32),
    Int(u32),
    Float(u32),
}

fn as_numeric(d: &DType) -> Option<Numeric> {
    Some(match d {
        DType::UInt8 => Numeric::UInt(8),
        DType::UInt16 => Numeric::UInt(16),
        DType::UInt32 => Numeric::UInt(32),
        DType::UInt64 => Numeric::UInt(64),
        DType::Int8 => Numeric::Int(8),
        DType::Int16 => Numeric::Int(16),
        DType::Int32 => Numeric::Int(32),
        DType::Int64 => Numeric::Int(64),
        DType::Float32 => Numeric::Float(32),
        DType::Float64 => Numeric::Float(64),
        _ => return None,
    })
}

fn from_numeric(n: Numeric) -> DType {
    match n {
        Numeric::UInt(8) => DType::UInt8,
        Numeric::UInt(16) => DType::UInt16,
        Numeric::UInt(32) => DType::UInt32,
        Numeric::UInt(_) => DType::UInt64,
        Numeric::Int(8) => DType::Int8,
        Numeric::Int(16) => DType::Int16,
        Numeric::Int(32) => DType::Int32,
        Numeric::Int(_) => DType::Int64,
        Numeric::Float(32) => DType::Float32,
        Numeric::Float(_) => DType::Float64,
    }
}

fn widen_numeric(a: Numeric, b: Numeric) -> Numeric {
    use Numeric::*;
    // Bit width of an operand regardless of kind.
    fn width(n: Numeric) -> u32 {
        match n {
            UInt(w) | Int(w) | Float(w) => w,
        }
    }
    let is_float = |n: Numeric| matches!(n, Float(_));
    if is_float(a) || is_float(b) {
        // f32 has a 24-bit mantissa: any ≥32-bit integer or an f64 forces f64.
        let needs_64 = width(a) >= 32 && !matches!(a, Float(32))
            || width(b) >= 32 && !matches!(b, Float(32))
            || matches!(a, Float(64))
            || matches!(b, Float(64));
        return Float(if needs_64 { 64 } else { 32 });
    }
    match (a, b) {
        (Int(x), Int(y)) => Int(x.max(y)),
        (UInt(x), UInt(y)) => UInt(x.max(y)),
        // Mixed sign → signed type large enough to hold the unsigned range.
        (Int(iw), UInt(uw)) | (UInt(uw), Int(iw)) => Int(iw.max((uw * 2).min(64))),
        _ => unreachable!("float handled above"),
    }
}

/// Serde wrapper for [`DType`] used as a map value in the schema
/// (reusing [`dtype_serde`]'s tagged representation).
#[derive(Debug, Clone, PartialEq)]
pub struct DTypeS(pub DType);

impl Serialize for DTypeS {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        dtype_serde::serialize(&self.0, s)
    }
}

impl<'de> Deserialize<'de> for DTypeS {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        dtype_serde::deserialize(d).map(DTypeS)
    }
}

/// Schema for a single named array within a dataset.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ArraySchema {
    /// Element type of this array.
    #[serde(with = "dtype_serde")]
    pub dtype: DType,
    /// Logical shape, one entry per axis.
    pub shape: Vec<usize>,
    /// On-disk chunk shape, same rank as `shape`. Equal to `shape` for
    /// single-chunk arrays.
    pub chunk_shape: Vec<usize>,
    /// Named dimensions, one per axis. Order matches `shape`.
    pub dimension_names: Vec<String>,
    /// Codec used when this array was first created; controls how new blocks are written.
    pub codec: Codec,
}

/// Serde helpers for [`DType`] (which uses rkyv, not serde).
pub(crate) mod dtype_serde {
    use array_format::DType;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(dtype: &DType, s: S) -> Result<S::Ok, S::Error> {
        DTypeRepr::from(dtype.clone()).serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<DType, D::Error> {
        DTypeRepr::deserialize(d).map(DType::from)
    }

    #[derive(Serialize, Deserialize)]
    #[serde(tag = "type", content = "args", rename_all = "snake_case")]
    enum DTypeRepr {
        Bool,
        Int8,
        Int16,
        Int32,
        Int64,
        UInt8,
        UInt16,
        UInt32,
        UInt64,
        Float32,
        Float64,
        String,
        Binary,
        #[serde(rename = "timestamp_nanoseconds")]
        TimestampNs,
        FixedSizeList {
            child: Box<DTypeRepr>,
            size: u32,
        },
        List {
            child: Box<DTypeRepr>,
        },
    }

    impl From<DType> for DTypeRepr {
        fn from(d: DType) -> Self {
            match d {
                DType::Bool => Self::Bool,
                DType::Int8 => Self::Int8,
                DType::Int16 => Self::Int16,
                DType::Int32 => Self::Int32,
                DType::Int64 => Self::Int64,
                DType::UInt8 => Self::UInt8,
                DType::UInt16 => Self::UInt16,
                DType::UInt32 => Self::UInt32,
                DType::UInt64 => Self::UInt64,
                DType::Float32 => Self::Float32,
                DType::Float64 => Self::Float64,
                DType::String => Self::String,
                DType::Binary => Self::Binary,
                DType::TimestampNs => Self::TimestampNs,
                DType::FixedSizeList { child, size } => Self::FixedSizeList {
                    child: Box::new((*child).into()),
                    size,
                },
                DType::List { child } => Self::List {
                    child: Box::new((*child).into()),
                },
            }
        }
    }

    impl From<DTypeRepr> for DType {
        fn from(d: DTypeRepr) -> Self {
            match d {
                DTypeRepr::Bool => Self::Bool,
                DTypeRepr::Int8 => Self::Int8,
                DTypeRepr::Int16 => Self::Int16,
                DTypeRepr::Int32 => Self::Int32,
                DTypeRepr::Int64 => Self::Int64,
                DTypeRepr::UInt8 => Self::UInt8,
                DTypeRepr::UInt16 => Self::UInt16,
                DTypeRepr::UInt32 => Self::UInt32,
                DTypeRepr::UInt64 => Self::UInt64,
                DTypeRepr::Float32 => Self::Float32,
                DTypeRepr::Float64 => Self::Float64,
                DTypeRepr::String => Self::String,
                DTypeRepr::Binary => Self::Binary,
                DTypeRepr::TimestampNs => Self::TimestampNs,
                DTypeRepr::FixedSizeList { child, size } => Self::FixedSizeList {
                    child: Box::new((*child).into()),
                    size,
                },
                DTypeRepr::List { child } => Self::List {
                    child: Box::new((*child).into()),
                },
            }
        }
    }
}
