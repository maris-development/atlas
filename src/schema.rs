use array_format::{DType, FillValue};
use serde::{Deserialize, Serialize};

use crate::config::Codec;

/// A per-dataset attribute value stored in `atlas.json`.
///
/// Atlas supports five attribute types — booleans, 64-bit signed integers,
/// 64-bit floats, UTF-8 strings, and nanosecond-precision timestamps. The
/// JSON form is untagged: each variant serializes as its natural JSON value
/// (`true`, `42`, `1.5`, `"hello"`, or an RFC 3339 string for the timestamp).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Attr {
    /// Boolean attribute. Listed first because `#[serde(untagged)]` tries
    /// variants in order and `bool` only matches JSON `true`/`false`.
    Bool(bool),
    /// Nanosecond-precision UTC timestamp. Stored as an RFC 3339 string;
    /// the deserializer parses strictly, so non-timestamp strings fall
    /// through to the `String` variant.
    #[serde(with = "timestamp_ns_serde")]
    TimestampNanoseconds(i64),
    /// UTF-8 string attribute.
    String(String),
    /// 64-bit signed integer attribute (JSON numbers without a decimal point).
    Int64(i64),
    /// 64-bit float attribute (JSON numbers with a decimal point or exponent).
    Float64(f64),
}

mod timestamp_ns_serde {
    use chrono::{DateTime, SecondsFormat, Utc};
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(nanos: &i64, s: S) -> Result<S::Ok, S::Error> {
        let dt = DateTime::<Utc>::from_timestamp_nanos(*nanos);
        // AutoSi: shortest faithful repr (drops trailing-zero subsecond digits).
        s.serialize_str(&dt.to_rfc3339_opts(SecondsFormat::AutoSi, true))
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<i64, D::Error> {
        let s = <&str>::deserialize(d)?;
        let dt = DateTime::parse_from_rfc3339(s)
            .map_err(serde::de::Error::custom)?
            .with_timezone(&Utc);
        dt.timestamp_nanos_opt().ok_or_else(|| {
            serde::de::Error::custom("timestamp out of nanosecond range (1677-09-21 .. 2262-04-11)")
        })
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
