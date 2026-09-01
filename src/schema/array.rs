//! The schema of a single named array, and its fill value.

use array_format::{DType, FillValue};
use serde::{Deserialize, Serialize};

/// Schema for one named array within a dataset.
///
/// This describes the array, not where its bytes are. Chunk addresses live in
/// the dataset's segment. A reader answers `array_meta` from here, and never
/// opens the segment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ArraySchema {
    /// Element type.
    #[serde(with = "super::dtype::dtype_serde")]
    pub dtype: DType,
    /// Logical shape, one entry per axis.
    pub shape: Vec<usize>,
    /// On-disk chunk shape, same rank as `shape`. Equal to `shape` when the
    /// array is stored as one chunk.
    pub chunk_shape: Vec<usize>,
    /// Named dimensions, one per axis, in the order of `shape`.
    pub dimension_names: Vec<String>,
    /// Value returned for elements that were never written.
    pub fill_value: Option<FillValueS>,
}

/// Serde mirror of [`array_format::FillValue`], which implements rkyv only.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FillValueS {
    /// Boolean fill value.
    Bool(bool),
    /// Signed integer fill value.
    Int(i64),
    /// Unsigned integer fill value.
    UInt(u64),
    /// Floating-point fill value.
    Float(f64),
    /// String fill value.
    String(String),
    /// Nanoseconds since the Unix epoch, for `TimestampNs` arrays.
    TimestampNs(i64),
}

/// Floats compare by bit pattern, as in [`array_format::FillValue`]. One NaN
/// fill therefore equals another. Otherwise two datasets with the same
/// NaN-filled array never share an interned schema.
impl PartialEq for FillValueS {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Bool(a), Self::Bool(b)) => a == b,
            (Self::Int(a), Self::Int(b)) => a == b,
            (Self::UInt(a), Self::UInt(b)) => a == b,
            (Self::Float(a), Self::Float(b)) => a.to_bits() == b.to_bits(),
            (Self::String(a), Self::String(b)) => a == b,
            (Self::TimestampNs(a), Self::TimestampNs(b)) => a == b,
            _ => false,
        }
    }
}

impl From<FillValue> for FillValueS {
    fn from(v: FillValue) -> Self {
        match v {
            FillValue::Bool(v) => Self::Bool(v),
            FillValue::Int(v) => Self::Int(v),
            FillValue::UInt(v) => Self::UInt(v),
            FillValue::Float(v) => Self::Float(v),
            FillValue::String(v) => Self::String(v),
            FillValue::TimestampNs(v) => Self::TimestampNs(v),
        }
    }
}

impl From<FillValueS> for FillValue {
    fn from(v: FillValueS) -> Self {
        match v {
            FillValueS::Bool(v) => Self::Bool(v),
            FillValueS::Int(v) => Self::Int(v),
            FillValueS::UInt(v) => Self::UInt(v),
            FillValueS::Float(v) => Self::Float(v),
            FillValueS::String(v) => Self::String(v),
            FillValueS::TimestampNs(v) => Self::TimestampNs(v),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn array_schema_roundtrip_via_msgpack() {
        let schema = ArraySchema {
            dtype: DType::Float64,
            shape: vec![10, 20],
            chunk_shape: vec![5, 5],
            dimension_names: vec!["lat".into(), "lon".into()],
            fill_value: Some(FillValueS::Float(f64::NAN)),
        };
        let packed = rmp_serde::to_vec(&schema).unwrap();
        let back: ArraySchema = rmp_serde::from_slice(&packed).unwrap();
        // NaN != NaN under f64, so compare the parts that settle it.
        assert_eq!(back.dtype, schema.dtype);
        assert_eq!(back.shape, schema.shape);
        assert_eq!(back.chunk_shape, schema.chunk_shape);
        assert_eq!(back.dimension_names, schema.dimension_names);
        assert!(matches!(back.fill_value, Some(FillValueS::Float(f)) if f.is_nan()));
    }

    #[test]
    fn nested_dtypes_survive_the_wire() {
        let schema = ArraySchema {
            dtype: DType::List {
                child: Box::new(DType::FixedSizeList {
                    child: Box::new(DType::Int16),
                    size: 3,
                }),
            },
            shape: vec![2],
            chunk_shape: vec![2],
            dimension_names: vec!["x".into()],
            fill_value: None,
        };
        let packed = rmp_serde::to_vec(&schema).unwrap();
        let back: ArraySchema = rmp_serde::from_slice(&packed).unwrap();
        assert_eq!(back, schema);
    }
}
