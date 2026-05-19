use array_format::DType;
use serde::{Deserialize, Serialize};

use crate::config::Codec;

/// A per-dataset attribute value stored in `_meta.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "value")]
pub enum Attr {
    Bool(bool),
    Int8(i8),
    Int16(i16),
    Int32(i32),
    Int64(i64),
    UInt8(u8),
    UInt16(u16),
    UInt32(u32),
    UInt64(u64),
    Float32(f32),
    Float64(f64),
    String(String),
}

/// Schema for a single named array within a dataset.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ArraySchema {
    #[serde(with = "dtype_serde")]
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub chunk_shape: Vec<usize>,
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
        FixedSizeList { child: Box<DTypeRepr>, size: u32 },
        List { child: Box<DTypeRepr> },
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
                DType::FixedSizeList { child, size } => {
                    Self::FixedSizeList { child: Box::new((*child).into()), size }
                }
                DType::List { child } => Self::List { child: Box::new((*child).into()) },
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
                DTypeRepr::FixedSizeList { child, size } => {
                    Self::FixedSizeList { child: Box::new((*child).into()), size }
                }
                DTypeRepr::List { child } => Self::List { child: Box::new((*child).into()) },
            }
        }
    }
}
