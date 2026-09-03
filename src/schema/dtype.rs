//! Serde support for [`DType`]. That type comes from `array_format`, which
//! implements rkyv and not serde.

use array_format::DType;
use serde::{Deserialize, Serialize};

/// Wire form of [`DType`](array_format::DType).
///
/// An externally tagged enum. In compact MessagePack a scalar dtype costs one
/// variant index. A nested list type recurses.
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
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

/// Serde helpers for a pool of dtypes, usable via
/// `#[serde(with = "dtype_pool_serde")]`.
pub(crate) mod dtype_pool_serde {
    use super::{DType, DTypeRepr};
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(pool: &[DType], s: S) -> Result<S::Ok, S::Error> {
        let wire: Vec<DTypeRepr> = pool.iter().cloned().map(DTypeRepr::from).collect();
        wire.serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<DType>, D::Error> {
        let wire = Vec::<DTypeRepr>::deserialize(d)?;
        Ok(wire.into_iter().map(DType::from).collect())
    }
}

#[cfg(test)]
mod tests {
    use array_format::DType;
    use serde::{Deserialize, Serialize};

    #[derive(Serialize, Deserialize, PartialEq, Debug)]
    struct Pool(#[serde(with = "super::dtype_pool_serde")] Vec<DType>);

    fn every_dtype() -> Vec<DType> {
        vec![
            DType::Bool,
            DType::Int8,
            DType::Int16,
            DType::Int32,
            DType::Int64,
            DType::UInt8,
            DType::UInt16,
            DType::UInt32,
            DType::UInt64,
            DType::Float32,
            DType::Float64,
            DType::String,
            DType::Binary,
            DType::TimestampNs,
            DType::FixedSizeList {
                child: Box::new(DType::Float32),
                size: 4,
            },
            DType::List {
                child: Box::new(DType::String),
            },
        ]
    }

    #[test]
    fn a_whole_pool_roundtrips_in_order() {
        let pool = Pool(every_dtype());
        let packed = rmp_serde::to_vec(&pool).unwrap();
        let back: Pool = rmp_serde::from_slice(&packed).unwrap();
        assert_eq!(back, pool);
    }
}
