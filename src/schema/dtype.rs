//! Serde support for [`DType`], which comes from `array_format` and implements
//! rkyv rather than serde.

/// Serde helpers for [`DType`](array_format::DType), usable via
/// `#[serde(with = "dtype_serde")]`.
///
/// The representation is an externally tagged enum: in compact MessagePack a
/// scalar dtype costs a variant index, and nested list types recurse.
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
}

#[cfg(test)]
mod tests {
    use array_format::DType;
    use serde::{Deserialize, Serialize};

    #[derive(Serialize, Deserialize, PartialEq, Debug)]
    struct Holder(#[serde(with = "super::dtype_serde")] DType);

    #[test]
    fn every_dtype_roundtrips_through_msgpack() {
        let cases = vec![
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
        ];
        for d in cases {
            let packed = rmp_serde::to_vec(&Holder(d.clone())).unwrap();
            let back: Holder = rmp_serde::from_slice(&packed).unwrap();
            assert_eq!(back.0, d);
        }
    }
}
