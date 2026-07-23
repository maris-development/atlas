//! Type widening across datasets, and JSON-friendly [`DType`] serialization.

use array_format::DType;
use serde::{Deserialize, Serialize};

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
    if let (Some(na), Some(nb)) = (Numeric::of(a), Numeric::of(b)) {
        return Some(na.widen(nb).into());
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

/// A numeric dtype as kind + bit width, the form widening reasons about.
#[derive(Clone, Copy)]
enum Numeric {
    UInt(u32),
    Int(u32),
    Float(u32),
}

impl Numeric {
    /// The numeric classification of `d`, or `None` for non-numeric dtypes.
    fn of(d: &DType) -> Option<Numeric> {
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

    fn width(self) -> u32 {
        match self {
            Numeric::UInt(w) | Numeric::Int(w) | Numeric::Float(w) => w,
        }
    }

    fn is_float(self) -> bool {
        matches!(self, Numeric::Float(_))
    }

    /// The smallest numeric type that holds both `self` and `other`.
    fn widen(self, other: Numeric) -> Numeric {
        use Numeric::*;
        if self.is_float() || other.is_float() {
            // f32 has a 24-bit mantissa: any ≥32-bit integer or an f64 forces
            // f64 to avoid silent precision loss.
            let forces_64 = |n: Numeric| n.width() >= 32 && !matches!(n, Float(32));
            let needs_64 = forces_64(self)
                || forces_64(other)
                || matches!(self, Float(64))
                || matches!(other, Float(64));
            return Float(if needs_64 { 64 } else { 32 });
        }
        match (self, other) {
            (Int(x), Int(y)) => Int(x.max(y)),
            (UInt(x), UInt(y)) => UInt(x.max(y)),
            // Mixed sign → signed type large enough to hold the unsigned range.
            (Int(iw), UInt(uw)) | (UInt(uw), Int(iw)) => Int(iw.max((uw * 2).min(64))),
            _ => unreachable!("float handled above"),
        }
    }
}

impl From<Numeric> for DType {
    fn from(n: Numeric) -> DType {
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
}

/// Serde wrapper for [`DType`], which serializes via rkyv rather than serde.
///
/// Used as a map value in the schema; delegates to [`dtype_serde`]'s tagged
/// representation so it round-trips through `atlas.json`.
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

/// Serde helpers for [`DType`], usable via `#[serde(with = "dtype_serde")]`.
///
/// `DType` comes from `array_format` and implements rkyv, not serde. This maps
/// it to a self-describing tagged enum so schemas are readable in `atlas.json`.
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
    use super::widen_dtype;
    use array_format::DType;

    #[test]
    fn numeric_widening() {
        assert_eq!(widen_dtype(&DType::Int8, &DType::Int32), Some(DType::Int32));
        assert_eq!(widen_dtype(&DType::UInt8, &DType::UInt16), Some(DType::UInt16));
        // Mixed sign promotes to a larger signed type.
        assert_eq!(widen_dtype(&DType::Int8, &DType::UInt8), Some(DType::Int16));
        assert_eq!(widen_dtype(&DType::Int32, &DType::UInt32), Some(DType::Int64));
        // Float with a ≥32-bit integer needs f64.
        assert_eq!(widen_dtype(&DType::Int32, &DType::Float32), Some(DType::Float64));
        assert_eq!(widen_dtype(&DType::Int8, &DType::Float32), Some(DType::Float32));
        assert_eq!(widen_dtype(&DType::Float32, &DType::Float64), Some(DType::Float64));
    }

    #[test]
    fn string_timestamp_widening() {
        assert_eq!(
            widen_dtype(&DType::String, &DType::TimestampNs),
            Some(DType::String)
        );
        assert_eq!(
            widen_dtype(&DType::TimestampNs, &DType::String),
            Some(DType::String)
        );
    }

    #[test]
    fn incompatible_types_collide() {
        assert_eq!(widen_dtype(&DType::Int32, &DType::String), None);
        assert_eq!(widen_dtype(&DType::Float64, &DType::Bool), None);
        assert_eq!(widen_dtype(&DType::Binary, &DType::String), None);
    }

    #[test]
    fn list_widening_is_elementwise() {
        let a = DType::List {
            child: Box::new(DType::Int8),
        };
        let b = DType::List {
            child: Box::new(DType::Int32),
        };
        assert_eq!(
            widen_dtype(&a, &b),
            Some(DType::List {
                child: Box::new(DType::Int32)
            })
        );
    }
}
