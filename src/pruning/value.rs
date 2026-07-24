//! Statistic values and their ordering.

use array_format::{AttributeValue, StatValue};
use serde::{Deserialize, Serialize};

/// A minimum or maximum, in the type the statistic was computed with.
///
/// A serde-compatible mirror of [`array_format::StatValue`], which serializes
/// via rkyv — the index keeps its own representation rather than pulling rkyv
/// into the metadata path.
///
/// Values are **not** converted to a common type. A column whose datasets
/// disagree (legal under
/// [`TypeMismatchPolicy::Warn`](crate::TypeMismatchPolicy::Warn)) holds each
/// dataset's own type; callers that want one buffer promote at the boundary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum StatVal {
    /// Signed integer.
    Int(i64),
    /// Unsigned integer; also carries `Bool`, as 0/1.
    UInt(u64),
    /// Floating point.
    Float(f64),
    /// String or binary, raw bytes in lexicographic order.
    Bytes(Vec<u8>),
    /// Nanoseconds since the Unix epoch.
    TimestampNs(i64),
}

impl From<&StatValue> for StatVal {
    fn from(v: &StatValue) -> Self {
        match v {
            StatValue::Int(i) => StatVal::Int(*i),
            StatValue::UInt(u) => StatVal::UInt(*u),
            StatValue::Float(f) => StatVal::Float(*f),
            StatValue::Bytes(b) => StatVal::Bytes(b.clone()),
            StatValue::TimestampNs(t) => StatVal::TimestampNs(*t),
        }
    }
}

/// Orders two values of the same variant; `None` when the variants differ,
/// which only happens in a column whose datasets disagree on type.
///
/// Floats use `total_cmp`, so NaN sorts consistently instead of poisoning every
/// comparison. This is what makes predicates read naturally:
///
/// ```
/// # use atlas::StatVal;
/// let threshold = StatVal::Float(25.0);
/// assert!(StatVal::Float(30.0) > threshold);
/// ```
impl PartialOrd for StatVal {
    fn partial_cmp(&self, other: &StatVal) -> Option<std::cmp::Ordering> {
        match (self, other) {
            (StatVal::Int(a), StatVal::Int(b)) => Some(a.cmp(b)),
            (StatVal::UInt(a), StatVal::UInt(b)) => Some(a.cmp(b)),
            (StatVal::Float(a), StatVal::Float(b)) => Some(a.total_cmp(b)),
            (StatVal::TimestampNs(a), StatVal::TimestampNs(b)) => Some(a.cmp(b)),
            (StatVal::Bytes(a), StatVal::Bytes(b)) => Some(a.cmp(b)),
            // Comparing across types is meaningless, so every comparison
            // operator answers `false` — the row is simply not a candidate.
            _ => None,
        }
    }
}

impl StatVal {
    /// A scalar attribute value as a `StatVal`, or `None` for a list-valued
    /// attribute (which has no single point for range pruning). An attribute is
    /// one value per dataset, so its pruning cell is a point range `[v, v]`.
    ///
    /// Widths collapse to the index's variants: any signed int → `Int`, any
    /// unsigned int (and `Bool` as 0/1) → `UInt`, either float → `Float`, and
    /// `String`/`Binary` → `Bytes` (lexicographic).
    pub(crate) fn scalar_from_attribute(value: &AttributeValue) -> Option<StatVal> {
        use AttributeValue as A;
        Some(match value {
            A::Bool(b) => StatVal::UInt(*b as u64),
            A::Int8(x) => StatVal::Int(*x as i64),
            A::Int16(x) => StatVal::Int(*x as i64),
            A::Int32(x) => StatVal::Int(*x as i64),
            A::Int64(x) => StatVal::Int(*x),
            A::UInt8(x) => StatVal::UInt(*x as u64),
            A::UInt16(x) => StatVal::UInt(*x as u64),
            A::UInt32(x) => StatVal::UInt(*x as u64),
            A::UInt64(x) => StatVal::UInt(*x),
            A::Float32(x) => StatVal::Float(*x as f64),
            A::Float64(x) => StatVal::Float(*x),
            A::String(s) => StatVal::Bytes(s.as_bytes().to_vec()),
            A::Binary(b) => StatVal::Bytes(b.clone()),
            // List-valued attributes carry no scalar range.
            _ => return None,
        })
    }

    /// The smaller of two values; keeps `self` when they aren't comparable.
    pub(crate) fn min_with(self, other: StatVal) -> StatVal {
        match self.partial_cmp(&other) {
            Some(std::cmp::Ordering::Greater) => other,
            _ => self,
        }
    }

    /// The larger of two values; keeps `self` when they aren't comparable.
    pub(crate) fn max_with(self, other: StatVal) -> StatVal {
        match self.partial_cmp(&other) {
            Some(std::cmp::Ordering::Less) => other,
            _ => self,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cmp::Ordering;

    #[test]
    fn compares_within_a_variant() {
        assert_eq!(
            StatVal::Int(1).partial_cmp(&StatVal::Int(2)),
            Some(Ordering::Less)
        );
        assert_eq!(
            StatVal::Bytes(b"apple".to_vec()).partial_cmp(&StatVal::Bytes(b"pear".to_vec())),
            Some(Ordering::Less),
            "byte comparison is lexicographic"
        );
    }

    #[test]
    fn mismatched_variants_are_incomparable() {
        assert_eq!(
            StatVal::Int(1).partial_cmp(&StatVal::Bytes(b"x".to_vec())),
            None
        );
        // ...and folding keeps the first rather than picking arbitrarily.
        assert_eq!(
            StatVal::Int(1).min_with(StatVal::Bytes(b"x".to_vec())),
            StatVal::Int(1)
        );
    }

    #[test]
    fn nan_does_not_poison_ordering() {
        let nan = StatVal::Float(f64::NAN);
        assert!(nan.partial_cmp(&StatVal::Float(1.0)).is_some());
        assert_eq!(StatVal::Float(1.0).min_with(nan), StatVal::Float(1.0));
    }

    #[test]
    fn folds_pick_the_extreme() {
        assert_eq!(StatVal::Int(5).min_with(StatVal::Int(2)), StatVal::Int(2));
        assert_eq!(StatVal::Int(5).max_with(StatVal::Int(2)), StatVal::Int(5));
    }
}
