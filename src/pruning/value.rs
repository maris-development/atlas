//! Statistic values and their ordering.

use array_format::StatValue;
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
