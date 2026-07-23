//! The schema of a single named array.

use array_format::DType;
use serde::{Deserialize, Serialize};

use crate::config::Codec;

/// Schema for a single named array within a dataset.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ArraySchema {
    /// Element type of this array.
    #[serde(with = "super::dtype::dtype_serde")]
    pub dtype: DType,
    /// Logical shape, one entry per axis.
    pub shape: Vec<usize>,
    /// On-disk chunk shape, same rank as `shape`. Equal to `shape` for
    /// single-chunk arrays.
    pub chunk_shape: Vec<usize>,
    /// Named dimensions, one per axis. Order matches `shape`.
    pub dimension_names: Vec<String>,
    /// Codec used when this array was first created; controls how new blocks
    /// are written.
    pub codec: Codec,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn array_schema_roundtrip_via_serde() {
        let schema = ArraySchema {
            dtype: DType::Float64,
            shape: vec![10, 20],
            chunk_shape: vec![5, 5],
            dimension_names: vec!["lat".into(), "lon".into()],
            codec: Codec::default(),
        };
        let json = serde_json::to_string(&schema).unwrap();
        let back: ArraySchema = serde_json::from_str(&json).unwrap();
        assert_eq!(schema, back);
    }
}
