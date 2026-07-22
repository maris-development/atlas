//! Per-dataset and collection-wide schema shapes.
//!
//! [`DatasetSchema`] records what one dataset holds (array schemas plus the
//! attribute-key namespace). [`MergedSchema`] is the widened union across the
//! whole store — descriptive only, written into `atlas.json` for external
//! tooling. Reads always use each dataset's own [`DatasetSchema`].

use std::sync::Arc;

use array_format::DType;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use crate::schema::{ArraySchema, DTypeS, widen_dtype};

/// Schema for a single dataset: its array schemas plus the attribute-key
/// namespace (which global/per-array attribute keys the dataset uses).
///
/// Attribute **values** are not stored here — they live in the per-array
/// `.af` files. Only the key names are recorded, so a reader knows which keys
/// to fetch from the array files. All maps preserve insertion order (via
/// [`IndexMap`]) so on-disk layouts and Python-side dict iteration mirror the
/// order arrays/attributes were added.
#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
pub struct DatasetSchema {
    /// Array name → schema. Insertion-ordered.
    #[serde(default)]
    pub arrays: IndexMap<String, ArraySchema>,
    /// Dataset-level (global) attribute key → its value type.
    #[serde(default)]
    pub global_attrs: IndexMap<String, DTypeS>,
    /// Array name → that array's per-variable attribute keys → value type.
    #[serde(default)]
    pub array_attrs: IndexMap<String, IndexMap<String, DTypeS>>,
}

/// Direct schema mutators, used only to build fixtures. Production writes go
/// through [`StoreMeta::record_global_attr`](super::StoreMeta::record_global_attr)
/// / [`record_array_attr`](super::StoreMeta::record_array_attr) so the type
/// index stays in sync.
#[cfg(test)]
impl DatasetSchema {
    pub(crate) fn register_global_attr(&mut self, key: &str, ty: DType) {
        self.global_attrs.insert(key.to_string(), DTypeS(ty));
    }

    pub(crate) fn register_array_attr(&mut self, array: &str, key: &str, ty: DType) {
        self.array_attrs
            .entry(array.to_string())
            .or_default()
            .insert(key.to_string(), DTypeS(ty));
    }
}

/// Content hash of a dataset schema, consistent with its `PartialEq`: two
/// schemas that compare equal hash equal. Used to intern identical schemas.
pub(super) fn schema_hash(schema: &DatasetSchema) -> u64 {
    use std::hash::{Hash, Hasher};

    // `DType` (from `array_format`) derives neither `Hash` nor `Eq` — it holds
    // no floats, this is just a missing derive — so spell the hash out.
    fn hash_dtype<H: Hasher>(dtype: &DType, state: &mut H) {
        std::mem::discriminant(dtype).hash(state);
        match dtype {
            DType::List { child } => hash_dtype(child, state),
            DType::FixedSizeList { child, size } => {
                hash_dtype(child, state);
                size.hash(state);
            }
            _ => {}
        }
    }

    let mut state = std::collections::hash_map::DefaultHasher::new();
    schema.arrays.len().hash(&mut state);
    for (name, array) in &schema.arrays {
        name.hash(&mut state);
        hash_dtype(&array.dtype, &mut state);
        array.shape.hash(&mut state);
        array.chunk_shape.hash(&mut state);
        array.dimension_names.hash(&mut state);
        array.codec.hash(&mut state);
    }
    schema.global_attrs.len().hash(&mut state);
    for (key, ty) in &schema.global_attrs {
        key.hash(&mut state);
        hash_dtype(&ty.0, &mut state);
    }
    schema.array_attrs.len().hash(&mut state);
    for (array, attrs) in &schema.array_attrs {
        array.hash(&mut state);
        attrs.len().hash(&mut state);
        for (key, ty) in attrs {
            key.hash(&mut state);
            hash_dtype(&ty.0, &mut state);
        }
    }
    state.finish()
}

/// A collection-wide, merged view of every unique array and attribute, with
/// types widened across all datasets. Purely descriptive — reads use each
/// dataset's own schema; this is a summary written into `atlas.json` for
/// external tools and quick inspection.
#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
pub struct MergedSchema {
    /// Array name → merged array description.
    #[serde(default)]
    pub arrays: IndexMap<String, MergedArray>,
    /// Global attribute key → merged (widened) type.
    #[serde(default)]
    pub global_attributes: IndexMap<String, DTypeS>,
}

/// One array's merged description across every dataset that declares it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MergedArray {
    /// Element type, widened across datasets.
    pub dtype: DTypeS,
    /// Named dimensions (from the first dataset that declared the array).
    pub dimension_names: Vec<String>,
    /// Per-variable attribute key → merged (widened) type.
    #[serde(default)]
    pub attributes: IndexMap<String, DTypeS>,
}

/// Merge one attribute type into a key→type map, widening on collision.
fn merge_type(map: &mut IndexMap<String, DTypeS>, key: &str, ty: &DType) {
    match map.get_mut(key) {
        Some(existing) => {
            if let Some(w) = widen_dtype(&existing.0, ty) {
                existing.0 = w;
            }
        }
        None => {
            map.insert(key.to_string(), DTypeS(ty.clone()));
        }
    }
}

/// Fold a set of dataset schemas into the collection-wide merged schema.
///
/// Type collisions widen where possible; insert-time validation (in
/// `DatasetView`) keeps a non-widenable type from displacing the first-seen one.
pub(super) fn compute_merged<'a>(
    schemas: impl Iterator<Item = &'a Arc<DatasetSchema>>,
) -> MergedSchema {
    let mut merged = MergedSchema::default();
    for schema in schemas {
        for (name, arr) in &schema.arrays {
            let entry = merged
                .arrays
                .entry(name.clone())
                .or_insert_with(|| MergedArray {
                    dtype: DTypeS(arr.dtype.clone()),
                    dimension_names: arr.dimension_names.clone(),
                    attributes: IndexMap::new(),
                });
            if let Some(w) = widen_dtype(&entry.dtype.0, &arr.dtype) {
                entry.dtype.0 = w;
            }
            if let Some(attrs) = schema.array_attrs.get(name) {
                for (k, ty) in attrs {
                    merge_type(&mut entry.attributes, k, &ty.0);
                }
            }
        }
        for (k, ty) in &schema.global_attrs {
            merge_type(&mut merged.global_attributes, k, &ty.0);
        }
    }
    merged
}

#[cfg(test)]
mod tests {
    use super::*;

    fn schema_with_array(array: &str, dtype: DType) -> DatasetSchema {
        let mut s = DatasetSchema::default();
        s.arrays.insert(
            array.to_string(),
            ArraySchema {
                dtype,
                shape: vec![1],
                chunk_shape: vec![1],
                dimension_names: vec!["x".into()],
                codec: crate::config::Codec::default(),
            },
        );
        s
    }

    #[test]
    fn merged_schema_widens_numeric_array_dtypes() {
        let datasets = [
            Arc::new(schema_with_array("temp", DType::Int16)),
            Arc::new(schema_with_array("temp", DType::Int32)),
            Arc::new({
                let mut c = schema_with_array("temp", DType::Float32);
                c.register_global_attr("region", DType::String);
                c
            }),
        ];
        let merged = compute_merged(datasets.iter());
        // Int16 ∪ Int32 ∪ Float32 → Float64 (float + ≥32-bit int).
        assert_eq!(merged.arrays["temp"].dtype.0, DType::Float64);
        assert_eq!(merged.global_attributes["region"].0, DType::String);
    }

    #[test]
    fn merged_schema_widens_string_and_timestamp_attr() {
        let mut a = DatasetSchema::default();
        a.register_global_attr("created", DType::TimestampNs);
        let mut b = DatasetSchema::default();
        b.register_global_attr("created", DType::String);
        let datasets = [Arc::new(a), Arc::new(b)];
        let merged = compute_merged(datasets.iter());
        assert_eq!(merged.global_attributes["created"].0, DType::String);
    }

    #[test]
    fn identical_schemas_hash_equal() {
        let a = schema_with_array("temp", DType::Int32);
        let b = schema_with_array("temp", DType::Int32);
        let c = schema_with_array("temp", DType::Float64);
        assert_eq!(schema_hash(&a), schema_hash(&b));
        assert_ne!(schema_hash(&a), schema_hash(&c));
    }
}
