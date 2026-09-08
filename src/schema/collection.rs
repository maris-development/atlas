//! What a whole collection declares.
//!
//! [`SchemaView`](crate::SchemaView) answers for one dataset. A collection
//! holds many, and they need not agree on a name. This module merges them, so
//! one call answers what the container holds.

use std::collections::HashMap;

use array_format::DType;
use smallvec::SmallVec;

use super::view::VALIDATED;
use crate::format::footer::CollectionFooter;

/// The types one name takes across a collection.
///
/// One entry is the normal case, and it stays inline. A second entry means two
/// datasets declare the same name at different types.
pub type DTypeSet<'a> = SmallVec<[&'a DType; 1]>;

/// Every name a collection declares, with every type it takes.
///
/// The three maps cover the three kinds of name: an array, a dataset-level
/// attribute, and an array attribute. Each borrows from the footer, so this
/// copies no name.
///
/// It reports every dataset in the container. The deletion mask hides none of
/// them.
#[derive(Debug, Default, Clone)]
pub struct CollectionSchema<'a> {
    /// Array name to the element types it has.
    pub arrays: HashMap<&'a str, DTypeSet<'a>>,
    /// Dataset-level attribute key to the value types it has.
    pub attributes: HashMap<&'a str, DTypeSet<'a>>,
    /// Array name to that array's attribute keys, each with its value types.
    /// An array nobody annotated has no entry.
    pub array_attributes: HashMap<&'a str, HashMap<&'a str, DTypeSet<'a>>>,
}

impl<'a> CollectionSchema<'a> {
    /// Merges every schema in the footer's pool.
    ///
    /// The walk covers the pool, not the datasets. Datasets that declare the
    /// same things share one entry, so ten thousand files of one convention
    /// cost one pass over one schema.
    pub(crate) fn new(footer: &'a CollectionFooter) -> Self {
        let mut out = Self::default();
        for schema in &footer.schema_pool {
            for &(name, dtype) in &schema.arrays {
                out.arrays.record(
                    footer.string(name).expect(VALIDATED),
                    footer.dtype(dtype).expect(VALIDATED),
                );
            }
            for &(key, dtype) in &schema.attrs {
                out.attributes.record(
                    footer.string(key).expect(VALIDATED),
                    footer.dtype(dtype).expect(VALIDATED),
                );
            }
            for (position, pairs) in &schema.array_attrs {
                // validate() proved the position names a declared array.
                let (name, _) = schema.arrays[*position as usize];
                let keyed = out
                    .array_attributes
                    .entry(footer.string(name).expect(VALIDATED))
                    .or_default();
                for &(key, dtype) in pairs {
                    keyed.record(
                        footer.string(key).expect(VALIDATED),
                        footer.dtype(dtype).expect(VALIDATED),
                    );
                }
            }
        }
        out
    }

    /// The types of one array, over every dataset that declares it. Empty for
    /// a name the collection does not hold.
    pub fn array_dtypes(&self, array: &str) -> &[&'a DType] {
        self.arrays.get(array).map_or(&[], SmallVec::as_slice)
    }

    /// The types of one dataset-level attribute. Empty for a key no dataset
    /// carries.
    pub fn attribute_dtypes(&self, key: &str) -> &[&'a DType] {
        self.attributes.get(key).map_or(&[], SmallVec::as_slice)
    }

    /// The types of one array's attribute. Empty for a pair no dataset
    /// carries.
    pub fn array_attribute_dtypes(&self, array: &str, key: &str) -> &[&'a DType] {
        self.array_attributes
            .get(array)
            .and_then(|keyed| keyed.get(key))
            .map_or(&[], SmallVec::as_slice)
    }
}

/// Collects the distinct types of one name.
trait Record<'a> {
    /// Records `dtype` under `name`. A type the entry already holds is
    /// dropped, so every set stays distinct.
    fn record(&mut self, name: &'a str, dtype: &'a DType);
}

impl<'a> Record<'a> for HashMap<&'a str, DTypeSet<'a>> {
    fn record(&mut self, name: &'a str, dtype: &'a DType) {
        let seen = self.entry(name).or_default();
        if !seen.contains(&dtype) {
            seen.push(dtype);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::format::footer::{CollectionFooter, Interner, VariableEntry};
    use crate::schema::Attr;
    use array_format::DType;
    use indexmap::IndexMap;
    use smol_str::SmolStr;

    /// Interns one dataset that declares `array` at `dtype`, with one
    /// dataset-level attribute and one array attribute.
    fn intern(interner: &mut Interner, array: &str, dtype: DType, attr: Attr) -> u32 {
        let mut arrays = IndexMap::new();
        arrays.insert(array.to_string(), dtype);
        let attrs = vec![("month".to_string(), Attr::Int64(1))];
        let array_attrs = vec![(0u32, vec![("units".to_string(), attr)])];
        interner.intern_schema(&arrays, &attrs, &array_attrs)
    }

    /// A footer over the interned schemas, with a segment per array name.
    fn footer(interner: Interner, schemas: &[u32], variables: &[&str]) -> CollectionFooter {
        let (string_pool, dtype_pool, schema_pool) = interner.into_pools();
        let variables = variables
            .iter()
            .map(|name| VariableEntry {
                name: string_pool
                    .iter()
                    .position(|s| s == name)
                    .expect("interned") as u32,
                seg_offset: 8,
                seg_len: 128,
            })
            .collect();
        CollectionFooter {
            version: crate::FORMAT_VERSION,
            codec: crate::Codec::Zstd,
            created_unix_ms: 0,
            string_pool,
            dtype_pool,
            schema_pool,
            variables,
            datasets: schemas
                .iter()
                .enumerate()
                .map(|(i, s)| (SmolStr::new(format!("ds{i}")), *s))
                .collect(),
        }
    }

    #[test]
    fn one_convention_gives_one_type_per_name() {
        let mut interner = Interner::default();
        let a = intern(
            &mut interner,
            "temperature",
            DType::Float32,
            Attr::String("K".into()),
        );
        let b = intern(
            &mut interner,
            "temperature",
            DType::Float32,
            Attr::String("K".into()),
        );
        assert_eq!(a, b);
        let f = footer(interner, &[a, b], &["temperature"]);

        let schema = f.collection_schema();
        assert_eq!(schema.arrays.len(), 1);
        assert_eq!(schema.array_dtypes("temperature"), [&DType::Float32]);
        assert_eq!(schema.attribute_dtypes("month"), [&DType::Int64]);
        assert_eq!(
            schema.array_attribute_dtypes("temperature", "units"),
            [&DType::String]
        );
    }

    #[test]
    fn datasets_that_disagree_give_both_types() {
        let mut interner = Interner::default();
        let a = intern(
            &mut interner,
            "temperature",
            DType::Float32,
            Attr::String("K".into()),
        );
        let b = intern(&mut interner, "temperature", DType::Float64, Attr::Int64(1));
        let f = footer(interner, &[a, b], &["temperature"]);

        let schema = f.collection_schema();
        // One entry per name, holding every type that name takes.
        assert_eq!(schema.arrays.len(), 1);
        assert_eq!(
            schema.array_dtypes("temperature"),
            [&DType::Float32, &DType::Float64]
        );
        assert_eq!(
            schema.array_attribute_dtypes("temperature", "units"),
            [&DType::String, &DType::Int64]
        );
    }

    #[test]
    fn every_array_of_the_collection_appears_once() {
        let mut interner = Interner::default();
        let a = intern(&mut interner, "temperature", DType::Float32, Attr::Int64(1));
        let b = intern(&mut interner, "salinity", DType::Float64, Attr::Int64(1));
        let f = footer(interner, &[a, b], &["temperature", "salinity"]);

        let schema = f.collection_schema();
        let mut names: Vec<&str> = schema.arrays.keys().copied().collect();
        names.sort_unstable();
        assert_eq!(names, ["salinity", "temperature"]);
        assert_eq!(schema.array_dtypes("salinity"), [&DType::Float64]);
        // The two share the attribute keys, so each map holds one key.
        assert_eq!(schema.attributes.len(), 1);
        assert_eq!(schema.array_attributes.len(), 2);
    }

    #[test]
    fn a_name_the_collection_does_not_hold_is_empty() {
        let f = footer(Interner::default(), &[], &[]);
        let schema = f.collection_schema();
        assert!(schema.arrays.is_empty());
        assert!(schema.array_dtypes("missing").is_empty());
        assert!(schema.attribute_dtypes("missing").is_empty());
        assert!(schema.array_attribute_dtypes("missing", "units").is_empty());
    }
}
