//! The container footer. It holds everything a reader needs before it touches
//! data.
//!
//! One [`CollectionFooter`] sits between the last segment and the trailer. It
//! holds every dataset name, every variable's byte range, and what each
//! dataset declares. An open reads this and nothing else.
//!
//! It holds nothing a segment already holds. Not an attribute value, and not a
//! statistic. Both sit on the array they belong to, inside that variable's
//! segment, and a reader takes them from there.
//!
//! A dataset is therefore one `u32`: an index into the schema pool. Its name
//! is the key that finds it.
//!
//! What the footer keeps is the schema, and that earns its place. It names
//! every array and every attribute key with its type, so a reader can answer
//! what a collection holds without opening anything, and can tell an `i64`
//! apart from a timestamp when it does.
//!
//! # A schema names things and nothing more
//!
//! An [`InternedSchema`] holds array names with their element types, and
//! attribute keys with their value types. It holds no shape, no chunk shape,
//! no dimension name, and no fill value. Those describe the data, and the
//! segment that holds the data already records them.
//!
//! That is what makes the pool small. Datasets whose arrays differ only in
//! length share one schema, because length is not in it. A directory of ten
//! thousand files of one convention therefore interns to one entry.
//!
//! # Three pools
//!
//! Every string interns once, whether it names an array or an attribute. Every
//! dtype interns once. Every distinct schema interns once. A schema is
//! therefore a list of `u32` pairs, and a dataset is one `u32` into the pool.
//!
//! The footer is MessagePack in compact (positional) form, then zstd. Compact
//! form drops the field names, so [`FORMAT_VERSION`](super::FORMAT_VERSION)
//! pins the field order. A change to any struct below is a format change.

use std::collections::{HashMap, HashSet};

use array_format::DType;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use smol_str::SmolStr;

/// Arrays a schema holds inline. A NetCDF convention rarely passes this, and
/// a schema that does spills to the heap once.
///
/// Inline storage pays here and not on [`DatasetEntry`]. The pool holds a
/// handful of schemas, so the bytes cost nothing. A dataset entry exists once
/// per dataset, and there the size decides.
const INLINE_ARRAYS: usize = 8;

/// Attribute keys a schema holds inline.
const INLINE_ATTRS: usize = 4;

/// One array's attribute keys with their value types.
pub(crate) type AttrKeys = SmallVec<[(u32, u32); INLINE_ATTRS]>;

use crate::schema::{Attr, SchemaView};
use crate::{Error, Result};

/// What one dataset declares. Names and types, in definition order.
///
/// Every field is a pool index, so the whole struct is `Hash` and `Eq`. That
/// is what lets [`Interner`] settle a schema with one map lookup.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub(crate) struct InternedSchema {
    /// `(array name, element dtype)`, in definition order.
    pub arrays: SmallVec<[(u32, u32); INLINE_ARRAYS]>,
    /// `(attribute key, value dtype)` at the dataset level, in the order
    /// somebody set them. [`DatasetEntry::global_attrs`] holds the values.
    pub attrs: AttrKeys,
    /// Per-array attributes as `(array position, [(key, value dtype)])`.
    /// [`DatasetEntry::array_attrs`] holds the values.
    pub array_attrs: Vec<(u32, AttrKeys)>,
}

/// One variable's segment.
///
/// A segment holds one array name across the whole collection. Inside it, each
/// dataset that declares the array stores it under the **dataset's** name. To
/// read one variable over every dataset therefore opens one file, and reads a
/// run of neighbouring blocks.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct VariableEntry {
    /// Array name, as a `string_pool` index.
    pub name: u32,
    /// Absolute offset of this variable's segment in the container.
    pub seg_offset: u64,
    /// Segment length in bytes.
    pub seg_len: u64,
    /// Absolute offset of the segment's statistics sidecar.
    ///
    /// `array-format` keeps the statistics beside its file, not inside it, and
    /// reads them once at open. The container embeds both, and
    /// [`SegmentStore`](super::segment_store::SegmentStore) serves each under
    /// the name that crate expects.
    pub stats_offset: u64,
    /// Sidecar length in bytes. Zero when the segment carries none.
    pub stats_len: u64,
}

/// The complete metadata of one collection.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct CollectionFooter {
    /// Footer schema version. It matches the trailer. The check runs again
    /// here, so a truncated read cannot pass as a valid footer.
    pub version: u32,
    /// `array-format` footer version of the embedded segments.
    pub segment_format: u32,
    /// Block codec the writer used. For information only. Every block records
    /// its own codec, so a reader never needs this field.
    pub codec: crate::Codec,
    /// Creation time, milliseconds since the Unix epoch.
    pub created_unix_ms: i64,
    /// Interned strings. Array names and attribute keys index this one pool.
    /// Dataset names do not. Each of those occurs once, so a pool entry would
    /// cost more than it saves.
    pub string_pool: Vec<SmolStr>,
    /// Interned types, for both array elements and attribute values.
    #[serde(with = "crate::schema::dtype::dtype_pool_serde")]
    pub dtype_pool: Vec<DType>,
    /// Interned schemas. [`DatasetEntry::schema`] indexes this.
    pub schema_pool: Vec<InternedSchema>,
    /// One segment per distinct array name, in the order the writer first saw
    /// each name. Every dataset that declares the array reads from here.
    pub variables: Vec<VariableEntry>,
    /// The datasets: a name to the schema it declares, in write order.
    ///
    /// A name is unique, and its position here is the dataset's **ordinal**.
    /// One map therefore carries both, and a lookup by name costs no scan.
    /// The deletion mask names that ordinal.
    ///
    /// A schema index is all a dataset needs. Its bytes are in the variable
    /// segments, its attribute values are on the arrays there, and so are its
    /// statistics.
    ///
    /// It goes on the wire as a sequence of `(name, schema)` pairs, not as a
    /// map. A map would let a repeated name collapse two datasets into one
    /// silently, and shift every ordinal after it.
    #[serde(with = "dataset_map_serde")]
    pub datasets: IndexMap<SmolStr, u32>,
}

/// Serde for [`CollectionFooter::datasets`].
///
/// The wire form is a sequence of `(name, schema)` pairs, so the field costs no
/// map framing and keeps its order. A repeated name is corruption: as a map it
/// would collapse two datasets into one and shift every later ordinal, while
/// the deletion mask still named the old ones. The writer refuses a repeat, so
/// one here means a damaged or foreign footer.
mod dataset_map_serde {
    use super::SmolStr;
    use indexmap::IndexMap;
    use serde::de::Error as _;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(
        datasets: &IndexMap<SmolStr, u32>,
        s: S,
    ) -> Result<S::Ok, S::Error> {
        let pairs: Vec<(&SmolStr, &u32)> = datasets.iter().collect();
        pairs.serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        d: D,
    ) -> Result<IndexMap<SmolStr, u32>, D::Error> {
        let pairs = Vec::<(SmolStr, u32)>::deserialize(d)?;
        let mut out = IndexMap::with_capacity(pairs.len());
        for (ordinal, (name, schema)) in pairs.into_iter().enumerate() {
            if let Some(first) = out.get_index_of(&name) {
                return Err(D::Error::custom(format!(
                    "dataset name '{name}' appears twice, at ordinal {first} and {ordinal}"
                )));
            }
            out.insert(name, schema);
        }
        Ok(out)
    }
}

/// Interns strings, dtypes, and schemas during a write.
///
/// The writer holds one, and hands out the indices the footer stores.
#[derive(Debug, Default)]
pub(crate) struct Interner {
    strings: Vec<SmolStr>,
    string_index: HashMap<SmolStr, u32>,
    dtypes: Vec<DType>,
    schemas: Vec<InternedSchema>,
    schema_index: HashMap<InternedSchema, u32>,
}

impl Interner {
    /// Index of `s` in the string pool. Adds it if it is new.
    pub(crate) fn intern_string(&mut self, s: &str) -> u32 {
        if let Some(&i) = self.string_index.get(s) {
            return i;
        }
        let idx = self.strings.len() as u32;
        self.string_index.insert(SmolStr::new(s), idx);
        self.strings.push(SmolStr::new(s));
        idx
    }

    /// Index of `dtype` in the dtype pool. Adds it if it is new.
    ///
    /// A scan, because a collection holds a handful of types. `DType`
    /// implements neither `Hash` nor `Eq`, so a map is out anyway.
    pub(crate) fn intern_dtype(&mut self, dtype: &DType) -> u32 {
        if let Some(i) = self.dtypes.iter().position(|d| d == dtype) {
            return i as u32;
        }
        self.dtypes.push(dtype.clone());
        (self.dtypes.len() - 1) as u32
    }

    /// Index of the schema these declarations describe. Adds it if it is new.
    ///
    /// `array_attrs` carries the array's position, not its name, because the
    /// caller already holds the arrays in definition order.
    pub(crate) fn intern_schema(
        &mut self,
        arrays: &IndexMap<String, DType>,
        attrs: &[(String, Attr)],
        array_attrs: &[(u32, Vec<(String, Attr)>)],
    ) -> u32 {
        let mut encoded = InternedSchema::default();
        for (name, dtype) in arrays {
            let name = self.intern_string(name);
            let dtype = self.intern_dtype(dtype);
            encoded.arrays.push((name, dtype));
        }
        for (key, value) in attrs {
            let key = self.intern_string(key);
            let dtype = self.intern_dtype(&value.dtype());
            encoded.attrs.push((key, dtype));
        }
        for (position, keyed) in array_attrs {
            let mut pairs = AttrKeys::with_capacity(keyed.len());
            for (key, value) in keyed {
                let key = self.intern_string(key);
                let dtype = self.intern_dtype(&value.dtype());
                pairs.push((key, dtype));
            }
            encoded.array_attrs.push((*position, pairs));
        }
        if let Some(&i) = self.schema_index.get(&encoded) {
            return i;
        }
        let idx = self.schemas.len() as u32;
        self.schema_index.insert(encoded.clone(), idx);
        self.schemas.push(encoded);
        idx
    }

    /// Consumes the interner into the three pools the footer stores.
    pub(crate) fn into_pools(self) -> (Vec<SmolStr>, Vec<DType>, Vec<InternedSchema>) {
        (self.strings, self.dtypes, self.schemas)
    }
}

impl CollectionFooter {
    /// Serializes to the bytes the container stores. Compact MessagePack,
    /// then zstd.
    pub(crate) fn encode(&self) -> Result<Vec<u8>> {
        let packed = rmp_serde::to_vec(self)?;
        Ok(zstd::stream::encode_all(packed.as_slice(), 0)?)
    }

    /// Reverses [`encode`](Self::encode). Then checks the collection agrees
    /// with itself.
    pub(crate) fn decode(bytes: &[u8]) -> Result<Self> {
        let packed = zstd::stream::decode_all(bytes)
            .map_err(|e| Error::CorruptCollection(format!("footer is not valid zstd: {e}")))?;
        let footer: Self = rmp_serde::from_slice(&packed)?;
        footer.validate()?;
        Ok(footer)
    }

    /// Rejects a footer whose indices do not resolve, whose value counts do
    /// not match their schema, or whose names repeat. One check here costs
    /// less than a dangling index at every use site.
    ///
    /// The schema pool is checked once, not once per dataset.
    fn validate(&self) -> Result<()> {
        if self.version != super::FORMAT_VERSION {
            return Err(Error::UnsupportedVersion {
                found: self.version,
                expected: super::FORMAT_VERSION,
            });
        }
        let mut variables: HashSet<u32> = HashSet::with_capacity(self.variables.len());
        for (i, variable) in self.variables.iter().enumerate() {
            let name = self.string(variable.name).ok_or_else(|| {
                Error::CorruptCollection(format!(
                    "variable {i} names an array {} out of {} strings",
                    variable.name,
                    self.string_pool.len()
                ))
            })?;
            if !variables.insert(variable.name) {
                return Err(Error::CorruptCollection(format!(
                    "variable '{name}' has two segments, the second at {i}"
                )));
            }
            if variable.seg_len == 0 {
                return Err(Error::CorruptCollection(format!(
                    "variable '{name}' has an empty segment"
                )));
            }
        }
        for (i, schema) in self.schema_pool.iter().enumerate() {
            for &(name, dtype) in &schema.arrays {
                let array = self.string(name).ok_or_else(|| {
                    Error::CorruptCollection(format!(
                        "schema {i} names an array {name} out of {} strings",
                        self.string_pool.len()
                    ))
                })?;
                // A declared array needs a segment to read from.
                if !variables.contains(&name) {
                    return Err(Error::CorruptCollection(format!(
                        "schema {i} declares array '{array}' but no segment holds it"
                    )));
                }
                self.check_dtype(i, dtype)?;
            }
            for &(key, dtype) in &schema.attrs {
                self.check_string(i, key)?;
                self.check_dtype(i, dtype)?;
            }
            for (position, pairs) in &schema.array_attrs {
                if *position as usize >= schema.arrays.len() {
                    return Err(Error::CorruptCollection(format!(
                        "schema {i} annotates array {position} but declares {}",
                        schema.arrays.len()
                    )));
                }
                for &(key, dtype) in pairs {
                    self.check_string(i, key)?;
                    self.check_dtype(i, dtype)?;
                }
            }
        }
        // `dataset_map_serde` already refused a repeated name, so every entry
        // below has a distinct name and a stable ordinal.
        for (name, schema) in self.datasets.iter() {
            if self.schema_pool.get(*schema as usize).is_none() {
                return Err(Error::CorruptCollection(format!(
                    "dataset '{name}' references schema {schema} but the pool holds {}",
                    self.schema_pool.len()
                )));
            }
        }
        Ok(())
    }

    fn check_string(&self, schema: usize, idx: u32) -> Result<()> {
        if self.string(idx).is_none() {
            return Err(Error::CorruptCollection(format!(
                "schema {schema} references string {idx} out of {}",
                self.string_pool.len()
            )));
        }
        Ok(())
    }

    fn check_dtype(&self, schema: usize, idx: u32) -> Result<()> {
        if self.dtype(idx).is_none() {
            return Err(Error::CorruptCollection(format!(
                "schema {schema} references dtype {idx} but the pool holds {}",
                self.dtype_pool.len()
            )));
        }
        Ok(())
    }

    /// The interned string at `idx`.
    pub(crate) fn string(&self, idx: u32) -> Option<&str> {
        self.string_pool.get(idx as usize).map(SmolStr::as_str)
    }

    /// Index of `s` in the string pool, if it holds it.
    ///
    /// One call turns a per-dataset string compare into an integer compare. A
    /// name the pool never held cannot occur anywhere in the footer, so a
    /// `None` here answers the whole collection at once.
    pub(crate) fn string_id(&self, s: &str) -> Option<u32> {
        self.string_pool
            .iter()
            .position(|p| p == s)
            .map(|i| i as u32)
    }

    /// The interned dtype at `idx`.
    pub(crate) fn dtype(&self, idx: u32) -> Option<&DType> {
        self.dtype_pool.get(idx as usize)
    }

    /// The schema at `index`.
    pub(crate) fn schema_of(&self, index: u32) -> &InternedSchema {
        // validate() proved the index resolves.
        &self.schema_pool[index as usize]
    }

    /// The schema at `index`, as the public borrowed view.
    pub(crate) fn schema_view(&self, index: u32) -> SchemaView<'_> {
        SchemaView::new(self, self.schema_of(index))
    }

    /// Which segment holds the array named by `name`, as a
    /// [`variables`](Self::variables) position.
    pub(crate) fn variable_index(&self, name: u32) -> Option<usize> {
        self.variables.iter().position(|v| v.name == name)
    }

    /// Where `array` sits in each interned schema, by pool index.
    ///
    /// `None` when no schema declares the name. Otherwise entry `i` is the
    /// position of `array` in schema `i`, or `None` if that schema omits it.
    ///
    /// One pass over the pool answers every dataset. Datasets that share a
    /// schema therefore resolve the name once between them, not once each.
    pub(crate) fn array_positions(&self, array: &str) -> Option<Vec<Option<u32>>> {
        let wanted = self.string_id(array)?;
        Some(
            self.schema_pool
                .iter()
                .map(|schema| {
                    schema
                        .arrays
                        .iter()
                        .position(|&(name, _)| name == wanted)
                        .map(|p| p as u32)
                })
                .collect(),
        )
    }

    /// The declared type of the dataset-level attribute `key` in each
    /// interned schema, as a `dtype_pool` index.
    ///
    /// `None` when no schema names the key. Otherwise entry `i` is the type
    /// schema `i` declares for it, or `None` if that schema omits it.
    ///
    /// One pass over the pool answers every dataset. A segment stores the
    /// value, and this says how to read it back: an `i64` is a timestamp when
    /// the schema says so.
    pub(crate) fn attr_dtypes(&self, key: &str) -> Option<Vec<Option<u32>>> {
        let wanted = self.string_id(key)?;
        Some(
            self.schema_pool
                .iter()
                .map(|schema| {
                    schema
                        .attrs
                        .iter()
                        .find(|&&(k, _)| k == wanted)
                        .map(|&(_, dtype)| dtype)
                })
                .collect(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// What one dataset declares: arrays, dataset attributes, array
    /// attributes.
    type Declarations = (
        IndexMap<String, DType>,
        Vec<(String, Attr)>,
        Vec<(u32, Vec<(String, Attr)>)>,
    );

    /// One array, one dataset attribute, one array attribute.
    fn declarations(dtype: DType) -> Declarations {
        let mut arrays = IndexMap::new();
        arrays.insert("temperature".to_string(), dtype);
        let attrs = vec![("month".to_string(), Attr::Int64(1))];
        let array_attrs = vec![(0u32, vec![("units".to_string(), Attr::String("K".into()))])];
        (arrays, attrs, array_attrs)
    }

    /// An interner holding one schema over `dtype`.
    fn one_schema(dtype: DType) -> Interner {
        let mut interner = Interner::default();
        let (arrays, attrs, array_attrs) = declarations(dtype);
        assert_eq!(interner.intern_schema(&arrays, &attrs, &array_attrs), 0);
        interner
    }

    /// A footer whose pools come from one interner, so every index resolves.
    /// Each name in `variables` gets a segment of 128 bytes.
    fn footer_with(
        datasets: Vec<(&str, u32)>,
        interner: Interner,
        variables: &[&str],
    ) -> CollectionFooter {
        let (string_pool, dtype_pool, schema_pool) = interner.into_pools();
        let variables = variables
            .iter()
            .enumerate()
            .map(|(i, name)| VariableEntry {
                name: string_pool
                    .iter()
                    .position(|s| s == name)
                    .expect("the interner holds every variable name") as u32,
                seg_offset: 8 + (i as u64) * 160,
                seg_len: 128,
                stats_offset: 8 + (i as u64) * 160 + 128,
                stats_len: 32,
            })
            .collect();
        CollectionFooter {
            version: super::super::FORMAT_VERSION,
            segment_format: super::super::SEGMENT_FORMAT,
            codec: crate::Codec::Zstd,
            created_unix_ms: 1_700_000_000_000,
            string_pool,
            dtype_pool,
            schema_pool,
            variables,
            datasets: datasets
                .into_iter()
                .map(|(name, schema)| (SmolStr::new(name), schema))
                .collect(),
        }
    }

    #[test]
    fn footer_roundtrips_through_msgpack_and_zstd() {
        let f = footer_with(
            vec![("a", 0), ("b", 0)],
            one_schema(DType::Float32),
            &["temperature"],
        );
        let bytes = f.encode().unwrap();
        assert_eq!(CollectionFooter::decode(&bytes).unwrap(), f);
    }

    #[test]
    fn identical_declarations_intern_to_one_pool_entry() {
        let mut interner = Interner::default();
        let (arrays, attrs, array_attrs) = declarations(DType::Float32);
        let a = interner.intern_schema(&arrays, &attrs, &array_attrs);
        let b = interner.intern_schema(&arrays, &attrs, &array_attrs);
        let (other, _, _) = declarations(DType::Int64);
        let c = interner.intern_schema(&other, &attrs, &array_attrs);
        assert_eq!(a, b);
        assert_ne!(a, c);
        let (_, _, schemas) = interner.into_pools();
        assert_eq!(schemas.len(), 2);
    }

    #[test]
    fn an_interned_schema_holds_only_names_and_types() {
        let (_, _, schemas) = one_schema(DType::Float32).into_pools();
        let schema = &schemas[0];
        // Names and dtypes, as pool indices. Nothing else fits in the type.
        assert_eq!(schema.arrays.len(), 1);
        assert_eq!(schema.attrs.len(), 1);
        assert_eq!(schema.array_attrs.len(), 1);
        assert_eq!(schema.array_attrs[0].0, 0);
    }

    #[test]
    fn array_names_and_attribute_keys_share_one_pool() {
        let (strings, dtypes, _) = one_schema(DType::Float32).into_pools();
        // temperature, month, units. One entry each, in the order interned.
        assert_eq!(
            strings,
            vec![
                "temperature".to_string(),
                "month".to_string(),
                "units".to_string()
            ]
        );
        // f32 for the array, i64 for month, utf8 for units.
        assert_eq!(dtypes, vec![DType::Float32, DType::Int64, DType::String]);
    }

    #[test]
    fn another_attribute_key_makes_another_schema() {
        // The schema names the keys, so a different key set is a different
        // schema. Two datasets of one convention still share theirs.
        let mut interner = Interner::default();
        let (arrays, attrs, array_attrs) = declarations(DType::Float32);
        let a = interner.intern_schema(&arrays, &attrs, &array_attrs);
        let more = vec![
            ("month".to_string(), Attr::Int64(1)),
            ("source".to_string(), Attr::String("buoy".into())),
        ];
        let b = interner.intern_schema(&arrays, &more, &array_attrs);
        assert_ne!(a, b);
    }

    #[test]
    fn the_schema_declares_the_type_of_every_attribute() {
        let f = footer_with(vec![("a", 0)], one_schema(DType::Float32), &["temperature"]);
        // The footer holds no value. It holds the key and its type, which is
        // what a reader needs to read the value out of a segment.
        let schema = &f.schema_pool[0];
        assert_eq!(schema.attrs.len(), 1);
        assert_eq!(f.string(schema.attrs[0].0), Some("month"));
        assert_eq!(f.dtype(schema.attrs[0].1), Some(&DType::Int64));

        let (position, keys) = &schema.array_attrs[0];
        assert_eq!(*position, 0);
        assert_eq!(f.string(keys[0].0), Some("units"));
        assert_eq!(f.dtype(keys[0].1), Some(&DType::String));
    }

    #[test]
    fn positions_answer_the_pool_not_the_datasets() {
        let mut interner = Interner::default();
        let (arrays, attrs, array_attrs) = declarations(DType::Float32);
        interner.intern_schema(&arrays, &attrs, &array_attrs);
        let mut two = arrays.clone();
        two.insert("salinity".to_string(), DType::Float32);
        interner.intern_schema(&two, &attrs, &array_attrs);
        let f = footer_with(
            vec![("a", 0), ("b", 1)],
            interner,
            &["temperature", "salinity"],
        );

        // One entry per pool schema, whatever the dataset count.
        assert_eq!(
            f.array_positions("temperature"),
            Some(vec![Some(0), Some(0)])
        );
        assert_eq!(f.array_positions("salinity"), Some(vec![None, Some(1)]));
        assert_eq!(f.array_positions("missing"), None);

        // The type of `month` in each schema, as a dtype pool index.
        let months = f.attr_dtypes("month").unwrap();
        assert_eq!(months.len(), f.schema_pool.len());
        for dtype in months {
            assert_eq!(f.dtype(dtype.unwrap()), Some(&DType::Int64));
        }
        assert_eq!(f.attr_dtypes("missing"), None);
    }

    #[test]
    fn a_variable_records_its_statistics_sidecar() {
        // `array-format` keeps the statistics beside its file, so the
        // container embeds both ranges and the reader serves both.
        let f = footer_with(vec![("a", 0)], one_schema(DType::Float32), &["temperature"]);
        let v = &f.variables[0];
        assert_eq!(v.seg_offset, 8);
        assert_eq!(v.seg_len, 128);
        assert_eq!(v.stats_offset, 136);
        assert_eq!(v.stats_len, 32);
        let bytes = f.encode().unwrap();
        assert_eq!(CollectionFooter::decode(&bytes).unwrap(), f);
    }

    #[test]
    fn a_dataset_is_one_index_into_the_schema_pool() {
        // Nothing else is per dataset. The bytes, the attribute values, and
        // the statistics all live in the segments.
        let f = footer_with(vec![("a", 0)], one_schema(DType::Float32), &["temperature"]);
        assert_eq!(f.datasets["a"], 0);
        assert_eq!(std::mem::size_of_val(&f.datasets["a"]), 4);
    }

    #[test]
    fn a_variable_resolves_to_its_segment() {
        let f = footer_with(vec![("a", 0)], one_schema(DType::Float32), &["temperature"]);
        let id = f.string_id("temperature").unwrap();
        assert_eq!(f.variable_index(id), Some(0));
        assert_eq!(f.variables[0].seg_len, 128);
    }

    /// Encodes `f`, decodes it, and returns the message of the expected
    /// corruption error.
    fn corruption(f: CollectionFooter) -> String {
        let bytes = f.encode().unwrap();
        let err = CollectionFooter::decode(&bytes).unwrap_err();
        assert!(matches!(err, Error::CorruptCollection(_)), "{err}");
        err.to_string()
    }

    #[test]
    fn a_dangling_schema_index_is_corruption() {
        let f = footer_with(vec![("a", 3)], one_schema(DType::Float32), &["temperature"]);
        assert!(corruption(f).contains("references schema"));
    }

    #[test]
    fn a_dangling_array_name_index_is_corruption() {
        let mut f = footer_with(vec![("a", 0)], one_schema(DType::Float32), &["temperature"]);
        f.schema_pool[0].arrays[0].0 = 9;
        assert!(corruption(f).contains("names an array"));
    }

    #[test]
    fn a_dangling_dtype_index_is_corruption() {
        let mut f = footer_with(vec![("a", 0)], one_schema(DType::Float32), &["temperature"]);
        f.schema_pool[0].arrays[0].1 = 9;
        assert!(corruption(f).contains("references dtype"));
    }

    #[test]
    fn an_array_with_no_segment_is_corruption() {
        let mut f = footer_with(vec![("a", 0)], one_schema(DType::Float32), &["temperature"]);
        f.variables.clear();
        assert!(corruption(f).contains("no segment holds it"));
    }

    #[test]
    fn a_repeated_variable_segment_is_corruption() {
        let mut f = footer_with(vec![("a", 0)], one_schema(DType::Float32), &["temperature"]);
        let repeat = f.variables[0].clone();
        f.variables.push(repeat);
        assert!(corruption(f).contains("two segments"));
    }

    #[test]
    fn an_empty_variable_segment_is_corruption() {
        let mut f = footer_with(vec![("a", 0)], one_schema(DType::Float32), &["temperature"]);
        f.variables[0].seg_len = 0;
        assert!(corruption(f).contains("empty segment"));
    }

    /// The footer's wire form, with the datasets as a plain sequence. It
    /// mirrors [`CollectionFooter`] field for field, so it encodes to the same
    /// bytes. Only this can express a repeated dataset name, which the real
    /// `IndexMap` refuses to hold.
    #[derive(Serialize)]
    struct RawFooter {
        version: u32,
        segment_format: u32,
        codec: crate::Codec,
        created_unix_ms: i64,
        string_pool: Vec<SmolStr>,
        #[serde(with = "crate::schema::dtype::dtype_pool_serde")]
        dtype_pool: Vec<DType>,
        schema_pool: Vec<InternedSchema>,
        variables: Vec<VariableEntry>,
        datasets: Vec<(SmolStr, u32)>,
    }

    #[test]
    fn a_repeated_dataset_name_is_corruption() {
        // Two datasets of one name would collapse into one on decode, and
        // shift every ordinal after them while the mask still named the old
        // ones. The decode refuses instead.
        let valid = footer_with(vec![("a", 0)], one_schema(DType::Float32), &["temperature"]);
        let raw = RawFooter {
            version: valid.version,
            segment_format: valid.segment_format,
            codec: valid.codec,
            created_unix_ms: valid.created_unix_ms,
            string_pool: valid.string_pool.clone(),
            dtype_pool: valid.dtype_pool.clone(),
            schema_pool: valid.schema_pool.clone(),
            variables: valid.variables.clone(),
            datasets: vec![(SmolStr::new("a"), 0), (SmolStr::new("a"), 0)],
        };
        let packed = rmp_serde::to_vec(&raw).unwrap();
        let bytes = zstd::stream::encode_all(packed.as_slice(), 0).unwrap();
        let err = CollectionFooter::decode(&bytes).unwrap_err();
        assert!(err.to_string().contains("appears twice"), "{err}");
    }

    #[test]
    fn the_wire_form_of_the_dataset_map_is_a_sequence() {
        // The mirror above only holds if the two encode alike. A change to
        // either field order or to the map framing breaks this.
        let f = footer_with(vec![("a", 0)], one_schema(DType::Float32), &["temperature"]);
        let raw = RawFooter {
            version: f.version,
            segment_format: f.segment_format,
            codec: f.codec,
            created_unix_ms: f.created_unix_ms,
            string_pool: f.string_pool.clone(),
            dtype_pool: f.dtype_pool.clone(),
            schema_pool: f.schema_pool.clone(),
            variables: f.variables.clone(),
            datasets: vec![(SmolStr::new("a"), 0)],
        };
        assert_eq!(
            rmp_serde::to_vec(&f).unwrap(),
            rmp_serde::to_vec(&raw).unwrap()
        );
    }

    #[test]
    fn a_dataset_position_is_its_ordinal() {
        let f = footer_with(
            vec![("first", 0), ("second", 0)],
            one_schema(DType::Float32),
            &["temperature"],
        );
        // The map carries the ordinal, so no entry has to repeat it.
        assert_eq!(f.datasets.get_index_of("first"), Some(0));
        assert_eq!(f.datasets.get_index_of("second"), Some(1));
        assert_eq!(f.datasets.get_index_of("missing"), None);
        assert_eq!(
            f.datasets.get_index(1).map(|(name, _)| name.as_str()),
            Some("second")
        );
        // The order survives the wire.
        let bytes = f.encode().unwrap();
        let back = CollectionFooter::decode(&bytes).unwrap();
        assert_eq!(
            back.datasets
                .keys()
                .map(SmolStr::as_str)
                .collect::<Vec<_>>(),
            vec!["first", "second"]
        );
    }

    #[test]
    fn distinct_dataset_names_pass() {
        let f = footer_with(
            vec![("a", 0), ("b", 0)],
            one_schema(DType::Float32),
            &["temperature"],
        );
        let bytes = f.encode().unwrap();
        assert_eq!(CollectionFooter::decode(&bytes).unwrap(), f);
    }

    #[test]
    fn garbage_bytes_are_corruption_not_a_panic() {
        assert!(matches!(
            CollectionFooter::decode(&[0xff; 32]),
            Err(Error::CorruptCollection(_))
        ));
    }

    #[test]
    fn empty_collection_roundtrips() {
        let f = footer_with(vec![], Interner::default(), &[]);
        let bytes = f.encode().unwrap();
        assert_eq!(CollectionFooter::decode(&bytes).unwrap(), f);
    }
}
