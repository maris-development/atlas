//! The container footer. It holds everything a reader needs before it touches
//! data.
//!
//! One [`CollectionFooter`] sits between the last segment and the trailer. It
//! holds every dataset name, its segment byte range, its schema, and its
//! attribute values. An open reads this and nothing else. To list the datasets
//! or to inspect a schema therefore costs one range read.
//!
//! Two pools keep the footer small when a collection holds many similar
//! datasets. Equal schemas intern by content hash. Attribute keys intern as
//! strings. Both pools are plain indices into a `Vec`.
//!
//! The footer is MessagePack in compact (positional) form, then zstd. Compact
//! form drops the field names, so [`FORMAT_VERSION`](super::FORMAT_VERSION)
//! pins the field order. A change to any struct below is a format change.

use std::collections::{HashMap, HashSet};

use array_format::{ArrayStats, DType, StatValue};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use crate::schema::{Attr, DatasetSchema};
use crate::{Error, Result};

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
    /// Interned dataset schemas. [`DatasetEntry::schema`] indexes this.
    pub schema_pool: Vec<DatasetSchema>,
    /// Interned attribute keys. Attribute entries index this.
    pub attr_key_pool: Vec<String>,
    /// One entry per dataset, in write order. A dataset's position here is its
    /// **ordinal**. The deletion mask names that ordinal.
    pub datasets: Vec<DatasetEntry>,
}

/// One dataset. Where its bytes are, what it holds, and what annotates it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct DatasetEntry {
    /// Dataset name, unique within the collection.
    pub name: String,
    /// Index into [`CollectionFooter::schema_pool`].
    pub schema: u32,
    /// Absolute offset of this dataset's segment in the container.
    pub seg_offset: u64,
    /// Segment length in bytes.
    pub seg_len: u64,
    /// Dataset-level attributes as `(attr_key_pool index, value)`.
    pub global_attrs: Vec<(u32, AttrS)>,
    /// Per-array attributes as `(array position in the schema, attributes)`.
    /// The entry stores a position, not a name, because datasets share the
    /// schema.
    pub array_attrs: Vec<(u32, Vec<(u32, AttrS)>)>,
    /// Per-array statistics as `(array position in the schema, stats)`.
    ///
    /// `array-format` computes these while the dataset stages. To record them
    /// here therefore costs nothing at write time, and makes them free at read
    /// time. They behave like any other footer metadata. An array that is
    /// declared but never written has no entry.
    pub array_stats: Vec<(u32, ArrayStatsS)>,
}

/// Content hash of a schema, in step with its `PartialEq`. The interner uses
/// it.
fn schema_hash(schema: &DatasetSchema) -> u64 {
    use std::hash::{Hash, Hasher};

    // DType derives neither Hash nor Eq, so the hash is explicit here.
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
        // A fill value holds a float. Hash the discriminant, and let
        // PartialEq settle a collision.
        std::mem::discriminant(&array.fill_value).hash(&mut state);
    }
    state.finish()
}

/// Interns schemas and attribute keys during a write.
///
/// The writer holds one, and hands out the indices the footer stores.
#[derive(Debug, Default)]
pub(crate) struct Interner {
    schemas: Vec<DatasetSchema>,
    schema_index: HashMap<u64, Vec<u32>>,
    keys: Vec<String>,
    key_index: HashMap<String, u32>,
}

impl Interner {
    /// Index of `schema` in the pool. Adds the schema on the first call. A
    /// hash collision falls back to `PartialEq` over the bucket.
    pub(crate) fn intern_schema(&mut self, schema: DatasetSchema) -> u32 {
        let bucket = self.schema_index.entry(schema_hash(&schema)).or_default();
        for &i in bucket.iter() {
            if self.schemas[i as usize] == schema {
                return i;
            }
        }
        let idx = self.schemas.len() as u32;
        bucket.push(idx);
        self.schemas.push(schema);
        idx
    }

    /// Index of `key` in the attribute-key pool. Adds the key if it is new.
    pub(crate) fn intern_key(&mut self, key: &str) -> u32 {
        if let Some(&i) = self.key_index.get(key) {
            return i;
        }
        let idx = self.keys.len() as u32;
        self.key_index.insert(key.to_string(), idx);
        self.keys.push(key.to_string());
        idx
    }

    /// Consumes the interner into the two pools the footer stores.
    pub(crate) fn into_pools(self) -> (Vec<DatasetSchema>, Vec<String>) {
        (self.schemas, self.keys)
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

    /// Rejects a footer whose indices do not resolve, or whose dataset names
    /// repeat. One check here costs less than a dangling index at every use
    /// site.
    fn validate(&self) -> Result<()> {
        if self.version != super::FORMAT_VERSION {
            return Err(Error::UnsupportedVersion {
                found: self.version,
                expected: super::FORMAT_VERSION,
            });
        }
        // A name resolves to the first dataset that carries it. A repeat would
        // therefore hide the second one from every lookup, while the counts
        // still include it. The writer refuses a repeat, so one here means a
        // damaged or foreign footer.
        let mut seen: HashSet<&str> = HashSet::with_capacity(self.datasets.len());
        for (ordinal, ds) in self.datasets.iter().enumerate() {
            if !seen.insert(ds.name.as_str()) {
                return Err(Error::CorruptCollection(format!(
                    "dataset name '{}' appears twice, the second time at ordinal {ordinal}",
                    ds.name
                )));
            }
            let schema = self.schema_pool.get(ds.schema as usize).ok_or_else(|| {
                Error::CorruptCollection(format!(
                    "dataset '{}' references schema {} but the pool holds {}",
                    ds.name,
                    ds.schema,
                    self.schema_pool.len()
                ))
            })?;
            for (key, _) in ds.global_attrs.iter() {
                self.key(*key).ok_or_else(|| {
                    Error::CorruptCollection(format!(
                        "dataset '{}' references attribute key {key} out of {}",
                        ds.name,
                        self.attr_key_pool.len()
                    ))
                })?;
            }
            for (array_pos, attrs) in ds.array_attrs.iter() {
                if schema.arrays.get_index(*array_pos as usize).is_none() {
                    return Err(Error::CorruptCollection(format!(
                        "dataset '{}' annotates array {array_pos} but its schema holds {}",
                        ds.name,
                        schema.arrays.len()
                    )));
                }
                for (key, _) in attrs.iter() {
                    self.key(*key).ok_or_else(|| {
                        Error::CorruptCollection(format!(
                            "dataset '{}' references attribute key {key} out of {}",
                            ds.name,
                            self.attr_key_pool.len()
                        ))
                    })?;
                }
            }
            for (array_pos, _) in ds.array_stats.iter() {
                if schema.arrays.get_index(*array_pos as usize).is_none() {
                    return Err(Error::CorruptCollection(format!(
                        "dataset '{}' has statistics for array {array_pos} but its schema holds {}",
                        ds.name,
                        schema.arrays.len()
                    )));
                }
            }
            if ds.seg_len == 0 {
                return Err(Error::CorruptCollection(format!(
                    "dataset '{}' (ordinal {ordinal}) has an empty segment",
                    ds.name
                )));
            }
        }
        Ok(())
    }

    /// The interned attribute key at `idx`.
    pub(crate) fn key(&self, idx: u32) -> Option<&str> {
        self.attr_key_pool.get(idx as usize).map(String::as_str)
    }

    /// The schema of the dataset at `ordinal`.
    pub(crate) fn schema_of(&self, entry: &DatasetEntry) -> &DatasetSchema {
        // validate() proved the index resolves.
        &self.schema_pool[entry.schema as usize]
    }

    /// Resolves an attribute list into a name-keyed map, in stored order.
    pub(crate) fn attrs_to_map(&self, attrs: &[(u32, AttrS)]) -> IndexMap<String, Attr> {
        attrs
            .iter()
            .filter_map(|(k, v)| Some((self.key(*k)?.to_string(), v.clone().into())))
            .collect()
    }
}

/// Serde mirror of [`array_format::StatValue`]. That type implements rkyv
/// only.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum StatValueS {
    /// Signed integer.
    Int(i64),
    /// Unsigned integer.
    UInt(u64),
    /// Floating point.
    Float(f64),
    /// String or binary, as raw bytes in lexicographic order.
    Bytes(Vec<u8>),
    /// Nanoseconds since the Unix epoch.
    TimestampNs(i64),
}

impl From<StatValue> for StatValueS {
    fn from(v: StatValue) -> Self {
        match v {
            StatValue::Int(v) => Self::Int(v),
            StatValue::UInt(v) => Self::UInt(v),
            StatValue::Float(v) => Self::Float(v),
            StatValue::Bytes(v) => Self::Bytes(v),
            StatValue::TimestampNs(v) => Self::TimestampNs(v),
        }
    }
}

impl From<StatValueS> for StatValue {
    fn from(v: StatValueS) -> Self {
        match v {
            StatValueS::Int(v) => Self::Int(v),
            StatValueS::UInt(v) => Self::UInt(v),
            StatValueS::Float(v) => Self::Float(v),
            StatValueS::Bytes(v) => Self::Bytes(v),
            StatValueS::TimestampNs(v) => Self::TimestampNs(v),
        }
    }
}

/// What `array-format` recorded about one array during the write.
///
/// `null_count` counts the elements equal to the array's fill value. That is
/// how the format stores a cell nobody wrote. `row_count` is the total element
/// count across every chunk that exists.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ArrayStatsS {
    /// Smallest value across the array, or `None` for a dtype with no ordering.
    pub min: Option<StatValueS>,
    /// Largest value across the array, or `None` for a dtype with no ordering.
    pub max: Option<StatValueS>,
    /// Elements equal to the fill value.
    pub null_count: u64,
    /// Total elements across all chunks.
    pub row_count: u64,
}

impl From<&ArrayStats> for ArrayStatsS {
    fn from(s: &ArrayStats) -> Self {
        Self {
            min: s.min.clone().map(Into::into),
            max: s.max.clone().map(Into::into),
            null_count: s.null_count,
            row_count: s.row_count,
        }
    }
}

impl ArrayStatsS {
    /// Folds another dataset's statistics for the same array into these.
    ///
    /// The counts add up. Each bound takes the wider of the two. A bound that
    /// is absent yields to a bound that is present.
    pub(crate) fn merge(&mut self, other: &Self) {
        self.null_count = self.null_count.saturating_add(other.null_count);
        self.row_count = self.row_count.saturating_add(other.row_count);
        self.min = pick(self.min.take(), other.min.clone(), Bound::Min);
        self.max = pick(self.max.take(), other.max.clone(), Bound::Max);
    }

    /// Rebuilds the `array-format` form, which needs the array's name.
    pub(crate) fn to_array_stats(&self, name: &str) -> ArrayStats {
        ArrayStats {
            name: name.to_string(),
            min: self.min.clone().map(Into::into),
            max: self.max.clone().map(Into::into),
            null_count: self.null_count,
            row_count: self.row_count,
        }
    }
}

/// Which end of the range [`pick`] keeps.
#[derive(Clone, Copy)]
enum Bound {
    Min,
    Max,
}

/// The smaller or the larger of two bounds. `None` means a dataset recorded no
/// bound, so the other one stands.
fn pick(a: Option<StatValueS>, b: Option<StatValueS>, bound: Bound) -> Option<StatValueS> {
    match (a, b) {
        (None, x) | (x, None) => x,
        (Some(a), Some(b)) => {
            let keep_a = match bound {
                Bound::Min => is_le(&a, &b),
                Bound::Max => !is_le(&a, &b),
            };
            Some(if keep_a { a } else { b })
        }
    }
}

/// Orders two bounds of one variant. Two variants mean two dtypes. The caller
/// excludes those before it merges.
fn is_le(a: &StatValueS, b: &StatValueS) -> bool {
    match (a, b) {
        (StatValueS::Int(a), StatValueS::Int(b)) => a <= b,
        (StatValueS::UInt(a), StatValueS::UInt(b)) => a <= b,
        (StatValueS::Float(a), StatValueS::Float(b)) => a.total_cmp(b).is_le(),
        (StatValueS::Bytes(a), StatValueS::Bytes(b)) => a <= b,
        (StatValueS::TimestampNs(a), StatValueS::TimestampNs(b)) => a <= b,
        _ => false,
    }
}

/// Serde mirror of [`Attr`].
///
/// [`Attr`] is the public type, and carries no serde impl of its own. This is
/// its wire form. A timestamp carries its own tag, so a string that looks like
/// a date stays a string on the way back.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) enum AttrS {
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
    Binary(Vec<u8>),
    TimestampNanoseconds(i64),
    BoolList(Vec<bool>),
    Int8List(Vec<i8>),
    Int16List(Vec<i16>),
    Int32List(Vec<i32>),
    Int64List(Vec<i64>),
    UInt8List(Vec<u8>),
    UInt16List(Vec<u16>),
    UInt32List(Vec<u32>),
    UInt64List(Vec<u64>),
    Float32List(Vec<f32>),
    Float64List(Vec<f64>),
    StringList(Vec<String>),
    BinaryList(Vec<Vec<u8>>),
}

impl From<Attr> for AttrS {
    fn from(a: Attr) -> Self {
        match a {
            Attr::Bool(v) => Self::Bool(v),
            Attr::Int8(v) => Self::Int8(v),
            Attr::Int16(v) => Self::Int16(v),
            Attr::Int32(v) => Self::Int32(v),
            Attr::Int64(v) => Self::Int64(v),
            Attr::UInt8(v) => Self::UInt8(v),
            Attr::UInt16(v) => Self::UInt16(v),
            Attr::UInt32(v) => Self::UInt32(v),
            Attr::UInt64(v) => Self::UInt64(v),
            Attr::Float32(v) => Self::Float32(v),
            Attr::Float64(v) => Self::Float64(v),
            Attr::String(v) => Self::String(v),
            Attr::Binary(v) => Self::Binary(v),
            Attr::TimestampNanoseconds(v) => Self::TimestampNanoseconds(v),
            Attr::BoolList(v) => Self::BoolList(v),
            Attr::Int8List(v) => Self::Int8List(v),
            Attr::Int16List(v) => Self::Int16List(v),
            Attr::Int32List(v) => Self::Int32List(v),
            Attr::Int64List(v) => Self::Int64List(v),
            Attr::UInt8List(v) => Self::UInt8List(v),
            Attr::UInt16List(v) => Self::UInt16List(v),
            Attr::UInt32List(v) => Self::UInt32List(v),
            Attr::UInt64List(v) => Self::UInt64List(v),
            Attr::Float32List(v) => Self::Float32List(v),
            Attr::Float64List(v) => Self::Float64List(v),
            Attr::StringList(v) => Self::StringList(v),
            Attr::BinaryList(v) => Self::BinaryList(v),
        }
    }
}

impl From<AttrS> for Attr {
    fn from(a: AttrS) -> Self {
        match a {
            AttrS::Bool(v) => Self::Bool(v),
            AttrS::Int8(v) => Self::Int8(v),
            AttrS::Int16(v) => Self::Int16(v),
            AttrS::Int32(v) => Self::Int32(v),
            AttrS::Int64(v) => Self::Int64(v),
            AttrS::UInt8(v) => Self::UInt8(v),
            AttrS::UInt16(v) => Self::UInt16(v),
            AttrS::UInt32(v) => Self::UInt32(v),
            AttrS::UInt64(v) => Self::UInt64(v),
            AttrS::Float32(v) => Self::Float32(v),
            AttrS::Float64(v) => Self::Float64(v),
            AttrS::String(v) => Self::String(v),
            AttrS::Binary(v) => Self::Binary(v),
            AttrS::TimestampNanoseconds(v) => Self::TimestampNanoseconds(v),
            AttrS::BoolList(v) => Self::BoolList(v),
            AttrS::Int8List(v) => Self::Int8List(v),
            AttrS::Int16List(v) => Self::Int16List(v),
            AttrS::Int32List(v) => Self::Int32List(v),
            AttrS::Int64List(v) => Self::Int64List(v),
            AttrS::UInt8List(v) => Self::UInt8List(v),
            AttrS::UInt16List(v) => Self::UInt16List(v),
            AttrS::UInt32List(v) => Self::UInt32List(v),
            AttrS::UInt64List(v) => Self::UInt64List(v),
            AttrS::Float32List(v) => Self::Float32List(v),
            AttrS::Float64List(v) => Self::Float64List(v),
            AttrS::StringList(v) => Self::StringList(v),
            AttrS::BinaryList(v) => Self::BinaryList(v),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::ArraySchema;
    use array_format::FillValue;

    fn schema(dtype: DType, shape: Vec<usize>) -> DatasetSchema {
        let mut arrays = IndexMap::new();
        arrays.insert(
            "temperature".to_string(),
            ArraySchema {
                dtype,
                chunk_shape: shape.clone(),
                shape,
                dimension_names: vec!["x".into()],
                fill_value: None,
            },
        );
        DatasetSchema { arrays }
    }

    fn footer_with(datasets: Vec<DatasetEntry>, pool: Vec<DatasetSchema>) -> CollectionFooter {
        CollectionFooter {
            version: super::super::FORMAT_VERSION,
            segment_format: super::super::SEGMENT_FORMAT,
            codec: crate::Codec::Zstd,
            created_unix_ms: 1_700_000_000_000,
            schema_pool: pool,
            attr_key_pool: vec!["units".to_string()],
            datasets,
        }
    }

    fn entry(name: &str, schema: u32, offset: u64) -> DatasetEntry {
        DatasetEntry {
            name: name.to_string(),
            schema,
            seg_offset: offset,
            seg_len: 128,
            global_attrs: vec![(0, AttrS::Int64(1))],
            array_attrs: vec![(0, vec![(0, AttrS::String("kelvin".into()))])],
            array_stats: vec![(
                0,
                ArrayStatsS {
                    min: Some(StatValueS::Float(-1.5)),
                    max: Some(StatValueS::Float(31.0)),
                    null_count: 2,
                    row_count: 32,
                },
            )],
        }
    }

    #[test]
    fn footer_roundtrips_through_msgpack_and_zstd() {
        let f = footer_with(
            vec![entry("a", 0, 8), entry("b", 0, 136)],
            vec![schema(DType::Float32, vec![4])],
        );
        let bytes = f.encode().unwrap();
        assert_eq!(CollectionFooter::decode(&bytes).unwrap(), f);
    }

    #[test]
    fn identical_schemas_intern_to_one_pool_entry() {
        let mut interner = Interner::default();
        let a = interner.intern_schema(schema(DType::Float32, vec![4]));
        let b = interner.intern_schema(schema(DType::Float32, vec![4]));
        let c = interner.intern_schema(schema(DType::Int64, vec![4]));
        assert_eq!(a, b);
        assert_ne!(a, c);
        let (schemas, _) = interner.into_pools();
        assert_eq!(schemas.len(), 2);
    }

    #[test]
    fn attribute_keys_intern() {
        let mut interner = Interner::default();
        assert_eq!(interner.intern_key("units"), 0);
        assert_eq!(interner.intern_key("source"), 1);
        assert_eq!(interner.intern_key("units"), 0);
        let (_, keys) = interner.into_pools();
        assert_eq!(keys, vec!["units".to_string(), "source".to_string()]);
    }

    #[test]
    fn every_attr_variant_roundtrips() {
        let cases = vec![
            Attr::Bool(true),
            Attr::Int8(-3),
            Attr::Int16(-300),
            Attr::Int32(-70_000),
            Attr::Int64(-5_000_000_000),
            Attr::UInt8(200),
            Attr::UInt16(60_000),
            Attr::UInt32(4_000_000_000),
            Attr::UInt64(18_000_000_000_000_000_000),
            Attr::Float32(2.5),
            Attr::Float64(-0.125),
            Attr::String("hello".into()),
            Attr::Binary(vec![0xde, 0xad]),
            Attr::TimestampNanoseconds(1_700_000_000_000_000_000),
            Attr::BoolList(vec![true, false]),
            Attr::Int8List(vec![1, -1]),
            Attr::Int16List(vec![2]),
            Attr::Int32List(vec![1, 2, 3]),
            Attr::Int64List(vec![4]),
            Attr::UInt8List(vec![5]),
            Attr::UInt16List(vec![6]),
            Attr::UInt32List(vec![7]),
            Attr::UInt64List(vec![8]),
            Attr::Float32List(vec![1.5]),
            Attr::Float64List(vec![0.0, 1.5]),
            Attr::StringList(vec!["a".into(), "b".into()]),
            Attr::BinaryList(vec![vec![1], vec![2]]),
        ];
        for v in cases {
            let wire = AttrS::from(v.clone());
            let packed = rmp_serde::to_vec(&wire).unwrap();
            let back: AttrS = rmp_serde::from_slice(&packed).unwrap();
            assert_eq!(Attr::from(back), v);
        }
    }

    #[test]
    fn rfc3339_strings_stay_strings() {
        // A timestamp has its own tag, so nothing guesses at a date-shaped
        // string.
        let v = Attr::String("2023-11-14T22:13:20Z".into());
        let back: Attr = AttrS::from(v.clone()).into();
        assert_eq!(back, v);
    }

    #[test]
    fn fill_values_roundtrip() {
        use crate::schema::FillValueS;
        let cases = vec![
            FillValue::Bool(true),
            FillValue::Int(-7),
            FillValue::UInt(7),
            FillValue::Float(f64::NAN),
            FillValue::String("".into()),
            FillValue::TimestampNs(i64::MIN),
        ];
        for v in cases {
            let wire = FillValueS::from(v.clone());
            let packed = rmp_serde::to_vec(&wire).unwrap();
            let back: FillValue = rmp_serde::from_slice::<FillValueS>(&packed).unwrap().into();
            // FillValue compares floats by bit pattern, so NaN == NaN here.
            assert_eq!(back, v);
        }
    }

    #[test]
    fn dangling_schema_index_is_corruption() {
        let f = footer_with(
            vec![entry("a", 3, 8)],
            vec![schema(DType::Float32, vec![4])],
        );
        let bytes = f.encode().unwrap();
        assert!(matches!(
            CollectionFooter::decode(&bytes),
            Err(Error::CorruptCollection(_))
        ));
    }

    #[test]
    fn dangling_attribute_key_is_corruption() {
        let mut e = entry("a", 0, 8);
        e.global_attrs = vec![(9, AttrS::Int64(1))];
        let f = footer_with(vec![e], vec![schema(DType::Float32, vec![4])]);
        let bytes = f.encode().unwrap();
        assert!(matches!(
            CollectionFooter::decode(&bytes),
            Err(Error::CorruptCollection(_))
        ));
    }

    #[test]
    fn dangling_array_position_is_corruption() {
        let mut e = entry("a", 0, 8);
        e.array_attrs = vec![(4, vec![])];
        let f = footer_with(vec![e], vec![schema(DType::Float32, vec![4])]);
        let bytes = f.encode().unwrap();
        assert!(matches!(
            CollectionFooter::decode(&bytes),
            Err(Error::CorruptCollection(_))
        ));
    }

    #[test]
    fn a_repeated_dataset_name_is_corruption() {
        // Two datasets of one name make the second unreachable, because every
        // lookup resolves to the first.
        let f = footer_with(
            vec![entry("a", 0, 8), entry("a", 0, 64)],
            vec![schema(DType::Float32, vec![4])],
        );
        let bytes = f.encode().unwrap();
        let err = CollectionFooter::decode(&bytes).unwrap_err();
        assert!(matches!(err, Error::CorruptCollection(_)));
        assert!(err.to_string().contains("appears twice"), "{err}");
    }

    #[test]
    fn distinct_dataset_names_pass() {
        let f = footer_with(
            vec![entry("a", 0, 8), entry("b", 0, 64)],
            vec![schema(DType::Float32, vec![4])],
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
        let f = footer_with(vec![], vec![]);
        let bytes = f.encode().unwrap();
        assert_eq!(CollectionFooter::decode(&bytes).unwrap(), f);
    }
}
