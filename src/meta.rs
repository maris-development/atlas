use std::sync::Arc;

use indexmap::IndexMap;
use object_store::{ObjectStore, ObjectStoreExt, path::Path};
use serde::{Deserialize, Serialize};
use tracing::warn;

use array_format::DType;

use crate::{
    Error, Result,
    config::{Codec, META_VARIANTS, MetaFormat},
    schema::{ArraySchema, DTypeS, widen_dtype},
};

/// Current on-disk store-format version. `atlas.json` written by an older
/// atlas (which inlined per-dataset attributes and duplicated schemas) is not
/// read by this version — see [`decode`].
pub(crate) const STORE_FORMAT_VERSION: u32 = 2;

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
/// through [`StoreMeta::record_global_attr`] / [`StoreMeta::record_array_attr`]
/// so the type index stays in sync.
#[cfg(test)]
impl DatasetSchema {
    /// Record (or update the type of) a global attribute key.
    pub(crate) fn register_global_attr(&mut self, key: &str, ty: DType) {
        self.global_attrs.insert(key.to_string(), DTypeS(ty));
    }

    /// Record (or update the type of) a per-variable attribute key on `array`.
    pub(crate) fn register_array_attr(&mut self, array: &str, key: &str, ty: DType) {
        self.array_attrs
            .entry(array.to_string())
            .or_default()
            .insert(key.to_string(), DTypeS(ty));
    }
}

/// Feed a [`DType`] into a hasher. `DType` comes from `array_format` and
/// derives neither `Hash` nor `Eq` (it holds no floats — this is just a
/// missing derive), so schema hashing spells it out here.
fn hash_dtype<H: std::hash::Hasher>(dtype: &DType, state: &mut H) {
    use std::hash::Hash;
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

/// Content hash of a dataset schema, consistent with its `PartialEq`: two
/// schemas that compare equal hash equal. Used to intern identical schemas.
fn schema_hash(schema: &DatasetSchema) -> u64 {
    use std::hash::{Hash, Hasher};
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

/// Fold every dataset's schema into the collection-wide merged schema.
/// Type collisions are widened where possible; insert-time validation
/// (in `DatasetView`) guarantees only widenable types ever reach here.
pub(crate) fn compute_merged(datasets: &IndexMap<String, Arc<DatasetSchema>>) -> MergedSchema {
    let mut merged = MergedSchema::default();
    for schema in datasets.values() {
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

/// What the insert-time type check should compare against.
enum Constraint {
    /// No other dataset uses this key — anything goes.
    Unconstrained,
    /// Compare against this merged type.
    Type(DType),
    /// The fast path can't answer exactly; scan every dataset. Only reachable
    /// for keys that hold mutually non-widenable types (see [`MergedType`]).
    NeedsScan,
}

/// The merged type recorded for one key across every dataset in the store,
/// plus how many datasets contribute it.
///
/// This is the collection's merged schema for that key, maintained
/// incrementally: each new declaration is folded into `dtype` rather than
/// recomputed by scanning every dataset. The fold is **first-seen-wins** — a
/// type that can't merge leaves `dtype` alone — matching [`compute_merged`] and
/// keeping a stored mismatch from becoming the reference type.
///
/// The insert-time check asks "what type do the *other* datasets use?", which
/// the merged type alone can't answer — widening is lossy, so a dataset's own
/// contribution can't be subtracted back out. `contributors` closes that gap
/// whenever all types recorded for the key are pairwise widenable: folding the
/// current dataset's own type in never changes the accept/reject decision
/// (widening within the numeric lattice is monotone, and `String`/`List`/`Bool`
/// are incompatible with numerics either way), so the only case that matters is
/// the current dataset being the sole contributor — a count of 1.
///
/// `TypeMismatchPolicy::Warn` (the default) breaks that premise: a mismatching
/// dataset is still stored, so a key can end up holding non-widenable types.
/// Once that happens, excluding a dataset genuinely can change the answer — if
/// the first-seen contributor is the one being excluded, the reference type
/// shifts to whatever another dataset holds. `conflicted` marks those keys and
/// sends them down the exact scan instead. It is set only by a real mismatch,
/// so the common case stays O(1).
#[derive(Debug, Default, Clone)]
pub(crate) struct MergedType {
    dtype: Option<DType>,
    contributors: usize,
    conflicted: bool,
}

impl MergedType {
    /// Fold one dataset's declaration into the merged type, first-seen-wins.
    fn add(&mut self, ty: &DType) {
        self.dtype = Some(match self.dtype.take() {
            None => ty.clone(),
            Some(a) => match widen_dtype(&a, ty) {
                Some(w) => w,
                None => {
                    // Storable under Warn, so record it and stop trusting the
                    // O(1) exclusion shortcut for this key.
                    self.conflicted = true;
                    a
                }
            },
        });
        self.contributors += 1;
    }

    fn constraint(&self, owner_declares: bool) -> Constraint {
        let others = self.contributors - usize::from(owner_declares);
        if others == 0 {
            return Constraint::Unconstrained;
        }
        if self.conflicted {
            return Constraint::NeedsScan;
        }
        match &self.dtype {
            Some(d) => Constraint::Type(d.clone()),
            None => Constraint::Unconstrained,
        }
    }
}

/// Per-array index entry: the merged dtype for this array name across all
/// datasets, plus the codec of the physical `.af` file that backs it.
#[derive(Debug, Default, Clone)]
pub(crate) struct ArrayIndexEntry {
    dtype: MergedType,
    codec: Codec,
}

/// Incrementally-maintained reverse index over every dataset schema in the
/// store, so adding dataset *N* costs the same as adding dataset 1.
///
/// Without it, each `define_array` / `set_attribute` / `set_array_attribute`
/// scanned all N datasets to find the type already recorded for that key
/// elsewhere, making a bulk ingest O(N² · keys-per-dataset) — the dominant
/// cost when writing thousands of small, attribute-heavy datasets (a NetCDF
/// folder ingest sets ~170 keys per file).
///
/// Kept in sync by the `record_*` / `unrecord_*` methods on [`StoreMeta`];
/// [`StoreMeta::rebuild_index`] recomputes it from scratch after a bulk load.
#[derive(Debug, Default, Clone)]
pub(crate) struct TypeIndex {
    arrays: std::collections::HashMap<String, ArrayIndexEntry>,
    global_attrs: std::collections::HashMap<String, MergedType>,
    array_attrs: std::collections::HashMap<(String, String), MergedType>,
}

#[derive(Debug, Default, Clone)]
pub(crate) struct StoreMeta {
    pub version: u32,
    /// Codec used when new arrays are defined in this store.
    pub codec: Codec,
    /// Dataset name → schema. Insertion-ordered.
    ///
    /// Mutate through the `record_*` / `unrecord_*` methods rather than
    /// directly, so [`StoreMeta::index`] stays consistent. Inserting an empty
    /// [`DatasetSchema`] directly is safe (it contributes nothing to the
    /// index); anything else must be followed by [`Self::rebuild_index`].
    pub datasets: IndexMap<String, Arc<DatasetSchema>>,
    /// Derived from `datasets` — never set directly; see [`Self::rebuild_index`].
    pub(crate) index: TypeIndex,
    /// Interning pool of distinct schemas, keyed by content hash.
    ///
    /// A collection of homogeneous datasets (the common case — one NetCDF
    /// folder, one variable layout) holds a single `DatasetSchema` allocation
    /// shared by every dataset, instead of one deep copy each. The on-disk
    /// format already interns this way; without the pool the in-memory
    /// representation was the larger of the two by far.
    ///
    /// Populated by [`Self::seal_dataset`]. Mutations use `Arc::make_mut`, so a
    /// dataset that is edited after interning transparently copies out of the
    /// pool first.
    pub(crate) schema_pool: std::collections::HashMap<u64, Vec<Arc<DatasetSchema>>>,
}

impl StoreMeta {
    /// Recompute the type index from `datasets`. O(total keys); called after a
    /// bulk load or a removal, never on the per-dataset write path.
    pub(crate) fn rebuild_index(&mut self) {
        let mut index = TypeIndex::default();
        for schema in self.datasets.values() {
            for (name, arr) in &schema.arrays {
                index
                    .arrays
                    .entry(name.clone())
                    .or_insert_with(|| ArrayIndexEntry {
                        dtype: MergedType::default(),
                        codec: arr.codec.clone(),
                    })
                    .dtype
                    .add(&arr.dtype);
            }
            for (key, ty) in &schema.global_attrs {
                index.global_attrs.entry(key.clone()).or_default().add(&ty.0);
            }
            for (array, attrs) in &schema.array_attrs {
                for (key, ty) in attrs {
                    index
                        .array_attrs
                        .entry((array.clone(), key.clone()))
                        .or_default()
                        .add(&ty.0);
                }
            }
        }
        self.index = index;
    }

    /// Merge a key's type across every dataset **except** `exclude`, by
    /// scanning them all. The exact reference definition; the type index is a
    /// fast path over it, and falls back here for conflicted keys.
    ///
    /// Folds like [`compute_merged`]: compatible types widen, and a type that
    /// can't merge leaves the accumulator alone so the **first-seen** type
    /// wins. Keeping the two in step matters — once a mismatching dataset is
    /// stored (under `TypeMismatchPolicy::Warn`), a last-wins fold here would
    /// silently adopt the odd type out and stop reporting further mismatches.
    fn scan_other<F>(&self, exclude: &str, mut pick: F) -> Option<DType>
    where
        F: FnMut(&DatasetSchema) -> Option<DType>,
    {
        let mut acc: Option<DType> = None;
        for (name, schema) in &self.datasets {
            if name == exclude {
                continue;
            }
            if let Some(t) = pick(schema) {
                acc = Some(match acc {
                    None => t,
                    Some(a) => widen_dtype(&a, &t).unwrap_or(a),
                });
            }
        }
        acc
    }

    /// The type constraint a write of `array` from dataset `exclude` must
    /// satisfy: the merged type across the collection, or `None` if `exclude`
    /// is the only dataset declaring it.
    pub(crate) fn other_array_dtype(&self, exclude: &str, array: &str) -> Option<DType> {
        let owner_declares = self
            .datasets
            .get(exclude)
            .is_some_and(|s| s.arrays.contains_key(array));
        match self.index.arrays.get(array)?.dtype.constraint(owner_declares) {
            Constraint::Unconstrained => None,
            Constraint::Type(d) => Some(d),
            Constraint::NeedsScan => {
                self.scan_other(exclude, |s| s.arrays.get(array).map(|a| a.dtype.clone()))
            }
        }
    }

    /// As [`Self::other_array_dtype`], for a dataset-global attribute key.
    pub(crate) fn other_global_attr_dtype(&self, exclude: &str, key: &str) -> Option<DType> {
        let owner_declares = self
            .datasets
            .get(exclude)
            .is_some_and(|s| s.global_attrs.contains_key(key));
        match self.index.global_attrs.get(key)?.constraint(owner_declares) {
            Constraint::Unconstrained => None,
            Constraint::Type(d) => Some(d),
            Constraint::NeedsScan => {
                self.scan_other(exclude, |s| s.global_attrs.get(key).map(|d| d.0.clone()))
            }
        }
    }

    /// As [`Self::other_array_dtype`], for a per-variable attribute key.
    pub(crate) fn other_array_attr_dtype(
        &self,
        exclude: &str,
        array: &str,
        key: &str,
    ) -> Option<DType> {
        let owner_declares = self
            .datasets
            .get(exclude)
            .and_then(|s| s.array_attrs.get(array))
            .is_some_and(|m| m.contains_key(key));
        match self
            .index
            .array_attrs
            .get(&(array.to_string(), key.to_string()))?
            .constraint(owner_declares)
        {
            Constraint::Unconstrained => None,
            Constraint::Type(d) => Some(d),
            Constraint::NeedsScan => self.scan_other(exclude, |s| {
                s.array_attrs
                    .get(array)
                    .and_then(|m| m.get(key))
                    .map(|d| d.0.clone())
            }),
        }
    }

    /// Codec of the physical file backing `array`, from the first dataset that
    /// declared it. O(1); replaces a scan over every dataset.
    pub(crate) fn array_file_codec(&self, array: &str) -> Option<Codec> {
        self.index.arrays.get(array).map(|e| e.codec.clone())
    }

    /// Declare `array` in `dataset`, folding its dtype into the index.
    pub(crate) fn record_array(&mut self, dataset: &str, array: &str, schema: ArraySchema) {
        let dtype = schema.dtype.clone();
        let codec = schema.codec.clone();
        let previous = self
            .schema_mut(dataset)
            .arrays
            .insert(array.to_string(), schema);
        if retyped(previous.as_ref().map(|p| &p.dtype), &dtype) {
            // Rare: this dataset changed the array's type. The merged type
            // can't un-widen, so recompute it exactly.
            self.rebuild_index();
            return;
        }
        if previous.is_none() {
            self.index
                .arrays
                .entry(array.to_string())
                .or_insert_with(|| ArrayIndexEntry {
                    dtype: MergedType::default(),
                    codec,
                })
                .dtype
                .add(&dtype);
        }
    }

    /// Record (or retype) a dataset-global attribute key, updating the index.
    pub(crate) fn record_global_attr(&mut self, dataset: &str, key: &str, ty: DType) {
        let previous = self
            .schema_mut(dataset)
            .global_attrs
            .insert(key.to_string(), DTypeS(ty.clone()));
        if retyped(previous.as_ref().map(|p| &p.0), &ty) {
            self.rebuild_index();
            return;
        }
        if previous.is_none() {
            self.index
                .global_attrs
                .entry(key.to_string())
                .or_default()
                .add(&ty);
        }
    }

    /// Record (or retype) a per-variable attribute key, updating the index.
    pub(crate) fn record_array_attr(&mut self, dataset: &str, array: &str, key: &str, ty: DType) {
        let previous = self
            .schema_mut(dataset)
            .array_attrs
            .entry(array.to_string())
            .or_default()
            .insert(key.to_string(), DTypeS(ty.clone()));
        if retyped(previous.as_ref().map(|p| &p.0), &ty) {
            self.rebuild_index();
            return;
        }
        if previous.is_none() {
            self.index
                .array_attrs
                .entry((array.to_string(), key.to_string()))
                .or_default()
                .add(&ty);
        }
    }

    /// Drop `array` (and its attribute keys) from `dataset`.
    pub(crate) fn unrecord_array(&mut self, dataset: &str, array: &str) {
        if self.datasets.contains_key(dataset) {
            let ds = self.schema_mut(dataset);
            ds.arrays.shift_remove(array);
            ds.array_attrs.shift_remove(array);
        }
        self.rebuild_index();
    }

    /// Remove `dataset` entirely. Returns its schema.
    pub(crate) fn unrecord_dataset(&mut self, dataset: &str) -> Option<Arc<DatasetSchema>> {
        let schema = self.datasets.shift_remove(dataset)?;
        self.rebuild_index();
        self.prune_schema_pool();
        Some(schema)
    }

    /// Mutable access to a dataset's schema, copying it out of the interning
    /// pool first if it is shared (`Arc::make_mut`). While a dataset is being
    /// built its schema is unshared, so this is a plain deref — the copy only
    /// happens when an already-sealed dataset is edited again.
    fn schema_mut(&mut self, dataset: &str) -> &mut DatasetSchema {
        Arc::make_mut(self.datasets.entry(dataset.to_string()).or_default())
    }

    /// Intern `dataset`'s schema: replace it with an identical one from the
    /// pool if there is one, otherwise add it to the pool.
    ///
    /// Called when a [`DatasetView`](crate::DatasetView) is dropped, i.e. once
    /// the dataset is fully written. Costs one hash plus one equality compare
    /// over the schema, both O(keys), once per dataset.
    pub(crate) fn seal_dataset(&mut self, dataset: &str) {
        let Some(schema) = self.datasets.get(dataset) else {
            return;
        };
        if Arc::strong_count(schema) > 1 {
            return; // already interned
        }
        let hash = schema_hash(schema);
        let bucket = self.schema_pool.entry(hash).or_default();
        match bucket.iter().find(|pooled| ***pooled == **schema) {
            Some(pooled) => {
                let shared = pooled.clone();
                self.datasets.insert(dataset.to_string(), shared);
            }
            None => bucket.push(schema.clone()),
        }
    }

    /// Drop pooled schemas no dataset references any more.
    fn prune_schema_pool(&mut self) {
        for bucket in self.schema_pool.values_mut() {
            bucket.retain(|s| Arc::strong_count(s) > 1);
        }
        self.schema_pool.retain(|_, bucket| !bucket.is_empty());
    }
}

/// `true` if `previous` exists and differs from `new` — the only case where an
/// incremental index update is not enough, since a merged type can widen but
/// never narrow back.
fn retyped(previous: Option<&DType>, new: &DType) -> bool {
    previous.is_some_and(|p| p != new)
}

/// On-disk wire form of [`StoreMeta`]. Identical dataset schemas are
/// **interned**: each distinct [`DatasetSchema`] is stored once in `schemas`
/// and every dataset references it by index. A collection of many homogeneous
/// datasets (same variables/shapes/attribute keys) therefore stores its schema
/// only once, no matter how many datasets share it.
#[derive(Serialize, Deserialize)]
struct StoreMetaWire {
    version: u32,
    #[serde(default)]
    codec: Codec,
    /// Pool of distinct dataset schemas, in first-seen order.
    #[serde(default)]
    schemas: Vec<DatasetSchema>,
    /// Dataset name → index into `schemas`. Insertion-ordered.
    #[serde(default)]
    datasets: IndexMap<String, usize>,
    /// Collection-wide merged schema (every unique array/attribute with
    /// widened types). Derived from `schemas`; written for external tooling
    /// and ignored on load (the per-dataset schemas are the source of truth).
    #[serde(default)]
    merged: MergedSchema,
}

/// Minimal probe used to read the format version before attempting a full
/// decode, so a store written by an older atlas produces a clear error rather
/// than an opaque parse failure.
#[derive(Deserialize)]
struct MetaVersion {
    #[serde(default)]
    version: u32,
}

/// Build the interned wire form from in-memory metadata.
///
/// Schemas are already interned in memory, so deduplication is a pointer-keyed
/// hash lookup. Distinct-but-equal schemas (possible only for datasets never
/// sealed) are caught by a fallback equality scan over the pool built so far.
fn to_wire(meta: &StoreMeta) -> StoreMetaWire {
    let mut schemas: Vec<DatasetSchema> = Vec::new();
    let mut by_ptr: std::collections::HashMap<*const DatasetSchema, usize> =
        std::collections::HashMap::new();
    let mut datasets: IndexMap<String, usize> = IndexMap::with_capacity(meta.datasets.len());
    for (name, schema) in &meta.datasets {
        let ptr = Arc::as_ptr(schema);
        let idx = match by_ptr.get(&ptr) {
            Some(&i) => i,
            None => {
                let i = match schemas.iter().position(|s| s == &**schema) {
                    Some(i) => i,
                    None => {
                        schemas.push((**schema).clone());
                        schemas.len() - 1
                    }
                };
                by_ptr.insert(ptr, i);
                i
            }
        };
        datasets.insert(name.clone(), idx);
    }
    let merged = compute_merged(&meta.datasets);
    StoreMetaWire {
        version: meta.version,
        codec: meta.codec,
        schemas,
        datasets,
        merged,
    }
}

/// Expand the interned wire form back into per-dataset schemas, sharing the
/// pooled schema for every dataset that references the same index.
fn from_wire(wire: StoreMetaWire) -> Result<StoreMeta> {
    // One allocation per distinct schema, shared by every dataset that uses it
    // — the load-side half of the in-memory interning.
    let pooled: Vec<Arc<DatasetSchema>> = wire.schemas.into_iter().map(Arc::new).collect();
    let mut datasets: IndexMap<String, Arc<DatasetSchema>> =
        IndexMap::with_capacity(wire.datasets.len());
    for (name, idx) in wire.datasets {
        let schema = pooled.get(idx).cloned().ok_or_else(|| {
            Error::ArrayFormat(array_format::Error::Storage(format!(
                "corrupt metadata: dataset '{name}' references schema index {idx} of {}",
                pooled.len()
            )))
        })?;
        datasets.insert(name, schema);
    }
    // Seed the interning pool so datasets added after a reopen can share these
    // schemas rather than starting a second copy of each.
    let mut schema_pool: std::collections::HashMap<u64, Vec<Arc<DatasetSchema>>> =
        std::collections::HashMap::new();
    for schema in pooled {
        schema_pool
            .entry(schema_hash(&schema))
            .or_default()
            .push(schema);
    }
    let mut meta = StoreMeta {
        version: wire.version,
        codec: wire.codec,
        datasets,
        schema_pool,
        ..Default::default()
    };
    meta.rebuild_index();
    meta.prune_schema_pool();
    Ok(meta)
}

/// Load store metadata, auto-detecting both the encoding format and the
/// compression from the on-disk filename.
///
/// Uses a single [`ObjectStore::list_with_delimiter`] to enumerate the
/// top-level files and matches them against the six known
/// `atlas.{json,msgpack}{,.zst,.lz4}` filenames. If more than one matches
/// (shouldn't happen unless the directory was hand-edited), the warning
/// names them and the priority order in
/// [`META_VARIANTS`](crate::config::META_VARIANTS) decides — uncompressed
/// before compressed within each format, JSON before MsgPack overall.
///
/// If no metadata file is found, returns the default (empty) metadata with
/// `(Json, Uncompressed)` so a freshly-created store gets the legacy
/// `atlas.json` filename on its first save.
///
/// The returned `(MetaFormat, Codec)` is what subsequent saves should use so
/// the same file is overwritten instead of leaving stale copies behind.
pub(crate) async fn load_meta(
    store: &Arc<dyn ObjectStore>,
) -> Result<(StoreMeta, MetaFormat, Codec)> {
    let listing = store
        .list_with_delimiter(None)
        .await
        .map_err(Error::ObjectStore)?;

    // Collect filenames present at the root.
    let present: std::collections::HashSet<&str> = listing
        .objects
        .iter()
        .filter_map(|o| o.location.filename())
        .collect();

    let matches: Vec<(MetaFormat, Codec)> = META_VARIANTS
        .iter()
        .copied()
        .filter(|&(fmt, comp)| present.contains(fmt.filename(comp)))
        .collect();

    let (format, compression) = match matches.as_slice() {
        [] => return Ok((StoreMeta::default(), MetaFormat::Json, Codec::Uncompressed)),
        [single] => *single,
        many => {
            let names: Vec<&str> = many.iter().map(|&(f, c)| f.filename(c)).collect();
            let chosen = many[0];
            warn!(
                "multiple metadata files present ({names:?}); loading {} by priority order",
                chosen.0.filename(chosen.1)
            );
            chosen
        }
    };

    let bytes = store
        .get(&Path::from(format.filename(compression)))
        .await
        .map_err(Error::ObjectStore)?
        .bytes()
        .await
        .map_err(Error::ObjectStore)?;
    let raw = decompress(&bytes, compression)?;
    let meta = decode(&raw, format)?;
    Ok((meta, format, compression))
}

fn decode(bytes: &[u8], format: MetaFormat) -> Result<StoreMeta> {
    // Read the version first so a store written by an older atlas fails with a
    // clear message instead of an opaque schema-shape parse error.
    let probe: MetaVersion = match format {
        MetaFormat::Json => serde_json::from_slice(bytes)?,
        MetaFormat::MsgPack => rmp_serde::from_slice(bytes)?,
    };
    if probe.version != STORE_FORMAT_VERSION {
        return Err(Error::UnsupportedVersion {
            found: probe.version,
            expected: STORE_FORMAT_VERSION,
        });
    }
    let wire: StoreMetaWire = match format {
        MetaFormat::Json => serde_json::from_slice(bytes)?,
        MetaFormat::MsgPack => rmp_serde::from_slice(bytes)?,
    };
    from_wire(wire)
}

fn encode(meta: &StoreMeta, format: MetaFormat) -> Result<Vec<u8>> {
    let wire = to_wire(meta);
    match format {
        MetaFormat::Json => Ok(serde_json::to_vec_pretty(&wire)?),
        MetaFormat::MsgPack => Ok(rmp_serde::to_vec_named(&wire)?),
    }
}

fn compress(bytes: Vec<u8>, codec: Codec) -> Result<Vec<u8>> {
    match codec {
        Codec::Uncompressed => Ok(bytes),
        // zstd default level (3) — good ratio at low CPU. Metadata is small,
        // so even level 19 would be sub-millisecond, but the default is fine.
        Codec::Zstd => Ok(zstd::stream::encode_all(bytes.as_slice(), 0)?),
        // lz4_flex compression is infallible; size prefix lets decode know the
        // output length without scanning.
        Codec::Lz4 => Ok(lz4_flex::compress_prepend_size(&bytes)),
    }
}

fn decompress(bytes: &[u8], codec: Codec) -> Result<Vec<u8>> {
    match codec {
        Codec::Uncompressed => Ok(bytes.to_vec()),
        Codec::Zstd => Ok(zstd::stream::decode_all(bytes)?),
        Codec::Lz4 => Ok(lz4_flex::decompress_size_prepended(bytes)?),
    }
}

pub(crate) async fn save_meta(
    store: &Arc<dyn ObjectStore>,
    meta: &StoreMeta,
    format: MetaFormat,
    compression: Codec,
) -> Result<()> {
    let encoded = encode(meta, format)?;
    let bytes = compress(encoded, compression)?;
    store
        .put(&Path::from(format.filename(compression)), bytes.into())
        .await
        .map_err(Error::ObjectStore)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Codec;
    use crate::schema::Attr;
    use array_format::{AttributeValue, DType};
    use object_store::memory::InMemory;

    /// Independent reference implementation of "merge this key's type across
    /// every dataset except `exclude`", written out separately from
    /// [`StoreMeta::scan_other`] so the test is a real cross-check rather than
    /// a tautology. First-seen-wins, matching [`compute_merged`].
    fn merged_other_by_scan<F>(meta: &StoreMeta, exclude: &str, mut pick: F) -> Option<DType>
    where
        F: FnMut(&DatasetSchema) -> Option<DType>,
    {
        let mut acc: Option<DType> = None;
        for (name, schema) in &meta.datasets {
            if name == exclude {
                continue;
            }
            if let Some(t) = pick(schema) {
                acc = Some(match acc {
                    None => t,
                    Some(a) => widen_dtype(&a, &t).unwrap_or(a),
                });
            }
        }
        acc
    }

    fn array_of(dtype: DType) -> ArraySchema {
        ArraySchema {
            dtype,
            shape: vec![1],
            chunk_shape: vec![1],
            dimension_names: vec!["i".into()],
            codec: Codec::default(),
        }
    }

    /// The index must give the same accept/reject answer as a full scan for
    /// every dataset/key combination, including the sole-contributor case and
    /// after deletions.
    #[test]
    fn index_matches_full_scan() {
        let dtypes = [
            DType::Int32,
            DType::Int64,
            DType::Float64,
            DType::UInt8,
            DType::String,
            DType::TimestampNs,
        ];
        let mut meta = StoreMeta::default();

        // Every dataset declares `x`, `k` and `units` with a rotating dtype.
        // Mismatching types are deliberately *kept*: under the default
        // `TypeMismatchPolicy::Warn` they really are stored, so the index has
        // to stay exact across conflicted keys too.
        for (i, dtype) in dtypes.iter().cycle().take(24).enumerate() {
            let ds = format!("ds{i}");
            meta.record_array(&ds, "x", array_of(dtype.clone()));
            meta.record_global_attr(&ds, "k", dtype.clone());
            meta.record_array_attr(&ds, "x", "units", dtype.clone());
        }
        assert!(meta.datasets.len() > 1, "fixture should hold datasets");
        assert!(
            meta.index.arrays["x"].dtype.conflicted,
            "fixture should exercise the conflicted path"
        );

        let check = |meta: &StoreMeta| {
            // Probe from every existing dataset plus one that isn't there.
            let mut names: Vec<String> = meta.datasets.keys().cloned().collect();
            names.push("absent".into());
            for name in &names {
                assert_eq!(
                    meta.other_array_dtype(name, "x"),
                    merged_other_by_scan(meta, name, |s| s.arrays.get("x").map(|a| a.dtype.clone())),
                    "array dtype mismatch for {name}"
                );
                assert_eq!(
                    meta.other_global_attr_dtype(name, "k"),
                    merged_other_by_scan(meta, name, |s| s.global_attrs.get("k").map(|d| d.0.clone())),
                    "global attr mismatch for {name}"
                );
                assert_eq!(
                    meta.other_array_attr_dtype(name, "x", "units"),
                    merged_other_by_scan(meta, name, |s| s
                        .array_attrs
                        .get("x")
                        .and_then(|m| m.get("units"))
                        .map(|d| d.0.clone())),
                    "array attr mismatch for {name}"
                );
            }
        };

        check(&meta);

        // Deleting must narrow the constraint back, not leave it widened.
        let victims: Vec<String> = meta.datasets.keys().take(2).cloned().collect();
        for v in victims {
            meta.unrecord_dataset(&v);
            check(&meta);
        }

        // Dropping the array from a dataset clears its attribute keys too.
        if let Some(name) = meta.datasets.keys().next().cloned() {
            meta.unrecord_array(&name, "x");
            check(&meta);
        }
    }

    /// Datasets with identical schemas must end up sharing one allocation, and
    /// editing one afterwards must not disturb the others.
    #[test]
    fn identical_schemas_share_one_allocation() {
        let mut meta = StoreMeta::default();
        for i in 0..100 {
            let ds = format!("ds{i}");
            meta.record_array(&ds, "temp", array_of(DType::Float64));
            meta.record_global_attr(&ds, "title", DType::String);
            meta.record_array_attr(&ds, "temp", "units", DType::String);
            meta.seal_dataset(&ds);
        }

        let first = meta.datasets["ds0"].clone();
        assert!(
            meta.datasets
                .values()
                .all(|s| Arc::ptr_eq(s, &first)),
            "all 100 identical schemas should share one allocation"
        );
        assert_eq!(meta.schema_pool.values().map(Vec::len).sum::<usize>(), 1);

        // Copy-on-write: editing one dataset must not touch its neighbours.
        meta.record_global_attr("ds7", "extra", DType::Int64);
        assert!(!Arc::ptr_eq(&meta.datasets["ds7"], &first));
        assert!(meta.datasets["ds7"].global_attrs.contains_key("extra"));
        assert!(!meta.datasets["ds0"].global_attrs.contains_key("extra"));
        assert!(Arc::ptr_eq(&meta.datasets["ds0"], &first));

        // A genuinely different schema gets its own pool entry.
        meta.seal_dataset("ds7");
        assert_eq!(meta.schema_pool.values().map(Vec::len).sum::<usize>(), 2);

        // Interning must not change what gets written to disk.
        let wire = to_wire(&meta);
        assert_eq!(wire.schemas.len(), 2);
        assert_eq!(wire.datasets.len(), 100);
        assert_ne!(wire.datasets["ds7"], wire.datasets["ds0"]);
    }

    /// Under `TypeMismatchPolicy::Warn` a key can hold non-widenable types.
    /// Excluding the *first-seen* contributor then genuinely shifts the
    /// reference type, which the merged-type fast path cannot represent — it
    /// must fall through to the exact scan.
    #[test]
    fn excluding_first_seen_contributor_of_a_conflicted_key() {
        let mut meta = StoreMeta::default();
        meta.record_global_attr("a", "k", DType::Int64);
        meta.record_global_attr("odd", "k", DType::String);

        // Merged (first-seen wins) is Int64, and that is what a third dataset
        // is checked against.
        assert_eq!(meta.other_global_attr_dtype("third", "k"), Some(DType::Int64));

        // But for "a" itself the constraint is only what the others hold —
        // String — not the merged type that "a" contributed to.
        assert_eq!(meta.other_global_attr_dtype("a", "k"), Some(DType::String));
        assert_eq!(meta.other_global_attr_dtype("odd", "k"), Some(DType::Int64));

        // All three agree with a full scan.
        for name in ["a", "odd", "third"] {
            assert_eq!(
                meta.other_global_attr_dtype(name, "k"),
                merged_other_by_scan(&meta, name, |s| s.global_attrs.get("k").map(|d| d.0.clone())),
                "mismatch for {name}"
            );
        }
    }

    /// A dataset that is the only one using a key has no constraint, so it may
    /// freely retype it — the behaviour the contributor count preserves.
    #[test]
    fn sole_contributor_can_retype() {
        let mut meta = StoreMeta::default();
        meta.record_global_attr("solo", "k", DType::Int64);
        assert_eq!(meta.other_global_attr_dtype("solo", "k"), None);

        meta.record_global_attr("solo", "k", DType::String);
        assert_eq!(meta.other_global_attr_dtype("solo", "k"), None);
        assert_eq!(meta.datasets["solo"].global_attrs["k"].0, DType::String);

        // A second dataset now sees the String constraint.
        assert_eq!(
            meta.other_global_attr_dtype("other", "k"),
            Some(DType::String)
        );
    }

    fn make_store() -> Arc<dyn ObjectStore> {
        Arc::new(InMemory::new())
    }

    fn sample_schema() -> DatasetSchema {
        let mut s = DatasetSchema {
            arrays: IndexMap::from([(
                "temp".into(),
                ArraySchema {
                    dtype: DType::Float32,
                    shape: vec![4, 8],
                    chunk_shape: vec![2, 4],
                    dimension_names: vec!["lat".into(), "lon".into()],
                    codec: Codec::default(),
                },
            )]),
            ..Default::default()
        };
        s.register_global_attr("month", DType::Int64);
        s.register_global_attr("active", DType::Bool);
        s.register_array_attr("temp", "units", DType::String);
        s
    }

    fn sample_meta() -> StoreMeta {
        let mut meta = StoreMeta {
            version: STORE_FORMAT_VERSION,
            ..Default::default()
        };
        meta.datasets.insert("ds1".into(), Arc::new(sample_schema()));
        meta
    }

    #[tokio::test]
    async fn load_meta_missing_returns_default_json_uncompressed() {
        let store = make_store();
        let (meta, format, compression) = load_meta(&store).await.unwrap();
        assert_eq!(meta.version, 0);
        assert!(meta.datasets.is_empty());
        assert_eq!(format, MetaFormat::Json);
        assert_eq!(compression, Codec::Uncompressed);
    }

    /// Roundtrip every (format, compression) pair through save_meta + load_meta.
    /// Asserts the detected pair matches what was written.
    #[tokio::test]
    async fn save_and_load_roundtrip_all_variants() {
        for &(format, compression) in &META_VARIANTS {
            let store = make_store();
            let meta = sample_meta();
            save_meta(&store, &meta, format, compression).await.unwrap();

            let (loaded, detected_fmt, detected_comp) = load_meta(&store).await.unwrap();
            assert_eq!(detected_fmt, format, "format mismatch for {format:?}/{compression:?}");
            assert_eq!(
                detected_comp, compression,
                "compression mismatch for {format:?}/{compression:?}"
            );
            assert_eq!(loaded.version, STORE_FORMAT_VERSION);
            let ds = &loaded.datasets["ds1"];
            assert_eq!(ds.arrays["temp"].dtype, DType::Float32);
            assert_eq!(ds.arrays["temp"].shape, vec![4, 8]);
            let global_keys: Vec<&String> = ds.global_attrs.keys().collect();
            assert_eq!(global_keys, vec!["month", "active"]);
            assert_eq!(ds.global_attrs["month"].0, DType::Int64);
            assert_eq!(ds.array_attrs["temp"]["units"].0, DType::String);
        }
    }

    /// Identical dataset schemas are interned to a single pool entry on the
    /// wire and reload as separate but equal in-memory schemas.
    #[tokio::test]
    async fn identical_schemas_are_interned() {
        let mut meta = StoreMeta {
            version: STORE_FORMAT_VERSION,
            ..Default::default()
        };
        // Three datasets share one schema; a fourth differs.
        meta.datasets.insert("a".into(), Arc::new(sample_schema()));
        meta.datasets.insert("b".into(), Arc::new(sample_schema()));
        meta.datasets.insert("c".into(), Arc::new(sample_schema()));
        let mut other = sample_schema();
        other.register_global_attr("extra", DType::Float64);
        meta.datasets.insert("d".into(), Arc::new(other));

        // Wire form pools identical schemas: two distinct entries, not four.
        let wire = to_wire(&meta);
        assert_eq!(wire.schemas.len(), 2);
        assert_eq!(wire.datasets["a"], wire.datasets["b"]);
        assert_eq!(wire.datasets["a"], wire.datasets["c"]);
        assert_ne!(wire.datasets["a"], wire.datasets["d"]);

        // The pooled JSON is far smaller than four inlined copies would be.
        let store = make_store();
        save_meta(&store, &meta, MetaFormat::Json, Codec::Uncompressed)
            .await
            .unwrap();
        let (loaded, _, _) = load_meta(&store).await.unwrap();
        assert_eq!(loaded.datasets.len(), 4);
        assert_eq!(loaded.datasets["a"], loaded.datasets["b"]);
        assert_eq!(loaded.datasets["a"].arrays["temp"].shape, vec![4, 8]);
        assert!(loaded.datasets["d"].global_attrs.contains_key("extra"));
    }

    #[tokio::test]
    async fn msgpack_is_smaller_than_json() {
        let meta = sample_meta();
        let json = encode(&meta, MetaFormat::Json).unwrap();
        let mp = encode(&meta, MetaFormat::MsgPack).unwrap();
        assert!(
            mp.len() < json.len(),
            "msgpack ({}) should be smaller than JSON ({})",
            mp.len(),
            json.len()
        );
    }

    /// Compression should shrink the encoded bytes. Uses a workload large
    /// enough to overcome compression framing overhead.
    #[tokio::test]
    async fn compression_shrinks_encoded_bytes() {
        let mut meta = StoreMeta {
            version: STORE_FORMAT_VERSION,
            ..Default::default()
        };
        for i in 0..30 {
            let mut ds = DatasetSchema::default();
            for j in 0..5 {
                ds.arrays.insert(
                    format!("arr_{j}"),
                    ArraySchema {
                        dtype: DType::Float32,
                        shape: vec![100, 200, 300],
                        chunk_shape: vec![10, 20, 30],
                        dimension_names: vec!["a".into(), "b".into(), "c".into()],
                        codec: Codec::default(),
                    },
                );
            }
            // Distinct global key per dataset to defeat interning here.
            ds.register_global_attr(&format!("k_{i}"), DType::Int64);
            meta.datasets.insert(format!("dataset_{i}"), Arc::new(ds));
        }

        for format in [MetaFormat::Json, MetaFormat::MsgPack] {
            let raw = encode(&meta, format).unwrap();
            let zstd = compress(raw.clone(), Codec::Zstd).unwrap();
            let lz4 = compress(raw.clone(), Codec::Lz4).unwrap();
            assert!(
                zstd.len() < raw.len(),
                "{format:?}: zstd ({}) should be smaller than raw ({})",
                zstd.len(),
                raw.len()
            );
            assert!(
                lz4.len() < raw.len(),
                "{format:?}: lz4 ({}) should be smaller than raw ({})",
                lz4.len(),
                raw.len()
            );
        }
    }

    #[tokio::test]
    async fn load_detects_msgpack_zstd_when_only_that_present() {
        let store = make_store();
        save_meta(&store, &sample_meta(), MetaFormat::MsgPack, Codec::Zstd)
            .await
            .unwrap();
        let (_, format, compression) = load_meta(&store).await.unwrap();
        assert_eq!(format, MetaFormat::MsgPack);
        assert_eq!(compression, Codec::Zstd);
    }

    /// When more than one metadata file is present, priority order picks
    /// uncompressed JSON over everything else.
    #[tokio::test]
    async fn load_priority_order_when_many_present() {
        let store = make_store();
        let mut a = sample_meta();
        a.datasets.insert("only_a".into(), Arc::new(DatasetSchema::default()));
        let b = sample_meta();
        let c = sample_meta();
        // Write three different files; uncompressed-JSON should win.
        save_meta(&store, &c, MetaFormat::MsgPack, Codec::Zstd).await.unwrap();
        save_meta(&store, &b, MetaFormat::Json, Codec::Zstd).await.unwrap();
        save_meta(&store, &a, MetaFormat::Json, Codec::Uncompressed).await.unwrap();

        let (loaded, format, compression) = load_meta(&store).await.unwrap();
        assert_eq!(format, MetaFormat::Json);
        assert_eq!(compression, Codec::Uncompressed);
        assert!(loaded.datasets.contains_key("only_a"));
    }

    #[tokio::test]
    async fn save_overwrites_previous_meta() {
        let store = make_store();
        let meta1 = StoreMeta {
            version: STORE_FORMAT_VERSION,
            ..Default::default()
        };
        save_meta(&store, &meta1, MetaFormat::Json, Codec::Uncompressed)
            .await
            .unwrap();

        let mut meta2 = StoreMeta {
            version: STORE_FORMAT_VERSION,
            ..Default::default()
        };
        meta2.datasets.insert("new_ds".into(), Arc::new(DatasetSchema::default()));
        save_meta(&store, &meta2, MetaFormat::Json, Codec::Uncompressed)
            .await
            .unwrap();

        let (loaded, _, _) = load_meta(&store).await.unwrap();
        assert!(loaded.datasets.contains_key("new_ds"));
    }

    /// A store written by an older atlas (version != 2) is rejected with a
    /// clear error rather than a silent misparse.
    #[tokio::test]
    async fn legacy_version_rejected() {
        let store = make_store();
        let legacy = StoreMeta {
            version: 1,
            ..Default::default()
        };
        save_meta(&store, &legacy, MetaFormat::Json, Codec::Uncompressed)
            .await
            .unwrap();
        let err = load_meta(&store).await.unwrap_err();
        assert!(
            matches!(err, Error::UnsupportedVersion { found: 1, expected: 2 }),
            "expected UnsupportedVersion, got {err:?}"
        );
    }

    #[test]
    fn attr_attributevalue_roundtrip() {
        let cases = vec![
            Attr::Bool(true),
            Attr::Int8(-3),
            Attr::Int64(-1_000_000),
            Attr::UInt32(42),
            Attr::Float32(2.5),
            Attr::Float64(2.5),
            Attr::String("hello".into()),
            Attr::Binary(vec![0xde, 0xad]),
            Attr::Int32List(vec![1, 2, 3]),
            Attr::Float64List(vec![0.0, 1.5]),
            Attr::StringList(vec!["a".into(), "b".into()]),
        ];
        for v in cases {
            let av: AttributeValue = v.clone().into();
            let back: Attr = av.into();
            assert_eq!(v, back);
        }
    }

    #[test]
    fn timestamp_attr_roundtrips_through_rfc3339_string() {
        let ts = Attr::TimestampNanoseconds(1_700_000_000_000_000_000);
        let av: AttributeValue = ts.clone().into();
        // Stored as an RFC 3339 string in the .af file.
        assert_eq!(
            av,
            AttributeValue::String("2023-11-14T22:13:20Z".into())
        );
        // A string that parses as RFC 3339 comes back as a timestamp...
        let back: Attr = av.into();
        assert_eq!(back, ts);
        // ...while a non-timestamp string stays a string.
        let plain: Attr = AttributeValue::String("not-a-date".into()).into();
        assert_eq!(plain, Attr::String("not-a-date".into()));
    }

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

    fn schema_with_array(name: &str, dtype: DType) -> DatasetSchema {
        DatasetSchema {
            arrays: IndexMap::from([(
                name.into(),
                ArraySchema {
                    dtype,
                    shape: vec![2],
                    chunk_shape: vec![2],
                    dimension_names: vec!["x".into()],
                    codec: Codec::default(),
                },
            )]),
            ..Default::default()
        }
    }

    #[test]
    fn merged_schema_widens_numeric_array_dtypes() {
        let mut datasets: IndexMap<String, Arc<DatasetSchema>> = IndexMap::new();
        datasets.insert("a".into(), Arc::new(schema_with_array("temp", DType::Int16)));
        datasets.insert("b".into(), Arc::new(schema_with_array("temp", DType::Int32)));
        let mut c = schema_with_array("temp", DType::Float32);
        c.register_global_attr("region", DType::String);
        datasets.insert("c".into(), Arc::new(c));

        let merged = compute_merged(&datasets);
        // Int16 ∪ Int32 ∪ Float32 → Float64 (float + ≥32-bit int).
        assert_eq!(merged.arrays["temp"].dtype.0, DType::Float64);
        assert_eq!(merged.global_attributes["region"].0, DType::String);
    }

    #[test]
    fn merged_schema_widens_string_and_timestamp_attr() {
        let mut datasets: IndexMap<String, Arc<DatasetSchema>> = IndexMap::new();
        let mut a = DatasetSchema::default();
        a.register_global_attr("created", DType::TimestampNs);
        let mut b = DatasetSchema::default();
        b.register_global_attr("created", DType::String);
        datasets.insert("a".into(), Arc::new(a));
        datasets.insert("b".into(), Arc::new(b));

        let merged = compute_merged(&datasets);
        assert_eq!(merged.global_attributes["created"].0, DType::String);
    }

    #[test]
    fn merged_schema_serialized_in_atlas_json() {
        let mut meta = StoreMeta {
            version: STORE_FORMAT_VERSION,
            ..Default::default()
        };
        meta.datasets
            .insert("a".into(), Arc::new(schema_with_array("temp", DType::Int32)));
        let json = String::from_utf8(encode(&meta, MetaFormat::Json).unwrap()).unwrap();
        assert!(json.contains("\"merged\""), "atlas.json must include merged schema:\n{json}");
    }
}

#[cfg(test)]
mod widen_tests {
    use crate::schema::widen_dtype;
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
        let a = DType::List { child: Box::new(DType::Int8) };
        let b = DType::List { child: Box::new(DType::Int32) };
        assert_eq!(
            widen_dtype(&a, &b),
            Some(DType::List { child: Box::new(DType::Int32) })
        );
    }
}
