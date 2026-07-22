//! [`StoreMeta`]: the in-memory metadata for a whole store.
//!
//! Holds the dataset list (tombstones included, addressed positionally), the
//! liveness mask, the incremental [`TypeIndex`], and the schema interning pool.
//! Every mutation goes through a `record_*` / `unrecord_*` method so the index
//! and the mask stay consistent; enumeration goes through the `live_*` methods
//! so tombstones never leak out.

use std::sync::Arc;

use array_format::DType;
use indexmap::IndexMap;

use super::STORE_FORMAT_VERSION;
use super::schema::{DatasetSchema, MergedSchema, compute_merged, schema_hash};
use super::type_index::{Constraint, TypeIndex};
use crate::{
    config::Codec,
    schema::{ArraySchema, DTypeS},
};

#[derive(Debug, Default, Clone)]
pub(crate) struct StoreMeta {
    pub version: u32,
    /// Codec used when new arrays are defined in this store.
    pub codec: Codec,
    /// Dataset name → schema, insertion-ordered and **including tombstones**.
    ///
    /// A dataset's position here is its permanent row ordinal in the pruning
    /// index, so deletes tombstone in place rather than removing — removing
    /// would shift every later dataset up one and silently re-point every row
    /// of a positional index.
    ///
    /// Private on purpose: enumerate through the `live_*` methods so dead
    /// entries don't leak, and mutate through `record_*` / `unrecord_*` so the
    /// index stays consistent. The sole exception is persistence, which reads
    /// [`entries`](Self::entries) to preserve tombstone ordinals on disk.
    datasets: IndexMap<String, Arc<DatasetSchema>>,
    /// Monotonic counter bumped on every save.
    ///
    /// The pruning index addresses rows positionally, so an index written
    /// against a different dataset list doesn't fail — every row silently means
    /// a different dataset. Both files carry this epoch; a mismatch means the
    /// index is stale and must be rebuilt rather than trusted.
    pub(crate) meta_epoch: u64,
    /// Liveness bit per entry of `datasets`, same order and length. The logical
    /// delete mask: `unrecord_dataset` clears a bit, `drop_tombstones` closes
    /// the holes.
    live: Vec<bool>,
    /// Derived from `datasets`; never set directly, see [`Self::rebuild_index`].
    index: TypeIndex,
    /// Interning pool of distinct schemas, keyed by content hash.
    ///
    /// A homogeneous collection (one NetCDF folder, one variable layout) holds
    /// a single [`DatasetSchema`] allocation shared by every dataset. Mutations
    /// use `Arc::make_mut`, so a dataset edited after interning copies out of
    /// the pool first.
    schema_pool: std::collections::HashMap<u64, Vec<Arc<DatasetSchema>>>,
}

impl StoreMeta {
    /// An empty store's metadata at the current format version.
    pub(crate) fn new(codec: Codec) -> Self {
        Self {
            version: STORE_FORMAT_VERSION,
            codec,
            ..Default::default()
        }
    }

    // ── Enumeration (tombstone-aware) ───────────────────────────────────

    /// Live datasets in ordinal order, as `(ordinal, name, schema)`.
    pub(crate) fn live_datasets(
        &self,
    ) -> impl Iterator<Item = (usize, &String, &Arc<DatasetSchema>)> {
        self.datasets
            .iter()
            .enumerate()
            .filter(|(i, _)| self.is_live_index(*i))
            .map(|(i, (name, schema))| (i, name, schema))
    }

    /// Live schemas only — the common case where the name isn't needed.
    pub(crate) fn live_schemas(&self) -> impl Iterator<Item = &Arc<DatasetSchema>> {
        self.live_datasets().map(|(_, _, schema)| schema)
    }

    /// `true` if `name` exists and is not tombstoned.
    pub(crate) fn is_live(&self, name: &str) -> bool {
        self.datasets
            .get_index_of(name)
            .is_some_and(|i| self.is_live_index(i))
    }

    /// Schema for `name`, or `None` if absent or tombstoned.
    pub(crate) fn live_schema(&self, name: &str) -> Option<&Arc<DatasetSchema>> {
        let i = self.datasets.get_index_of(name)?;
        self.is_live_index(i)
            .then(|| self.datasets.get_index(i).map(|(_, schema)| schema))?
    }

    /// Number of row slots, live and tombstoned — the pruning index's row
    /// count. The invariant `pruning.rows == row_slots()` must always hold.
    pub(crate) fn row_slots(&self) -> usize {
        self.datasets.len()
    }

    /// How many datasets actually exist (tombstones excluded).
    pub(crate) fn live_count(&self) -> usize {
        self.live.iter().filter(|l| **l).count()
    }

    /// Names of the live datasets, in ordinal order.
    pub(crate) fn live_names(&self) -> Vec<String> {
        self.live_datasets().map(|(_, name, _)| name.clone()).collect()
    }

    /// Ordinal of `name` if it is live.
    pub(crate) fn live_ordinal(&self, name: &str) -> Option<usize> {
        let i = self.datasets.get_index_of(name)?;
        self.is_live_index(i).then_some(i)
    }

    /// Dataset name per row slot; `None` for tombstones.
    pub(crate) fn names_by_row(&self) -> Vec<Option<String>> {
        self.datasets
            .keys()
            .enumerate()
            .map(|(i, name)| self.is_live_index(i).then(|| name.clone()))
            .collect()
    }

    /// The liveness mask over row slots, for applying to a pruning index.
    pub(crate) fn live_mask(&self) -> Vec<bool> {
        self.live.clone()
    }

    fn is_live_index(&self, i: usize) -> bool {
        self.live.get(i).copied().unwrap_or(false)
    }

    // ── Type queries ────────────────────────────────────────────────────

    /// The type constraint an array write from `exclude` must satisfy: the
    /// merged type across the collection, or `None` if `exclude` is the only
    /// dataset declaring it.
    pub(crate) fn other_array_dtype(&self, exclude: &str, array: &str) -> Option<DType> {
        let owner_declares = self
            .live_schema(exclude)
            .is_some_and(|s| s.arrays.contains_key(array));
        self.resolve(self.index.array_constraint(array, owner_declares), exclude, |s| {
            s.arrays.get(array).map(|a| a.dtype.clone())
        })
    }

    /// As [`Self::other_array_dtype`], for a dataset-global attribute key.
    pub(crate) fn other_global_attr_dtype(&self, exclude: &str, key: &str) -> Option<DType> {
        let owner_declares = self
            .live_schema(exclude)
            .is_some_and(|s| s.global_attrs.contains_key(key));
        self.resolve(
            self.index.global_attr_constraint(key, owner_declares),
            exclude,
            |s| s.global_attrs.get(key).map(|d| d.0.clone()),
        )
    }

    /// As [`Self::other_array_dtype`], for a per-variable attribute key.
    pub(crate) fn other_array_attr_dtype(
        &self,
        exclude: &str,
        array: &str,
        key: &str,
    ) -> Option<DType> {
        let owner_declares = self
            .live_schema(exclude)
            .and_then(|s| s.array_attrs.get(array))
            .is_some_and(|m| m.contains_key(key));
        self.resolve(
            self.index.array_attr_constraint(array, key, owner_declares),
            exclude,
            |s| {
                s.array_attrs
                    .get(array)
                    .and_then(|m| m.get(key))
                    .map(|d| d.0.clone())
            },
        )
    }

    /// Turn an index [`Constraint`] into a concrete type, scanning only for the
    /// conflicted keys the fast path can't answer.
    fn resolve<F>(&self, constraint: Option<Constraint>, exclude: &str, pick: F) -> Option<DType>
    where
        F: Fn(&DatasetSchema) -> Option<DType>,
    {
        match constraint? {
            Constraint::Unconstrained => None,
            Constraint::Type(d) => Some(d),
            Constraint::NeedsScan => self.scan_other(exclude, pick),
        }
    }

    /// Merge a key's type across every live dataset **except** `exclude`, by
    /// scanning them all. The exact reference the type index is a fast path
    /// over, used only for keys holding non-widenable types.
    ///
    /// First-seen-wins, matching [`compute_merged`](super::schema::compute_merged):
    /// once a mismatching dataset is stored (under `TypeMismatchPolicy::Warn`),
    /// a last-wins fold would silently adopt the odd type out.
    fn scan_other<F>(&self, exclude: &str, pick: F) -> Option<DType>
    where
        F: Fn(&DatasetSchema) -> Option<DType>,
    {
        let mut acc: Option<DType> = None;
        for (_, name, schema) in self.live_datasets() {
            if name == exclude {
                continue;
            }
            if let Some(t) = pick(schema) {
                acc = Some(match acc {
                    None => t,
                    Some(a) => crate::schema::widen_dtype(&a, &t).unwrap_or(a),
                });
            }
        }
        acc
    }

    /// Codec of the physical file backing `array`, from the first dataset that
    /// declared it. O(1); replaces a scan over every dataset.
    pub(crate) fn array_file_codec(&self, array: &str) -> Option<Codec> {
        self.index.array_codec(array)
    }

    /// The collection-wide merged schema over the live datasets.
    pub(crate) fn merged_schema(&self) -> MergedSchema {
        compute_merged(self.live_schemas())
    }

    // ── Mutation ────────────────────────────────────────────────────────

    /// Registers `name` as a live dataset, returning its ordinal.
    ///
    /// Reviving a tombstoned name reuses its slot (and pruning-index row) and
    /// starts from an empty schema, so nothing of the previous occupant
    /// survives; a genuinely new name appends.
    pub(crate) fn add_dataset(&mut self, name: &str) -> usize {
        match self.datasets.get_index_of(name) {
            Some(i) => {
                self.live[i] = true;
                self.datasets[i] = Arc::new(DatasetSchema::default());
                self.rebuild_index();
                i
            }
            None => self.push_new(name, DatasetSchema::default()),
        }
    }

    /// Declare `array` in `dataset`, folding its dtype into the index.
    pub(crate) fn record_array(&mut self, dataset: &str, array: &str, schema: ArraySchema) {
        let (dtype, codec) = (schema.dtype.clone(), schema.codec);
        let previous = self.schema_mut(dataset).arrays.insert(array.to_string(), schema);
        // A retype can't be applied incrementally — a merged type widens but
        // never narrows — so recompute the whole index. A fresh key just folds
        // in. Re-inserting the same type is a no-op for the index.
        if retyped(previous.as_ref().map(|p| &p.dtype), &dtype) {
            self.rebuild_index();
        } else if previous.is_none() {
            self.index.record_array(array, &dtype, &codec);
        }
    }

    /// Record (or retype) a dataset-global attribute key.
    pub(crate) fn record_global_attr(&mut self, dataset: &str, key: &str, ty: DType) {
        let previous = self
            .schema_mut(dataset)
            .global_attrs
            .insert(key.to_string(), DTypeS(ty.clone()));
        if retyped(previous.as_ref().map(|p| &p.0), &ty) {
            self.rebuild_index();
        } else if previous.is_none() {
            self.index.record_global_attr(key, &ty);
        }
    }

    /// Record (or retype) a per-variable attribute key.
    pub(crate) fn record_array_attr(&mut self, dataset: &str, array: &str, key: &str, ty: DType) {
        let previous = self
            .schema_mut(dataset)
            .array_attrs
            .entry(array.to_string())
            .or_default()
            .insert(key.to_string(), DTypeS(ty.clone()));
        if retyped(previous.as_ref().map(|p| &p.0), &ty) {
            self.rebuild_index();
        } else if previous.is_none() {
            self.index.record_array_attr(array, key, &ty);
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

    /// Tombstone `dataset`, returning its schema.
    ///
    /// The entry keeps its slot so every later dataset keeps its ordinal;
    /// [`drop_tombstones`](Self::drop_tombstones) is what actually removes it.
    pub(crate) fn unrecord_dataset(&mut self, dataset: &str) -> Option<Arc<DatasetSchema>> {
        let i = self.datasets.get_index_of(dataset)?;
        if !self.live[i] {
            return None; // already dead — behaves as "not found"
        }
        self.live[i] = false;
        let schema = self.datasets[i].clone();
        self.rebuild_index();
        Some(schema)
    }

    /// Drops every tombstone, closing the ordinal holes.
    ///
    /// The only operation that renumbers, so it invalidates any externally
    /// cached row ordinal and forces a full pruning-index rebuild.
    pub(crate) fn drop_tombstones(&mut self) {
        if self.live.iter().all(|l| *l) {
            return;
        }
        let mut live = self.live.iter();
        self.datasets.retain(|_, _| *live.next().unwrap_or(&true));
        self.live = vec![true; self.datasets.len()];
        self.rebuild_index();
        self.prune_schema_pool();
    }

    /// Intern `dataset`'s schema: share an identical one from the pool if there
    /// is one, otherwise add this to the pool.
    ///
    /// Called when a [`DatasetView`](crate::DatasetView) is dropped, i.e. once
    /// the dataset is fully written. One hash plus one equality compare per
    /// dataset.
    pub(crate) fn seal_dataset(&mut self, dataset: &str) {
        let Some(schema) = self.datasets.get(dataset) else {
            return;
        };
        if Arc::strong_count(schema) > 1 {
            return; // already interned
        }
        let bucket = self.schema_pool.entry(schema_hash(schema)).or_default();
        match bucket.iter().find(|pooled| ***pooled == **schema) {
            Some(pooled) => {
                let shared = pooled.clone();
                self.datasets.insert(dataset.to_string(), shared);
            }
            None => bucket.push(schema.clone()),
        }
    }

    /// Recompute the type index from the live datasets. O(total keys); runs
    /// after a bulk load, a removal, or a retype — never on the fast write path.
    pub(crate) fn rebuild_index(&mut self) {
        let mut index = TypeIndex::default();
        for (_, _, schema) in self.live_datasets() {
            for (name, arr) in &schema.arrays {
                index.record_array(name, &arr.dtype, &arr.codec);
            }
            for (key, ty) in &schema.global_attrs {
                index.record_global_attr(key, &ty.0);
            }
            for (array, attrs) in &schema.array_attrs {
                for (key, ty) in attrs {
                    index.record_array_attr(array, key, &ty.0);
                }
            }
        }
        self.index = index;
    }

    /// Append a brand-new dataset with `schema`, returning its ordinal.
    fn push_new(&mut self, name: &str, schema: DatasetSchema) -> usize {
        self.datasets.insert(name.to_string(), Arc::new(schema));
        self.live.push(true);
        self.datasets.len() - 1
    }

    /// Mutable access to a dataset's schema, copying it out of the interning
    /// pool first if shared. Creating an entry here also allocates its liveness
    /// bit, or `live` and `datasets` would drift out of step.
    fn schema_mut(&mut self, dataset: &str) -> &mut DatasetSchema {
        let index = self
            .datasets
            .get_index_of(dataset)
            .unwrap_or_else(|| self.push_new(dataset, DatasetSchema::default()));
        Arc::make_mut(&mut self.datasets[index])
    }

    /// Drop pooled schemas no dataset references any more.
    fn prune_schema_pool(&mut self) {
        for bucket in self.schema_pool.values_mut() {
            bucket.retain(|s| Arc::strong_count(s) > 1);
        }
        self.schema_pool.retain(|_, bucket| !bucket.is_empty());
    }

    // ── Persistence seam (used only by `super::persist`) ────────────────

    /// Raw dataset entries in ordinal order, tombstones included.
    ///
    /// The one read that must *not* filter by liveness: a dataset's position is
    /// its row ordinal, so persistence writes tombstones in place to keep every
    /// later ordinal stable across a reload.
    pub(super) fn entries(&self) -> &IndexMap<String, Arc<DatasetSchema>> {
        &self.datasets
    }

    /// Ordinals of tombstoned rows, for persisting the mask.
    pub(super) fn deleted_ordinals(&self) -> Vec<u32> {
        self.live
            .iter()
            .enumerate()
            .filter(|(_, live)| !**live)
            .map(|(i, _)| i as u32)
            .collect()
    }

    /// Rebuild a `StoreMeta` from decoded wire pieces, seeding the interning
    /// pool and the type index.
    pub(super) fn from_loaded(
        version: u32,
        codec: Codec,
        meta_epoch: u64,
        datasets: IndexMap<String, Arc<DatasetSchema>>,
        live: Vec<bool>,
    ) -> Self {
        let mut meta = StoreMeta {
            version,
            codec,
            meta_epoch,
            datasets,
            live,
            ..Default::default()
        };
        meta.seed_schema_pool();
        meta.rebuild_index();
        meta
    }

    /// Fill the interning pool from the loaded datasets: one entry per distinct
    /// `Arc` (distinct schemas share a pointer after `from_wire` pools them).
    fn seed_schema_pool(&mut self) {
        let mut seen = std::collections::HashSet::new();
        for schema in self.datasets.values() {
            if seen.insert(Arc::as_ptr(schema)) {
                self.schema_pool
                    .entry(schema_hash(schema))
                    .or_default()
                    .push(schema.clone());
            }
        }
    }
}

/// Test-only fixture builder — insert a fully-formed schema, keeping `live` and
/// the index in step.
#[cfg(test)]
impl StoreMeta {
    pub(crate) fn insert_dataset(&mut self, name: &str, schema: DatasetSchema) {
        match self.datasets.get_index_of(name) {
            Some(i) => {
                self.datasets[i] = Arc::new(schema);
                self.live[i] = true;
            }
            None => {
                self.push_new(name, schema);
            }
        }
        self.rebuild_index();
    }
}

/// `true` if `previous` exists and differs from `new` — the only case where an
/// incremental index update won't do, since a merged type can widen but never
/// narrow back.
fn retyped(previous: Option<&DType>, new: &DType) -> bool {
    previous.is_some_and(|p| p != new)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Independent reference for "merge this key's type across every live
    /// dataset except `exclude`", written separately from
    /// [`StoreMeta::scan_other`] so the check is real rather than a tautology.
    fn merged_by_scan<F>(meta: &StoreMeta, exclude: &str, pick: F) -> Option<DType>
    where
        F: Fn(&DatasetSchema) -> Option<DType>,
    {
        let mut acc: Option<DType> = None;
        for (_, name, schema) in meta.live_datasets() {
            if name == exclude {
                continue;
            }
            if let Some(t) = pick(schema) {
                acc = Some(match acc {
                    None => t,
                    Some(a) => crate::schema::widen_dtype(&a, &t).unwrap_or(a),
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

    /// The index must give the same answer as a full scan for every dataset/key
    /// combination — including the sole-contributor case, conflicted keys, and
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

        // Mismatching types are deliberately kept: under the default
        // `TypeMismatchPolicy::Warn` they really are stored, so the index has
        // to stay exact across conflicted keys too.
        for (i, dtype) in dtypes.iter().cycle().take(24).enumerate() {
            let ds = format!("ds{i}");
            meta.record_array(&ds, "x", array_of(dtype.clone()));
            meta.record_global_attr(&ds, "k", dtype.clone());
            meta.record_array_attr(&ds, "x", "units", dtype.clone());
        }
        assert!(meta.row_slots() > 1, "fixture should hold datasets");
        assert!(
            meta.index.array_is_conflicted("x"),
            "fixture should exercise the conflicted path"
        );

        let check = |meta: &StoreMeta| {
            let mut names = meta.live_names();
            names.push("absent".into());
            for name in &names {
                assert_eq!(
                    meta.other_array_dtype(name, "x"),
                    merged_by_scan(meta, name, |s| s.arrays.get("x").map(|a| a.dtype.clone())),
                    "array dtype mismatch for {name}"
                );
                assert_eq!(
                    meta.other_global_attr_dtype(name, "k"),
                    merged_by_scan(meta, name, |s| s.global_attrs.get("k").map(|d| d.0.clone())),
                    "global attr mismatch for {name}"
                );
                assert_eq!(
                    meta.other_array_attr_dtype(name, "x", "units"),
                    merged_by_scan(meta, name, |s| s
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
        let victims: Vec<String> = meta.live_names().into_iter().take(2).collect();
        for v in victims {
            meta.unrecord_dataset(&v);
            check(&meta);
        }

        // Dropping the array from a dataset clears its attribute keys too.
        if let Some(name) = meta.live_names().first().cloned() {
            meta.unrecord_array(&name, "x");
            check(&meta);
        }
    }

    /// Identical schemas share one allocation, and editing one must not disturb
    /// the others.
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
            meta.datasets.values().all(|s| Arc::ptr_eq(s, &first)),
            "all 100 identical schemas should share one allocation"
        );
        assert_eq!(meta.schema_pool.values().map(Vec::len).sum::<usize>(), 1);

        // Copy-on-write: editing one dataset must not touch its neighbours.
        meta.record_global_attr("ds7", "extra", DType::Int64);
        assert!(!Arc::ptr_eq(&meta.datasets["ds7"], &first));
        assert!(meta.datasets["ds7"].global_attrs.contains_key("extra"));
        assert!(!meta.datasets["ds0"].global_attrs.contains_key("extra"));

        meta.seal_dataset("ds7");
        assert_eq!(meta.schema_pool.values().map(Vec::len).sum::<usize>(), 2);
    }

    /// Under `Warn` a key can hold non-widenable types. Excluding the
    /// first-seen contributor then shifts the reference type — which the fast
    /// path can't represent, so it must fall through to the scan.
    #[test]
    fn excluding_first_seen_contributor_of_a_conflicted_key() {
        let mut meta = StoreMeta::default();
        meta.record_global_attr("a", "k", DType::Int64);
        meta.record_global_attr("odd", "k", DType::String);

        assert_eq!(meta.other_global_attr_dtype("third", "k"), Some(DType::Int64));
        assert_eq!(meta.other_global_attr_dtype("a", "k"), Some(DType::String));
        assert_eq!(meta.other_global_attr_dtype("odd", "k"), Some(DType::Int64));

        for name in ["a", "odd", "third"] {
            assert_eq!(
                meta.other_global_attr_dtype(name, "k"),
                merged_by_scan(&meta, name, |s| s.global_attrs.get("k").map(|d| d.0.clone())),
                "mismatch for {name}"
            );
        }
    }

    /// The sole user of a key has no constraint, so it may freely retype it.
    #[test]
    fn sole_contributor_can_retype() {
        let mut meta = StoreMeta::default();
        meta.record_global_attr("solo", "k", DType::Int64);
        assert_eq!(meta.other_global_attr_dtype("solo", "k"), None);

        meta.record_global_attr("solo", "k", DType::String);
        assert_eq!(meta.other_global_attr_dtype("solo", "k"), None);
        assert_eq!(meta.live_schema("solo").unwrap().global_attrs["k"].0, DType::String);

        assert_eq!(meta.other_global_attr_dtype("other", "k"), Some(DType::String));
    }
}
