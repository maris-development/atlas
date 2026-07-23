//! [`PendingAttrs`]: the in-memory buffer for attribute writes.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use indexmap::IndexMap;

use crate::schema::Attr;

/// Buffered attribute writes, applied to the `.af` files at flush time.
///
/// Setting an attribute never touches disk: it lands here and is drained into
/// the array files by [`Atlas::flush`](crate::Atlas::flush) /
/// [`Atlas::compact`](crate::Atlas::compact), preserving atlas's single
/// durability boundary. Keyed by physical array-file name (`_global` for
/// dataset-global attrs, else the array name) and dataset name.
///
/// # Interning
///
/// Attribute names and values repeat heavily across datasets — a NetCDF
/// collection is ~94% duplicate values (`"SeaDataNet quality flag"` on every
/// variable of every file, and so on). Holding a fresh copy per dataset made
/// the buffer the bulk of resident memory during a large ingest.
///
/// So the buffer interns: file names, dataset names, and attribute keys are
/// shared [`Arc<str>`], and values are shared [`Arc<Attr>`] deduplicated by
/// content. A value stored 10 000 times costs one allocation plus 10 000
/// pointers. The pools are cleared on [`drain_all`](Self::drain_all) (the buffer
/// is empty afterwards) and pruned on removal.
/// One dataset's attributes in one physical file: interned key → interned value.
pub(crate) type FileAttrs = IndexMap<Arc<str>, Arc<Attr>>;

/// A drained `(file, dataset)` group, as handed to the flush-time writer.
pub(crate) type DrainedEntry = ((Arc<str>, Arc<str>), FileAttrs);

#[derive(Default)]
pub(crate) struct PendingAttrs {
    entries: HashMap<(Arc<str>, Arc<str>), FileAttrs>,
    /// Interned file names, dataset names, and attribute keys.
    strings: HashSet<Arc<str>>,
    /// Interned values, bucketed by content hash.
    values: HashMap<u64, Vec<Arc<Attr>>>,
}

impl PendingAttrs {
    pub(super) fn set(&mut self, file: &str, dataset: &str, key: &str, value: Attr) {
        let file = self.intern_str(file);
        let dataset = self.intern_str(dataset);
        let key = self.intern_str(key);
        let value = self.intern_value(value);
        self.entries.entry((file, dataset)).or_default().insert(key, value);
    }

    pub(super) fn get(&self, file: &str, dataset: &str, key: &str) -> Option<Attr> {
        self.entries
            .get(&(Arc::from(file), Arc::from(dataset)))
            .and_then(|m| m.get(key))
            .map(|v| (**v).clone())
    }

    /// Drops every buffered write for one `(file, dataset)` pair.
    pub(super) fn remove(&mut self, file: &str, dataset: &str) {
        if self.entries.remove(&(Arc::from(file), Arc::from(dataset))).is_some() {
            self.prune_pools();
        }
    }

    /// Drops every buffered write for `dataset` across all files.
    pub(crate) fn remove_dataset(&mut self, dataset: &str) {
        let before = self.entries.len();
        self.entries.retain(|(_, ds), _| ds.as_ref() != dataset);
        if self.entries.len() != before {
            self.prune_pools();
        }
    }

    /// Snapshot of all buffered writes as `((file, dataset), key→value)`,
    /// consuming the buffer. Used by the flush-time drain.
    ///
    /// The interning pools are cleared: every value now lives only in the
    /// returned [`Arc`]s, so nothing is leaked and the next ingest starts fresh.
    pub(crate) fn drain_all(&mut self) -> Vec<DrainedEntry> {
        self.strings.clear();
        self.values.clear();
        self.entries.drain().collect()
    }

    /// Intern a string, sharing an existing `Arc` when the content matches.
    fn intern_str(&mut self, s: &str) -> Arc<str> {
        if let Some(existing) = self.strings.get(s) {
            return existing.clone();
        }
        let arc: Arc<str> = Arc::from(s);
        self.strings.insert(arc.clone());
        arc
    }

    /// Intern a value, sharing an existing `Arc` when the content matches.
    fn intern_value(&mut self, value: Attr) -> Arc<Attr> {
        let bucket = self.values.entry(hash_attr(&value)).or_default();
        if let Some(existing) = bucket.iter().find(|a| ***a == value) {
            return existing.clone();
        }
        let arc = Arc::new(value);
        bucket.push(arc.clone());
        arc
    }

    /// Drop pooled strings/values no entry references any more. The buffer holds
    /// one strong ref per live use, so `strong_count == 1` means pool-only.
    fn prune_pools(&mut self) {
        self.strings.retain(|s| Arc::strong_count(s) > 1);
        for bucket in self.values.values_mut() {
            bucket.retain(|v| Arc::strong_count(v) > 1);
        }
        self.values.retain(|_, bucket| !bucket.is_empty());
    }
}

/// Content hash of an attribute value, consistent with its `PartialEq`.
///
/// `Attr` holds floats so it can't derive `Hash`/`Eq`; floats hash by bit
/// pattern here, matching the derived bitwise `PartialEq`.
fn hash_attr(value: &Attr) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    std::mem::discriminant(value).hash(&mut h);
    match value {
        Attr::Bool(v) => v.hash(&mut h),
        Attr::Int8(v) => v.hash(&mut h),
        Attr::Int16(v) => v.hash(&mut h),
        Attr::Int32(v) => v.hash(&mut h),
        Attr::Int64(v) => v.hash(&mut h),
        Attr::UInt8(v) => v.hash(&mut h),
        Attr::UInt16(v) => v.hash(&mut h),
        Attr::UInt32(v) => v.hash(&mut h),
        Attr::UInt64(v) => v.hash(&mut h),
        Attr::Float32(v) => v.to_bits().hash(&mut h),
        Attr::Float64(v) => v.to_bits().hash(&mut h),
        Attr::String(v) => v.hash(&mut h),
        Attr::Binary(v) => v.hash(&mut h),
        Attr::TimestampNanoseconds(v) => v.hash(&mut h),
        Attr::BoolList(v) => v.hash(&mut h),
        Attr::Int8List(v) => v.hash(&mut h),
        Attr::Int16List(v) => v.hash(&mut h),
        Attr::Int32List(v) => v.hash(&mut h),
        Attr::Int64List(v) => v.hash(&mut h),
        Attr::UInt8List(v) => v.hash(&mut h),
        Attr::UInt16List(v) => v.hash(&mut h),
        Attr::UInt32List(v) => v.hash(&mut h),
        Attr::UInt64List(v) => v.hash(&mut h),
        Attr::Float32List(v) => v.iter().for_each(|x| x.to_bits().hash(&mut h)),
        Attr::Float64List(v) => v.iter().for_each(|x| x.to_bits().hash(&mut h)),
        Attr::StringList(v) => v.hash(&mut h),
        Attr::BinaryList(v) => v.hash(&mut h),
    }
    h.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Identical values across datasets share a single allocation.
    #[test]
    fn duplicate_values_are_interned_to_one_allocation() {
        let mut p = PendingAttrs::default();
        for i in 0..100 {
            let ds = format!("ds{i}");
            p.set("_global", &ds, "flag", Attr::String("SeaDataNet quality flag".into()));
        }
        // 100 entries, but one pooled value and one pooled key/name.
        let drained = p.drain_all();
        assert_eq!(drained.len(), 100);
        let first = &drained[0].1["flag"];
        assert!(
            drained.iter().all(|(_, m)| Arc::ptr_eq(&m["flag"], first)),
            "every dataset's value must be the same Arc"
        );
    }

    #[test]
    fn distinct_values_get_distinct_allocations() {
        let mut p = PendingAttrs::default();
        p.set("_global", "a", "k", Attr::String("x".into()));
        p.set("_global", "b", "k", Attr::String("y".into()));
        p.set("_global", "c", "k", Attr::Int64(1));
        let drained: Vec<_> = p.drain_all();
        let get = |ds: &str| {
            drained
                .iter()
                .find(|((_, d), _)| d.as_ref() == ds)
                .map(|(_, m)| m["k"].clone())
                .unwrap()
        };
        assert!(!Arc::ptr_eq(&get("a"), &get("b")));
        assert_eq!(*get("c"), Attr::Int64(1));
    }

    #[test]
    fn get_reads_back_the_value() {
        let mut p = PendingAttrs::default();
        p.set("v", "ds", "units", Attr::String("m/s".into()));
        assert_eq!(p.get("v", "ds", "units"), Some(Attr::String("m/s".into())));
        assert_eq!(p.get("v", "ds", "missing"), None);
        assert_eq!(p.get("v", "other", "units"), None);
    }

    #[test]
    fn overwriting_a_key_replaces_the_value() {
        let mut p = PendingAttrs::default();
        p.set("v", "ds", "units", Attr::String("m".into()));
        p.set("v", "ds", "units", Attr::String("km".into()));
        assert_eq!(p.get("v", "ds", "units"), Some(Attr::String("km".into())));
    }

    /// Removing entries prunes the pools so a long-lived store doesn't leak
    /// interned values.
    #[test]
    fn remove_dataset_prunes_the_pools() {
        let mut p = PendingAttrs::default();
        p.set("_global", "a", "k", Attr::String("shared".into()));
        p.set("_global", "b", "k", Attr::String("shared".into()));
        p.set("v", "a", "u", Attr::String("only-a".into()));

        p.remove_dataset("a");
        // "only-a" is gone; "shared" stays (b still uses it).
        assert!(!p.values.values().flatten().any(|v| **v == Attr::String("only-a".into())));
        assert!(p.values.values().flatten().any(|v| **v == Attr::String("shared".into())));
        assert!(!p.strings.iter().any(|s| s.as_ref() == "only-a" || s.as_ref() == "u"));
        assert!(p.strings.iter().any(|s| s.as_ref() == "b"));
    }

    #[test]
    fn float_values_intern_by_bit_pattern() {
        let mut p = PendingAttrs::default();
        p.set("v", "a", "range", Attr::Float64List(vec![0.0, 120.0]));
        p.set("v", "b", "range", Attr::Float64List(vec![0.0, 120.0]));
        let drained = p.drain_all();
        let a = drained.iter().find(|((_, d), _)| d.as_ref() == "a").unwrap().1["range"].clone();
        let b = drained.iter().find(|((_, d), _)| d.as_ref() == "b").unwrap().1["range"].clone();
        assert!(Arc::ptr_eq(&a, &b), "equal float lists must share one Arc");
    }
}
