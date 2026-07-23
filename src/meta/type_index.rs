//! The incremental type index: what type each array/attribute key holds across
//! the whole store, maintained as datasets are written.
//!
//! Without it, every `define_array` / `set_attribute` would scan all N datasets
//! to find the type already recorded for that key elsewhere, making a bulk
//! ingest O(N² · keys-per-dataset) — the dominant cost when writing thousands
//! of small, attribute-heavy datasets (a NetCDF folder ingest sets ~170 keys
//! per file). This turns each lookup into a hash-map hit.
//!
//! [`StoreMeta`](super::StoreMeta) owns a [`TypeIndex`] and keeps it in sync;
//! it asks the index for a [`Constraint`] and, only for the rare conflicted
//! key, falls back to an exact scan it does itself.

use std::collections::HashMap;

use array_format::DType;

use crate::{config::Codec, schema::widen_dtype};

/// What an insert-time type check should compare a new value against.
pub(super) enum Constraint {
    /// No other dataset uses this key — anything goes.
    Unconstrained,
    /// The new type must widen with this merged type.
    Type(DType),
    /// The index can't answer exactly; the caller must scan every dataset.
    /// Only reachable for keys holding mutually non-widenable types.
    NeedsScan,
}

/// The merged type recorded for one key, plus how many datasets contribute it.
///
/// The fold is **first-seen-wins**: a type that can't merge leaves `dtype`
/// alone, matching [`compute_merged`](super::schema::compute_merged) so a stored
/// mismatch never becomes the reference type.
///
/// The insert-time check asks "what type do the *other* datasets use?", which
/// the merged type alone can't answer — widening is lossy, so a dataset's own
/// contribution can't be subtracted back out. `contributors` closes that gap
/// whenever all recorded types are pairwise widenable: folding the current
/// dataset's own type in never changes the accept/reject decision (widening
/// within the numeric lattice is monotone, and `String`/`List`/`Bool` are
/// incompatible with numerics either way), so the only case that matters is the
/// current dataset being the sole contributor — a count of 1.
///
/// `TypeMismatchPolicy::Warn` (the default) breaks that premise: a mismatching
/// dataset is still stored, so a key can hold non-widenable types. Then
/// excluding a dataset genuinely can change the answer, and `conflicted` routes
/// the key to the exact scan instead. It is set only by a real mismatch, so the
/// common case stays O(1).
#[derive(Debug, Default, Clone)]
struct MergedType {
    dtype: Option<DType>,
    contributors: usize,
    conflicted: bool,
}

impl MergedType {
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

/// Per-array entry: the merged dtype across all datasets, plus the codec of the
/// physical `.af` file that backs the array name.
#[derive(Debug, Default, Clone)]
struct ArrayEntry {
    dtype: MergedType,
    codec: Codec,
}

/// Reverse index over every dataset schema: key → the type it holds store-wide.
#[derive(Debug, Default, Clone)]
pub(super) struct TypeIndex {
    arrays: HashMap<String, ArrayEntry>,
    global_attrs: HashMap<String, MergedType>,
    array_attrs: HashMap<(String, String), MergedType>,
}

impl TypeIndex {
    /// Fold an array declaration into the index. `codec` is only used the first
    /// time the array name is seen, fixing the backing file's codec.
    pub(super) fn record_array(&mut self, array: &str, dtype: &DType, codec: &Codec) {
        self.arrays
            .entry(array.to_string())
            .or_insert_with(|| ArrayEntry {
                dtype: MergedType::default(),
                codec: *codec,
            })
            .dtype
            .add(dtype);
    }

    /// Fold a dataset-global attribute declaration into the index.
    pub(super) fn record_global_attr(&mut self, key: &str, ty: &DType) {
        self.global_attrs.entry(key.to_string()).or_default().add(ty);
    }

    /// Fold a per-variable attribute declaration into the index.
    pub(super) fn record_array_attr(&mut self, array: &str, key: &str, ty: &DType) {
        self.array_attrs
            .entry((array.to_string(), key.to_string()))
            .or_default()
            .add(ty);
    }

    /// The constraint an array write from a dataset that `owner_declares` (or
    /// not) must satisfy, or `None` if the array name is unknown.
    pub(super) fn array_constraint(&self, array: &str, owner_declares: bool) -> Option<Constraint> {
        Some(self.arrays.get(array)?.dtype.constraint(owner_declares))
    }

    /// As [`array_constraint`](Self::array_constraint), for a global attribute.
    pub(super) fn global_attr_constraint(
        &self,
        key: &str,
        owner_declares: bool,
    ) -> Option<Constraint> {
        Some(self.global_attrs.get(key)?.constraint(owner_declares))
    }

    /// As [`array_constraint`](Self::array_constraint), for a per-variable
    /// attribute.
    pub(super) fn array_attr_constraint(
        &self,
        array: &str,
        key: &str,
        owner_declares: bool,
    ) -> Option<Constraint> {
        Some(
            self.array_attrs
                .get(&(array.to_string(), key.to_string()))?
                .constraint(owner_declares),
        )
    }

    /// Codec of the physical file backing `array`, from the first dataset that
    /// declared it.
    pub(super) fn array_codec(&self, array: &str) -> Option<Codec> {
        self.arrays.get(array).map(|e| e.codec)
    }

    /// `true` if this array holds mutually non-widenable types — the test
    /// fixtures assert the conflicted path is exercised.
    #[cfg(test)]
    pub(super) fn array_is_conflicted(&self, array: &str) -> bool {
        self.arrays.get(array).is_some_and(|e| e.dtype.conflicted)
    }
}
