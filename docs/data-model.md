# 3. Data Model

## Datasets and arrays

A **dataset** is a named collection of **arrays** (N-dimensional, typed) plus
attributes. Think of one NetCDF file → one dataset; its variables → arrays.

```
   dataset "GL_PR_BO_JLKU"
   ├── array  TIME       datetime64[ns]   dims: [time]
   ├── array  LATITUDE   float32          dims: [time]
   ├── array  TEMP       float64          dims: [time, depth]
   ├── attr   platform_code = "BO"        (dataset-global)
   └── TEMP.units = "degree_Celsius"      (per-array attribute)
```

## Array-name sharing (the core idea)

Two datasets that declare an array with the **same name** are stored in the
**same physical file**, keyed by dataset name. The store holds *N* physical
arrays where *N* = number of distinct array names, regardless of dataset count.

```
   ds "jan"  ── TEMP, PSAL ──┐
   ds "feb"  ── TEMP, PSAL ──┤──►  TEMP/data.af   { jan, feb, mar }
   ds "mar"  ── TEMP, PSAL ──┘     PSAL/data.af   { jan, feb, mar }
```

This is what makes "read one variable across all datasets" and the pruning
index cheap: the variable's data (and stats) are already gathered in one file.

**Corollary — normalize names.** If `jan` calls it `TEMP` and `feb` calls it
`TEMP_SBE`, they don't co-locate — you get a file per variant. For heterogeneous
ingest (per-file/station-specific names), map names to a shared schema *before*
writing. A store with a handful of shared array names scales to millions of
datasets; a store with per-dataset-unique names does not.

## Ordinals — a dataset's fixed row number

Every dataset has an **ordinal**: its position in the insertion-ordered dataset
list. The ordinal is a dataset's row in every cross-dataset view (the pruning
index, `read_array_across`).

```
   ordinal   0        1        2        3
   dataset   ds0      ds1      ds2      ds3
```

Ordinals are **stable**: deleting a dataset *tombstones* its slot (a liveness
bit flips to false) rather than removing it, so no later dataset's ordinal
shifts. The one operation that renumbers is [`compact`](write-path.md), which
drops tombstones and closes the gaps.

```
   delete ds1:
   ordinal   0        1(dead)  2        3
   dataset   ds0      —        ds2      ds3      ← ds2, ds3 keep ordinals 2, 3

   compact:
   ordinal   0        1        2
   dataset   ds0      ds2      ds3               ← renumbered, gaps closed
```

## Attributes

Two scopes, both key→value:

- **Dataset-global** (`set_attribute`) — e.g. `platform_code`, `cruise_id`.
  Stored on the dataset's entry in `_global/data.af`.
- **Per-array** (`set_array_attribute`) — e.g. `units`, `valid_range`.
  Stored on the array's entry in `<array>/data.af`.

Values are scalars or lists (`Attr`): bool, ints, floats, string, binary, and
list variants. The pruning index can range-prune on **scalar** attribute values
(a per-dataset value becomes a point range `[v, v]`); list-valued attributes are
tracked as present without a range.

## The type system

Atlas array dtypes (`DType`):

```
   UInt8/16/32/64   Int8/16/32/64   Float32/64
   String           TimestampNs (nanoseconds since epoch)
   List { child }
```

The numpy/xarray mapping (in `atlas-python`) covers these plus
`datetime64[ns]` → `TimestampNs` and `timedelta64` → `Int64` (+ a marker for
round-trip). Object/bytes/unicode → `String`.

### Type widening across datasets

Datasets sharing an array name may disagree on exact dtype; Atlas computes a
**merged type** for the collection:

```
   widen(Int32, Int64)      = Int64
   widen(Int32, Float32)    = Float64   (int + float promotes to f64)
   widen(UInt16, UInt32)    = UInt32
   widen(String, TimestampNs) = String  (timestamps render as strings)
   widen(List<Int8>, List<Int16>) = List<Int16>   (element-wise)
```

Incompatible combinations (e.g. a numeric vs. a string array under the same
name) are **rejected at define time** under `TypeMismatchPolicy::Error`, or
warned under `::Warn`. The merged schema is what `merged_schema()` and the
pruning index's collection-wide summaries report.
