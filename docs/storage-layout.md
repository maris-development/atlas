# 2. Storage Layout

A store is a directory (or an object-store prefix). Everything Atlas needs is
inside it.

```
   my_store/
   ├── atlas.json            ← the collection schema + ordinals (see metadata.md)
   │                            (or atlas.msgpack / atlas.msgpack.zst — see MetaFormat)
   │
   ├── _global/
   │   └── data.af           ← reserved file: dataset-level attribute VALUES,
   │                            one entry per dataset that set a global attribute
   │
   ├── temperature/
   │   └── data.af           ← ArrayFile for the "temperature" array:
   │                            one entry per dataset + per-variable attributes + stats
   │
   ├── salinity/
   │   └── data.af
   │
   └── time/
       └── data.af
```

**One directory per distinct array name**, not per dataset. The number of
physical files is bounded by the number of distinct array/variable names in the
collection — *not* by the number of datasets. (This is the property to protect;
see the note at the end.)

## `atlas.json` — schema only

Holds the *structure* of the collection, never bulk data or attribute values:

- each distinct per-dataset schema, **interned** (a homogeneous collection —
  e.g. one NetCDF folder — stores a single schema shared by every dataset);
- the dataset list in insertion order (position = **ordinal**), with tombstones;
- the attribute-key namespace and a collection-wide **merged schema** (every
  array/attribute with its type widened across datasets);
- the `meta_epoch` counter and the array codec.

Serialization is pluggable via `MetaFormat`: `atlas.json` (default, human
readable), `atlas.msgpack`, or `atlas.msgpack.zst`/`.lz4` (compact). Details in
[metadata.md](metadata.md).

## `<array>/data.af` — one array, all datasets

Each `data.af` is an [`array-format`](https://crates.io) `ArrayFile`. Inside a
single file, many datasets' data for that array live side by side, keyed by
dataset name:

```
   temperature/data.af
   ┌────────────────────────────────────────────────────────────────┐
   │  base layer + delta layers (appended on each flush)             │
   │                                                                 │
   │   entry "ds0000" → chunked, compressed array data + attributes  │
   │   entry "ds0001" → …                                            │
   │   entry "ds0002" → …                                            │
   │       …                                                         │
   │                                                                 │
   │   StatsFile:  per-entry min / max / row_count / null_count      │
   │   footer:     directory of entries, chunk index, dtype, fill    │
   └────────────────────────────────────────────────────────────────┘
```

What array-format gives Atlas for free:

- **Chunking + per-chunk compression** (Zstd / LZ4 / none, per `Codec`).
- **Delta layers**: a write buffers into a *pending* layer that spills full
  4 MiB blocks to a tempfile; `flush` commits it as an immutable sidecar layer
  appended to the file. `compact` merges layers back into one base. This is what
  bounds write memory (see [write-path.md](write-path.md)).
- **A `StatsFile`**: per-dataset-entry `min`/`max`/`row_count`/`null_count`,
  computed as a byproduct of writing. This *is* the source of the pruning index
  (see [pruning-index.md](pruning-index.md)) — Atlas stores no separate index.
- **Per-entry attributes**: an array's per-variable attributes live on its own
  entry; dataset-global attributes live on the entry in `_global/data.af`.

## Where each kind of thing lives

| Thing | Lives in |
|-------|----------|
| Which datasets exist, their schema, ordinals | `atlas.json` |
| An array's actual data (per dataset) | `<array>/data.af` |
| An array's per-dataset statistics | `<array>/data.af` (StatsFile) |
| A per-variable attribute value | `<array>/data.af` (on the entry) |
| A dataset-global attribute value | `_global/data.af` (on the entry) |
| The pruning index | **nowhere** — assembled on demand from the StatsFiles |

## The property to protect

Because there is one physical file per array **name**, a collection is healthy
when datasets **share** names (`TEMP`, `PSAL`, `TIME`…). If every dataset
invents unique variable names, you get one file per dataset — millions of tiny
files — which defeats the layout, exhausts file descriptors on write, and makes
cross-dataset reads slow. When ingesting heterogeneous sources, **normalize
variable names to a shared schema.** See [data-model.md](data-model.md).
