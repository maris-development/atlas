# 1. Architecture

Atlas is three layers stacked on object storage. Each layer has one job.

```
   ┌──────────────────────────────────────────────────────────────────────┐
   │  atlas-python  (PyO3 bindings + xarray integration)                    │
   │  • Atlas / DatasetView wrappers        • ds.atlas.write(store)          │
   │  • numpy ⇄ atlas dtype mapping         • add_xarray_dataset(ds, name)   │
   │  Releases the GIL, moves array data zero-copy into the core.           │
   └───────────────────────────────────┬──────────────────────────────────┘
                                       │
   ┌───────────────────────────────────▼──────────────────────────────────┐
   │  atlas-rust  (the core — this crate)                                   │
   │                                                                        │
   │   store/     Atlas: create/open, flush, compact, delete, queries       │
   │   dataset/   DatasetView: define/write/read arrays, get/set attrs      │
   │   meta/      StoreMeta: the schema, ordinals, type index (atlas.json)  │
   │   pruning/   the flat statistics table, assembled on demand            │
   │   schema/    DType, Attr, ArraySchema — the type system                │
   │   array.rs   AtlasArray: the lazy per-array-file handle + cache        │
   │   config.rs  Codec, MetaFormat, StoreConfig, TypeMismatchPolicy        │
   │   error.rs   Error / Result                                            │
   └───────────────────────────────────┬──────────────────────────────────┘
                                       │
   ┌───────────────────────────────────▼──────────────────────────────────┐
   │  array-format  (sibling crate — the columnar file format)              │
   │   ArrayFile: one physical file holding many datasets' entries for one   │
   │   array name. Chunked, per-chunk compressed, append-friendly via delta  │
   │   layers, with a per-dataset StatsFile and per-entry attributes.        │
   └───────────────────────────────────┬──────────────────────────────────┘
                                       │
   ┌───────────────────────────────────▼──────────────────────────────────┐
   │  object_store  (local FS, S3, GCS, Azure, in-memory)                   │
   └──────────────────────────────────────────────────────────────────────┘
```

## Who owns what

| Layer | Owns | Does **not** touch |
|-------|------|--------------------|
| `atlas-python` | ergonomics, numpy/xarray glue | storage, stats |
| `atlas-rust` | the *collection* — which datasets exist, their schema, cross-dataset queries | how one array's bytes are laid out |
| `array-format` | one physical array file — chunking, compression, stats, attributes | anything about *other* arrays or the collection |
| `object_store` | bytes in and out of a backend | structure |

The boundary that matters: **atlas-rust knows about the collection; array-format
knows about a single file.** Atlas never reaches into an array's byte layout;
array-format never knows two arrays belong to the same dataset.

## The central types

```
   Atlas ──────────────► one store (a directory / prefix)
     │  create_dataset(name) -> DatasetView
     │  open_dataset(name)   -> DatasetView
     │  flush() / compact()
     │  pruning_index(cols)  -> PruningIndex     (cross-dataset stats)
     │  read_array_across(…) -> stacked arrays   (one var, many datasets)
     │
     ├── StoreMeta  (in memory, persisted as atlas.json)
     │     datasets: name → schema  (insertion order = ordinal, tombstoned)
     │     schema_pool, type index, codec, meta_epoch
     │
     ├── ArrayCache  (name → AtlasArray, lazy)
     │     one handle per physical array file, opened on first use
     │
     └── ordinal_map cache   (name → ordinal, for the pruning pivot)

   DatasetView ────────► one dataset inside a store
     │  define_array / write_array / read_array
     │  set_attribute / get_attribute  (dataset-global)
     │  set_array_attribute / …        (per-array)
     └── all mutations buffer in memory; nothing hits disk until Atlas::flush()
```

## How a call flows

**Write** (`view.write_array("temp", …)`):

```
   DatasetView.write_array
     └─ ArrayCache.get_or_insert("temp")     → AtlasArray (lazy handle)
         └─ AtlasArray.get().await           → opens/creates temp/data.af
             └─ ArrayFile.write_array        → buffers into the pending delta
                                               (spills 4 MiB blocks to a tempfile;
                                                nothing durable yet)
```

**Durability boundary** (`atlas.flush()`): commits every touched `ArrayFile`'s
pending delta to a sidecar layer, computes each array's `StatsFile`, and writes
`atlas.json`. See [write-path.md](write-path.md).

**Cross-dataset query** (`atlas.pruning_index([temp])`): reads `temp/data.af`'s
`StatsFile`, pivots the per-dataset entries into a flat length-N column by
ordinal. **No separate index is stored** — see [pruning-index.md](pruning-index.md).

## Runtime & concurrency

- Everything is `async` on Tokio. `Atlas` is `Send + Sync`.
- Each physical array file is guarded by its own `tokio::sync::RwLock`, so
  different arrays are independent — the basis for the parallel pruning reads.
- Store metadata is a `parking_lot::Mutex` held only for short, non-`await`
  critical sections.
