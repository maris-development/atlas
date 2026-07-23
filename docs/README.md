# Atlas — Architecture Documentation

Atlas (**A**ggregated **T**ensor **L**arge **A**rray **S**tore) is a
directory-based store for **thousands to millions of named datasets**, each a
collection of N-dimensional arrays with attributes, backed by object storage
(local filesystem, S3, …).

This folder explains how the pieces fit together. Read the docs in order for a
full picture, or jump to a topic.

| # | Doc | What it covers |
|---|-----|----------------|
| 1 | [architecture.md](architecture.md) | The crates, the layers, and how a call flows top to bottom |
| 2 | [storage-layout.md](storage-layout.md) | What's on disk: `atlas.json`, `<array>/data.af`, `_global` |
| 3 | [data-model.md](data-model.md) | Datasets, arrays, array-name sharing, ordinals, attributes, dtypes |
| 4 | [metadata.md](metadata.md) | `atlas.json` / `StoreMeta`: schema interning, the type index, tombstones |
| 5 | [write-path.md](write-path.md) | create → write → flush → compact, buffering, and the durability boundary |
| 6 | [pruning-index.md](pruning-index.md) | The flat statistics table, **built on demand** from the array files |
| 7 | [python-xarray.md](python-xarray.md) | The `atlas-python` bindings and the xarray integration |

## The 10,000-foot view

```
        ┌───────────────────────────────────────────────────────────────┐
        │  Consumers:  Rust API   •   Python (xarray)   •   CLI/notebooks │
        └───────────────────────────────┬───────────────────────────────┘
                                        │
        ┌───────────────────────────────▼───────────────────────────────┐
        │  atlas-rust  — the store                                        │
        │                                                                 │
        │   Atlas         one store: create/open, flush, compact,         │
        │                 pruning_index (query stats across datasets)     │
        │   DatasetView   one dataset: define/write/read arrays + attrs   │
        │   StoreMeta     the schema + ordinals (atlas.json)              │
        │   Pruning       a flat stats table, assembled on demand         │
        └───────────────────────────────┬───────────────────────────────┘
                                        │  typed columnar arrays + stats
        ┌───────────────────────────────▼───────────────────────────────┐
        │  array-format — one physical file per array name                │
        │  chunked · compressed · delta layers · per-dataset statistics   │
        └───────────────────────────────┬───────────────────────────────┘
                                        │
        ┌───────────────────────────────▼───────────────────────────────┐
        │  object_store —  local FS  •  S3  •  GCS  •  Azure  •  memory    │
        └───────────────────────────────────────────────────────────────┘
```

## The one idea that makes Atlas different

**Datasets that share an array name are co-located in one physical file, keyed
by dataset name.** A thousand sensors that each record `temperature` don't make
a thousand files — they make **one** `temperature/data.af`, with a thousand
entries inside. This is what lets Atlas scan a variable *across* datasets
cheaply, and it's the backbone of the [pruning index](pruning-index.md).

```
   1,000 datasets ── each declares "temperature" and "salinity" ──►

   store/
   ├── temperature/data.af   ← 1,000 entries (one per dataset)
   └── salinity/data.af      ← 1,000 entries (one per dataset)

   NOT: store/ds0/…, store/ds1/…, …  (1,000 directories)
```

See [data-model.md](data-model.md) for why, and [pruning-index.md](pruning-index.md)
for what it buys you.
