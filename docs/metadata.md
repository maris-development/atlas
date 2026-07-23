# 4. Metadata (`atlas.json` / `StoreMeta`)

`StoreMeta` is the in-memory model of the collection's *structure*; `atlas.json`
is its on-disk form. It holds no bulk data and no attribute values — only
schema, ordinals, and the type index.

```
   StoreMeta
   ├── version            format version
   ├── codec              default Codec for new arrays (Zstd / Lz4 / none)
   ├── meta_epoch         monotonic counter, bumped on every save
   │
   ├── datasets:  IndexMap<name, Arc<DatasetSchema>>     ← insertion order = ordinal
   │              (includes tombstones — never removed except by compact)
   ├── live:      Vec<bool>                              ← liveness bit per ordinal
   │
   ├── schema_pool: HashMap<hash, Vec<Arc<DatasetSchema>>>   ← interning pool
   └── index:     TypeIndex                              ← derived: name → merged type
```

## Schema interning

Most collections are **homogeneous** — every dataset from one NetCDF folder has
the same variables with the same dtypes. Storing that schema once per dataset
would be wasteful, so identical `DatasetSchema`s are **interned**: deduplicated
by content hash into `schema_pool`, and every dataset holds an `Arc` to the
shared instance.

```
   1,000 identical datasets  ──►  schema_pool: { hash → [ one DatasetSchema ] }
                                  datasets:    1,000 × Arc(same schema)
```

Editing a dataset's schema after interning uses `Arc::make_mut`, copying it out
of the pool first (copy-on-write), so shared instances are never mutated.

A `DatasetSchema` is:

```
   DatasetSchema
   ├── arrays:       IndexMap<name, ArraySchema>          (dtype, shape, chunks, dims, codec, fill)
   ├── global_attrs: IndexMap<key, DTypeS>                (attribute-key namespace)
   └── array_attrs:  IndexMap<array, IndexMap<key, DTypeS>>
```

## Tombstones and ordinals

Deletes are logical. `unrecord_dataset` clears the liveness bit; the entry stays
in `datasets` so ordinals below and above it don't shift. Enumeration goes
through the `live_*` methods so dead entries never leak:

```
   live_datasets()  → (ordinal, name, schema) for live rows only
   row_slots()      → total slots incl. tombstones  = the pruning row count
   live_count()     → live datasets only
   live_mask()      → Vec<bool>, applied by pruning views to hide deleted rows
```

`compact` is the only thing that physically drops tombstones and renumbers (see
[write-path.md](write-path.md)).

## The type index

`TypeIndex` is **derived** from `datasets` (never set directly). It maps each
array name / attribute key to its collection-wide **merged type**, so that:

- defining an array can be checked against what other datasets already declared
  (widen or reject — see [data-model.md](data-model.md));
- `merged_schema()` can report the whole collection's unified shape without
  re-scanning every dataset.

It is rebuilt whenever the dataset set changes.

## `meta_epoch`

A monotonic counter bumped on every `save` (flush/compact). Historically it also
tagged a persisted pruning index so a stale index could be detected. **The
pruning index is now built on demand** (there's nothing persisted to go stale),
so `meta_epoch` is purely a save/version counter today.

## On-disk format (`MetaFormat`)

| Format | File | Use |
|--------|------|-----|
| `Json` (default) | `atlas.json` | human-readable, diff-able |
| `MsgPack` | `atlas.msgpack` | compact binary |
| `MsgPack` + compression | `atlas.msgpack.zst` / `.lz4` | large collections |

On `open`, the format and codec are **auto-detected** from the files present —
no config needed to reopen.
