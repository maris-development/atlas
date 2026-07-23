# Datasets and arrays

## Mental model

An [`Atlas`](../reference/atlas.md) is a directory-backed handle. It owns:

1. An in-memory **`StoreMeta`** — the collection's schema: every dataset and
   every array schema, plus the attribute-key namespace (which global and
   per-variable attribute keys exist). Distinct dataset schemas are interned,
   so thousands of identical schemas cost one copy. Loaded once at
   `open`/`create`, mutated by every schema write, persisted to `atlas.json`
   on `flush()`.
2. A set of **array file caches** — one buffer per array name across the
   whole store. Pending array writes accumulate here until `flush()`.
3. A **pending-attribute buffer** — attribute *values* set via `set_attribute`
   / `set_array_attribute`, drained into the `.af` files on `flush()`.

A [`DatasetView`](../reference/dataset-view.md) is a typed handle into a
single logical dataset. It exposes the per-dataset array schemas, the
per-array statistics, dataset-level (global) attributes, and per-variable
attributes. Mutations through the view (`define_array`, `write_array`,
`set_attribute`, `set_array_attribute`, …) update the parent atlas's in-memory
state; attribute *values* live in the `.af` files, and nothing reaches disk
until you flush the *atlas*.

```text
Atlas ── StoreMeta (in-memory) ─┬─ DatasetView "jan_2024"
                                ├─ DatasetView "feb_2024"
                                └─ DatasetView ...
       │
       └─ array caches ───────── temperature/data.af  (shared by all datasets)
                                 pressure/data.af
                                 ...
```

## Lifecycle

```python
import atlas

# Create or open
atlas = atlas.Atlas.create("/tmp/store", codec="zstd")    # new store
atlas = atlas.Atlas.open("/tmp/store")                    # existing store

# Datasets are cheap — mutations stay in-memory until flush.
jan = atlas.create_dataset("jan_2024")
feb = atlas.create_dataset("feb_2024")

atlas.list_datasets()                # ["jan_2024", "feb_2024"]
atlas.dataset_exists("jan_2024")     # True

# Reopen an existing dataset (no disk I/O for the registry — it's already in memory).
jan = atlas.open_dataset("jan_2024")

# Remove (in-memory; persisted on next flush).
atlas.delete_dataset("feb_2024")
```

## Deleting datasets

`delete_dataset` is **logical**. The dataset immediately disappears from
`list_datasets()`, reads, the merged schema, and the pruning index — but under
the hood it is *tombstoned*: its slot stays in `atlas.json` and its bytes stay
in the shared array files. This keeps every other dataset's **row ordinal
stable** (`dataset_row(name)` never shifts under you, which is what lets a
cached [pruning index](stats.md) stay valid across a delete).

```python
atlas.delete_dataset("feb_2024")
atlas.dataset_exists("feb_2024")   # False — invisible immediately
atlas.row_slots()                  # still counts the dead slot until compact
```

Call [`atlas.compact()`](durability.md) to actually reclaim the space: it
rewrites the array files without the tombstoned regions and **renumbers** the
surviving datasets to close the ordinal holes — the only operation that changes
a row ordinal. Re-creating a deleted name reuses its slot with a clean schema
(the revived dataset never inherits the old one's data or stats).

## Declaring an array

`define_array` records the schema (dtype, dims, shape, chunking, fill
value) but allocates no data. The dtype is enforced on every later
`write_array`.

```python
jan.define_array(
    "temperature",
    dtype="float32",                  # see Supported dtypes
    dims=["lat", "lon"],
    shape=[8, 16],                    # logical extent on each axis
    chunk_shape=[4, 8],               # optional; defaults to shape (= 1 chunk)
    fill_value=float("nan"),          # optional; returned for unwritten cells
)
```

`chunk_shape` controls both compression granularity and partial-read
performance. A chunk shape equal to the full shape stores the array as one
block (no slice push-down). For chunked storage, partial reads only
decompress the chunks that touch the requested slice. See
[Codecs and metadata](codecs-and-meta.md) for codec choice and the
[Quickstart](../quickstart.md) for a typical example.

`fill_value` must match the array dtype:

- Integer / `timestamp_*` arrays — Python `int`, range-checked.
  `OverflowError` if out of range, `TypeError` on a `str`/`float`.
- Float arrays — Python `float` (or `int`, coerced).
- String arrays — Python `str`.

Reading an unwritten cell returns the fill value. Any *written* cell equal
to the fill value is counted as a null in [`array_stats`](stats.md).

## Writing

```python
import numpy as np
jan.write_array(
    "temperature",
    start=[0, 0],
    data=np.full((4, 8), 20.0, dtype=np.float32),
)
```

Rules:

- The numpy dtype must match the stored dtype exactly. `int32`-into-`int64`
  is not auto-promoted.
- The array must be C-contiguous. Pass `np.ascontiguousarray(data)` if
  you're unsure.
- `start` + `data.shape` must fit inside the declared `shape`.
- Writes are buffered into the array cache; `flush()` is the durability
  boundary.

## Reading

```python
full   = jan.read_array("temperature")                       # entire array
slice_ = jan.read_array("temperature", [0, 0], [4, 8])       # partial
missing = jan.read_array("not_defined_here")                 # -> None
```

For multi-array reads inside a hot loop (e.g. one dask worker), use the
bulk path:

```python
result = jan.read_arrays(["temperature", "pressure"], start=[0, 0], shape=[4, 8])
# {"temperature": np.ndarray, "pressure": np.ndarray}
```

See [Bulk reads](bulk-reads.md) for the cross-dataset variants.

## Inspecting schema

```python
jan.list_arrays()                       # ["temperature", "pressure", ...]
jan.array_meta("temperature")           # {"dtype", "shape", "chunk_shape", "dimension_names"}
jan.array_fill_value("temperature")     # the fill value passed to define_array, or None
```

`array_meta(name)` returns `None` if the array doesn't exist in *this*
dataset — useful for "does this dataset declare it?" checks without raising.

## Deleting arrays

```python
jan.delete_array("temperature")    # tombstone within this dataset
```

The array's bytes inside the shared physical file are tombstoned; reclaim
the space with [`atlas.compact()`](durability.md).

## Merged schema and type widening

`atlas.json` also carries a collection-wide **merged schema** — every unique
array (with its dtype, dimensions, and per-variable attribute types) and every
global attribute type, folded across all datasets. Read it with
`atlas.merged_schema()`:

```python
merged = store.merged_schema()
merged["arrays"]["temperature"]["dtype"]        # widened dtype across datasets
merged["arrays"]["temperature"]["attributes"]   # {attr_key: dtype}
merged["global_attributes"]                     # {attr_key: dtype}
```

When the same array name or attribute key appears in multiple datasets with
different types, the merged type is **widened** — but only within numeric types
(`int16` ∪ `int32` → `int32`, `int32` ∪ `float32` → `float64`) or between
`string` and `timestamp` (→ `string`).

Anything else can't merge (e.g. an `int32` array in one dataset and a `string`
array under the same name in another). The dataset is **still stored** under its
own type and reads back normally; the merged schema just keeps the
**first-seen** type. `on_type_mismatch` decides how that's reported:

```python
# "warn" (default): stored, logs a warning, merged keeps the first type
store = atlas.Atlas.create("/tmp/store")
store = atlas.Atlas.open("/tmp/store")

# "error": the mismatching define_array / set_attribute raises ValueError
store = atlas.Atlas.create("/tmp/store", on_type_mismatch="error")
store = atlas.Atlas.open("/tmp/store", on_type_mismatch="error")
```

It's a per-session choice — it isn't stored in `atlas.json`, so pass it each
time you open the collection.
