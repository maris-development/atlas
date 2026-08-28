# Datasets and arrays

## Mental model

```text
collection                       one data.atlas file
├── dataset "jan_2024"           one contiguous byte range inside it
│   ├── array "temperature"      float32, [4, 8], chunked [2, 4]
│   ├── array "lat"              float64, [4]
│   └── attributes               month=1, source="buoy"
└── dataset "feb_2024"
    └── …
```

A **dataset** is what a NetCDF file or an `xarray.Dataset` holds: named
N-dimensional arrays that share dimensions, plus attributes. A **collection**
holds many of them, and is the unit of a file.

An **array** belongs to exactly one dataset. Two datasets may both declare
`temperature`; those are separate arrays that happen to share a name. Nothing is
shared between them, and they need not agree on dtype, shape, or chunking.

## The two halves of the API

```python
AtlasWriter  ──add_dataset()──▶  DatasetWriter    # building, once
Atlas        ──dataset()──────▶  DatasetView      # reading metadata
```

Writing and reading are different objects because they are different phases.
A collection is built once and then fixed; see [Immutability](immutability.md).

## Building

```python
import numpy as np
import atlas

with atlas.AtlasWriter.create("/tmp/collection", codec="zstd") as w:
    ds = w.add_dataset("jan_2024")

    ds.define_array(
        "temperature",
        dtype="float32",
        dims=["lat", "lon"],
        shape=[4, 8],
        chunk_shape=[2, 4],       # optional; defaults to shape
        fill_value=float("nan"),  # optional
    )
    ds.write_array("temperature", start=[0, 0], data=block)
    ds.set_attribute("month", 1)
    ds.set_array_attribute("temperature", "units", "celsius")

    ds.finish()   # the dataset enters the file here
```

`add_dataset` returns an owned writer, so several may be open at once — useful
when ingesting many files. Each enters the collection when it finishes.

A `DatasetWriter` also works as a context manager: a clean exit finishes it, an
exception discards it.

```python
with w.add_dataset("jan_2024") as ds:
    ds.define_array(...)
```

### define_array

Declaring an array allocates nothing. It records the type, shape, chunking,
dimension names, and fill value.

| Argument | Meaning |
|---|---|
| `name` | Unique within this dataset |
| `dtype` | See [Supported dtypes](dtypes.md) |
| `dims` | One name per axis, in `shape` order |
| `shape` | Logical shape |
| `chunk_shape` | Storage granularity. Defaults to `shape` — one chunk |
| `fill_value` | Returned for cells never written. Type-checked against `dtype` |

**Chunking is the decision that matters.** It is the granularity at which a
reader fetches: a region read pulls only the chunks it overlaps. An array stored
as one chunk is read whole or not at all. For a large array being sliced, chunk
it; for a small coordinate vector, don't bother.

### write_array

```python
ds.write_array("temperature", start=[0, 0], data=block)
```

Writes `block` with its origin at `start`. The region may span chunks and need
not be chunk-aligned — partially covered chunks are handled for you. Write in
any order, in as many slabs as you like:

```python
ds.define_array("x", dtype="int32", dims=["i"], shape=[8], chunk_shape=[3])
ds.write_array("x", start=[0], data=np.array([0, 1], dtype=np.int32))
ds.write_array("x", start=[2], data=np.array([2, 3, 4, 5], dtype=np.int32))
ds.write_array("x", start=[6], data=np.array([6, 7], dtype=np.int32))
```

`data` must be a C-contiguous numpy array whose dtype matches the declared one.
Numeric types are read zero-copy. A mismatch raises `TypeError`; a strided array
raises `ValueError`.

Regions never written cost no bytes and read back as the fill value. Declaring
an array and never writing it is legitimate and cheap.

## Reading metadata

```python
collection = atlas.Atlas.open("/tmp/collection")

collection.list_datasets()          # ['jan_2024', 'feb_2024']
collection.list_arrays()            # every distinct array name, sorted
collection.dataset_exists("jan_2024")
collection.dataset_count()
len(collection); "jan_2024" in collection; list(collection)

view = collection.dataset("jan_2024")
view.name; view.ordinal; view.segment_range
view.list_arrays()
view.array_meta("temperature")
view.array_fill_value("temperature")
```

All of it comes from the footer that `open` already read — no further I/O.
Array *values* are read from Rust; see [Reading data](reading-data.md).

## Ordinals

A dataset's position in the collection, assigned in write order:

```python
collection.dataset("jan_2024").ordinal   # 0
```

Ordinals are fixed for the life of the container. Deleting a dataset does not
renumber the others, so an ordinal you recorded stays valid.

## Names

Dataset and array names must be non-empty, must not contain `/`, must not be
`.` or `..`, and must not start with `_` (that prefix is reserved). Anything
else raises `ValueError`.

A duplicate dataset name raises `FileExistsError`, as does defining the same
array twice in one dataset.

## Deleting

```python
collection.delete_dataset("feb_2024")
```

Hides the dataset by writing a small mask file. The container is untouched, no
space is reclaimed, and no ordinal moves. See
[Immutability](immutability.md#deleting).
