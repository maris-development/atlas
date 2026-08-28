# Immutability

A collection is written once and never modified. This page is about what that
means in practice, because it is the constraint everything else follows from.

## There is no flush

Atlas 0.14 had a durability boundary: writes accumulated in memory, and
`flush()` was the moment they reached disk. That concept is gone. A collection
is a single file that either has a valid trailer or does not exist as a
collection at all.

```python
with atlas.AtlasWriter.create(path) as w:
    ds = w.add_dataset("jan")
    ...
    ds.finish()     # this dataset is now in the file
# the footer and trailer are written here; only now is anything readable
```

Two commit points:

| Call | Effect |
|---|---|
| `DatasetWriter.finish()` | The dataset's bytes are appended to the container |
| `AtlasWriter.finish()` (or a clean `with` exit) | Footer and trailer written; the collection becomes readable |

Before the second, `Atlas.open(path)` raises. There is no partially valid state
to detect, and nothing to clean up after a crash.

## What you cannot do

| Not available | What to do instead |
|---|---|
| Add a dataset to a finished collection | Rewrite the collection |
| Change an array's values | Rewrite the collection |
| Add an array to an existing dataset | Rewrite the collection |
| `flush()` / `compact()` | Nothing to flush; no layers to compact |
| Reclaim space from a deleted dataset | Rewrite the collection |

If that sounds expensive, note what it replaces: no delta layers to resolve on
read, no tombstones interleaved with data, no ordinal shifting under a reader,
no compaction to schedule. Collections are cheap to rebuild precisely because
writing one is a single forward pass.

## Deleting

The one operation on a finished collection:

```python
collection = atlas.Atlas.open(path)
collection.delete_dataset("feb_2024")
```

It writes a small `deleted.mask` file beside the container listing the ordinals
of deleted datasets. The container is not touched.

Consequences worth knowing:

- **No space is reclaimed.** The deleted dataset's bytes stay where they are.
- **Ordinals do not shift.** `DatasetView.ordinal` is stable for the life of the
  container, so a recorded ordinal stays valid.
- **Deleting twice raises `KeyError`.** The dataset is already hidden.
- **Concurrent deletes are last-writer-wins.** Two processes deleting different
  datasets at the same moment can lose one of the deletions. Serialize them if
  that matters.

Deleting an absent mask file is the same as an empty one, and a mask naming a
dataset the collection does not have is ignored with a warning rather than
failing the open.

## Rewriting

The idiom for "change one dataset in a collection":

```python
import atlas
import xarray as xr

old = atlas.Atlas.open(src)

with atlas.AtlasWriter.create(dst) as w:
    for name in old.list_datasets():
        if name == "the_one_to_replace":
            w.add_xarray_dataset(new_version, name)
        else:
            # Copying a dataset across needs its array data, which means the
            # Rust API. See "Reading data".
            ...
```

Copying datasets you are not changing needs to read their arrays, and Python
cannot. In practice collections are rebuilt from their original sources — the
NetCDF files, the database query — rather than from a previous collection, which
sidesteps this entirely.

## Failure during a write

An exception anywhere inside the `with` block abandons the whole collection:

```python
try:
    with atlas.AtlasWriter.create(path) as w:
        w.add_xarray_dataset(good, "a")
        w.add_xarray_dataset(broken, "b")   # raises
except NotImplementedError:
    pass

atlas.Atlas.open(path)   # ValueError: not an atlas collection
```

If you want a bad dataset to be skipped rather than to sink the write, catch it
inside the block — `add_xarray_dataset` aborts only its own dataset, and the
writer carries on:

```python
with atlas.AtlasWriter.create(path) as w:
    for nc in files:
        try:
            w.add_xarray_dataset(xr.open_dataset(nc), name=nc.stem)
        except (NotImplementedError, TypeError) as e:
            print(f"skipping {nc.stem}: {e}")
```
