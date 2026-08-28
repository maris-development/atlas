# Dask streaming

Dask-backed variables stream into a collection one block at a time, so peak
memory is one block per variable rather than the whole array. This is what makes
it possible to write a dataset far larger than RAM.

This is a write-side integration only. Reading a collection back as a
dask-backed `xr.Dataset` is not offered; see [Reading data](reading-data.md).

## Streaming writes

If a variable's `.data` is a `dask.array.Array` — from
`xr.open_dataset(path, chunks=...)` or `ds.chunk({...})` —
`add_xarray_dataset` walks the dask chunk grid and issues one `write_array` per
block. The array is never materialised whole.

```python
import xarray as xr
import atlas

ds = xr.open_dataset("big.nc", chunks={"time": 100, "lat": -1, "lon": -1})

with atlas.AtlasWriter.create("/tmp/collection") as w:
    w.add_xarray_dataset(ds, "big")     # streams block by block
```

Peak memory is roughly one dask block per variable plus dask's own graph
overhead. For a 10 GB file chunked at 100 MB, that is 100 MB per variable, not
10 GB.

Blocks are prefetched on a background thread — batches of 8, two deep — so
compression and the next block's computation overlap.

## Chunk shape

The dask chunking becomes the on-disk `chunk_shape`, one to one, with no
realignment. Override per variable to decouple the write-time memory budget from
the read-side layout:

```python
w.add_xarray_dataset(ds, "big", chunks={"temperature": [50, 50, 24]})
```

Chunk shape is the granularity at which a later reader fetches: a region read
pulls only the chunks it overlaps. Choose it for how the data will be read, not
for how it happened to be chunked in memory.

```python
view = atlas.Atlas.open("/tmp/collection").dataset("big")
view.array_meta("temperature")["chunk_shape"]
```

## Non-dask variables

A numpy-backed variable is written as a single full-shape block, and stored as
one chunk unless `chunks=` says otherwise. Eager and lazy variables mix freely
in one dataset:

```python
ds = xr.Dataset({
    "eager": xr.DataArray(np.arange(8, dtype=np.int32), dims=["x"]),
    "lazy":  xr.DataArray(dask_array.arange(8, dtype=np.int32, chunks=2), dims=["x"]),
})
# eager -> chunk_shape [8]; lazy -> chunk_shape [2]
```

## Scheduler

Writing uses dask only to compute blocks, on whatever scheduler is active.
The default threaded scheduler is fine; the writes themselves happen in Rust
with the GIL released.
