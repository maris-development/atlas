# Dask streaming and lazy reads

Atlas integrates with dask at both ends of the pipeline:

- **Writes** — dask-backed `xr.Dataset` variables stream one block at a
  time into atlas; peak memory ≈ one chunk per variable.
- **Reads** — variables stored with `chunk_shape != shape` come back as
  dask arrays, one task per on-disk chunk.

## Streaming writes

If a variable's `.data` is a `dask.array.Array` (e.g. from
`xr.open_dataset(path, chunks=...)` or `ds.chunk({...})`),
`atlas.add_xarray_dataset` / `ds.atlas.write` iterates the dask chunk grid and
calls `view.write_array(start=..., data=chunk)` once per chunk. The whole
array is never materialised.

```python
import xarray as xr
import atlas

ds = xr.open_dataset("big.nc", chunks={"time": 100, "lat": -1, "lon": -1})

with atlas.Atlas.create("/tmp/store") as atlas:
    atlas.add_xarray_dataset(ds, "big")     # streams chunk-by-chunk
```

The dask chunk shape is also used as the atlas on-disk `chunk_shape`, so
the layout maps 1:1 with no extra alignment cost. Override per-variable
with `chunks={"var_name": [...]}`:

```python
atlas.add_xarray_dataset(ds, "big", chunks={"temperature": [50, 50, 24]})
```

This decouples write-time memory budget from read-side chunk layout.

Peak memory ≈ one dask chunk per variable, plus dask's own task graph
overhead. For a 10 GB chunked NetCDF with 100 MB chunks, you pay 100 MB of
RAM per variable, not 10 GB.

## Lazy reads

`atlas.open_as_xarray_dataset(name)` returns each variable dask-backed when it was
stored with non-trivial chunking (`chunk_shape != shape`). The dask
`chunks` tuple mirrors the on-disk chunk grid one-to-one, and each
on-disk chunk becomes a **single** dask task. Full-shape arrays and 0-D
scalars come back eager as numpy.

```python
ds = xr.open_dataset("big.nc", chunks={"time": 100, "lat": -1, "lon": -1})
with atlas.Atlas.create("/tmp/store") as atlas:
    atlas.add_xarray_dataset(ds, "big")

ds_back = atlas.Atlas.open("/tmp/store").open_as_xarray_dataset("big")
type(ds_back["temperature"].data)        # -> dask.array.Array
ds_back["temperature"][0:100].compute()  # reads exactly one chunk
```

`.compute()` materialises the requested slice. Use xarray's normal
operators (`isel`, `sel`, `mean`, `map_blocks`, …) to operate lazily.

## Scheduler restrictions

The dask graph captures the `DatasetView` directly, so dask's default
**threaded** scheduler works out of the box. Distributed and multiprocessing
schedulers aren't supported in this release — the `DatasetView` isn't
picklable across process boundaries.

```python
# Works
ds_back["temperature"][0:100].compute(scheduler="threads")

# Does NOT work — DatasetView is not picklable
ds_back["temperature"][0:100].compute(scheduler="processes")
ds_back["temperature"][0:100].compute(scheduler=Client(...))   # dask.distributed
```

If you need cross-process parallelism, either:

1. **Materialise first** — `.compute()` (on the threaded scheduler) and
   hand off numpy arrays to your workers.
2. **Use the bulk-read APIs** — `Atlas.read_array_across_stacked` and
   `DatasetView.read_arrays` are PyO3 calls into Rust that return eager
   numpy arrays, with parallelism handled internally by the tokio runtime.
   See [Bulk reads](bulk-reads.md).

## When chunking is worth it

| Storage shape | Result |
|---|---|
| `chunk_shape == shape` (default) | Whole array stored as one block. Reads are eager numpy. Slice reads decompress the full array. |
| `chunk_shape != shape` | One block per chunk. Reads come back as dask arrays. Slice reads decompress only the touching chunks. |

For small per-dataset shapes (e.g. the `profile` benchmark case — `(50,
168)` per variable) chunking adds more overhead than it saves; keep the
default single-chunk layout. For large per-dataset shapes (e.g. the
`gridded` case — `(100, 100, 48)`), chunking is what unlocks the
decompression-only-what-you-touch behaviour. See
[Benchmarks](../benchmarks.md) for the actual numbers.
