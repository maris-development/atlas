# atlas-python

Python bindings for **ATLAS** (Aggregated Tensor Large Array Store) — a directory-based store
for many similarly-shaped N-dimensional arrays, backed by local files or any object store
(S3 / GCS / Azure / HTTP). Built on a Rust core with a synchronous, NumPy-native API and
first-class [xarray](https://docs.xarray.dev) integration.

```bash
pip install atlas-python
```

```python
import atlas
```

| Extra | Install | Adds |
| --- | --- | --- |
| cloud | `pip install "atlas-python[cloud]"` | S3 / GCS / Azure / HTTP backends via [obstore](https://github.com/developmentseed/obstore) |

`numpy`, `xarray`, and `dask` are installed automatically.

## Quick start

```python
import numpy as np
import atlas

# The `with` block flushes (== close) on exit. Nothing is persisted before that.
with atlas.Atlas.create("/tmp/my_store", codec="zstd") as store:   # "zstd" | "lz4" | "none"
    ds = store.create_dataset("jan_2024")
    ds.define_array(
        "temperature",
        dtype="float32",
        dims=["lat", "lon"],
        shape=[8, 16],
        chunk_shape=[4, 8],
        fill_value=float("nan"),   # unwritten cells read back as NaN; NaN cells count as nulls in stats
    )
    ds.write_array("temperature", start=[0, 0], data=np.full((8, 16), 20.0, dtype=np.float32))
    ds.set_attribute("month", 1)
    ds.set_attribute("station", "KNMI")

# Reopen and read
store = atlas.Atlas.open("/tmp/my_store")
ds = store.open_dataset("jan_2024")
arr = ds.read_array("temperature")                    # full read -> np.ndarray
chunk = ds.read_array("temperature", [0, 0], [4, 8])  # partial read
stats = ds.array_stats("temperature")                 # {"row_count", "null_count", "min", "max"}
month = ds.get_attribute("month")                     # 1
```

## Durability model

This is the one concept to internalise: **writes are buffered in memory and only hit disk on
`flush()`.**

The store's metadata is loaded once on `open`/`create`. Every subsequent mutation — creating
datasets, defining arrays, `write_array`, `set_attribute` — updates in-memory state only.
**Nothing reaches disk until `store.flush()` (equivalently `store.close()`, or the `with store:`
block exiting).** Dropping an `Atlas` without flushing abandons every pending write.

The payoff: N consecutive writes amortise to a single flush — one delta file per touched array
name and one metadata rewrite, no matter how many datasets you touched.

```python
store = atlas.Atlas.create("/tmp/my_store")
# ... many create_dataset / write_array calls ...
store.flush()   # the single durability boundary
```

## xarray integration

Importing `atlas` registers an accessor at `xr.Dataset.atlas`, so the integration is always
available. The store must exist first; you then append xarray datasets to it.

```python
import numpy as np, xarray as xr, atlas

ds = xr.Dataset(
    data_vars={
        "temperature": (["lat", "lon"], np.arange(8 * 16, dtype=np.float32).reshape(8, 16),
                        {"units": "C", "long_name": "surface temperature"}),
    },
    coords={"lat": np.arange(8, dtype=np.float32), "lon": np.arange(16, dtype=np.float32)},
    attrs={"month": 1, "station": "KNMI"},
)

with atlas.Atlas.create("/tmp/my_store") as store:
    store.add_xarray_dataset(ds, "jan_2024")     # store-side method
    ds.atlas.write(store, "jan_2025")        # xarray accessor (same effect)

# Read back as an xr.Dataset
store = atlas.Atlas.open("/tmp/my_store")
ds_back = store.open_as_xarray_dataset("jan_2024")
xr.testing.assert_identical(ds, ds_back)
```

### Bulk ingestion

`add_xarray_dataset` never flushes by itself — N consecutive calls accumulate in memory and a single
`flush()` (or the `with` exit) persists everything.

```python
import glob, os, atlas, xarray as xr

with atlas.Atlas.create("/tmp/store") as store:
    for nc_path in sorted(glob.glob("*.nc")):
        name = os.path.splitext(os.path.basename(nc_path))[0]
        store.add_xarray_dataset(xr.open_dataset(nc_path), name)
# One delta file per array name across the whole batch (not one per file).
```

### Streaming dask-backed writes

If a variable's `.data` is a `dask.array.Array` (e.g. from `xr.open_dataset(path, chunks=...)`
or `ds.chunk({...})`), `add_xarray_dataset` / `ds.atlas.write` stream **one dask block at a time**
into the store rather than materialising the whole array. The dask chunk shape becomes the
on-disk `chunk_shape`, so the layout maps 1:1. Peak memory ≈ one chunk per variable.

```python
ds = xr.open_dataset("big.nc", chunks={"time": 100, "lat": -1, "lon": -1})
with atlas.Atlas.create("/tmp/store") as store:
    store.add_xarray_dataset(ds, "big")     # streams chunk-by-chunk
```

Pass `chunks={var: [...]}` to `add_xarray_dataset` / `ds.atlas.write` to override the on-disk chunk
shape independently of dask's chunking.

### Lazy dask-backed reads

`store.open_as_xarray_dataset(name)` returns each variable dask-backed whenever it was stored with non-trivial
chunking (`chunk_shape != shape`); the dask `chunks` tuple mirrors the on-disk chunk grid and each
on-disk chunk is one dask task. Full-shape arrays (and 0-D scalars) come back eager as numpy. Call
`.compute()` to materialise, or slice / `map_blocks` to operate lazily.

```python
ds_back = atlas.Atlas.open("/tmp/store").open_as_xarray_dataset("big")
ds_back["temperature"].data              # -> dask.array.Array
ds_back["temperature"][0:100].compute()  # reads exactly one chunk
```

Reads run under dask's **threaded** scheduler only — the `DatasetView` captured in the graph isn't
picklable, so call `.compute()` before handing off to distributed/multiprocessing schedulers.

### How xarray maps onto the store

| Item | How it's stored |
| --- | --- |
| Each coord / data variable | A separate array, with `dims` mapped 1:1. |
| Dataset attrs | Dataset-level (global) attributes, plain keys. |
| Per-variable attrs | Real per-variable attributes on each variable's array. |
| Per-variable `_FillValue` | Consumed by `define_array` as a typed fill value (source `Dataset.attrs` is not mutated). See [Missing data](#missing-data). |
| Coord vs data_var distinction | JSON list in the internal `_pyatlas_coords` attr. |
| Non-scalar attr values (list, ndarray) | JSON-encoded string with a `json:` prefix marker. |

Each `add_xarray_dataset` / `ds.atlas.write` creates a *new* dataset — there is no append-into-existing
mode.

### Missing data

When a dataset is opened with `mask_and_scale=True` (xarray's default), masked cells become `NaN`
(floats) / `NaT` (datetimes) and `_FillValue` is moved into `var.encoding`. So those cells are
recorded as **null** (counted in `null_count`, excluded from min/max stats), arrays default to a
sentinel fill on write: `NaN` for floats, `NaT` for `datetime64[ns]`, and `""` for strings (integers
have none). Missing string cells (`None`/`NaN`) are substituted with the string fill and a warning is
emitted, since a string can't be stored as null directly.

Override per write with `fill_value` — a bare scalar (numeric arrays) or a `{var: scalar}` dict
(`None` disables the default for that var); an explicit CF `_FillValue` attribute still takes
precedence over the default:

```python
store.add_xarray_dataset(ds, "jan_2024", fill_value={"temperature": -9999.0})
```

See the [xarray guide](https://github.com/maris-development/atlas/blob/main/atlas-python/docs/guides/xarray.md#fill-values-and-missing-data)
for the full resolution order.

## Supported dtypes

| numpy dtype | atlas dtype |
| --- | --- |
| `int8/16/32/64`, `uint8/16/32/64`, `float32/64` | matching numeric |
| `datetime64[ns]` | `timestamp_nanoseconds` (aliases: `timestamp_ns`, `datetime64[ns]`) |
| `object` (`str`/`bytes`), `\|S<n>`, `\|U<n>` | `string` (variable-length; reads return Python `str`) |

- 0-D scalar arrays (`shape=[]`) are supported for every dtype above.
- `bool` is available as an *attribute* type but not as an array dtype.
- `binary`, `list[...]`, `fixed_size_list[...,N]` are reserved for a later release.

## Cloud / object storage

With the `cloud` extra, `Atlas.open` / `Atlas.create` accept an
[obstore](https://github.com/developmentseed/obstore)-constructed S3 / GCS / Azure / HTTP store
handle instead of a local path. The path-based local-filesystem API works without it. See the
[cloud storage guide](https://github.com/maris-development/atlas/blob/main/atlas-python/docs/guides/cloud-storage.md).

## API reference

### `atlas.Atlas`

| Method | Description |
| --- | --- |
| `Atlas.create(path, codec="zstd")` | Create a new store at `path`. |
| `Atlas.open(path)` | Open an existing store. |
| `create_dataset(name) -> DatasetView` | New dataset (in-memory until flush). |
| `open_dataset(name) -> DatasetView` | Existing dataset. |
| `delete_dataset(name)` | Remove a dataset (persisted on next `flush`). |
| `list_datasets() -> list[str]` | All dataset names. |
| `list_arrays() -> list[str]` | Distinct array names across datasets. |
| `dataset_exists(name) -> bool` | Existence check. |
| `add_xarray_dataset(ds, name, chunks=None, fill_value=None)` | Append an `xarray.Dataset` (does **not** flush). `fill_value` overrides the per-array fill (scalar or `{var: scalar}`); see [Missing data](#missing-data). |
| `open_as_xarray_dataset(name) -> xr.Dataset` | Read a dataset back (chunked vars come back dask-backed). |
| `open_as_many_xarray_dataset(names, concat_dim="dataset") -> xr.Dataset` | Open many datasets stacked along `concat_dim` (eager numpy). |
| `flush()` | The single durability boundary — persist everything. |
| `close()` | Alias for `flush()`; also the `with`-block exit. |
| `compact()` | Reclaim tombstoned space across cached array files. |
| `__enter__` / `__exit__` | Context-manager support (`__exit__` calls `close()`). |

### `atlas.DatasetView`

| Method | Description |
| --- | --- |
| `name` (property) | Dataset name. |
| `list_arrays() -> list[str]` | Array names in this dataset. |
| `define_array(name, dtype, dims, shape, chunk_shape=None, fill_value=None)` | Declare a new array. `fill_value` is a Python scalar matching the dtype; unwritten cells read back as it, and *written* cells equal to it count as nulls in `array_stats`. Dtype is enforced (`TypeError` on mismatch, `OverflowError` for out-of-range ints). |
| `write_array(name, start, data)` | Write a numpy ndarray (matching the stored dtype). |
| `read_array(name, start=None, shape=None) -> np.ndarray \| None` | Read full or partial; `None` if the array isn't in this dataset. |
| `delete_array(name)` | Tombstone the array within this dataset. |
| `array_meta(name) -> dict \| None` | `{"dtype", "shape", "chunk_shape", "dimension_names"}`. |
| `array_stats(name) -> dict \| None` | `{"row_count", "null_count", "min", "max"}` — populated after `flush()`. |
| `set_attribute(key, value, dtype=None)` | Set a dataset-level (global) attribute. Type inferred from the Python value; pass `dtype` to override (e.g. `"int8"`, `"float32"`, `"timestamp_nanoseconds"`). Values are stored in the `_global` `.af` file. |
| `get_attribute(key)` / `attributes()` | Single global attribute or dict of all. |
| `set_array_attribute(array, key, value, dtype=None)` | Set a per-variable attribute on `array` (e.g. `units`); `KeyError` if the array isn't defined. |
| `get_array_attribute(array, key)` / `array_attributes(array)` | Single per-variable attribute or dict of all, for `array`. |

`DatasetView` does **not** expose its own `flush` / `compact` — both go through the parent `Atlas`.

## Examples

Runnable, self-contained scripts (each writes to a temp directory):

- [01_basics.py](https://github.com/maris-development/atlas/blob/main/atlas-python/examples/01_basics.py) — create a store, define arrays, set attributes, reopen, read back.
- [02_xarray.py](https://github.com/maris-development/atlas/blob/main/atlas-python/examples/02_xarray.py) — round-trip an `xr.Dataset` via both `store.add_xarray_dataset(...)` and the `ds.atlas.write(...)` accessor.
- [03_dask_streaming.py](https://github.com/maris-development/atlas/blob/main/atlas-python/examples/03_dask_streaming.py) — stream a dask-chunked `xr.Dataset` in one chunk at a time.
- [09_missing_data.py](https://github.com/maris-development/atlas/blob/main/atlas-python/examples/09_missing_data.py) — masked/missing cells with default `NaN` / `NaT` / `""` fills, `null_count`, and `fill_value=` overrides.

## Performance

ATLAS is tuned for collections of many similarly-shaped datasets. On a "1000 datasets" benchmark
against netCDF4 and Zarr v3, the bulk read paths (`Atlas.open_as_many_xarray_dataset` /
`Atlas.read_array_across_stacked`) beat Zarr by ~2.8× on large chunked slice reads, and on small
per-dataset workloads ATLAS leads on both reads and writes. See the
[benchmarks](https://github.com/maris-development/atlas/tree/main/atlas-python/benchmarks) for the full
methodology, numbers, and an API picker for the fastest read path per workload.

## Links

- Source & issues: <https://github.com/maris-development/atlas>
- License: Apache-2.0
