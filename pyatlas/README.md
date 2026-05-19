# pyatlas

Python bindings for [ATLAS](../README.md) (Aggregated Tensor Large Array Store), a directory-based store for many similarly-shaped N-dimensional arrays.

## Status

- Local filesystem only (wraps `Atlas::open_path` / `Atlas::create_path`)
- Sync API backed by an internal multi-threaded tokio runtime; the GIL is released during blocking calls
- Supported array dtypes: `int8/16/32/64`, `uint8/16/32/64`, `float32/64`, `timestamp_nanoseconds` (aliases: `timestamp_ns`, `datetime64[ns]`), `string` (variable-length; `|S<n>` / `|U<n>` fixed-size inputs are accepted and stored as vlen strings)
  - `bool` is exposed as an attribute type but not as an array dtype (limitation of the underlying `array-format` crate)
  - `binary`, `list[...]`, `fixed_size_list[...,N]` are reserved for a later release
  - 0-D scalar arrays (`shape=[]`) are supported for every dtype above

## Install (development)

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install maturin numpy
cd pyatlas
maturin develop --release
```

## Quick start

```python
import numpy as np
import pyatlas

# Use the context manager so atlas.close() (== flush) runs on exit.
with pyatlas.Atlas.create("/tmp/my_store", codec="zstd") as s:   # "zstd" | "lz4" | "none"
    ds = s.create_dataset("jan_2024")
    ds.define_array(
        "temperature",
        dtype="float32",
        dims=["lat", "lon"],
        shape=[8, 16],
        chunk_shape=[4, 8],
    )
    ds.write_array(
        "temperature",
        start=[0, 0],
        data=np.full((8, 16), 20.0, dtype=np.float32),
    )
    ds.set_attribute("month", 1)
    ds.set_attribute("station", "KNMI")

# Reopen
s2 = pyatlas.Atlas.open("/tmp/my_store")
ds2 = s2.open_dataset("jan_2024")
arr = ds2.read_array("temperature")                 # full read -> np.ndarray
chunk = ds2.read_array("temperature", [0, 0], [4, 8])  # partial read
stats = ds2.array_stats("temperature")              # {"row_count", "null_count", "min", "max"}
```

## Durability model

`atlas.json` is loaded **once** when the store is opened or created. Every subsequent mutation only updates an in-memory `StoreMeta`; array writes buffer inside the per-array cache. **Nothing reaches disk until `atlas.flush()` (or `atlas.close()`, or the `with atlas:` block exits).** Dropping the `Atlas` without flushing abandons every pending write.

This means N consecutive `add_xr_dataset` / `create_dataset` calls amortise to a single flush — one delta file per touched array name, one `atlas.json` rewrite.

## API

### `pyatlas.Atlas`

| Method | Description |
| --- | --- |
| `Atlas.create(path, codec="zstd")` | Create a new store at `path`. |
| `Atlas.open(path)` | Open an existing store (reads `atlas.json` once). |
| `create_dataset(name) -> DatasetView` | New dataset (mutates in-memory meta only). |
| `open_dataset(name) -> DatasetView` | Existing dataset (reads in-memory meta; no disk I/O for the registry). |
| `delete_dataset(name)` | Remove a dataset (in-memory; persisted on next `flush`). |
| `list_datasets() -> list[str]` | All dataset names. |
| `list_arrays() -> list[str]` | Distinct array names across datasets. |
| `dataset_exists(name) -> bool` | Existence check. |
| `add_xr_dataset(ds, name, chunks=None)` | Append an `xarray.Dataset` (does **not** flush). |
| `to_xarray(name) -> xr.Dataset` | Read a dataset back as an `xarray.Dataset` (eager). |
| `flush()` | The single durability boundary — persist atlas.json + every cached array file. |
| `close()` | Alias for `flush()`; also runs as the `with`-block exit. |
| `compact()` | Reclaim tombstoned space across every cached array file. |
| `__enter__` / `__exit__` | Context-manager support — `__exit__` calls `close()`. |

### `pyatlas.DatasetView`

| Method | Description |
| --- | --- |
| `name` (property) | Dataset name. |
| `list_arrays() -> list[str]` | Array names in this dataset. |
| `define_array(name, dtype, dims, shape, chunk_shape=None)` | Declare a new array (in-memory). |
| `write_array(name, start, data)` | Write a numpy ndarray (matching stored dtype). |
| `read_array(name, start=None, shape=None) -> np.ndarray \| None` | Read full or partial. |
| `delete_array(name)` | Tombstone the array within this dataset. |
| `array_meta(name) -> dict` | `{"dtype", "shape", "chunk_shape", "dimension_names"}`. |
| `array_stats(name) -> dict \| None` | `{"row_count", "null_count", "min", "max"}` (populated after `atlas.flush()`). |
| `set_attribute(key, value, dtype=None)` | Type inferred from Python type; pass `dtype` to override (e.g. `"int8"`, `"float32"`, `"timestamp_nanoseconds"`). All integer hints are coerced to `int64`, all float hints to `float64` — the on-disk attribute types are bool, int64, float64, string, and timestamp_nanoseconds. |
| `get_attribute(key)` / `attributes()` | Single attribute or dict of all. |

`DatasetView` does **not** expose its own `flush` / `compact` — both go through the parent `Atlas`.

## xarray integration

`xarray` and `dask` ship as required dependencies. Importing `pyatlas` also registers an xarray accessor at `xr.Dataset.atlas`, so the integration is always available with no extra setup.

The atlas must exist first; you then append xarray datasets to it.

```python
import numpy as np, xarray as xr, pyatlas

ds = xr.Dataset(
    data_vars={
        "temperature": (["lat", "lon"], np.arange(8 * 16, dtype=np.float32).reshape(8, 16),
                        {"units": "C", "long_name": "surface temperature"}),
    },
    coords={"lat": np.arange(8, dtype=np.float32),
            "lon": np.arange(16, dtype=np.float32)},
    attrs={"month": 1, "station": "KNMI"},
)

# Use `with` so atlas.close() (== flush) runs on exit.
with pyatlas.Atlas.create("/tmp/my_store") as atlas:
    # Two equivalent ways to write the Dataset:
    atlas.add_xr_dataset(ds, "jan_2024")     # atlas-side method
    ds.atlas.write(atlas, "jan_2025")        # xarray accessor (same effect)

# Read back as xr.Dataset
atlas2 = pyatlas.Atlas.open("/tmp/my_store")
ds_back = atlas2.to_xarray("jan_2024")
xr.testing.assert_identical(ds, ds_back)
```

### Bulk ingestion

`add_xr_dataset` never flushes by itself — N consecutive calls accumulate in memory and a single `atlas.flush()` (or the `with` block exit) persists everything.

```python
import glob, os
import pyatlas, xarray as xr

with pyatlas.Atlas.create("/tmp/store") as atlas:
    for nc_path in sorted(glob.glob("*.nc")):
        name = os.path.splitext(os.path.basename(nc_path))[0]
        atlas.add_xr_dataset(xr.open_dataset(nc_path), name)
# One delta file per array name across the whole batch (not one per file).
```

### Storage conventions

| Item | How it's stored in atlas |
| --- | --- |
| Each coord / data variable | A separate atlas array, with `dims` mapped 1:1. |
| Dataset attrs | Atlas dataset attrs, plain keys. |
| Per-variable attrs | Flattened as `{var}.{attr}` at the dataset attr level. |
| Coord vs data_var distinction | JSON list in the internal `_pyatlas_coords` attr. |
| Non-scalar attr values (list, ndarray) | JSON-encoded string with a `json:` prefix marker. |

Reading back without the `_pyatlas_coords` marker falls back to a 1-D-same-name-as-dim heuristic, so atlas datasets written via the raw API still load cleanly into xarray.

### Streaming dask-backed Datasets

If a variable's `.data` is a `dask.array.Array` (e.g. from `xr.open_dataset(path, chunks=...)` or `ds.chunk({...})`), `atlas.add_xr_dataset` / `ds.atlas.write` stream **one dask block at a time** into atlas rather than materialising the whole array. The dask chunk shape is also used as the atlas on-disk `chunk_shape`, so the layout maps 1:1.

```python
ds = xr.open_dataset("big.nc", chunks={"time": 100, "lat": -1, "lon": -1})
with pyatlas.Atlas.create("/tmp/store") as atlas:
    atlas.add_xr_dataset(ds, "big")     # streams chunk-by-chunk
```

Peak memory ≈ one dask chunk per variable (plus dask's task graph). Pass `chunks={var: [...]}` to `add_xr_dataset` (or `ds.atlas.write`) to override the on-disk chunk shape independently of dask's chunking.

### Supported xarray variable dtypes

| numpy dtype | atlas dtype |
| --- | --- |
| `int8`/`int16`/`int32`/`int64`, `uint*`, `float32`/`float64` | matching numeric |
| `datetime64[ns]` | `timestamp_nanoseconds` (round-trips back to `datetime64[ns]`) |
| `object` (Python `str`/`bytes`), `\|S<n>`, `\|U<n>` | `string` (variable-length; reads return Python `str`) |

0-D scalar variables (e.g. a NetCDF `TRAJECTORY` identifier) round-trip natively.

### Limitations

- Eager **reads** — `atlas.to_xarray(name)` always pulls the full dataset into memory.
- Each call to `add_xr_dataset` / `ds.atlas.write` creates a *new* atlas dataset; there is no append-into-existing mode.
- `bool`, `binary`, `list[...]`, `fixed_size_list[...,N]` are not yet exposed as array element types.

## Examples

Runnable scripts live in [examples/](examples/). Each is self-contained and writes to a temp directory.

| File | Shows |
| --- | --- |
| [examples/01_basics.py](examples/01_basics.py) | Create a store, define arrays, set attributes, reopen, read back. |
| [examples/02_xarray.py](examples/02_xarray.py) | Round-trip an `xr.Dataset` through atlas using both `atlas.add_xr_dataset(...)` and the `ds.atlas.write(...)` accessor. |
| [examples/03_dask_streaming.py](examples/03_dask_streaming.py) | Stream a dask-chunked `xr.Dataset` into atlas one chunk at a time, preserving the chunk shape on disk. |

Run any of them with:

```bash
python pyatlas/examples/01_basics.py
```

## Testing

```bash
pytest pyatlas/tests/ -v
```
