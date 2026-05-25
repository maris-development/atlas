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
        fill_value=float("nan"),   # unwritten cells read back as NaN; NaN cells count as nulls in stats
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
| `to_xarray(name) -> xr.Dataset` | Read a dataset back; chunked variables come back dask-backed (see below), full-shape variables eager. |
| `flush()` | The single durability boundary — persist atlas.json + every cached array file. |
| `close()` | Alias for `flush()`; also runs as the `with`-block exit. |
| `compact()` | Reclaim tombstoned space across every cached array file. |
| `__enter__` / `__exit__` | Context-manager support — `__exit__` calls `close()`. |

### `pyatlas.DatasetView`

| Method | Description |
| --- | --- |
| `name` (property) | Dataset name. |
| `list_arrays() -> list[str]` | Array names in this dataset. |
| `define_array(name, dtype, dims, shape, chunk_shape=None, fill_value=None)` | Declare a new array (in-memory). `fill_value` is a Python scalar matching the array dtype; unwritten cells read back as this value, and any *written* cell equal to it is counted as a null in `array_stats`. The dtype is enforced (`TypeError` on mismatch, `OverflowError` for out-of-range ints). |
| `write_array(name, start, data)` | Write a numpy ndarray (matching stored dtype). |
| `read_array(name, start=None, shape=None) -> np.ndarray \| None` | Read full or partial. Returns `None` if the array isn't in this dataset. |
| `delete_array(name)` | Tombstone the array within this dataset. |
| `array_meta(name) -> dict \| None` | `{"dtype", "shape", "chunk_shape", "dimension_names"}`, or `None` if the array isn't in this dataset. |
| `array_stats(name) -> dict \| None` | `{"row_count", "null_count", "min", "max"}`, or `None` if the array isn't in this dataset or stats haven't been computed yet (populated after `atlas.flush()`). |
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
| Per-variable `_FillValue` | Consumed by `define_array` as a typed fill value (not flattened as a regular attr). The source `Dataset.attrs` is not mutated. |
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

### Lazy dask-backed reads

`atlas.to_xarray(name)` returns each variable dask-backed whenever it was stored with a non-trivial chunking (`chunk_shape != shape`); the dask `chunks` tuple mirrors the on-disk chunk grid one-to-one and each on-disk chunk becomes a single dask task. Full-shape arrays (and 0-D scalars) still come back eager as numpy. Call `.compute()` to materialise, or slice/`map_blocks` to operate lazily.

```python
ds = xr.open_dataset("big.nc", chunks={"time": 100, "lat": -1, "lon": -1})
with pyatlas.Atlas.create("/tmp/store") as atlas:
    atlas.add_xr_dataset(ds, "big")

ds_back = pyatlas.Atlas.open("/tmp/store").to_xarray("big")
ds_back["temperature"].data            # -> dask.array.Array
ds_back["temperature"][0:100].compute()  # reads exactly one chunk
```

The graph captures the `DatasetView` directly, so dask's default threaded scheduler works out of the box. Distributed/multiprocessing schedulers aren't supported in this release — call `.compute()` before crossing a process boundary.

### Supported xarray variable dtypes

| numpy dtype | atlas dtype |
| --- | --- |
| `int8`/`int16`/`int32`/`int64`, `uint*`, `float32`/`float64` | matching numeric |
| `datetime64[ns]` | `timestamp_nanoseconds` (round-trips back to `datetime64[ns]`) |
| `object` (Python `str`/`bytes`), `\|S<n>`, `\|U<n>` | `string` (variable-length; reads return Python `str`) |

0-D scalar variables (e.g. a NetCDF `TRAJECTORY` identifier) round-trip natively.

### Limitations

- Each call to `add_xr_dataset` / `ds.atlas.write` creates a *new* atlas dataset; there is no append-into-existing mode.
- Lazy reads run under dask's threaded scheduler only — the `DatasetView` captured in the dask graph is not picklable, so `.compute()` before handing off to distributed/multiprocessing schedulers.
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

## Benchmarks

A reproducible comparison against `netCDF4` and Zarr v3 lives in [`benchmarks/`](benchmarks/). The harness writes the **same** deterministic data through each backend, then measures write time, slice-read time, and on-disk size. Each backend uses its canonical "many datasets" layout: atlas = one store with N datasets, netcdf = N separate `.nc` files (read via `xr.open_mfdataset`), zarr = N separate `.zarr` stores (also via `open_mfdataset`).

Headline numbers on a typical Apple Silicon laptop, 1000 datasets each:

**`--case gridded`** — `(100, 100, 48)` per variable × 3 variables, chunks `(50, 50, 24)`, slice 25%. Decompression-dominated; ~1.8 GB raw. All three backends push the slice down to chunk-level reads.

| Backend | Read slice (s) | Write (s) | Storage (MiB) |
|---|---:|---:|---:|
| **atlas-bulk** (`read_array_across_stacked` + slice push-down) | **2.12** | 59 | 6387 |
| **atlas + `--use-dask`** (per-dataset `view.read_arrays(...)`) | **3.21** | 60 | 6387 |
| zarr (`open_mfdataset(parallel=True).isel(...)`) | 5.99 | 38 | 6392 |
| atlas (default, serial `to_xarray(...).isel(...).load()`) | 10.23 | 51 | 6387 |
| netcdf (`open_mfdataset(parallel=True).isel(...)`) | 13.91 | 122 | 5596 |

**`--case profile`** — `(50, 168)` per variable × 2 variables, slice 25%. Overhead-dominated; ~67 MB raw.

| Backend | Read slice (s) | Write (s) | Storage (MiB) |
|---|---:|---:|---:|
| **atlas-bulk** (`to_xarray_many`) | **0.08** | 0.77 | 55.5 |
| **atlas (default, serial)** | **0.32** | 0.91 | 55.5 |
| atlas + `--use-dask` | 2.03 | 2.98 | 55.6 |
| netcdf | 4.27 | 2.98 | 62.3 |
| zarr | 4.07 | 11.43 | 61.6 |

**TL;DR**:

- On large per-dataset workloads (`gridded` with realistic chunking + slice push-down): **`atlas-bulk` beats zarr by 2.8×** on slice reads, and **`atlas + --use-dask` beats zarr by 1.9×**. zarr remains the fastest writer.
- On small per-dataset workloads (`profile`): atlas wins on everything — reads ~50× faster than zarr, writes ~12× faster. Per-dataset overhead is atlas's home court.
- **API picker for reads** (in rough order of speed):
  - Cross-dataset slice of the same vars across many datasets → `Atlas.to_xarray_many` / `Atlas.read_array_across_stacked` (the `atlas-bulk` path; one Rust call per variable).
  - Per-dataset slice reads inside a dask worker → `view.read_arrays(vars, start, shape)` (returns `dict[str, np.ndarray]`; skips xr.Dataset + per-chunk dask graph). This is what `bench_atlas` with `--use-dask` does internally.
  - Natural xarray code → `to_xarray(name).isel(...).load()`. Most ergonomic but pays per-chunk dask graph build overhead on chunked storage.
- `--use-dask` is workload-dependent: helps when per-dataset decompression is the bottleneck, hurts otherwise.

See the [top-level README's Benchmarks section](../README.md#benchmarks) for the full breakdown and caveats.

Install + run:

```bash
pip install -e "pyatlas[bench]"
python pyatlas/benchmarks/bench_collection.py --case gridded --datasets 1000
```

See [`benchmarks/README.md`](benchmarks/README.md) for all flags (`--case sensors|gridded|profile`, `--use-dask`, `--atlas-bulk`, `--netcdf-groups`, `--zarr-groups`, …).

## Testing

```bash
pytest pyatlas/tests/ -v
```
