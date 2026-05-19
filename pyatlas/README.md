# pyatlas

Python bindings for [ATLAS](../README.md) (Aggregated Tensor Large Array Store), a directory-based store for many similarly-shaped N-dimensional arrays.

## Status

- Local filesystem only (wraps `Atlas::open_path` / `Atlas::create_path`)
- Sync API backed by an internal multi-threaded tokio runtime; the GIL is released during blocking calls
- Supported dtypes: `bool`*, `int8/16/32/64`, `uint8/16/32/64`, `float32/64`
  - `*bool` is exposed as an attribute type but not as an array dtype (limitation of the underlying `array-format` crate)
  - `string`, `binary`, `list[...]`, `fixed_size_list[...,N]` are reserved for a later release

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

s = pyatlas.Atlas.create("/tmp/my_store", codec="zstd")   # "zstd" | "lz4" | "none"
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
ds.flush()

# Reopen
s2 = pyatlas.Atlas.open("/tmp/my_store")
ds2 = s2.open_dataset("jan_2024")
arr = ds2.read_array("temperature")                 # full read -> np.ndarray
chunk = ds2.read_array("temperature", [0, 0], [4, 8])  # partial read
stats = ds2.array_stats("temperature")              # {"row_count", "null_count", "min", "max"}
```

## API

### `pyatlas.Atlas`
| Method | Description |
| --- | --- |
| `Atlas.create(path, codec="zstd")` | Create a new store at `path`. |
| `Atlas.open(path)` | Open an existing store. |
| `create_dataset(name) -> DatasetView` | New dataset. |
| `open_dataset(name) -> DatasetView` | Existing dataset. |
| `delete_dataset(name)` | Remove a dataset. |
| `list_datasets() -> list[str]` | All dataset names. |
| `list_arrays() -> list[str]` | Distinct array names across datasets. |
| `dataset_exists(name) -> bool` | Existence check. |

### `pyatlas.DatasetView`
| Method | Description |
| --- | --- |
| `name` (property) | Dataset name. |
| `list_arrays() -> list[str]` | Array names in this dataset. |
| `define_array(name, dtype, dims, shape, chunk_shape=None)` | Declare a new array. |
| `write_array(name, start, data)` | Write a numpy ndarray (matching stored dtype). |
| `read_array(name, start=None, shape=None) -> np.ndarray \| None` | Read full or partial. |
| `delete_array(name)` | Remove. |
| `array_meta(name) -> dict` | `{"dtype", "shape", "chunk_shape", "dimension_names"}`. |
| `array_stats(name) -> dict \| None` | `{"row_count", "null_count", "min", "max"}`. |
| `set_attribute(key, value, dtype=None)` | Type inferred from Python type; pass `dtype` to override (e.g. `"int8"`). |
| `get_attribute(key)` / `attributes()` | Single attribute or dict of all. |
| `flush()` | Persist writes + recompute stats. |
| `compact()` | Reclaim deleted space. |

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

atlas = pyatlas.Atlas.create("/tmp/my_store")

# Two equivalent ways to write the Dataset:
atlas.add_xr_dataset(ds, "jan_2024")     # atlas-side method
ds.atlas.write(atlas, "jan_2025")        # xarray accessor (same effect)

# Read back as xr.Dataset
atlas2 = pyatlas.Atlas.open("/tmp/my_store")
ds_back = atlas2.to_xarray("jan_2024")
xr.testing.assert_identical(ds, ds_back)
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
atlas = pyatlas.Atlas.create("/tmp/store")
atlas.add_xr_dataset(ds, "big")     # streams chunk-by-chunk
```

Peak memory ≈ one dask chunk per variable (plus dask's task graph). Pass `chunks={var: [...]}` to `add_xr_dataset` (or `ds.atlas.write`) to override the on-disk chunk shape independently of dask's chunking.

### Limitations
- Numeric dtypes only (matches pyatlas core).
- Eager **reads** — `atlas.to_xarray(name)` always pulls the full dataset into memory.
- Each call to `add_xr_dataset` / `ds.atlas.write` creates a *new* atlas dataset; there is no append-into-existing mode.

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
