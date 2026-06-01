# xarray integration

`xarray` and `dask` are required dependencies, and importing `atlas`
registers an accessor at `xr.Dataset.atlas`, so the integration is on
without any extra setup.

```python
import atlas        # registers ds.atlas on import
import xarray as xr
```

## Writing an `xr.Dataset`

Atlas must exist first; you then append `xr.Dataset`s to it. There are
two equivalent entry points:

```python
import numpy as np, xarray as xr, atlas

ds = xr.Dataset(
    data_vars={
        "temperature": (["lat", "lon"],
                        np.arange(8 * 16, dtype=np.float32).reshape(8, 16),
                        {"units": "C", "long_name": "surface temperature"}),
    },
    coords={"lat": np.arange(8, dtype=np.float32),
            "lon": np.arange(16, dtype=np.float32)},
    attrs={"month": 1, "station": "KNMI"},
)

with atlas.Atlas.create("/tmp/store") as atlas:
    atlas.add_xr_dataset(ds, "jan_2024")     # method on Atlas
    ds.atlas.write(atlas, "feb_2024")        # accessor on xr.Dataset (same effect)
```

`add_xr_dataset` doesn't flush — N consecutive calls accumulate and one
`atlas.flush()` (or the `with` block exit) persists everything. See
[Bulk ingestion](#bulk-ingestion) below.

## Reading back

```python
atlas = atlas.Atlas.open("/tmp/store")
ds_back = atlas.to_xarray("jan_2024")
xr.testing.assert_identical(ds, ds_back)     # round-trip is bit-identical
```

Variables stored with `chunk_shape != shape` come back **dask-backed** (one
dask task per on-disk chunk); full-shape and 0-D variables come back eager
as numpy. See [Dask streaming and lazy reads](dask.md).

## Storage conventions

| `xr.Dataset` item | How it's stored in atlas |
|---|---|
| Each `data_var` and each `coord` | A separate atlas array; `dims` map 1:1 onto atlas dimension names. |
| `Dataset.attrs` | Atlas dataset attributes, plain keys. |
| Per-variable `attrs` | Flattened as `{var}.{attr}` at the dataset attribute level. |
| Per-variable `_FillValue` | Consumed by `define_array` as a typed fill value (**not** flattened as a regular attr). The source `Dataset.attrs` is not mutated. |
| Coord vs data_var distinction | JSON list in the internal `_pyatlas_coords` attribute. |
| Non-scalar attr values (list, ndarray, dict) | JSON-encoded with a `json:` string prefix. |

Reading back without the `_pyatlas_coords` marker falls back to a
"1-D-array-named-after-its-dim-is-a-coord" heuristic, so atlas datasets
written via the raw `DatasetView` API still load cleanly into xarray.

## Supported variable dtypes

| numpy dtype | atlas dtype |
|---|---|
| `int8`/`int16`/`int32`/`int64`, `uint*`, `float32`/`float64` | matching numeric |
| `datetime64[ns]` | `timestamp_nanoseconds` (round-trips to `datetime64[ns]`) |
| `object` (Python `str`/`bytes`), `\|S<n>`, `\|U<n>` | `string` (variable-length; reads return Python `str`) |

0-D scalar variables (e.g. a NetCDF `TRAJECTORY` identifier) round-trip
natively. See [Supported dtypes](dtypes.md) for the full list and the
restrictions.

## Bulk ingestion

`add_xr_dataset` never flushes by itself — N consecutive calls accumulate
in memory and one flush at the end persists everything. One delta file is
written per **array name** touched across the whole batch, not one per
input file:

```python
import glob, os
import atlas, xarray as xr

with atlas.Atlas.create("/tmp/store") as atlas:
    for nc_path in sorted(glob.glob("*.nc")):
        name = os.path.splitext(os.path.basename(nc_path))[0]
        atlas.add_xr_dataset(xr.open_dataset(nc_path), name)
# Single flush on `with` exit.
```

If the NetCDF files are chunked (`xr.open_dataset(path, chunks=...)`), the
dask chunks become the atlas on-disk chunks automatically — see
[Dask streaming and lazy reads](dask.md).

## Overriding the on-disk chunk shape

```python
atlas.add_xr_dataset(ds, "jan_2024", chunks={"temperature": [4, 8]})
ds.atlas.write(atlas, "feb_2024", chunks={"temperature": [4, 8]})
```

This is independent of dask's chunking — the dask chunks are still what
gets streamed at write time, but the on-disk `chunk_shape` is whatever you
pass. Pick this when you want a different read-side chunk layout from
your write-side memory budget.

## Limitations

- **No append-into-existing.** Each call to `add_xr_dataset` /
  `ds.atlas.write` creates a *new* atlas dataset. Append-style updates to
  an existing dataset go through the raw `DatasetView` API.
- **Threaded scheduler only for lazy reads.** The `DatasetView` captured
  in the dask graph isn't picklable, so distributed / multiprocessing
  schedulers don't work. Call `.compute()` before crossing a process
  boundary, or use the [bulk-read APIs](bulk-reads.md) which return eager
  numpy.
- **No `bool` / `binary` / `list` arrays yet** (see [Supported dtypes](dtypes.md)).
