# xarray integration

`xarray` and `dask` are required dependencies, and importing `atlas` registers
an accessor at `xr.Dataset.atlas`, so the integration is on with no extra setup.

```python
import atlas        # registers ds.atlas on import
import xarray as xr
```

This is the write path. Reading a collection back as an `xr.Dataset` is not
offered — see [Reading data](reading-data.md) for why.

## Writing an `xr.Dataset`

Two equivalent entry points:

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

with atlas.AtlasWriter.create("/tmp/collection") as w:
    w.add_xarray_dataset(ds, "jan_2024")   # method on AtlasWriter
    ds.atlas.write(w, "feb_2024")          # accessor on xr.Dataset, same effect
```

## What lands where

| xarray | atlas |
|---|---|
| coordinate or data variable | an array of the same name |
| `var.dims` | `dimension_names` |
| `var.shape` | `shape` |
| dask chunking | `chunk_shape`, unless `chunks=` overrides it |
| `var.attrs` | per-array attributes |
| `ds.attrs` | dataset-level attributes |
| `_FillValue` | the array's fill value, not an attribute |
| which variables were coords | the `_pyatlas_coords` marker |

Coordinates are written first, then data variables, so the on-disk order is
predictable.

Read it back with the collection-level accessors, which decode the conventions:

```python
collection = atlas.Atlas.open("/tmp/collection")

collection.coords("jan_2024")                           # ['lat', 'lon']
collection.attributes("jan_2024")                       # {'month': 1, 'station': 'KNMI'}
collection.array_attributes("jan_2024", "temperature")  # {'units': 'C', 'long_name': ...}
collection.dataset("jan_2024").array_meta("temperature")
```

## dtypes

| numpy | atlas | note |
|---|---|---|
| int / uint widths, `float32`, `float64` | the same | straight through |
| `datetime64[ns]` | `timestamp_nanoseconds` | only `[ns]`; convert others first |
| `timedelta64[*]` | `int64` | normalized to ns, tagged `_pyatlas_timedelta` |
| `object` / `S` / `U` | `string` | variable-length UTF-8 |
| anything else | — | `NotImplementedError` |

Atlas has no duration type, so a timedelta becomes int64 nanoseconds plus a
marker attribute naming the unit. Surrogate-escaped strings — common from
netCDF backends — are sanitised on the way in.

See [Supported dtypes](dtypes.md).

## Chunking

Precedence, highest first:

1. `chunks={"var": [d0, d1, ...]}` passed to `add_xarray_dataset`
2. the variable's dask chunking, if it has one
3. one full-shape chunk

```python
w.add_xarray_dataset(ds, "jan", chunks={"temperature": [4, 8]})
```

Chunking is what makes a later partial read cheap. See [Dask](dask.md).

## Fill values and missing data

Reading a NetCDF file with `mask_and_scale=True` (xarray's default) leaves `NaN`
and `NaT` where data is missing, and moves `_FillValue` into `var.encoding`.
Atlas records those cells as never-written by defaulting each array to a
sentinel:

| dtype | default fill |
|---|---|
| float | `NaN` |
| `datetime64[ns]` | `NaT` |
| string | `""` |
| integer | none |

Override it:

```python
w.add_xarray_dataset(ds, "jan", fill_value=-999)                  # every numeric array
w.add_xarray_dataset(ds, "jan", fill_value={"counts": -1})        # one variable
w.add_xarray_dataset(ds, "jan", fill_value={"temp": None})        # opt out
```

Missing **string** cells are the one lossy case: atlas cannot store a null
string, so `None` and `NaN` are replaced with the fill and a `UserWarning` names
the count.

## Bulk ingestion

One collection from a directory of files:

```python
from pathlib import Path

paths = sorted(Path("/data/nc").glob("*.nc"))

with atlas.AtlasWriter.create("/data/collection") as w:
    for p in paths:
        w.add_xarray_dataset(xr.open_dataset(p), name=p.stem)
```

Nothing is readable until the `with` block exits, and an exception inside it
abandons the whole collection. To skip bad files instead, catch per file:

```python
with atlas.AtlasWriter.create("/data/collection") as w:
    for p in paths:
        try:
            w.add_xarray_dataset(xr.open_dataset(p), name=p.stem)
        except (NotImplementedError, TypeError) as e:
            print(f"skipping {p.stem}: {e}")
```

`add_xarray_dataset` is atomic per dataset: one that fails partway — an
unsupported dtype after several good variables — never enters the collection,
and the writer carries on.

## Attributes that are not scalars

xarray attrs are often nested dicts, ragged lists, or numpy arrays. Anything
atlas cannot store natively is JSON-encoded behind a `json:` prefix and decoded
again by `Atlas.attributes()`:

```python
ds.attrs = {"nested": {"a": 1, "b": [2, 3]}}
# ...
collection.attributes("jan")["nested"]     # {'a': 1, 'b': [2, 3]}
```

A value that cannot be JSON-serialised raises `TypeError`.
