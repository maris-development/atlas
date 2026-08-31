# Supported dtypes

What a NetCDF variable becomes when `atlas create` ingests it.

## The mapping

| numpy | atlas | notes |
|---|---|---|
| `int8` … `int64` | `int8` … `int64` | straight through |
| `uint8` … `uint64` | `uint8` … `uint64` | straight through |
| `float32`, `float64` | `float32`, `float64` | IEEE-754 |
| `datetime64[ns]` | `timestamp_nanoseconds` | int64 ns since the epoch, bit-identical |
| `timedelta64[*]` | `int64` | normalized to ns, tagged with a unit marker |
| `object` / `S<n>` / `U<n>` | `string` | variable-length UTF-8 |
| anything else | — | `NotImplementedError` |

`atlas show` prints the atlas name, so a `datetime64[ns]` variable appears as
`timestamp_nanoseconds`.

### datetime and timedelta

Only the `[ns]` resolution of `datetime64` is supported. Other resolutions are
rejected — convert with `.astype("datetime64[ns]")` before writing the NetCDF
file.

Atlas has no duration type, so `timedelta64` is stored as int64 nanoseconds
plus a `_pyatlas_timedelta` marker attribute recording the unit. The marker is
how the duration could be reconstructed; it shows up as an ordinary attribute.

### Strings

`object` arrays of Python `str` or `bytes`, and fixed-size `|S5` / `|U10`
arrays, all become variable-length UTF-8 strings. Surrogate-escaped strings —
common from netCDF backends that decoded bytes with `errors="surrogateescape"`
— are sanitised on the way in.

Atlas cannot store a **null** string. Missing cells (`None`, `NaN`) are replaced
with the fill value, and a `UserWarning` names how many.

## Fill values

The value a read returns for cells that were never written. They cost no bytes
on disk, and they are what `null_count` counts in the
[statistics](inspecting.md#statistics).

Reading a NetCDF file with `mask_and_scale=True` — xarray's default — leaves
`NaN` and `NaT` where data is missing and moves `_FillValue` into
`var.encoding`. Atlas records those cells as never-written by defaulting each
array to a sentinel:

| dtype | default fill |
|---|---|
| `float32`, `float64` | `NaN` |
| `timestamp_nanoseconds` | `NaT` (`i64::MIN`) |
| `string` | `""` |
| integers | none |

A variable's own `_FillValue` attribute wins over the default, and is stored as
the array's fill rather than as an attribute.

Check what landed:

```python
arrays = {a["name"]: a for a in atlas.describe(collection, "2024-01")["arrays"]}
arrays["temperature"]["fill_value"]
```

## 0-D scalars

Every dtype works at `shape=[]`. NetCDF products use these for things like a
`TRAJECTORY` identifier — single values that are logically array data rather
than an attribute. They come through unchanged:

```text
	string TRAJECTORY() ;
		// stats: count=1  min="6801adf"  max="6801adf"
```

## Not yet available as array types

| Type | Status |
|---|---|
| `bool` | The underlying `array-format` crate does not support it as an element type |
| `binary` | Not exposed yet |
| `list[...]`, `fixed_size_list[..., N]` | Not exposed yet |

A NetCDF file with a boolean variable will fail to ingest. Store it as `uint8`
and document the convention.

All four work as **attribute** values, which is where they usually appear
anyway.

## Attribute types

Attributes are typed separately from arrays, and the range is wider: any
scalar — `bool`, the integer widths, floats, strings, bytes, and a nanosecond
timestamp — plus a homogeneous list of any of those.

xarray attributes that are not scalars (nested dicts, ragged lists, numpy
arrays) are JSON-encoded on the way in and decoded again by `describe`:

```python
atlas.describe(collection, "2024-01")["attributes"]
# {'month': 1, 'bounds': [1.0, 2.0], 'nested': {'a': 1}}
```

A value that cannot be JSON-serialised fails the ingest.

## Reading values

Array data is read from Rust. The table above tells you what type to ask for:
`"float32"` is `f32`, `"timestamp_nanoseconds"` is `TimestampNs`, `"string"` is
`String`. See [Reading data](reading-data.md).
