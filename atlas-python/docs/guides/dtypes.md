# Supported dtypes

What a NetCDF variable becomes when `atlas create` ingests it.

## The mapping

| numpy | atlas | notes |
|---|---|---|
| `int8` … `int64` | `int8` … `int64` | straight through |
| `uint8` … `uint64` | `uint8` … `uint64` | straight through |
| `float32`, `float64` | `float32`, `float64` | IEEE-754 |
| `datetime64[ns]` | `timestamp_nanoseconds` | int64 ns from the epoch, bit for bit |
| `timedelta64[*]` | `int64` | converted to ns, with a unit marker |
| `object` / `S<n>` / `U<n>` | `string` | variable-length UTF-8 |
| anything else | none | `NotImplementedError` |

`atlas show` prints the atlas name, so a `datetime64[ns]` variable appears as
`timestamp_nanoseconds`.

### Calendars that decode to cftime

xarray decodes a time axis to `datetime64[ns]` only when the calendar allows
it. A Julian, `360_day`, or `noleap` calendar, or a date outside the
`datetime64[ns]` range, decodes to a `cftime` object instead. Those arrive as
a numpy `object` array, and atlas cannot store one:

```text
atlas: profile.nc: variable 'JULD' holds cftime objects (DatetimeJulian),
which atlas cannot store...
```

Atlas refuses rather than converts, because a Julian date and a Gregorian date
of the same number are days apart. A silent conversion would move every
timestamp.

Three ways forward:

**Keep the raw numbers.** The time axis stores as an integer, with its `units`
and `calendar` attributes beside it. Nothing is lost, and a reader decodes it
with `cftime.num2date`:

```bash
atlas create /data/nc /data/collection --no-decode-times
```

```python
atlas.create("/data/nc", dest, decode_times=False)
```

**Convert the calendar first.** This gives a real `timestamp_nanoseconds`
array, and shifts the dates onto the standard calendar:

```python
xr.open_dataset(path).convert_calendar("standard").to_netcdf(clean_path)
```

**Drop the axis.** `--skip-unsupported` leaves the time array out and keeps
the rest of the dataset.

### datetime and timedelta

Atlas supports the `[ns]` resolution of `datetime64` alone. It rejects every
other resolution. Convert with `.astype("datetime64[ns]")` before you write the
NetCDF file.

Atlas has no duration type. A `timedelta64` therefore stores as int64
nanoseconds, with a `_pyatlas_timedelta` marker attribute that records the
unit. That marker is enough to rebuild the duration. It appears as a plain
attribute.

### Strings

An `object` array of Python `str` or `bytes` becomes a variable-length UTF-8
string. So does a fixed-size `|S5` or `|U10` array. A surrogate-escaped string
is common from a netCDF backend that decoded bytes with
`errors="surrogateescape"`. Atlas cleans one on the way in.

Atlas cannot store a **null** string. Each missing cell, `None` or `NaN`, takes
the fill value instead. A `UserWarning` names the count.

## Fill values

A fill value is what a read returns for a cell nobody wrote. Such a cell costs
no byte on disk. `null_count` in the [statistics](inspecting.md#statistics)
counts them.

xarray defaults to `mask_and_scale=True`. A read of a NetCDF file then leaves
`NaN` and `NaT` where data is missing, and moves `_FillValue` into
`var.encoding`. Atlas records those cells as never written, and gives each
array a default sentinel:

| dtype | default fill |
|---|---|
| `float32`, `float64` | `NaN` |
| `timestamp_nanoseconds` | `NaT` (`i64::MIN`) |
| `string` | `""` |
| integers | none |

The `_FillValue` attribute of a variable beats the default. It stores as the
fill of the array, and not as an attribute.

Check what landed:

```python
arrays = {a["name"]: a for a in atlas.describe(collection, "2024-01.nc")["arrays"]}
arrays["temperature"]["fill_value"]
```

## 0-D scalars

Every dtype works at `shape=[]`. A NetCDF product uses one for a value such as
a `TRAJECTORY` identifier. Those are single values, and they belong to the
array data, not to the attributes. They come through unchanged:

```text
	string TRAJECTORY() ;
		// stats: count=1  min="6801adf"  max="6801adf"
```

## Not yet available as array types

| Type | Status |
|---|---|
| `bool` | The `array-format` crate below supports no `bool` element type |
| `binary` | Not exposed yet |
| `list[...]`, `fixed_size_list[..., N]` | Not exposed yet |

A NetCDF file with a boolean variable fails the ingest. Store it as `uint8`,
and write the convention down. To keep the rest of that file, pass
`--skip-unsupported`, or `on_unsupported="skip"`. That leaves out the one array
and lands everything else. See
[Creating a collection](creating.md#one-bad-array-not-one-bad-file).

All four work as an **attribute** value, which is where they usually appear.

## Attribute types

An attribute takes its type apart from an array, and the range is wider. Any
scalar works: `bool`, the integer widths, a float, a string, bytes, and a
nanosecond timestamp. A list of one of those works too.

Some xarray attributes are no scalar. A nested dict, a ragged list, and a numpy
array each encode as JSON on the way in. `describe` decodes them again:

```python
atlas.describe(collection, "2024-01.nc")["attributes"]
# {'month': 1, 'bounds': [1.0, 2.0], 'nested': {'a': 1}}
```

A value JSON cannot serialize fails the ingest.

## Reading values

Rust reads array data. The table above names the type to ask for. `"float32"`
is `f32`. `"timestamp_nanoseconds"` is `TimestampNs`. `"string"` is `String`.
See [Reading data](reading-data.md).
