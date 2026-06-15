# Attributes

Each [`DatasetView`](../reference/dataset-view.md) carries a typed
attribute dict. Attributes are stored alongside the dataset's array
schemas inside `atlas.json` (or `atlas.msgpack`) — they're cheap to read
and don't load any array bytes.

## Setting and getting

```python
ds = atlas.create_dataset("jan_2024")

ds.set_attribute("month", 1)            # inferred int
ds.set_attribute("station", "KNMI")     # inferred str
ds.set_attribute("calibrated", True)    # inferred bool

ds.get_attribute("month")               # -> 1
ds.get_attribute("missing")             # -> None
ds.attributes()                         # -> {"month": 1, "station": "KNMI", "calibrated": True}
```

## The on-disk type system

Atlas attributes are stored as one of **five** wire types:

| On-disk type | Python type returned on read |
|---|---|
| `bool` | `bool` |
| `int64` | `int` |
| `float64` | `float` |
| `string` | `str` |
| `timestamp_nanoseconds` | `numpy.datetime64[ns]` |

Type is inferred from the Python value by default — `int` → `int64`,
`float` → `float64`, `str` → `string`, `bool` → `bool`. Numpy scalars are
unwrapped first (`np.int32(5)` is stored as `int64`).

## Overriding inferred types

Pass `dtype=` to narrow or force a specific type:

```python
ds.set_attribute("sensor_id", 7, dtype="int8")          # range-checked → int64 on disk
ds.set_attribute("ratio", 0.5, dtype="float32")         # widened → float64 on disk
ds.set_attribute("observed_at",
                 np.datetime64("2024-01-15T10:00", "ns"),
                 dtype="timestamp_nanoseconds")
```

Important caveats:

- **All integer hints are coerced to `int64`** on disk. `dtype="int8"`
  range-checks the value but doesn't shrink the storage representation.
- **All float hints are coerced to `float64`** on disk. `dtype="float32"`
  is accepted but doesn't change the bytes.
- The narrower hints exist so you can express *intent* and get runtime
  validation; the storage representation is always the wider type.

## Per-variable xarray attributes

When you write an `xr.Dataset` via `atlas.add_xarray_dataset(ds, name)`,
per-variable attributes are **flattened** as `{var}.{attr}` and stored
alongside the dataset's own `attrs`:

```python
ds = xr.Dataset(
    data_vars={"temperature": xr.DataArray(arr, dims=["lat", "lon"],
                                            attrs={"units": "C"})},
    attrs={"station": "KNMI"},
)
atlas.add_xarray_dataset(ds, "jan_2024")

# In atlas:
view = atlas.open_dataset("jan_2024")
view.attributes()
# {"station": "KNMI", "temperature.units": "C"}
```

On read, `atlas.open_as_xarray_dataset("jan_2024")` reverses the flattening so the
per-variable attrs end up back on the right `DataArray`. The dataset-level
attrs come through directly.

See [xarray integration](xarray.md) for the full storage convention.

## JSON-encoded "complex" attributes

xarray attribute values are sometimes lists, ndarrays, or nested dicts —
none of which fit into atlas's five on-disk types. The xarray bridge handles
this by JSON-encoding the value and prefixing the resulting string with
`json:`. On read the prefix is detected and decoded transparently. You
generally don't need to think about this, but it's why some attribute
values come back as strings starting with `json:` if you read them through
the raw `DatasetView.attributes()` API.

## What attributes can't do

- They're per-dataset, not per-array. Per-variable attrs in xarray
  round-trip via the `{var}.{attr}` flattening, but the raw `set_attribute`
  / `get_attribute` API has no array key. Use the array name as a prefix
  if you need that.
- There's no list-valued attribute type. Use a JSON-encoded string
  (`set_attribute("tags", "json:" + json.dumps([...]))`) or, better, the
  xarray bridge which handles this automatically.
