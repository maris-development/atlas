# Attributes

Each [`DatasetView`](../reference/dataset-view.md) carries typed attributes at
two levels: **dataset-level** (global) attributes and **per-variable**
attributes on individual arrays. Attribute *values* are stored in the `.af`
files — global values in a reserved `_global` file, per-variable values on the
variable's own file. Only the attribute *key names* are recorded in
`atlas.json` (as part of the schema), so listing which attributes exist is
cheap and doesn't load any array bytes.

## Dataset-level (global) attributes

```python
ds = atlas.create_dataset("jan_2024")

ds.set_attribute("month", 1)            # inferred int
ds.set_attribute("station", "KNMI")     # inferred str
ds.set_attribute("calibrated", True)    # inferred bool

ds.get_attribute("month")               # -> 1
ds.get_attribute("missing")             # -> None
ds.attributes()                         # -> {"month": 1, "station": "KNMI", "calibrated": True}
```

## Per-variable attributes

Attributes can also attach to a specific array (e.g. `units` on `temperature`):

```python
ds.define_array("temperature", dtype="float32", dims=["lat", "lon"], shape=[4, 8])
ds.set_array_attribute("temperature", "units", "degC")
ds.set_array_attribute("temperature", "valid_range", [-40.0, 60.0], dtype="f64")

ds.get_array_attribute("temperature", "units")   # -> "degC"
ds.array_attributes("temperature")               # -> {"units": "degC", "valid_range": [...]}
```

`set_array_attribute` raises `KeyError` if the array isn't defined in the
dataset.

## Durability

Attribute writes are buffered in memory and only reach disk on
[`atlas.flush()`](durability.md) (or leaving a `with atlas:` block), together
with the array data. The reserved `_global/data.af` file is created lazily —
only once a dataset actually sets a global attribute.

## The on-disk type system

Atlas's `Attr` type mirrors the underlying `array-format` `AttributeValue`:

| On-disk type | Python type returned on read |
|---|---|
| `bool` | `bool` |
| `int8` / `int16` / `int32` / `int64` | `int` |
| `uint8` / `uint16` / `uint32` / `uint64` | `int` |
| `float32` / `float64` | `float` |
| `string` | `str` |
| `binary` | `bytes` |
| `timestamp_nanoseconds` | `int` (nanoseconds; stored as an RFC 3339 string) |
| any of the above as a list | `list` |

Type is inferred from the Python value by default — `int` → `int64`,
`float` → `float64`, `str` → `string`, `bytes` → `binary`, `bool` → `bool`.

## Overriding inferred types

Pass `dtype=` to narrow or force a specific type (works on both
`set_attribute` and `set_array_attribute`):

```python
ds.set_attribute("sensor_id", 7, dtype="int8")          # stored as int8, range-checked
ds.set_attribute("ratio", 0.5, dtype="float32")         # stored as float32
ds.set_attribute("observed_at",
                 np.datetime64("2024-01-15T10:00", "ns").astype("int64").item(),
                 dtype="timestamp_nanoseconds")
```

Unlike earlier versions, width hints now preserve the storage type: `dtype="int8"`
stores an 8-bit integer, not a widened `int64`. A `timestamp_nanoseconds`
attribute is stored as an RFC 3339 string and restored to a timestamp on read.

## Per-variable xarray attributes

When you write an `xr.Dataset` via `atlas.add_xarray_dataset(ds, name)`, each
variable's `attrs` are stored as **real per-variable attributes** on that
variable's array, and the dataset's own `attrs` become dataset-level
attributes:

```python
ds = xr.Dataset(
    data_vars={"temperature": xr.DataArray(arr, dims=["lat", "lon"],
                                            attrs={"units": "C"})},
    attrs={"station": "KNMI"},
)
atlas.add_xarray_dataset(ds, "jan_2024")

view = atlas.open_dataset("jan_2024")
view.attributes()                        # {"station": "KNMI"}
view.array_attributes("temperature")     # {"units": "C"}
```

On read, `atlas.open_as_xarray_dataset("jan_2024")` puts each variable's attrs
back on the right `DataArray` and the global attrs back on the Dataset.

See [xarray integration](xarray.md) for the full storage convention.

## JSON-encoded "complex" attributes

xarray attribute values are sometimes nested dicts or other structures that
don't map to atlas's on-disk types. The xarray bridge JSON-encodes those and
prefixes the string with `json:`, decoding transparently on read. You generally
don't need to think about this, but it's why some values come back as strings
starting with `json:` if you read them through the raw attribute API. Simple
lists of scalars are stored natively as typed-list attributes.
