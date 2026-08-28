# Attributes

Attributes annotate a dataset or one of its arrays with a typed value. They are
set on a [`DatasetWriter`](../reference/dataset-writer.md) while building, and
read from a [`DatasetView`](../reference/dataset-view.md) afterwards.

Values live in the collection **footer**, not in the array data. That is why
reading them costs nothing beyond opening the collection — and why attribute
filtering across a whole collection is free.

## Setting

```python
ds = writer.add_dataset("jan_2024")

# Dataset-level
ds.set_attribute("month", 1)
ds.set_attribute("station", "KNMI")
ds.set_attribute("calibrated", True)
ds.set_attribute("bounds", [10.0, 60.0])

# Per-array — the array must already be defined
ds.define_array("temperature", dtype="float32", dims=["lat", "lon"], shape=[4, 8])
ds.set_array_attribute("temperature", "units", "degC")
ds.set_array_attribute("temperature", "valid_range", [-40.0, 60.0])
```

Setting a key twice replaces the value. Order is preserved.
`set_array_attribute` raises `KeyError` if the array is not defined.

## Reading

```python
view = collection.dataset("jan_2024")

view.get_attribute("month")                        # 1
view.get_attribute("missing")                      # None
view.attributes()                                  # {'month': 1, 'station': 'KNMI', ...}

view.get_array_attribute("temperature", "units")   # 'degC'
view.array_attributes("temperature")               # {'units': 'degC', ...}
```

For datasets written from xarray, prefer the collection-level accessors — they
decode the encoding conventions and hide the internal markers:

```python
collection.attributes("jan_2024")                       # decoded, marker hidden
collection.array_attributes("jan_2024", "temperature")
collection.coords("jan_2024")                           # which vars were coords
```

## Types

Any scalar: `bool`, the signed and unsigned integer widths, `float32`,
`float64`, `str`, `bytes`, and a nanosecond timestamp. Plus a **homogeneous
list** of any of those.

Inference from a Python value:

| Python | atlas |
|---|---|
| `bool` | `bool` |
| `int` | `int64` |
| `float` | `float64` |
| `str` | `string` |
| `bytes` | `binary` |
| `list` / `tuple` of one of the above | the matching list type |

A list must hold one type throughout; a mixed list raises `ValueError`. An
integer among floats is accepted and widened, since that direction is lossless.

### Narrowing with `dtype=`

Inference gives you `int64` and `float64`. To store something narrower, or a
timestamp:

```python
ds.set_attribute("small", 7, dtype="int32")
ds.set_attribute("ratio", 0.5, dtype="float32")
ds.set_attribute("created", 1_700_000_000_000_000_000, dtype="timestamp_nanoseconds")
```

An out-of-range value raises `OverflowError`.

### Timestamps are a real type

They have their own tag on the wire, so a string that happens to look like a
date stays a string:

```python
ds.set_attribute("when", 1_700_000_000_000_000_000, dtype="timestamp_nanoseconds")
ds.set_attribute("looks_like", "2023-11-14T22:13:20Z")

view.get_attribute("when")        # 1700000000000000000
view.get_attribute("looks_like")  # '2023-11-14T22:13:20Z'  — still a string
```

Atlas 0.14 encoded timestamps as RFC 3339 strings and turned any string that
parsed as one back into a timestamp. That guess is gone.

## Complex values from xarray

xarray attrs are often not scalars — nested dicts, ragged lists, numpy arrays.
`add_xarray_dataset` JSON-encodes anything it cannot store natively and prefixes
it with `json:`. `Atlas.attributes()` decodes them again:

```python
ds_attrs = {"nested": {"a": 1, "b": [2, 3]}}
# ... written via add_xarray_dataset ...
collection.attributes("jan")["nested"]     # {'a': 1, 'b': [2, 3]}
```

The raw `DatasetView.attributes()` returns what is stored, prefix and all.

## Filtering a collection

Because attributes are in the footer, this reads nothing beyond the open:

```python
collection = atlas.Atlas.open(path)

northern = [
    name for name in collection.list_datasets()
    if collection.dataset(name).get_attribute("site") == "north"
]
```

For a collection of ten thousand datasets that is still one request.

## Durability

Attributes are buffered on the `DatasetWriter` and written into the footer when
the collection finishes. Like everything else, they are not readable until then.
See [Immutability](immutability.md).
