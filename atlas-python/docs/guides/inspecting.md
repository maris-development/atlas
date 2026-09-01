# Inspecting a collection

Three operations. The footer the open already read answers all of them. None
fetches array data, so a catalogue of ten thousand datasets costs one request.

## What is in here: `list_datasets` and `ls`

```python
atlas.list_datasets("/data/collection")
# ['2024-01', '2024-02', '2024-03']
```

```bash
atlas ls /data/collection
```

The names, in write order. A removed dataset does not appear.

## The collection as a whole: `info`

```python
atlas.info("/data/collection")
```

```bash
atlas info /data/collection
```

```python
{
    "source": "/data/collection",
    "format_version": 1,
    "created_unix_ms": 1788165177965,
    "codec": "zstd",
    "container_bytes": 5571,
    "dataset_count": 2,        # live
    "deleted_count": 1,        # hidden by the mask, bytes still present
    "total_datasets": 3,       # written
    "distinct_arrays": ["lat", "lon", "station", "temperature"],
    "array_stats": {
        "lat": {"min": 0.0, "max": 3.0, "null_count": 0, "row_count": 8},
        "lon": {"min": 0.0, "max": 5.0, "null_count": 0, "row_count": 12},
        "station": {"min": b"a", "max": b"d", "null_count": 0, "row_count": 8},
        "temperature": {"min": 1.0, "max": 26.0, "null_count": 0, "row_count": 48},
    },
    "interned_schemas": 1,
}
```

`deleted_count` and `total_datasets` together say how much of the file is dead
weight. `interned_schemas` is how many distinct schemas the datasets share
between them. A fleet of a thousand datasets of one shape shows `1`.

`array_stats` gives one set of statistics per array, for the whole collection.
The counts add up over every live dataset that holds the array. The minimum is
the smallest of the minimums. The maximum is the largest of the maximums. A
removed dataset counts for nothing. The value is `None` if no live dataset
records statistics for that array. A dataset that declares the same name with a
different dtype stays out, because two dtypes do not compare.

Use `describe` for the statistics of one dataset on its own.

## One dataset in detail: `describe` and `show`

```python
atlas.describe("/data/collection", "2024-01")
```

```bash
atlas show /data/collection 2024-01
```

The CLI prints it like `ncdump -h`. The library returns the structure. Both
give the dimensions. For every array both give the type, the shape, and the
chunk shape. They also give the dimension names, the fill value, the
attributes, the coordinate flag, and the statistics.

`name` is a dataset name, or the NetCDF path the dataset came from:

```python
atlas.describe(collection, "/data/nc/2024-01.nc")   # same as "2024-01"
```

### Statistics

Each array carries what the write recorded:

```python
{"min": 1.0, "max": 24.0, "null_count": 0, "row_count": 24}
```

| Field | Meaning |
|---|---|
| `row_count` | Total elements across every chunk |
| `null_count` | Elements equal to the fill value. That is how a cell nobody wrote is stored |
| `min` / `max` | The two bounds. `None` for a dtype with no order |

These come from the footer, and not from the data. To read them is free.

Two points matter:

**An array somebody declared and never wrote reports
`row_count == null_count`.** Every element is a hole. That is how you find a
variable the NetCDF header names, and no data fills.

**String extremes are bytes**, compared lexicographically: `b"alpha"` and
`b"gamma"`. In `--json` output they are decoded to text so the output stays
valid JSON.

`info` reports the same four fields for the whole collection. The two answer
different questions. `describe` says what one month holds. `info` says what the
collection holds.

### Coordinates

The ingest records which variables were xarray coordinates, and reports them
back:

```python
detail = atlas.describe(collection, "2024-01")
detail["coordinates"]              # ['lat', 'lon']
[a["name"] for a in detail["arrays"] if a["is_coordinate"]]
```

The CLI marks them with `// coordinate`.

### Segment range

```python
detail["segment_range"]   # [8, 1691]
```

The bytes this dataset occupies in `data.atlas`. They are a complete
`array-format` file that stands alone:

```python
start, end = detail["segment_range"]
blob = open(f"{collection}/data.atlas", "rb").read()[start:end]
open("2024-01.af", "wb").write(blob)
```

## Filtering a fleet

The footer holds both the attributes and the statistics. To filter across a
whole collection therefore costs one request:

```python
import atlas

collection = "s3://bucket/2024"
hot = [
    name
    for name in atlas.list_datasets(collection)
    for array in atlas.describe(collection, name)["arrays"]
    if array["name"] == "temperature" and array["stats"]["max"] > 30
]
```

That reads the footer once per `describe` call. To read it once in total, work
from the `--json` output. Or accept the repeat, because each one is a single
range read.

A question about the collection as a whole needs no loop at all:

```python
atlas.info(collection)["array_stats"]["temperature"]["max"]   # highest anywhere
```

## Reading array values

Not from here. See [Reading data](reading-data.md).
