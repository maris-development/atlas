# Inspecting a collection

Three operations, all answered from the footer that opening already read. None
of them fetch array data, so cataloguing a collection of ten thousand datasets
costs one request.

## What is in here — `list_datasets` / `ls`

```python
atlas.list_datasets("/data/collection")
# ['2024-01', '2024-02', '2024-03']
```

```bash
atlas ls /data/collection
```

Names in write order. Removed datasets are not listed.

## The collection as a whole — `info`

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
    "interned_schemas": 1,
}
```

`deleted_count` and `total_datasets` together tell you how much of the file is
dead weight. `interned_schemas` is how many distinct schemas the datasets share
between them — a fleet of a thousand identically-shaped datasets shows `1`.

## One dataset in detail — `describe` / `show`

```python
atlas.describe("/data/collection", "2024-01")
```

```bash
atlas show /data/collection 2024-01
```

The CLI renders it like `ncdump -h`; the library returns the structure. Either
way you get dimensions, and for every array its type, shape, chunk shape,
dimension names, fill value, attributes, whether it was a coordinate, and its
statistics.

`name` may be a dataset name or the NetCDF path it came from:

```python
atlas.describe(collection, "/data/nc/2024-01.nc")   # same as "2024-01"
```

### Statistics

Each array carries what was recorded when it was written:

```python
{"min": 1.0, "max": 24.0, "null_count": 0, "row_count": 24}
```

| Field | Meaning |
|---|---|
| `row_count` | Total elements across every chunk |
| `null_count` | Elements equal to the fill value — how a never-written cell is stored |
| `min` / `max` | Extremes. `None` for a dtype with no ordering |

These come from the footer, not from the data, so reading them is free.

Two things worth knowing:

**An array declared and never written reports `row_count == null_count`.** Every
element is a hole. That is how you spot a variable that was present in the
NetCDF header but carried no data.

**String extremes are bytes**, compared lexicographically: `b"alpha"` and
`b"gamma"`. In `--json` output they are decoded to text so the output stays
valid JSON.

### Coordinates

Which variables were xarray coordinates is recorded at ingest and reported
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

The bytes this dataset occupies in `data.atlas`. They are a complete,
standalone `array-format` file:

```python
start, end = detail["segment_range"]
blob = open(f"{collection}/data.atlas", "rb").read()[start:end]
open("2024-01.af", "wb").write(blob)
```

## Filtering a fleet

Because attributes and statistics are both in the footer, filtering across a
whole collection costs one request:

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

That reads the footer once per `describe` call. To read it exactly once, work
from `--json` output, or accept the repeat — it is a single range read either
way.

## Reading array values

Not from here. See [Reading data](reading-data.md).
