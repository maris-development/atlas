# Reading data

Python reads the **metadata** of a collection. It reads no array value.

```python
atlas.list_datasets(collection)            # ✓
atlas.describe(collection, "jan")          # ✓  types, shapes, attrs, stats
atlas.info(collection)                     # ✓  counts, size, collection stats

atlas.read_array(collection, "jan", "t")   # ✗ does not exist
```

## Why

The design gives each language one job. Python builds collections from xarray,
and serves a catalogue of what they hold. It is good at both, and neither needs
an array byte. To read array data means to decompress chunks and to assemble
regions. That is the job of Rust, and it never belonged across the GIL.

The result is that the read side of the Python API is *free*. The footer that
`open` already fetched answers every call above. A service that lists a
thousand datasets and their schemas therefore issues one request.

## What metadata gives you

It is enough to build a catalogue, to check an ingest, or to choose the
datasets worth a fetch:

```python
detail = atlas.describe(collection, "2024-01")

detail["dimensions"]     # {'lat': 4, 'lon': 6}
detail["coordinates"]    # ['lat', 'lon']
detail["attributes"]     # decoded, markers hidden
detail["ordinal"]        # stable position in the collection
detail["segment_range"]  # [start, end] byte offsets in data.atlas

for array in detail["arrays"]:
    array["dtype"], array["shape"], array["chunk_shape"]
    array["fill_value"], array["attributes"], array["is_coordinate"]
    array["stats"]       # {'min', 'max', 'null_count', 'row_count'}
```

The statistics deserve a mention. Each array records a minimum, a maximum, and
how many elements are missing. That is often the real question. *Does this
dataset hold anything above 30?* The answer costs no data byte.

`info` answers the same question for the collection:

```python
stats = atlas.info(collection)["array_stats"]

stats["temperature"]   # {'min': 1.0, 'max': 26.0, 'null_count': 0, 'row_count': 72}
```

The counts add up over every live dataset that holds the array. The minimum is
the smallest of the minimums. The maximum is the largest of the maximums.
One call covers a thousand datasets. It costs one range read.

To filter across a whole collection costs nothing beyond the footer reads:

```python
hot = [
    name
    for name in atlas.list_datasets(collection)
    for array in atlas.describe(collection, name)["arrays"]
    if array["name"] == "temperature" and array["stats"]["max"] > 30
]
```

## Reading values, in Rust

```rust
use atlas::Atlas;

let atlas = Atlas::open_path("/data/weather").await?;
let ds = atlas.dataset("jan_2024")?;

// The whole array.
let all = ds.read_array::<f32>("temperature", vec![], vec![]).await?;

// A window: only the chunks it overlaps are fetched.
let window = ds.read_array::<f32>("temperature", vec![1, 3], vec![2, 2]).await?;
```

The first read on a dataset opens its segment. Every later read reuses it. See
`docs/read-path.md` in the repository for the cost of each call.

## If you need values in Python anyway

Three options, from the least work to the most.

**Keep the source data.** A collection comes from NetCDF files, and those files
are the obvious place to read a value. `xr.open_dataset` on the original is
simpler than anything else here.

**Extract a segment.** `segment_range` gives the byte offsets of a complete
`array-format` file that stands alone. Cut it out, and any `array-format`
reader opens it:

```python
start, end = atlas.describe(collection, "2024-01")["segment_range"]
blob = open(f"{collection}/data.atlas", "rb").read()[start:end]
open("2024-01.af", "wb").write(blob)
```

**Write a small Rust service.** The values often travel over a network anyway.
To read them in Rust and serve the result is then simpler than a path through
Python.
