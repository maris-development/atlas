# Reading data

Python reads a collection's **metadata**. It does not read array values.

```python
atlas.list_datasets(collection)            # ✓
atlas.describe(collection, "jan")          # ✓  types, shapes, attrs, stats
atlas.info(collection)                     # ✓

atlas.read_array(collection, "jan", "t")   # ✗ does not exist
```

## Why

The design puts one job in each language. Python builds collections from xarray
and serves a catalogue of what they hold — both of which it is good at, and
neither of which needs array bytes. Reading array data means decompressing
chunks and assembling regions, which is Rust's job and never wanted to cross the
GIL.

The practical upshot is that the read side of the Python API is *free*: every
call above is answered from the footer that `open` already fetched, so a service
listing a thousand datasets and their schemas issues one request.

## What metadata gives you

Enough to build a catalogue, validate an ingest, or decide which datasets are
worth fetching:

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

Statistics deserve a mention: minimum, maximum, and how many elements are
missing, for every array, recorded when it was written. Often that is the
question you actually had — *does this dataset contain anything above 30?* —
and it is answered without fetching a byte of data.

Filtering across a whole collection costs nothing beyond the footer reads:

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

The first read on a dataset opens its segment; subsequent reads reuse it. See
`docs/read-path.md` in the repository for what each call costs.

## If you need values in Python anyway

Three options, in order of how much work they are.

**Keep the source data.** A collection is built from NetCDF files, and those
files are the obvious place to read values from. `xr.open_dataset` on the
original is simpler than anything else here.

**Extract a segment.** `segment_range` gives the byte offsets of a complete,
standalone `array-format` file. Cut it out and any `array-format` reader opens
it directly:

```python
start, end = atlas.describe(collection, "2024-01")["segment_range"]
blob = open(f"{collection}/data.atlas", "rb").read()[start:end]
open("2024-01.af", "wb").write(blob)
```

**Write a small Rust service.** If the values are wanted over a network anyway,
reading them in Rust and serving the result is usually simpler than moving them
through Python.
