# Reading data

Python reads a collection's **metadata**. It does not read array values.

```python
collection = atlas.Atlas.open(path)

collection.list_datasets()                 # ✓
collection.dataset("jan").array_meta("t")  # ✓
collection.attributes("jan")               # ✓

collection.dataset("jan").read_array("t")  # ✗ does not exist
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
view = collection.dataset("jan_2024")

view.list_arrays()            # ['lat', 'lon', 'temperature']
view.array_meta("temperature")
# {'dtype': 'float32', 'shape': [4, 8], 'chunk_shape': [2, 4],
#  'dimension_names': ['lat', 'lon'], 'fill_value': nan}

view.array_fill_value("temperature")
view.ordinal                  # stable position in the collection
view.segment_range            # (start, end) byte offsets in data.atlas

collection.coords("jan_2024")                       # which vars were coords
collection.attributes("jan_2024")                   # decoded dataset attrs
collection.array_attributes("jan_2024", "temperature")
```

Attribute filtering across a whole collection costs nothing beyond the open:

```python
northern = [
    name for name in collection.list_datasets()
    if collection.dataset(name).get_attribute("site") == "north"
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

**Keep the source data.** Collections are usually built from NetCDF files or a
database query. If you need values in Python, read the source rather than the
collection.

**Extract a segment.** `segment_range` gives the byte offsets of a complete,
standalone `array-format` file. Cut it out and any `array-format` reader opens
it directly:

```python
start, end = collection.dataset("jan_2024").segment_range
blob = open(f"{path}/data.atlas", "rb").read()[start:end]
open("jan_2024.af", "wb").write(blob)
```

**Write a small Rust service.** If the values are wanted over a network anyway,
reading them in Rust and serving the result is usually simpler than moving them
through Python.
