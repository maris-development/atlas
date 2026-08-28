# Quickstart

The smallest useful collection: one dataset, one 2-D array, a couple of
attributes. Then reopen it and inspect what landed.

## Write

```python
import numpy as np
import atlas

with atlas.AtlasWriter.create("/tmp/my_collection", codec="zstd") as w:   # (1)
    ds = w.add_dataset("jan_2024")                                        # (2)
    ds.define_array(
        "temperature",
        dtype="float32",
        dims=["lat", "lon"],
        shape=[8, 16],
        chunk_shape=[4, 8],                                               # (3)
        fill_value=float("nan"),                                          # (4)
    )
    ds.write_array(
        "temperature",
        start=[0, 0],
        data=np.full((8, 16), 20.0, dtype=np.float32),                    # (5)
    )
    ds.set_attribute("month", 1)
    ds.set_array_attribute("temperature", "units", "celsius")
    ds.finish()                                                           # (6)
# (7)
```

1. **`AtlasWriter.create(path, codec=...)`** — codec is `"zstd"` (default),
   `"lz4"`, or `"none"`. See [Codecs](guides/codecs.md).
2. **`add_dataset(name)`** returns a
   [`DatasetWriter`](reference/dataset-writer.md).
3. **`chunk_shape`** is the storage granularity, and it is what makes a later
   partial read cheap. Defaults to `shape` — one chunk, read whole or not at all.
4. **`fill_value`** is what a read returns for cells never written. They cost no
   bytes.
5. **numpy in, zero copy** — a C-contiguous array whose dtype matches the
   declared one.
6. **`ds.finish()`** is when the dataset enters the file. Drop the writer
   instead and it never does.
7. **The `with` exit writes the footer.** Nothing at the path is readable before
   that, and an exception inside the block leaves nothing at all.

## Reopen

```python
collection = atlas.Atlas.open("/tmp/my_collection")   # one range read

collection.list_datasets()      # ['jan_2024']
collection.list_arrays()        # ['temperature']
len(collection)                 # 1
"jan_2024" in collection        # True

view = collection.dataset("jan_2024")
view.list_arrays()              # ['temperature']
view.array_meta("temperature")
# {'dtype': 'float32', 'shape': [8, 16], 'chunk_shape': [4, 8],
#  'dimension_names': ['lat', 'lon'], 'fill_value': nan}

collection.attributes("jan_2024")                       # {'month': 1}
collection.array_attributes("jan_2024", "temperature")  # {'units': 'celsius'}
```

Every call after `open` is answered from the footer, with no further I/O.

## Reading the data

You don't — not from Python. There is no `read_array` on `Atlas` or
`DatasetView`. Array data is read through the Rust API:

```rust
let atlas = Atlas::open_path("/tmp/my_collection").await?;
let ds = atlas.dataset("jan_2024")?;
let window = ds.read_array::<f32>("temperature", vec![0, 0], vec![4, 8]).await?;
```

See [Reading data](guides/reading-data.md) for why, and what to do if you need
values in Python.

## Deleting

```python
collection.delete_dataset("jan_2024")
```

Writes a small `deleted.mask` beside the container and hides the dataset. The
container itself is untouched, so this reclaims no space and moves no ordinals.
Rewrite the collection to get the bytes back.

## Next steps

- [**Datasets and arrays**](guides/datasets-and-arrays.md) — the mental model
  and the full `define_array` / `write_array` surface.
- [**xarray integration**](guides/xarray.md) — write a whole `xr.Dataset` with
  one call.
- [**Dask streaming**](guides/dask.md) — datasets larger than memory.
