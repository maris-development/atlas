# Quickstart

This page builds the smallest useful atlas store: one dataset with one
2-D array and a couple of attributes, then closes the store and reopens it
to confirm everything persisted.

## Create, write, flush

```python
import numpy as np
import atlas

with atlas.Atlas.create("/tmp/my_store", codec="zstd") as store:   # (1)
    ds = store.create_dataset("jan_2024")                            # (2)
    ds.define_array(
        "temperature",
        dtype="float32",
        dims=["lat", "lon"],
        shape=[8, 16],
        chunk_shape=[4, 8],
        fill_value=float("nan"),                                     # (3)
    )
    ds.write_array(
        "temperature",
        start=[0, 0],
        data=np.full((8, 16), 20.0, dtype=np.float32),
    )
    ds.set_attribute("month", 1)
    ds.set_attribute("station", "KNMI")
# (4) `with` exit calls store.close() == flush().
```

1. **`Atlas.create(path, codec=...)`** — codec is one of `"zstd"`,
   `"lz4"`, `"none"`. See [Codecs and metadata](guides/codecs-and-meta.md).
2. **`create_dataset(name)`** returns a [`DatasetView`](reference/dataset-view.md);
   `define_array` declares a typed N-D array within it.
3. **`fill_value`** is the scalar returned for unwritten cells. Any *written*
   cell equal to the fill value is counted as a null in [`array_stats`](guides/stats.md).
4. **No I/O until `flush`** — `define_array` and `write_array` only mutate
   in-memory state. The `with` exit (or an explicit `store.flush()` /
   `store.close()`) is the **single durability boundary**.
   See [Durability and flushing](guides/durability.md).

## Reopen and read back

```python
atlas2 = atlas.Atlas.open("/tmp/my_store")                # auto-detects codec / meta format
ds2 = atlas2.open_dataset("jan_2024")

arr = ds2.read_array("temperature")                         # full read -> np.ndarray
chunk = ds2.read_array("temperature", [0, 0], [4, 8])       # partial read

stats = ds2.array_stats("temperature")                      # {row_count, null_count, min, max}
attrs = ds2.attributes()                                    # {"month": 1, "station": "KNMI"}
```

## Next steps

- [**Datasets and arrays**](guides/datasets-and-arrays.md) — the mental model
  and the full `define_array` / `write_array` / `read_array` surface.
- [**xarray integration**](guides/xarray.md) — round-trip whole `xr.Dataset`s
  with one call.
- [**Bulk reads**](guides/bulk-reads.md) — when you need the same slice from
  many datasets at once.
