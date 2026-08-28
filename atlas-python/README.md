# atlas-python

Python bindings for **ATLAS** (Aggregated Tensor Large Array Store) — thousands
of N-dimensional datasets in one immutable file, on local disk or any object
store (S3 / GCS / Azure / HTTP). A Rust core with a synchronous, NumPy-native
write API and first-class [xarray](https://docs.xarray.dev) integration.

```bash
pip install atlas-python
```

| Extra | Install | Adds |
|---|---|---|
| cloud | `pip install "atlas-python[cloud]"` | S3 / GCS / Azure / HTTP via [obstore](https://github.com/developmentseed/obstore) |

`numpy`, `xarray`, and `dask` are installed automatically.

## Quick start

```python
import numpy as np
import atlas

# Nothing is readable until the `with` block exits and the footer is written.
with atlas.AtlasWriter.create("/tmp/my_collection", codec="zstd") as w:
    ds = w.add_dataset("jan_2024")
    ds.define_array(
        "temperature",
        dtype="float32",
        dims=["lat", "lon"],
        shape=[8, 16],
        chunk_shape=[4, 8],
        fill_value=float("nan"),
    )
    ds.write_array("temperature", start=[0, 0],
                   data=np.full((8, 16), 20.0, dtype=np.float32))
    ds.set_attribute("month", 1)
    ds.set_array_attribute("temperature", "units", "celsius")
    ds.finish()

# Reopen. One range read, whatever the collection size.
collection = atlas.Atlas.open("/tmp/my_collection")
collection.list_datasets()              # ['jan_2024']
collection.attributes("jan_2024")       # {'month': 1}

view = collection.dataset("jan_2024")
view.array_meta("temperature")
# {'dtype': 'float32', 'shape': [8, 16], 'chunk_shape': [4, 8],
#  'dimension_names': ['lat', 'lon'], 'fill_value': nan}
```

## Two things to internalise

**A collection is written once.** There is no append, no in-place update, no
compaction, and no `flush` — the file either has a valid trailer or it is not a
collection. To change a dataset you rewrite the collection. The one exception is
`delete_dataset`, which writes a small mask file and never touches the container.

**Python writes; Rust reads array data.** From Python you build collections and
read their *metadata* — dataset names, array names, dtypes, shapes, chunk
shapes, fill values, attributes. There is no `read_array`. Array values are read
through the Rust API.

That split is why the read side is free: every metadata call is answered from
the footer that `open` already fetched, so cataloguing a thousand datasets is
one request.

## xarray

```python
import atlas
import xarray as xr
from pathlib import Path

with atlas.AtlasWriter.create("/data/collection") as w:
    for p in sorted(Path("/data/nc").glob("*.nc")):
        w.add_xarray_dataset(xr.open_dataset(p), name=p.stem)
```

Coordinates and data variables become arrays, variable attrs become per-array
attributes, dataset attrs become dataset-level attributes, and which variables
were coordinates is recorded so `Atlas.coords()` can tell you afterwards.

Dask-backed variables stream one block at a time, so a dataset far larger than
memory writes without trouble, and the dask chunking becomes the on-disk chunk
shape.

The write is atomic per dataset: one that fails partway never enters the
collection, and the writer carries on.

The accessor form is equivalent:

```python
ds.atlas.write(w, "jan_2024")
```

## Cloud storage

```python
import obstore as obs

store = obs.store.S3Store("my-bucket", prefix="collections/2024", region="us-east-1")

with atlas.AtlasWriter.create(store) as w:
    ...

collection = atlas.Atlas.open(store)
```

Writing is one multipart upload; opening is one range read. Credentials are
obstore's business — atlas never sees them.

## dtypes

| numpy | atlas |
|---|---|
| int / uint widths, `float32`, `float64` | the same |
| `datetime64[ns]` | `timestamp_nanoseconds` |
| `timedelta64[*]` | `int64` nanoseconds, plus a unit marker |
| `object` / `S` / `U` | `string` |

`bool`, `binary`, and the list types work as *attribute* values but are not yet
available as array element types.

## Documentation

Full docs at **<https://maris-development.github.io/atlas/>** — guides for
[datasets and arrays], [immutability], [reading data], [attributes], [dtypes],
[xarray], [dask], [codecs], and [cloud storage], plus the API reference.

The format itself is documented in [`docs/`](https://github.com/maris-development/atlas/tree/main/docs)
in the repository.

[datasets and arrays]: https://maris-development.github.io/atlas/guides/datasets-and-arrays/
[immutability]: https://maris-development.github.io/atlas/guides/immutability/
[reading data]: https://maris-development.github.io/atlas/guides/reading-data/
[attributes]: https://maris-development.github.io/atlas/guides/attributes/
[dtypes]: https://maris-development.github.io/atlas/guides/dtypes/
[xarray]: https://maris-development.github.io/atlas/guides/xarray/
[dask]: https://maris-development.github.io/atlas/guides/dask/
[codecs]: https://maris-development.github.io/atlas/guides/codecs/
[cloud storage]: https://maris-development.github.io/atlas/guides/cloud-storage/

## Migrating from 0.14

0.15 cannot read a 0.14 store — the layouts share no bytes. Migration means
reading with 0.14 and writing with 0.15, in two environments. See
[docs/migration.md](https://github.com/maris-development/atlas/blob/main/docs/migration.md).

## License

Apache-2.0. See [LICENSE](https://github.com/maris-development/atlas/blob/main/LICENSE).
