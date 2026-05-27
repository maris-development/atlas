# Atlas

`atlas` is the Python binding for **ATLAS** (Aggregated Tensor Large Array
Store), a directory-based store for many similarly-shaped N-dimensional
arrays. One store holds N logical datasets; arrays of the same name across
datasets share a single physical file on disk, so the on-disk footprint grows
with *schemas* rather than *datasets*.

```python
import numpy as np
import atlas

with atlas.Atlas.create("/tmp/my_store", codec="zstd") as atlas:
    ds = atlas.create_dataset("jan_2024")
    ds.define_array(
        "temperature",
        dtype="float32",
        dims=["lat", "lon"],
        shape=[8, 16],
        chunk_shape=[4, 8],
    )
    ds.write_array("temperature", start=[0, 0],
                   data=np.full((8, 16), 20.0, dtype=np.float32))
    ds.set_attribute("month", 1)
# `with` exit == atlas.close() == single flush to disk
```

## What's here

- **Multi-dataset stores** — one `Atlas` directory holds many named
  datasets, each with their own arrays and attributes.
- **Shared physical arrays** — N datasets that all declare `temperature`
  share one `temperature/data.af` file, so the 1000th dataset is just an
  append.
- **Compression** — zstd / lz4 / uncompressed array codecs and json /
  msgpack metadata, optionally compressed.
- **xarray + dask integration** — `atlas.add_xr_dataset(ds, name)` to
  ingest, `atlas.to_xarray(name)` to read back. Dask-backed variables
  stream chunk-by-chunk on write and come back lazily on read.
- **Bulk cross-dataset APIs** — `to_xarray_many` and
  `read_array_across_stacked` collapse N per-dataset reads into one PyO3
  call, sharing a single file handle.
- **Sync API, GIL-released** — a multi-threaded tokio runtime backs every
  blocking call; the GIL is released so other Python threads can run.
- **Local fs or cloud storage** — pass a path string for local, or an
  [obstore](https://github.com/developmentseed/obstore) handle for S3 /
  GCS / Azure / HTTP via `pip install "atlas-python[cloud]"`. The full
  read / write / flush / compact / bulk-read API works identically against
  either. See [Cloud storage](guides/cloud-storage.md).

## How does this compare to Zarr / netCDF?

netCDF and Zarr both put one logical "dataset" in one file (or one chunk
directory); a fleet of N similar datasets becomes N stores. Atlas inverts
that layout — **one** store, **N** datasets, with arrays of the same name
sharing a physical file across all of them. This is the design choice that
makes atlas dramatically faster on "many small datasets, same schema"
workloads and on cross-dataset slice reads.

See [vs Zarr / netCDF](vs-zarr-netcdf.md) for the head-to-head, and
[Benchmarks](benchmarks.md) for the numbers.

## Next steps

- [**Installation**](installation.md) — install the wheel, or build from source.
- [**Quickstart**](quickstart.md) — first store in five minutes.
- [**Guides**](guides/datasets-and-arrays.md) — the mental model, dtypes,
  attributes, xarray, dask, bulk reads, stats.
- [**API reference**](reference/atlas.md) — auto-generated from the
  source-level docstrings.

## Status

- Local filesystem **and** any [`object_store`](https://docs.rs/object_store)
  backend (S3 / GCS / Azure / HTTP / local) via the optional
  [obstore](https://github.com/developmentseed/obstore) dependency.
  `pip install "atlas-python[cloud]"` enables it; see
  [Cloud storage (S3, GCS, Azure)](guides/cloud-storage.md).
- Lazy dask reads use the threaded scheduler only; `.compute()` before
  crossing into distributed/multiprocessing schedulers.
- `bool`, `binary`, `list[...]`, `fixed_size_list[...,N]` are reserved for
  a later release as array element types.
