# Atlas

Python bindings to **[`atlas-rust`](https://github.com/maris-development/atlas)** —
ATLAS (Aggregated Tensor Large Array Store). Think of it as a *zip for
N-dimensional data*: thousands of NetCDF-style datasets in one immutable file,
written from [xarray](https://docs.xarray.dev), with a catalogue you can read
back in a single request.

```python
import atlas
import xarray as xr

with atlas.AtlasWriter.create("/tmp/weather", codec="zstd") as w:
    for path in nc_files:
        w.add_xarray_dataset(xr.open_dataset(path), name=path.stem)

collection = atlas.Atlas.open("/tmp/weather")
collection.list_datasets()          # every dataset, from one range read
collection.attributes("jan_2024")   # its attributes, no further I/O
```

## The shape of it

A collection is **one write-once file**. Every dataset occupies a contiguous
byte range; a footer at the end records where each one lives, along with its
schema and attributes.

```text
my_collection/
├── data.atlas      ATLS │ segment │ segment │ … │ footer │ trailer
└── deleted.mask    optional
```

That buys two things:

- **Metadata is one read.** Opening fetches the footer and nothing else. Listing
  datasets, inspecting schemas, and reading attributes are then free — for ten
  datasets or a million.
- **Data is fetched by the chunk.** Reading a region of an array fetches only
  the chunks it overlaps.

And it costs one: a collection **cannot be modified after it is written**. No
append, no in-place update, no compaction. To change a dataset you rewrite the
collection. The one exception is deleting a dataset, which writes a small mask
file and never touches the container.

## Python writes; Rust reads data

From Python you build collections and read their **metadata**: dataset names,
array names, dtypes, shapes, chunk shapes, fill values, attributes. There is no
`read_array` — array data is read through the Rust API.

That split is deliberate. Building collections from xarray and serving a
catalogue of what they hold is what Python is good for here; pulling array bytes
through the GIL never was.

## What's here

- **One file, many datasets** — a collection holds thousands of named datasets,
  each with its own arrays and attributes.
- **xarray + dask ingest** — `add_xarray_dataset(ds, name)` maps a whole
  `xr.Dataset` across. Dask-backed variables stream block by block, so a dataset
  larger than memory writes without trouble.
- **numpy arrays** — the low-level `write_array` takes a C-contiguous ndarray,
  zero-copy for numeric dtypes.
- **Free metadata** — schemas and attributes come from the footer that opening
  already read. Filtering a fleet of datasets by an attribute costs no I/O.
- **Compression** — zstd, lz4, or none. Blocks record their own codec, so a
  reader is never told which was used.
- **Local or cloud** — a path string, or an
  [obstore](https://github.com/developmentseed/obstore) handle for S3 / GCS /
  Azure / HTTP via `pip install "atlas-python[cloud]"`. See
  [Cloud storage](guides/cloud-storage.md).
- **Sync API, GIL released** — a multi-threaded tokio runtime backs every
  blocking call, so other Python threads keep running.

## How does this compare to Zarr / netCDF?

netCDF and Zarr put one logical dataset in one file or one chunk directory, so a
fleet of N similar datasets becomes N stores — N sets of metadata to open, N
handles, N objects to list. Atlas puts all N in one file with one footer, which
is what makes "many small datasets, same schema" cheap to catalogue.

See [vs Zarr / netCDF](vs-zarr-netcdf.md) for the head-to-head.

## Next steps

- [**Installation**](installation.md) — install the wheel, or build from source.
- [**Quickstart**](quickstart.md) — first collection in five minutes.
- [**Guides**](guides/datasets-and-arrays.md) — the mental model, dtypes,
  attributes, xarray, dask, cloud storage.
- [**API reference**](reference/atlas.md) — generated from the source docstrings.

## Status

- Local filesystem and any [`object_store`](https://docs.rs/object_store)
  backend (S3 / GCS / Azure / HTTP) via the optional obstore dependency.
- Array data is read from Rust only. Python is metadata-only on the read side.
- `bool`, `binary`, `list[...]`, and `fixed_size_list[...,N]` are not yet
  available as array element types from Python. All of them work as *attribute*
  values.
