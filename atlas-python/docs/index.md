# atlas

Thousands of NetCDF datasets in one immutable file, on local disk or object
storage.

```bash
pip install atlas-python
atlas create /data/nc /data/collection
```

```text
$ atlas ls /data/collection
2024-01
2024-02
2024-03
```

## The shape of it

A collection is **one write-once file**. Every dataset occupies a contiguous
byte range; a footer at the end records where each one lives, along with its
schema, attributes, and statistics.

```text
my_collection/
├── data.atlas      ATLS │ segment │ segment │ … │ footer │ trailer
└── deleted.mask    optional
```

That buys one thing above all: **metadata is a single read.** Opening fetches
the footer and nothing else, so listing datasets, inspecting schemas, and
reading statistics are free — for three datasets or a million, on a local disk
or across the Atlantic.

It costs one thing: a collection **cannot be modified once written**. No append,
no in-place update. To change a dataset you rebuild the collection. The one
exception is removing datasets, which writes a small mask file and leaves the
container alone.

## Five operations

| Library | Command | Does |
|---|---|---|
| `atlas.create` | `atlas create` | Build a collection from a directory of NetCDF files |
| `atlas.remove` | `atlas rm` | Remove datasets, in one call |
| `atlas.list_datasets` | `atlas ls` | What the collection holds |
| `atlas.describe` | `atlas show` | One dataset in detail, `ncdump` style |
| `atlas.info` | `atlas info` | The collection as a whole |

Every one takes a local path or a URL — `s3://`, `gs://`, `az://`, `https://` —
so the same call works against a bucket.

```python
import atlas

atlas.create("/data/nc", "s3://bucket/2024", region="eu-west-1")
atlas.list_datasets("s3://bucket/2024", region="eu-west-1")
```

## What `show` gives you

```text
$ atlas show /data/collection 2024-01
dataset 2024-01 {
dimensions:
	lat = 4 ;
	lon = 6 ;
variables:
	float32 temperature(lat, lon) ;
		temperature:units = "celsius" ;
		// stats: count=24  min=1.0  max=24.0
	string station(lat) ;
		// stats: count=4  min="a"  max="d"

// global attributes:
		:month = 1 ;
}
```

Types, shapes, chunking, fill values, attributes, and the statistics recorded
when each array was written — minimum, maximum, and how many elements are
missing. All from the footer, so it needs no more I/O than `ls` did.

## Python writes; Rust reads data

Array *values* are read through the Rust API, not from Python. What Python
gives you is the structure: which datasets exist, what arrays they hold, and
everything recorded about them.

That split is deliberate. Building collections from NetCDF and serving a
catalogue of what they hold is what Python is good for here; pulling array
bytes through the GIL never was. See [Reading data](guides/reading-data.md).

## Compared to Zarr and netCDF

netCDF and Zarr put one dataset in one file or one chunk directory, so N
similar datasets become N stores — N sets of metadata to open, N objects to
list. Atlas puts all N in one file with one footer, which is what makes "many
small datasets, same schema" cheap to catalogue.

It is a poor fit when you need to modify data, when you have one big array
rather than many small datasets, or when you need the values back in Python.
See [vs Zarr / netCDF](vs-zarr-netcdf.md) for the honest version.

## Next

- [**Installation**](installation.md)
- [**Quickstart**](quickstart.md) — a collection in five minutes
- [**The `atlas` command**](cli.md) — every subcommand and flag
- [**Guides**](guides/creating.md) — creating, inspecting, removing, cloud storage
- [**API reference**](reference/api.md)

## Status

- Local filesystem and any [`object_store`](https://docs.rs/object_store)
  backend (S3 / GCS / Azure / HTTP) via the optional obstore dependency.
- NetCDF is the only ingest route from Python.
- Array data is read from Rust only.
- `bool`, `binary`, and the list types are not yet available as array element
  types. They work as attribute values.
