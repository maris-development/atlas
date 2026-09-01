# atlas

Thousands of NetCDF datasets in one immutable file. It sits on local disk or on
object storage.

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

A collection is **one write-once file**. Every dataset occupies one contiguous
byte range. A footer at the end records where each one lives, with its schema,
its attributes, and its statistics.

```text
my_collection/
├── data.atlas      ATLS │ segment │ segment │ … │ footer │ trailer
└── deleted.mask    optional
```

That buys one thing above all. **Metadata is one read.** An open fetches the
footer and nothing else. To list the datasets, to inspect a schema, and to read
the statistics are then free. Three datasets and a million cost the same, on a
local disk and across an ocean.

It costs one thing. A collection **cannot change after a write**. There is no
append, and no in-place update. To change a dataset, rebuild the collection. A
remove is the one exception. It writes a small mask file, and leaves the
container alone.

## Five operations

| Library | Command | Does |
|---|---|---|
| `atlas.create` | `atlas create` | Build a collection from a directory of NetCDF files |
| `atlas.remove` | `atlas rm` | Remove datasets, in one call |
| `atlas.list_datasets` | `atlas ls` | What the collection holds |
| `atlas.describe` | `atlas show` | One dataset in detail, `ncdump` style |
| `atlas.info` | `atlas info` | The collection as a whole |

Every one takes a local path or a URL: `s3://`, `gs://`, `az://`, or
`https://`. The same call therefore works against a bucket.

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

Types, shapes, chunking, fill values, attributes, and the statistics of the
write. Those statistics are the minimum, the maximum, and how many elements are
missing. It all comes from the footer, so it needs no more I/O than `ls`.

## Python writes. Rust reads data

The Rust API reads array *values*. Python does not. Python gives the structure.
Which datasets exist, what arrays they hold, and everything on record about
them.

That split is deliberate. Python builds collections from NetCDF, and serves a
catalogue of what they hold. Array bytes through the GIL were never the fast
path. See [Reading data](guides/reading-data.md).

## Compared to Zarr and netCDF

netCDF and Zarr put one dataset in one file or one chunk directory. N similar
datasets therefore become N stores. That is N sets of metadata to open, and N
objects to list. Atlas puts all N in one file, behind one footer. Many small
datasets of one schema are therefore cheap to catalogue.

Atlas suits you poorly in three cases. You must change data. You have one large
array instead of many small datasets. Or you need the values back in Python.
See [vs Zarr / netCDF](vs-zarr-netcdf.md) for the full account.

## Next

- [**Installation**](installation.md)
- [**Quickstart**](quickstart.md). A collection in five minutes
- [**The `atlas` command**](cli.md). Every subcommand and flag
- [**Guides**](guides/creating.md). Creating, inspecting, removing, cloud storage
- [**API reference**](reference/api.md)

## Status

- The local filesystem, and any [`object_store`](https://docs.rs/object_store)
  backend: S3, GCS, Azure, or HTTP. Those need the optional obstore package.
- NetCDF is the one ingest route from Python.
- Rust alone reads array data.
- `bool`, `binary`, and the list types work as an attribute value. No one of
  them works yet as an array element type.
