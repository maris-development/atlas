# atlas-python

Thousands of NetCDF datasets in one immutable file. It sits on local disk or on
object storage: S3, GCS, Azure, or HTTP. A Rust core, five operations, and one
command.

```bash
pip install atlas-python
```

```bash
atlas create /data/nc /data/collection
atlas ls     /data/collection
atlas show   /data/collection 2024-01.nc
atlas info   /data/collection
atlas rm     /data/collection 2024-02.nc 2024-03.nc
```

`python -m atlas` runs the same command without a PATH lookup, for a shell that
cannot find `atlas`.

| Extra | Install | Adds |
|---|---|---|
| cloud | `pip install "atlas-python[cloud]"` | S3 / GCS / Azure / HTTP via [obstore](https://github.com/developmentseed/obstore) |

`numpy`, `xarray`, and `dask` install automatically.

## Five operations

The same five as a library:

```python
import atlas

atlas.create("/data/nc", "/data/collection")   # from a directory of NetCDF files
atlas.list_datasets("/data/collection")        # ['2024-01.nc', '2024-02.nc', '2024-03.nc']
atlas.describe("/data/collection", "2024-01.nc")  # types, shapes, attrs, statistics
atlas.info("/data/collection")                 # counts, size, codec, statistics
atlas.remove("/data/collection", ["2024-02.nc"])  # updates the mask
```

Every one takes a local path, a URL, or an obstore handle:

```bash
atlas ls s3://my-bucket/collections/2024 --region eu-west-1
```

```python
atlas.list_datasets("s3://my-bucket/collections/2024", region="eu-west-1")
```

## Two things to internalise

**One write builds a collection.** There is no append, no in-place update, and
no `flush`. The file has a valid trailer, or it is no collection. To change a
dataset, rebuild the collection. `remove` is the one exception. It writes a
small mask file, and never touches the container. It therefore reclaims no
space, and moves no ordinal.

**Python writes. Rust reads array data.** From Python you build a collection
and read its *metadata*. That is the dataset names, the array types, the
shapes, the chunk shapes, the fill values, the attributes, and the statistics
of the write. There is no `read_array`. Array values come from the Rust API.

That split makes the read side free. The footer an open already fetched answers
every metadata call. A catalogue of a thousand datasets is therefore one
request.

## What `show` gives you

```text
$ atlas show /data/collection 2024-01.nc
dataset 2024-01.nc {
dimensions:
	lat = 4 ;
	lon = 6 ;
variables:
	float32 temperature(lat, lon) ;
		temperature:_FillValue = nan ;
		temperature:units = "celsius" ;
		// stats: count=24  min=1.0  max=24.0
	string station(lat) ;
		// stats: count=4  min="a"  max="d"

// global attributes:
		:month = 1 ;

// ordinal 0, segment bytes 8..1691
}
```

The shape follows `ncdump -h`. It adds the statistics of the write: the
minimum, the maximum, and how many elements are missing. `--json` on any read
command gives the same content as a structure.

## Ingest

`create` scans a directory for `.nc`, `.nc4`, `.cdf`, and `.netcdf`, and
descends into every subdirectory. It sorts them, and writes one dataset per
file, named after the file. Each coordinate
and data variable becomes an array. Each variable attribute becomes a per-array
attribute. `_FillValue` becomes the fill of the array.

Each file opens with dask chunking. A file far larger than memory therefore
streams block by block. `--chunk-size` sets the block budget, and defaults to
128 MiB. It is about the memory ceiling per variable. Those blocks also become
the stored chunk shape.

Nothing at the destination is readable until every file lands. A failure
part-way leaves no collection, and not a partial one. `on_error="skip"`, or
`--skip-errors`, trades that for progress.

Atlas cannot store every numpy dtype, and `bool` is the common case. One such
variable fails the whole file by default. `on_unsupported="skip"`, or
`--skip-unsupported`, leaves out that one array and lands the rest.

`--log-file PATH`, or `atlas.log_to_file(path)`, appends every error and
warning to a file, each with its reason and the file it came from.

## dtypes

| numpy | atlas |
|---|---|
| int / uint widths, `float32`, `float64` | the same |
| `datetime64[ns]` | `timestamp_nanoseconds` |
| `timedelta64[*]` | `int64` nanoseconds, plus a unit marker |
| `object` / `S` / `U` | `string` |

`bool`, `binary`, and the list types work as an *attribute* value. No one of
them works yet as an array element type.

## Documentation

The full docs sit at **<https://maris-development.github.io/atlas/>**. They
hold the [command reference], and guides for [creating], [inspecting],
[removing], [dtypes], [reading data], and [cloud storage].

The format itself is documented in
[`docs/`](https://github.com/maris-development/atlas/tree/main/docs).

[command reference]: https://maris-development.github.io/atlas/cli/
[creating]: https://maris-development.github.io/atlas/guides/creating/
[inspecting]: https://maris-development.github.io/atlas/guides/inspecting/
[removing]: https://maris-development.github.io/atlas/guides/removing/
[dtypes]: https://maris-development.github.io/atlas/guides/dtypes/
[reading data]: https://maris-development.github.io/atlas/guides/reading-data/
[cloud storage]: https://maris-development.github.io/atlas/guides/cloud-storage/

## License

Apache-2.0. See [LICENSE](https://github.com/maris-development/atlas/blob/main/LICENSE).
