# atlas-python

Thousands of NetCDF datasets in one immutable file, on local disk or object
storage (S3 / GCS / Azure / HTTP). A Rust core, five operations, and a command.

```bash
pip install atlas-python
```

```bash
atlas create /data/nc /data/collection
atlas ls     /data/collection
atlas show   /data/collection 2024-01
atlas info   /data/collection
atlas rm     /data/collection 2024-02 2024-03
```

| Extra | Install | Adds |
|---|---|---|
| cloud | `pip install "atlas-python[cloud]"` | S3 / GCS / Azure / HTTP via [obstore](https://github.com/developmentseed/obstore) |

`numpy`, `xarray`, and `dask` install automatically.

## Five operations

The same five as a library:

```python
import atlas

atlas.create("/data/nc", "/data/collection")   # from a directory of NetCDF files
atlas.list_datasets("/data/collection")        # ['2024-01', '2024-02', '2024-03']
atlas.describe("/data/collection", "2024-01")  # types, shapes, attrs, statistics
atlas.info("/data/collection")                 # counts, size, codec
atlas.remove("/data/collection", ["2024-02"])  # updates the mask
```

Every one takes a local path, a URL, or an obstore handle:

```bash
atlas ls s3://my-bucket/collections/2024 --region eu-west-1
```

```python
atlas.list_datasets("s3://my-bucket/collections/2024", region="eu-west-1")
```

## Two things to internalise

**A collection is written once.** No append, no in-place update, no `flush` —
the file either has a valid trailer or it is not a collection. To change a
dataset, rebuild the collection. The one exception is `remove`, which writes a
small mask file and never touches the container, so it reclaims no space and
moves no ordinals.

**Python writes; Rust reads array data.** From Python you build collections and
read their *metadata* — dataset names, array types, shapes, chunk shapes, fill
values, attributes, and the statistics recorded at write time. There is no
`read_array`; array values come from the Rust API.

That split is why the read side is free: every metadata call is answered from
the footer that opening already fetched, so cataloguing a thousand datasets is
one request.

## What `show` gives you

```text
$ atlas show /data/collection 2024-01
dataset 2024-01 {
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

Shaped like `ncdump -h`, plus the statistics — minimum, maximum, and how many
elements are missing — computed when each array was written. `--json` on any
read command gives the same thing as a structure.

## Ingest

`create` scans a directory for `.nc`, `.nc4`, `.cdf`, and `.netcdf`, sorts
them, and writes one dataset per file named after the stem. Coordinates and
data variables become arrays; variable attrs become per-array attributes;
`_FillValue` becomes the array's fill.

Files are opened with dask chunking, so one far larger than memory streams
block by block rather than being read whole. `--chunk-size` (default 128 MiB)
sets the block budget and is roughly the memory ceiling per variable; those
blocks also become the stored chunk shape.

Nothing is readable at the destination until every file is written — a failure
part-way leaves no collection, not a partial one. `on_error="skip"`
(`--skip-errors`) trades that for progress.

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

Full docs at **<https://maris-development.github.io/atlas/>** — the
[command reference], plus guides for [creating], [inspecting], [removing],
[dtypes], [reading data], and [cloud storage].

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
