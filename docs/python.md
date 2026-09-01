# The Python package

## What it is

Five operations, exposed as a library and as the `atlas` command:

| Operation | Command | Does |
|---|---|---|
| `create` | `atlas create` | Build a collection from a directory of NetCDF files |
| `remove` | `atlas rm` | Remove datasets, in one call |
| `list_datasets` | `atlas ls` | What the collection holds |
| `describe` | `atlas show` | One dataset in detail, `ncdump` style |
| `info` | `atlas info` | The collection as a whole |

Every one takes a local path, a URL (`s3://`, `gs://`, `az://`, `https://`), or
an obstore handle. The same call therefore works against a bucket.

There is nothing else. No writer object, no `define_array`, and no
`read_array`. NetCDF is the one way data goes in. Array values come back out
through the Rust API.

## The layers

| Layer | Lives in | Does |
|---|---|---|
| `atlas` | `atlas-python/python/atlas/` | The five operations, the CLI, the xarray mapping |
| `atlas._atlas` | `atlas-python/src/*.rs` | PyO3: numpy ⇄ ndarray, Python ⇄ `Attr`, error mapping |

`_atlas` is private. It still exposes the writer classes, because the ingest
path needs them. They are no part of the package surface, and nothing outside
`atlas._ops` touches them.

**The format is Rust only.** `atlas-python` holds no format knowledge. Grep it
for `ATLS`, and you get nothing.

## Modules

| File | Role |
|---|---|
| `__init__.py` | The five operations and the two error types. `__all__` is the contract |
| `__init__.pyi` | The typed surface, with the full docstrings |
| `_ops.py` | The operations themselves |
| `_cli.py` | Argument parsing and output formatting |
| `_source.py` | Turning a path or URL into something the bindings accept |
| `xarray.py` | The xarray → atlas mapping. Internal |

## Ingest

`create` scans a directory for `.nc`, `.nc4`, `.cdf`, and `.netcdf` files, and
descends into every subdirectory. It sorts them, and writes one dataset per
file, named after the file. `2024-01.nc` becomes `2024-01.nc`, suffix and all.
The sort makes the ordinals reproducible. Build the same directory twice, and
every dataset lands at the same position.

A dataset name carries no directory, because a name may hold no `/`. Two files
of one name in two subdirectories therefore collide. `recursive=False` scans
the top directory alone.

One writer does all of it. Nothing at the destination is readable until the
last file lands, with the footer. A failure part-way leaves no collection. That
is the behaviour a job needs when it dies overnight.

`on_error="skip"` trades that for progress. The result records each file that
fails, and the rest continue.

`on_unsupported` works one level down, at the array. Atlas stores no `bool`
array, so a NetCDF file with one fails by default. `on_unsupported="skip"`
leaves that array out of the schema and lands the rest of the dataset. Atlas
resolves every dtype before it defines the first array, so a skip never leaves
a half-written array behind. The result lists each one under
`skipped_arrays`.

## Logging

Every module logs to the `atlas` logger, and the package attaches no handler of
its own. A library user therefore sees nothing until they add one.

`atlas.log_to_file(path)` attaches a file handler and returns it. The `atlas`
command does the same for `--log-file PATH`. The file records each skipped
file, each skipped array, and every error, with the reason and the source file
name. It opens in append mode.

`log_to_file` also captures Python warnings, such as the one about missing
string cells. That moves them off stderr, because `logging.captureWarnings` is
process-wide. A second call for a path that is already attached returns the
handler it already has, so no line lands twice.

The Rust core logs through `tracing`, which is separate. `init_tracing` sends
that stream to stderr.

### The mapping

| xarray | atlas |
|---|---|
| coordinate or data variable | an array of the same name |
| `var.dims` | `dimension_names` |
| `var.shape` | `shape` |
| dask chunking | `chunk_shape`, unless `chunks=` overrides it |
| `var.attrs` | per-array attributes |
| `ds.attrs` | dataset-level attributes |
| `_FillValue` | the array's fill value, not an attribute |
| which variables were coords | the `_pyatlas_coords` marker |

The coordinates land first, then the data variables. The on-disk order is
therefore predictable.

### dtypes

Integer widths and `float32`/`float64` map straight through. Beyond that:

| numpy | atlas | note |
|---|---|---|
| `datetime64[ns]` | `timestamp_nanoseconds` | only `[ns]` |
| `timedelta64[*]` | `int64` | normalized to ns, tagged `_pyatlas_timedelta` |
| `object` / `S` / `U` | `string` | variable length |
| anything else | none | `NotImplementedError` |

Atlas has no duration type. A timedelta therefore becomes int64 nanoseconds,
with a marker that names the unit. A surrogate-escaped string is common from a
netCDF backend. Atlas cleans one on the way in.

### Fill values

xarray defaults to `mask_and_scale=True`. A read of a NetCDF file then leaves
`NaN` and `NaT` where data is missing, and moves `_FillValue` into
`var.encoding`. Atlas records those cells as never written. Each array takes a
default sentinel: `NaN` for a float, `NaT` for a datetime, `""` for a string,
and none for an integer.

A missing *string* cell is the one lossy case. Atlas cannot store a null
string. Each one takes the fill instead, and a warning names the count.

### Streaming

Each file opens with dask chunking, under `chunks="auto"` by default. Every
variable therefore arrives as blocks, and not whole. The blocks land one at a
time. A background thread prefetches them, so NetCDF reads overlap atlas
writes.

**Bytes** size the prefetch batch, not the block count. A batch sized by count
suits the many-small-chunks case, where it covers the dask scheduler overhead.
It ruins the large-block case. Eight 128 MiB blocks per batch, with two batches
in flight, holds 2 GiB in memory. `_batch_size_for` computes the count from the
block size against a 64 MiB budget. A variable chunked at 128 MiB therefore
holds two blocks in flight, not sixteen.

On a 500 MiB variable, peak RSS falls from about 1.6 GiB for a whole read to
about 500 MiB under a 16 MiB block budget. It tracks the budget, not the file.

The blocks a read uses also become the stored chunk shape, unless `chunks=`
overrides it. That is one decision, not two. `open_chunks` picks the strategy.
`"auto"` lets dask size the blocks to `chunk_size`. `"native"` uses the
encoding of the file. `None` reads each variable whole. A dict sets an explicit
size per dimension.

## Conventions Rust does not know about

Three things the Python layer writes as ordinary attributes:

- `_pyatlas_coords`. A JSON list of the variables that were coordinates.
- `_pyatlas_timedelta`. The unit marker above.
- A `json:` prefix on any attribute value atlas cannot store as it is, such as
  a nested dict or a ragged list. The JSON sits behind the prefix.

A Rust reader sees these as plain string attributes. Only Python reads their
meaning. They sit on top of the format, and are no part of it.
`tests/cross_fixture.rs` asserts that, and reads them as raw strings.

`describe` and `info` decode them and hide the markers.

## Reading array data

Not from Python. `describe` gives the type, the shape, the chunking, the fill
value, the attributes, and the write statistics of every array. That is enough
to catalogue a collection, or to check an ingest. It all comes from the footer
the open already read.

`info` answers the same question for the whole collection. Its `array_stats`
maps each array name to one set of statistics. The counts add up over every
live dataset that holds the array. The minimum is the smallest of the minimums.
The maximum is the largest of the maximums. A dataset that declares the same
name with a different dtype stays out, because two dtypes do not compare.

Array *values* come from the Rust API. See [read-path.md](read-path.md).

## Errors

| Rust | Python |
|---|---|
| `DatasetNotFound`, `ArrayNotFound` | `KeyError`, wrapped as `AtlasError` by the operations |
| `InvalidName`, `NotAnAtlasCollection`, `UnsupportedVersion` | `ValueError` |
| `Io` | `OSError` |
| `CorruptCollection`, `CorruptMask`, `ObjectStore` | `RuntimeError` |

An operation raises `AtlasError` for anything it can explain itself. An empty
directory, a duplicate name, or a dataset that is absent. It raises
`SourceError` when a URL does not resolve. The CLI turns both into a one-line
message and exit code 1.
