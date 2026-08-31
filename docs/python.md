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

Every one of them takes a local path, a URL (`s3://`, `gs://`, `az://`,
`https://`), or an obstore handle, so the same call works against a bucket.

There is nothing else. No writer object, no `define_array`, no `read_array`.
NetCDF is the only way data goes in, and array values come back out through the
Rust API.

## The layers

| Layer | Lives in | Does |
|---|---|---|
| `atlas` | `atlas-python/python/atlas/` | The five operations, the CLI, the xarray mapping |
| `atlas._atlas` | `atlas-python/src/*.rs` | PyO3: numpy ⇄ ndarray, Python ⇄ `Attr`, error mapping |

`_atlas` is private. It still exposes writer classes, because the ingest path
needs them, but they are not part of the package's surface and nothing outside
`atlas._ops` touches them.

**The format is Rust only.** `atlas-python` holds no format knowledge — grep it
for `ATLS` and you get nothing.

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

`create` scans a directory for `.nc`, `.nc4`, `.cdf`, and `.netcdf` files,
sorts them, and writes one dataset per file named after the file stem. Sorting
is what makes ordinals reproducible: build the same directory twice and every
dataset lands at the same position.

Everything happens inside one writer, so nothing at the destination is readable
until the last file is written and the footer lands. A failure part-way leaves
no collection at all, which is the behaviour you want when a job dies at 3 a.m.

`on_error="skip"` trades that for progress: a file that fails is recorded in
the result and the rest carry on.

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

Coordinates are written first, then data variables, so the on-disk order is
predictable.

### dtypes

Integer widths and `float32`/`float64` map straight through. Beyond that:

| numpy | atlas | note |
|---|---|---|
| `datetime64[ns]` | `timestamp_nanoseconds` | only `[ns]` |
| `timedelta64[*]` | `int64` | normalized to ns, tagged `_pyatlas_timedelta` |
| `object` / `S` / `U` | `string` | variable length |
| anything else | — | `NotImplementedError` |

Atlas has no duration type, so a timedelta becomes int64 nanoseconds plus a
marker naming the unit. Surrogate-escaped strings — common from netCDF backends
— are sanitised on the way in.

### Fill values

Reading a NetCDF file with `mask_and_scale=True`, xarray's default, leaves `NaN`
and `NaT` where data is missing and moves `_FillValue` into `var.encoding`.
Atlas records those cells as never-written by defaulting each array to a
sentinel: `NaN` for floats, `NaT` for datetimes, `""` for strings, and none for
integers.

Missing *string* cells are the one lossy case — atlas cannot store a null
string, so they are replaced with the fill and a warning names the count.

### Streaming

Dask-backed variables are written one block at a time, prefetched on a
background thread, so peak memory is one block per variable rather than the
whole array. A dataset far larger than memory ingests without trouble.

## Conventions Rust does not know about

Three things the Python layer writes as ordinary attributes:

- `_pyatlas_coords` — a JSON list of which variables were coordinates
- `_pyatlas_timedelta` — the unit marker described above
- a `json:` prefix on any attribute value too complex to store natively
  (nested dicts, ragged lists), JSON-encoded behind it

A Rust reader sees these as plain string attributes; only Python interprets
them. They are conventions layered on the format, not part of it —
`tests/cross_fixture.rs` asserts exactly that by reading them as raw strings.

`describe` and `info` decode them and hide the markers.

## Reading array data

Not from Python. `describe` gives you every array's type, shape, chunking, fill
value, attributes, and the statistics recorded when it was written — enough to
catalogue a collection or validate an ingest, all from the footer that opening
already read.

Array *values* come from the Rust API. See [read-path.md](read-path.md).

## Errors

| Rust | Python |
|---|---|
| `DatasetNotFound`, `ArrayNotFound` | `KeyError`, wrapped as `AtlasError` by the operations |
| `InvalidName`, `NotAnAtlasCollection`, `UnsupportedVersion` | `ValueError` |
| `Io` | `OSError` |
| `CorruptCollection`, `CorruptMask`, `ObjectStore` | `RuntimeError` |

The operations raise `AtlasError` for anything they can explain themselves — an
empty directory, a duplicate stem, a dataset that is not there — and
`SourceError` when a URL cannot be resolved. The CLI turns both into a one-line
message and exit code 1.
