# Python bindings and the xarray mapping

## The split

The file format is Rust. `atlas-python` is a binding layer plus the xarray
conventions — it holds no format knowledge and cannot produce or parse a byte
of a container.

| Layer | Lives in | Does |
|---|---|---|
| `atlas._atlas` | `atlas-python/src/*.rs` | PyO3 classes, numpy ⇄ ndarray, Python ⇄ `Attr`, error mapping |
| `atlas` | `atlas-python/python/atlas/` | The facade, xarray conversion, attribute encoding |

Four classes cross the boundary:

```python
AtlasWriter   ──add_dataset()──▶  DatasetWriter    # building
Atlas         ──dataset()──────▶  DatasetView      # reading metadata
```

## Writing is full, reading is metadata-only

Python builds collections. It does not read array data back — there is no
`read_array` on `Atlas` or `DatasetView`, and no
`open_as_xarray_dataset`. Use the Rust API for data.

What Python does get from an open collection: dataset names, array names,
dtypes, shapes, chunk shapes, dimension names, fill values, and all attributes.
That is enough to serve a catalogue, and it costs one range read.

## Arrays are numpy arrays

`write_array` takes a C-contiguous numpy `ndarray` whose dtype matches the
declared one, and reads it zero-copy for every numeric type:

```python
ds.define_array("temperature", dtype="float32", dims=["lat", "lon"], shape=[4, 8])
ds.write_array("temperature", start=[0, 0], data=np.zeros((4, 8), np.float32))
```

The exceptions are strings, which are extracted element by element into a
`Vec<String>` (unavoidable — Python strings are not a contiguous buffer), and
`datetime64[ns]`, which is passed as `arr.view(np.int64)`: a zero-copy
reinterpretation, since numpy distinguishes the dtype kinds where atlas does
not.

## The xarray mapping

`add_xarray_dataset(ds, name)` writes coordinates first, then data variables.

| xarray | atlas |
|---|---|
| variable | array, named the same |
| `var.dims` | `dimension_names` |
| `var.shape` | `shape` |
| dask chunking | `chunk_shape`, unless `chunks=` overrides |
| `var.attrs` | per-array attributes |
| `ds.attrs` | dataset-level attributes |
| `_FillValue` | the array's fill value, not an attribute |
| which vars were coords | the `_pyatlas_coords` marker attribute |

### dtypes

Signed and unsigned integer widths and `float32`/`float64` map straight through.
Beyond that:

| numpy | atlas | note |
|---|---|---|
| `datetime64[ns]` | `timestamp_nanoseconds` | only `[ns]` |
| `timedelta64[*]` | `int64` | normalized to ns, tagged `_pyatlas_timedelta` |
| `object` / `S` / `U` | `string` | variable length |
| anything else | — | `NotImplementedError` |

Atlas has no duration type, so a timedelta becomes int64 nanoseconds plus a
marker attribute naming the unit — the same trick datetime uses, but recorded
explicitly because the target type is not distinctive.

### Fill values

Reading a NetCDF file with `mask_and_scale=True` leaves `NaN` and `NaT` where
data is missing, and moves `_FillValue` into `var.encoding`. Atlas records those
cells as never-written by defaulting each array to a sentinel:

| dtype | default fill |
|---|---|
| float | `NaN` |
| datetime64 | `NaT` (`i64::MIN`) |
| string | `""` |
| integer | none |

Override per variable with `fill_value={"var": scalar}`, all at once with a bare
scalar, or opt out with `{"var": None}`.

Missing string cells are the one lossy case: atlas cannot store a null string,
so `None`/`NaN` are replaced with the fill and a warning names the count.

### Streaming

A dask-backed variable is written one block at a time — `_iter_blocks` walks the
chunk grid, prefetching batches of 8 blocks two deep on a background thread — so
peak memory is one block per variable rather than the whole array. A numpy-backed
variable is written as a single full-shape block.

### Conventions Rust does not know about

Three things the Python layer writes as ordinary attributes:

- `_pyatlas_coords` — a JSON list of which variables were coordinates
- `_pyatlas_timedelta` — the unit marker described above
- a `json:` prefix on any attribute value too complex to store natively
  (nested dicts, ragged lists), JSON-encoded behind it

A Rust reader sees these as plain string attributes; only Python interprets
them. They are conventions layered on the format, not part of it —
`tests/cross_fixture.rs` asserts exactly that by reading them as raw strings.

`Atlas.coords()` and `Atlas.attributes()` decode them; `DatasetView.attributes()`
returns what is stored.

## Atomicity

`add_xarray_dataset` builds a `DatasetWriter` and aborts it on any exception, so
a dataset that fails partway — an unsupported dtype after several good variables
— never enters the collection, and the writer carries on with the next one.

The collection as a whole is atomic in the same way: use `AtlasWriter` as a
context manager and an exception abandons the write entirely, leaving nothing at
the target that opens as a collection.

```python
with atlas.AtlasWriter.create(path) as w:
    for nc in paths:
        w.add_xarray_dataset(xr.open_dataset(nc), name=nc.stem)
# finish() runs on a clean exit; nothing is readable before it
```

## Errors

| Rust | Python |
|---|---|
| `DatasetNotFound`, `ArrayNotFound` | `KeyError` |
| `DatasetAlreadyExists`, `ArrayAlreadyExists` | `FileExistsError` |
| `InvalidName`, `NotAnAtlasCollection`, `UnsupportedVersion`, `WriterFinished` | `ValueError` |
| `Io` | `OSError` |
| `CorruptCollection`, `CorruptMask`, `Internal`, `ObjectStore`, `ArrayFormat` | `RuntimeError` |

Type and range failures on a fill value raise `TypeError` and `OverflowError`
before any I/O happens.
