# 7. Python & xarray Integration

`atlas-python` is a thin PyO3 layer over the Rust core plus an xarray
integration. The split is deliberate:

```
   atlas-python/python/atlas/
   ├── store.py     Atlas facade — forwards primitives to the Rust core,
   │                adds the high-level xarray convenience methods
   └── xarray.py    pure-Python xarray ⇄ atlas conversion (no rebuild to change)

   atlas-python/src/    the PyO3 bindings (Rust) exposing the core primitives
```

> **The Rust core owns performance; Python owns ergonomics.** The per-chunk write
> loop runs against the **core** `DatasetView` directly, so the xarray
> convenience adds no per-chunk overhead. Adding a high-level method is a
> pure-Python edit — no recompile.

## Writing an xarray Dataset

```python
import atlas, xarray as xr

with atlas.Atlas.create("my_store") as store:
    for path in nc_paths:
        ds = xr.open_dataset(path)
        store.add_xarray_dataset(ds, name=path.stem)   # or: ds.atlas.write(store, name)
    # store.close() (== flush) runs on __exit__ — the single durability boundary
```

Each variable/coord becomes an atlas array; coords are recorded so the
distinction round-trips; dataset attrs → global attributes; per-variable attrs →
per-array attributes.

### dtype mapping

| numpy | atlas |
|-------|-------|
| int8/16/32/64, uint8/16/32/64, float32/64 | same |
| `datetime64[ns]` | `TimestampNs` (viewed as int64) |
| `timedelta64` (any unit) | `Int64` nanoseconds + a `_pyatlas_timedelta` marker, restored to a duration on read |
| object / bytes / unicode | `String` |

Missing values map to typed null sentinels: `NaN` (float), `NaT`/`i64::MIN`
(datetime & timedelta), `""` (string), so masked cells are recorded as null.

### Atomic inserts

`add_xarray_dataset` is transactional: if populating the dataset fails partway
(e.g. an unsupported dtype), it rolls back the just-created dataset with
`delete_dataset`, so a later `flush()`/`close()` can't persist a half-written
record.

### Chunking

Large variables can be opened dask-chunked (`chunks="auto"`); atlas streams them
chunk-by-chunk into the store instead of loading fully into memory, and the
on-disk chunk grid mirrors the dask chunks. Full-shape variables come back eager
as numpy on read; chunked ones come back dask-backed.

## Reading back

```python
store = atlas.Atlas.open("my_store")
ds  = store.open_as_xarray_dataset("jan")                    # one dataset
big = store.open_as_many_xarray_dataset(["jan","feb",…],     # many, stacked
                                        concat_dim="dataset")
```

`open_as_many_xarray_dataset` uses the Rust `read_array_across` path: it reads
each variable across all named datasets in one call (sharing a single read lock
on the physical file), returning a pre-stacked `(N, *shape)` array — much faster
than N per-dataset round trips.

## Bulk ingestion at scale

For very large runs, the reference `examples`/`batch_to_atlas.py`-style pipeline
should:

- **close each `xr.Dataset` after ingest** — frees the source NetCDF file handle
  (critical for the lazy `chunks="auto"` path, or you exhaust file descriptors);
- **raise the OS open-file limit** where possible;
- **normalize variable names to a shared schema** — the single most important
  thing for scaling (see [data-model.md](data-model.md)); unique per-file names
  create one physical array file per dataset, which is the failure mode the
  layout is designed to avoid.

See [write-path.md](write-path.md) for what `flush`/`close` actually do.
