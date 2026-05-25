# Cloud storage (S3, GCS, Azure)

[`Atlas.create`](../reference/atlas.md) and
[`Atlas.open`](../reference/atlas.md) both accept an
[obstore](https://github.com/developmentseed/obstore) store handle in
place of a filesystem path. obstore is a thin Python binding around the
Rust [`object_store`](https://docs.rs/object_store) crate, so any
backend obstore supports — S3, GCS, Azure Blob, HTTP, local fs — works
with atlas, transparently. pyatlas itself never sees the credentials.

## Install

`obstore` is an optional dependency:

```bash
pip install "pyatlas[cloud]"
```

Equivalent to `pip install pyatlas obstore`. Without it, atlas continues
to work against local filesystem paths exactly as before.

## Quickstart — S3

```python
import numpy as np
import obstore as obs
import pyatlas

# Construct the obstore handle. Credentials are loaded from the standard
# AWS env vars / ~/.aws/credentials by default; pass them explicitly to
# override.
store = obs.store.S3Store(
    "my-bucket",
    prefix="stores/jan_2024",
    region="us-east-1",
)

# Pass the handle into Atlas.create / Atlas.open exactly where you would
# have passed a path. Everything below this line — define_array,
# write_array, set_attribute, flush, add_xr_dataset, to_xarray, all of
# the bulk reads — works identically against S3.
with pyatlas.Atlas.create(store, codec="zstd") as atlas:
    ds = atlas.create_dataset("jan_2024")
    ds.define_array(
        "temperature",
        dtype="float32",
        dims=["lat", "lon"],
        shape=[8, 16],
        chunk_shape=[4, 8],
    )
    ds.write_array(
        "temperature",
        start=[0, 0],
        data=np.full((8, 16), 20.0, dtype=np.float32),
    )
    ds.set_attribute("month", 1)

# Reopen — codec / meta format / meta compression are auto-detected just
# like on local fs.
atlas2 = pyatlas.Atlas.open(store)
arr = atlas2.open_dataset("jan_2024").read_array("temperature")
```

## Other backends

The same pattern works for every obstore backend:

```python
import obstore as obs
import pyatlas

# Google Cloud Storage
gcs = obs.store.GCSStore("my-bucket", prefix="stores/jan_2024")
pyatlas.Atlas.open(gcs)

# Azure Blob Storage
azure = obs.store.AzureStore(container_name="my-container", prefix="stores/jan_2024")
pyatlas.Atlas.open(azure)

# Generic HTTP (read-only)
http = obs.store.HttpStore.from_url("https://example.com/atlas-store/")
pyatlas.Atlas.open(http)

# Plain local filesystem via obstore (equivalent to passing a path string)
local = obs.store.LocalStore("/tmp/my_store")
pyatlas.Atlas.open(local)
```

See [obstore's
documentation](https://developmentseed.org/obstore/latest/) for the full
list of backends and their credential / region / endpoint options.

A complete runnable script — create / write / `add_xr_dataset` /
`to_xarray` / `array_stats`, then reopen and verify — lives at
[`pyatlas/examples/08_object_store.py`](https://github.com/robinskil/atlas/blob/main/pyatlas/examples/08_object_store.py).
It uses `LocalStore` by default so it runs without credentials; swap in
the S3 / GCS / Azure block at the top to point at a real backend.

## Read/write parity

Everything atlas exposes works identically against any obstore backend:

| Operation | Local fs | Cloud (S3 / GCS / Azure) |
|---|---|---|
| `create_dataset` / `open_dataset` / `delete_dataset` | ✓ | ✓ |
| `define_array` / `write_array` / `read_array` / `delete_array` | ✓ | ✓ |
| `set_attribute` / `get_attribute` / `attributes` | ✓ | ✓ |
| `flush` (the single durability boundary) | fsync + rename | atomic PutObject |
| `compact` | rewrite array files | rewrite array files |
| `read_array_across_stacked` / `to_xarray_many` | tokio fan-out | tokio fan-out, capped at `num_cpus` |
| Dask streaming `add_xr_dataset` | ✓ | ✓ |
| Dask-backed lazy reads (`to_xarray`) | threaded scheduler only | threaded scheduler only |

The [single-flush durability model](durability.md) carries over cleanly:
on S3, `PutObject` is atomic for the full object, so the in-memory
buffering of `flush()` maps onto one `PutObject` per touched array file
plus one for the metadata. Flush latency will be noticeably higher than
on local fs — call `flush()` at coarse grain (per-batch, not
per-dataset).

## Limitations specific to cloud backends

- **Bulk-read concurrency.** `read_array_across_stacked` and
  `to_xarray_many` dispatch on the shared tokio runtime via a `JoinSet`
  capped at `num_cpus`. On S3 the bottleneck is usually HTTP
  concurrency, not CPU — let us know if you'd find a configurable cap
  useful.
- **`MemoryStore` can't be reconstructed across the FFI boundary.**
  `obstore.store.MemoryStore` keeps its data in process memory and can't
  be cloned through `pyo3-object_store`'s external-store path. Use
  `obstore.store.LocalStore` (against a `tempfile.TemporaryDirectory`)
  if you want an ephemeral in-tests backend.
- **A warning on first store reconstruction.** You will see one
  `RuntimeWarning: Successfully reconstructed a store defined in another
  Python module. Connection pooling will not be shared across store
  instances.` per store. This is expected — pyatlas re-creates obstore's
  ObjectStore on its side of the FFI boundary using the constructor
  arguments. Connection pools are independent but functionality is
  identical.
- **Single writer.** atlas assumes a single process is writing to any
  given store at a time. The cloud backends inherit that constraint —
  concurrent writers from multiple processes are not coordinated.

## Where Zarr / netCDF still win on the cloud

See [vs Zarr / netCDF](../vs-zarr-netcdf.md) — Zarr is still the natural
choice for write-heavy workloads with many independent processes (chunk
files are independent and parallel-writable), and netCDF has the
longest tooling lineage. atlas is a focused fit for "N small datasets,
same schema" workloads where the bulk-read APIs pay back the per-store
constraint.
