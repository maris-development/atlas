# Examples

Runnable scripts live in
[`atlas-python/examples/`](https://github.com/maris-development/atlas/tree/main/atlas-python/examples).
Each is self-contained, writes to a temp directory, and runs in seconds.

| File | What it shows |
|---|---|
| [`01_basics.py`](https://github.com/maris-development/atlas/blob/main/atlas-python/examples/01_basics.py) | Build a collection, define arrays, set attributes, reopen it, inspect the metadata, delete a dataset. The minimal end-to-end loop. |
| [`02_xarray.py`](https://github.com/maris-development/atlas/blob/main/atlas-python/examples/02_xarray.py) | Write three `xr.Dataset`s into one collection, then read back the schema, coordinates, and attributes. Shows the dtype mapping. |
| [`03_dask_streaming.py`](https://github.com/maris-development/atlas/blob/main/atlas-python/examples/03_dask_streaming.py) | Stream a dask-chunked `xr.Dataset` block by block, and confirm the dask chunking became the on-disk chunk shape. |
| [`05_codecs.py`](https://github.com/maris-development/atlas/blob/main/atlas-python/examples/05_codecs.py) | Compare `zstd` / `lz4` / `none` on a smooth `float32` field, and confirm a reader needs no codec argument. |
| [`08_object_store.py`](https://github.com/maris-development/atlas/blob/main/atlas-python/examples/08_object_store.py) | Pass an [obstore](https://github.com/developmentseed/obstore) handle instead of a path. Uses `LocalStore` so it needs no credentials; the S3 / GCS / Azure lines are commented in at the top. Requires `pip install "atlas-python[cloud]"`. |
| [`09_missing_data.py`](https://github.com/maris-development/atlas/blob/main/atlas-python/examples/09_missing_data.py) | Masked cells across float / datetime / string / integer variables: the default `NaN` / `NaT` / `""` fills, the warning for missing strings, and overriding with `fill_value=`. |

## Rust examples

Reading array data happens in Rust, so the read-side examples live in
[`examples/`](https://github.com/maris-development/atlas/tree/main/examples)
at the repository root:

```bash
cargo run --example lifecycle       # build, read, delete, reopen
cargo run --example sensor_fleet    # many small datasets in one file
cargo run --example weather_store   # an object store, read lazily
```

## Running

From a clone of the repository:

```bash
python atlas-python/examples/01_basics.py
```

Each script is standalone, with no shared state, writing into a
`tempfile.TemporaryDirectory()` that cleans up on exit.

## See also

- [Quickstart](quickstart.md) — the same ground as `01_basics.py`, line by line.
- [Datasets and arrays](guides/datasets-and-arrays.md) — the model behind every
  script.
- [Reading data](guides/reading-data.md) — why the read examples are in Rust.
