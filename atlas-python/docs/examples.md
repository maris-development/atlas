# Examples

Runnable scripts live in
[`atlas-python/examples/`](https://github.com/robinskil/atlas/tree/main/atlas-python/examples).
Each one is self-contained, writes to a temp directory, and runs in under
a minute.

| File | What it shows |
|---|---|
| [`01_basics.py`](https://github.com/robinskil/atlas/blob/main/atlas-python/examples/01_basics.py) | Create a store, define arrays, set attributes, reopen, read back. The minimal end-to-end loop. |
| [`02_xarray.py`](https://github.com/robinskil/atlas/blob/main/atlas-python/examples/02_xarray.py) | Round-trip an `xr.Dataset` through atlas using both `atlas.add_xr_dataset(...)` and the `ds.atlas.write(...)` accessor. |
| [`03_dask_streaming.py`](https://github.com/robinskil/atlas/blob/main/atlas-python/examples/03_dask_streaming.py) | Stream a dask-chunked `xr.Dataset` into atlas one chunk at a time, preserving the chunk shape on disk; read back lazily. |
| [`04_meta_formats.py`](https://github.com/robinskil/atlas/blob/main/atlas-python/examples/04_meta_formats.py) | Compare the six metadata format × compression combinations (`atlas.json`, `atlas.msgpack.zst`, …) on a 30-dataset store. Confirms auto-detection on reopen. |
| [`05_codecs.py`](https://github.com/robinskil/atlas/blob/main/atlas-python/examples/05_codecs.py) | Compare `zstd` / `lz4` / `none` array codecs on a smooth `float32` field. Demonstrates per-array codec recording. |
| [`06_stats_scan.py`](https://github.com/robinskil/atlas/blob/main/atlas-python/examples/06_stats_scan.py) | Find the dataset with the highest peak reading across a fleet of 32 sensors by scanning `array_stats` only — no raw data read. |
| [`07_shared_arrays.py`](https://github.com/robinskil/atlas/blob/main/atlas-python/examples/07_shared_arrays.py) | 50 datasets sharing 2 physical array files. Prints the on-disk directory listing to confirm the layout. |
| [`08_object_store.py`](https://github.com/robinskil/atlas/blob/main/atlas-python/examples/08_object_store.py) | Pass an [obstore](https://github.com/developmentseed/obstore) handle into `Atlas.create` / `Atlas.open` and run the full create / read / xarray / stats loop against it. Demos with `obstore.store.LocalStore` (no credentials); S3 / GCS / Azure variants are commented in at the top. Requires `pip install "atlas-python[cloud]"`. |

## Running

From a clone of the repository:

```bash
python atlas-python/examples/01_basics.py
```

Every script can be run standalone; there's no shared state between them.
Each writes into a `tempfile.TemporaryDirectory()` that cleans up on exit.

## See also

- [Quickstart](quickstart.md) — the same idea as `01_basics.py`, walked
  through line-by-line.
- [Datasets and arrays](guides/datasets-and-arrays.md) — the mental model
  behind every script.
- [Benchmarks](benchmarks.md) — the cross-backend harness, also runnable.
