# Examples

The runnable scripts sit in
[`atlas-python/examples/`](https://github.com/maris-development/atlas/tree/main/atlas-python/examples).
Each one stands alone, writes to a temporary directory, and runs in seconds.

| File | What it shows |
|---|---|
| [`01_library.py`](https://github.com/maris-development/atlas/blob/main/atlas-python/examples/01_library.py) | All five operations as a library. It builds a collection from NetCDF files, lists it, summarizes it, and inspects one dataset with its statistics. It then removes datasets, and shows the container stay put. |
| [`02_object_store.py`](https://github.com/maris-development/atlas/blob/main/atlas-python/examples/02_object_store.py) | The same five, against an obstore handle. It uses `LocalStore`, so it needs no credential. The S3, GCS, and Azure lines sit in a comment at the top. It needs `pip install "atlas-python[cloud]"`. |

```bash
python atlas-python/examples/01_library.py
```

## The same thing on the command line

```bash
atlas create /data/nc /data/collection
atlas ls     /data/collection
atlas info   /data/collection
atlas show   /data/collection 2024-01.nc
atlas rm     /data/collection 2024-02.nc 2024-03.nc
```

See [The `atlas` command](cli.md).

## Rust examples

Rust reads array data, so the read-side examples sit in
[`examples/`](https://github.com/maris-development/atlas/tree/main/examples)
at the repository root:

```bash
cargo run --example lifecycle       # build, read, delete, reopen
cargo run --example sensor_fleet    # many small datasets in one file
cargo run --example weather_store   # an object store, read lazily
```

## See also

- [Quickstart](quickstart.md). The same ground, line by line.
- [Creating a collection](guides/creating.md). Chunking, errors, and memory.
- [Reading data](guides/reading-data.md). Why the read examples are in Rust.
