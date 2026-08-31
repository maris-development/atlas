# Examples

Runnable scripts live in
[`atlas-python/examples/`](https://github.com/maris-development/atlas/tree/main/atlas-python/examples).
Each is self-contained, writes to a temp directory, and runs in seconds.

| File | What it shows |
|---|---|
| [`01_library.py`](https://github.com/maris-development/atlas/blob/main/atlas-python/examples/01_library.py) | All five operations as a library: build a collection from NetCDF files, list it, summarise it, inspect a dataset with its statistics, then remove datasets and see the container stay put. |
| [`02_object_store.py`](https://github.com/maris-development/atlas/blob/main/atlas-python/examples/02_object_store.py) | The same five, against an obstore handle. Uses `LocalStore` so it needs no credentials; the S3 / GCS / Azure lines are commented in at the top. Requires `pip install "atlas-python[cloud]"`. |

```bash
python atlas-python/examples/01_library.py
```

## The same thing on the command line

```bash
atlas create /data/nc /data/collection
atlas ls     /data/collection
atlas info   /data/collection
atlas show   /data/collection 2024-01
atlas rm     /data/collection 2024-02 2024-03
```

See [The `atlas` command](cli.md).

## Rust examples

Reading array data happens in Rust, so the read-side examples live in
[`examples/`](https://github.com/maris-development/atlas/tree/main/examples)
at the repository root:

```bash
cargo run --example lifecycle       # build, read, delete, reopen
cargo run --example sensor_fleet    # many small datasets in one file
cargo run --example weather_store   # an object store, read lazily
```

## See also

- [Quickstart](quickstart.md) — the same ground, line by line.
- [Creating a collection](guides/creating.md) — chunking, errors, memory.
- [Reading data](guides/reading-data.md) — why the read examples are in Rust.
