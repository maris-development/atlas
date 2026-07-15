# ATLAS — Aggregated Tensor Large Array Store

[![CI](https://github.com/maris-development/atlas/actions/workflows/ci.yaml/badge.svg)](https://github.com/maris-development/atlas/actions/workflows/ci.yaml)
[![crates.io](https://img.shields.io/crates/v/atlas-rust.svg?logo=rust)](https://crates.io/crates/atlas-rust)
[![docs.rs](https://img.shields.io/docsrs/atlas-rust?logo=docsdotrs&label=docs.rs)](https://docs.rs/atlas-rust)
[![PyPI](https://img.shields.io/pypi/v/atlas-python.svg?logo=pypi&logoColor=white)](https://pypi.org/project/atlas-python/)
[![Python docs](https://img.shields.io/badge/docs-atlas--python-blue?logo=materialformkdocs&logoColor=white)](https://maris-development.github.io/atlas/)
[![License](https://img.shields.io/crates/l/atlas-rust.svg)](LICENSE)

A directory-based store for thousands of named datasets, each holding N-dimensional typed arrays. Built on top of the [`array-format`](https://github.com/robinskil/array-format) (`.af`) binary format, with configurable compression (Zstd, LZ4, or none), chunked I/O, and an [`object_store`](https://crates.io/crates/object_store) backend that works on local disk, S3, GCS, Azure Blob, and in-memory.

**Think of it as a "zip" for N-dimensional datasets.** Where a `.zip` bundles many files into one archive, ATLAS gathers many NetCDF / Zarr-style datasets — anything that's a set of named N-dimensional arrays — into a single high-performance collection. But it's more than a bundle: instead of storing each dataset whole, ATLAS lays the data out **variable-first**, so every dataset's `temperature` lives in one file. That makes it cheap to scan or slice a single variable across thousands of datasets at once, while still reading back any individual dataset as a normal NetCDF/xarray-like object.

> **Looking for Python?** The Python bindings live in [`atlas-python/`](atlas-python/) — `pip install atlas-python`, then `import atlas`. They add a NumPy-native API and first-class [xarray](https://docs.xarray.dev) integration. See the [atlas-python README](atlas-python/README.md) for usage.
>
> The rest of this document covers **how the format works and the Rust crate**.

---

## What it does

`atlas` is designed for workloads where you have a large collection of similarly-shaped datasets — such as one dataset per time step, sensor station, or simulation run — and you want to query a single variable (e.g. `temperature`) across all of them efficiently.

Each dataset is a named group of N-dimensional arrays with typed attributes — both dataset-level (global) attributes and per-variable attributes. Datasets that share an array name (e.g. every `jan_2024`, `feb_2024`, … all have `temperature`) are stored together in the same physical file, keyed by dataset name inside the file.

```text
my_store/
├── atlas.json               ← interned per-dataset schema + attribute-key namespace (JSON)
├── _global/
│   └── data.af         ← dataset-level (global) attribute values, one entry per dataset
├── temperature/
│   └── data.af         ← one ArrayFile: temperature + its per-variable attributes, per dataset
├── pressure/
│   └── data.af
└── time/
    └── data.af
```

`atlas.json` holds only the **schema** of the collection. Attribute **values** live in the `.af` files: per-variable attributes on the variable's own file, and dataset-level attributes in the reserved `_global` file.

---

## Durability model

`atlas.json` is read **once** when the store is opened or created. Every subsequent mutation — `create_dataset`, `define_array`, `set_attribute`, `set_array_attribute`, `delete_array`, `delete_dataset` — only touches in-memory state; array writes buffer inside the per-array in-memory layer and attribute writes buffer in a pending-attribute map. **Nothing reaches disk until `Atlas::flush()` (or `Atlas::close()`).** Dropping an `Atlas` without flushing abandons every pending in-memory write.

A single `Atlas::flush()` drains the buffered attributes into their `.af` files, walks every cached `ArrayFile` (writing deltas + stats), and then serialises the in-memory schema to `atlas.json`. This gives one durability boundary for the whole store: N datasets ⇒ one delta file per touched array name (not one per dataset) and one `atlas.json` rewrite (not N).

`DatasetView` is a borrowed handle into the atlas's shared meta — it has no `flush()` of its own.

---

## File format

### `atlas.json`

The schema is a plain JSON file written on `Atlas::flush()` / `Atlas::close()`. It stores:

- **Store version** — for format upgrades. The current version is `2`; stores written by an older atlas are rejected on open (re-export to upgrade).
- **Interned schema pool** — each *distinct* per-dataset schema is stored once and referenced by index. A dataset's schema is its array schemas (per array: dtype, shape, chunk shape, named dimensions, codec) plus its **attribute-key namespace** (the names of its global and per-variable attribute keys). A collection of thousands of homogeneous datasets stores its schema a single time.
- **Dataset registry** — a map of dataset name → schema-pool index.
- **Merged schema** — a collection-wide summary listing every unique array (with its dtype widened across all datasets, its dimensions, and its per-variable attribute types) and every global attribute type. Descriptive only — reads always use each dataset's own schema — but handy for tools that want one view of "what's in here." Also exposed programmatically via `Atlas::merged_schema()`.

When the same array name (or attribute key) appears in more than one dataset with different types, the merged type is **widened**. Widening is allowed only within numeric types (e.g. `int16` ∪ `int32` → `int32`; `int32` ∪ `float32` → `float64`) or between `string` and `timestamp` (→ `string`).

Any other combination — e.g. an `int32` array in one dataset and a `string` array under the same name in another — can't merge. The dataset is **still stored** under its own type (each dataset keeps the type it declared, and its data reads back normally); the merged schema simply keeps the **first-seen** type. Whether that's reported as a warning or an error is set by `StoreConfig::on_type_mismatch`:

| `TypeMismatchPolicy` | Behaviour |
| --- | --- |
| `Warn` (default) | Logs a `tracing` warning and carries on — an ingest of many heterogeneous files won't abort because one disagrees. |
| `Error` | `define_array` / `set_attribute` / `set_array_attribute` return `Error::TypeMismatch`. |

The policy is a **per-session** choice, not an on-disk property: pass it to `Atlas::create`, or to `Atlas::open_with_config` / `Atlas::open_path_with_config` when opening an existing collection (plain `open` / `open_path` use the default). Note that these open variants honour only `on_type_mismatch` — the codec and metadata format are always detected from disk.

Attribute **values** are **not** in `atlas.json` — only their key names are (as part of the schema). Values live in the `.af` files and are read via the `array-format` attribute API. Because `atlas.json` is human-readable and self-describing, you can still inspect or audit the collection's schema with any JSON tool without needing the library.

### `<array_name>/data.af`

Each array variable gets its own subdirectory with a single `data.af` binary file. The `.af` format (from the `array-format` crate) is a columnar, chunk-oriented binary format:

- **Multiple datasets in one file** — every dataset that owns this variable is stored as a named entry inside the same file.
- **Chunked layout** — arrays are split into chunks of a user-specified shape, so partial reads and writes touch only the relevant blocks.
- **Configurable compression** — each block is compressed with the codec set when the store was created (default: Zstd; also LZ4 and uncompressed). The codec is persisted in `atlas.json` and restored automatically on `open` — no need to pass it again. Block target size is 8 MiB.
- **Per-array fill value** — `define_array` accepts an optional `FillValue` (one of `Bool`/`Int`/`UInt`/`Float`/`String`). Unwritten cells read back as the fill value, and any *written* cell equal to it is counted as a null in the persisted stats (see below). Fill values are stored in the per-array footer; `atlas.json` is not extended.
- **Attribute values** — per-variable attributes (e.g. `units`) are stored in the footer of the variable's own `data.af`, keyed by dataset. Dataset-level (global) attributes live the same way in the reserved `_global/data.af`. Atlas's `Attr` type mirrors `array-format`'s `AttributeValue` (bool, sized ints/uints, `f32`/`f64`, string, binary, and typed lists) plus a nanosecond timestamp variant stored as an RFC 3339 string.
- **Persisted statistics** — on `Atlas::flush()`, min, max, null count, and row count are computed per array per dataset and stored alongside the data. Cells equal to the array's fill value are tallied in `null_count` and excluded from `min`/`max` (NaN fills match NaN cells by bit pattern). Statistics survive store reopening.
- **In-memory caches** — a 256 MiB decoded block cache and a 64 MiB raw I/O cache sit in front of the object store for repeated reads.

---

## Quick start

```rust
use atlas::{Atlas, Attr, FillValue, StoreConfig};
use ndarray::Array2;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a new store — codec is persisted to atlas.json
    let mut s = Atlas::create_path("/tmp/my_store", StoreConfig::default()).await?;

    {
        // Create a dataset and write arrays
        let mut ds = s.create_dataset("jan_2024").await?;
        ds.define_array::<f32>(
            "temperature",
            vec!["lat".into(), "lon".into()],
            vec![8, 16],
            Some(vec![4, 8]),         // chunk shape
            Some(FillValue::Float(f64::NAN)),  // returned for unwritten cells; counted as nulls in stats
        ).await?;

        let data = Array2::<f32>::from_elem([8, 16], 20.0).into_dyn();
        ds.write_array("temperature", vec![0, 0], data.view()).await?;

        ds.set_attribute("month", Attr::UInt32(1));
        ds.set_attribute("station", Attr::String("KNMI".into()));
    }

    // One flush persists atlas.json + every cached array file.
    s.flush().await?;

    // Reopen — codec is read from atlas.json, no StoreConfig needed
    let s2 = Atlas::open_path("/tmp/my_store").await?;
    let ds2 = s2.open_dataset("jan_2024").await?;

    // Full read
    let temp = ds2.read_array::<f32>("temperature", vec![], vec![]).await?.unwrap();

    // Partial read — one chunk region
    let chunk = ds2.read_array::<f32>("temperature", vec![0, 0], vec![4, 8]).await?.unwrap();

    // Query persisted statistics
    let stats = ds2.array_stats("temperature").await.unwrap();
    println!("rows={} min={:?} max={:?}", stats.row_count, stats.min, stats.max);

    Ok(())
}
```

---

## Key concepts

| Concept | Description |
| --- | --- |
| **Store** | The root directory, managed by `Atlas`. |
| **Dataset** | A named group of arrays + typed attributes, accessed via `DatasetView`. |
| **Array** | An N-dimensional typed array with named dimensions and an optional chunk shape. |
| **Attribute** | A typed scalar attached to a dataset (metadata, not array data). |
| **Array file** | One `data.af` file per variable name, shared across all datasets that define that variable. |
| **Flush** | `Atlas::flush()` — the single durability boundary. Persists every cached array file and rewrites `atlas.json` from the in-memory `StoreMeta`. Must be called explicitly (or via `Atlas::close()` / Python's `with atlas:`). |
| **Compact** | `Atlas::compact()` rewrites every cached `.af` file to reclaim space after deletes. |
| **StoreConfig** | Configuration passed to `Atlas::create`. Currently holds the compression `Codec`. |
| **Codec** | Compression codec for new array blocks: `Codec::Zstd` (default), `Codec::Lz4`, or `Codec::Uncompressed`. Persisted in `atlas.json`; `open` reads it automatically. |

---

## Supported dtypes

`bool`, `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`, `f32`, `f64`, `TimestampNs`, `String`, `Binary`, `List<T>`, `FixedSizeList<T, N>`

0-D scalar arrays (`shape = []`) are supported for any element type — useful for single-valued metadata variables like a trajectory identifier.

---

## Comparison with NetCDF and Zarr

### Similarities

- All three formats are chunked, N-dimensional array stores.
- All support named dimensions, per-variable metadata, and compression.
- All are designed for scientific/numerical data.

### Layout: the key difference

The critical architectural distinction is how datasets and variables are organized on disk.

**NetCDF / Zarr** use a *dataset-first* layout:

```text
zarr_store/
├── jan_2024/
│   ├── temperature/   ← chunks for temperature
│   └── pressure/      ← chunks for pressure
└── feb_2024/
    ├── temperature/
    └── pressure/
```

Reading `temperature` for 1 000 time steps means opening 1 000 separate directories/files.

**ATLAS** uses a *variable-first* layout:

```text
atlas_store/
├── temperature/
│   └── data.af        ← temperature for ALL datasets in one file
└── pressure/
    └── data.af
```

Reading `temperature` for 1 000 datasets means opening exactly **one file**. This is the primary design goal of this format.

### Feature comparison

| Feature | NetCDF-4 | Zarr v3 | ATLAS |
| --- | --- | --- | --- |
| Layout | Dataset-first | Dataset-first | Variable-first |
| Compression | Deflate / Zstd / … | Any codec plugin | Zstd / LZ4 / None |
| Chunking | Yes | Yes | Yes |
| Cloud object store | No (needs FUSE/etc) | Yes (native) | Yes (via `object_store`) |
| Multiple datasets in one file | No | No | Yes (all datasets per variable) |
| Metadata format | Binary (HDF5) | JSON | JSON |
| Cross-dataset column scan | Slow (N file opens) | Slow (N directory opens) | Fast (1 file open) |
| Partial reads | Yes | Yes | Yes |
| Statistics (min/max/nulls) | No | No | Yes (persisted on flush) |
| Self-describing metadata | Yes | Yes | Yes (`atlas.json`) |
| Language support | C/Python/Julia/… | Python/Java/… | Rust |
| Mutable after write | Limited | Yes | Yes (chunked overwrites + compact) |

### When to choose ATLAS

- You have many homogeneous datasets (same variable schema, different instances — time steps, stations, runs).
- Your primary query is "give me variable X across all datasets" — a column scan across the dataset dimension.
- You want a simple on-disk layout with no special runtime dependencies and human-readable metadata.
- You are working in Rust with an async runtime and an `object_store`-compatible backend.

### When to choose Zarr or NetCDF

- You need wide ecosystem support (Python, Julia, C libraries, GIS tools).
- Your primary query pattern is "give me all variables for one dataset" (dataset-first access).
- You need hierarchical group nesting beyond two levels.
- You need a codec ecosystem with many options (blosc, gzip, numcodecs, …).

---

## Performance characteristics

### Cross-dataset scans (where this format excels)

Because all datasets share a single `.af` file per variable, scanning `temperature` across N datasets costs:

- **1 file open** regardless of N.
- Sequential reads within one file, which are friendly to OS read-ahead and object-store range requests.
- Decompression only for the blocks actually read.

In a dataset-first format, the same scan requires N file or directory opens, which is bounded by metadata latency, not throughput — especially painful on object stores where each `HEAD`/`GET` has ~10 ms overhead.

### Chunked I/O

Chunk shapes are set per-array at definition time. A chunk shape equal to the full array shape (the default when `chunk_shape` is omitted) gives a single compressed block per dataset entry, minimising overhead for small arrays. Smaller chunks allow fine-grained partial reads and updates at the cost of more blocks.

### In-memory caches

Two caches sit in front of the object store:

| Cache | Default size | What it holds |
| --- | --- | --- |
| Decoded block cache | 256 MiB | Decompressed array chunks, ready for use |
| I/O cache | 64 MiB | Raw compressed bytes from the object store |

The decoded cache means repeated reads of the same chunk cost only a hash-map lookup. Both caches are shared across all `DatasetView`s that open the same `Atlas`.

### Persisted statistics

Min, max, null count, and row count are computed and persisted on every `Atlas::flush()`. Downstream systems can read these statistics from the opened `DatasetView` without touching array data at all — useful for query planning, dashboards, or data-quality checks.

### Compression

The codec is chosen once at store creation via `StoreConfig` and written into `atlas.json`. `Atlas::open` reads it from there, so callers never need to repeat the codec choice. Each array also records its own codec in `atlas.json`, so a store can theoretically hold arrays written with different codecs if the schema is migrated.

| Codec | Trade-off |
| --- | --- |
| `Codec::Zstd` (default) | Best compression ratio; moderate CPU cost |
| `Codec::Lz4` | Faster compression/decompression; larger files |
| `Codec::Uncompressed` | Fastest write path; no size reduction |

Choose LZ4 when decompression throughput matters more than storage size (e.g. large in-memory analytics). Choose Uncompressed for already-compressed data or when profiling shows decompression is the bottleneck.

### Write path

Writes are buffered in-memory across the whole atlas. Calling `Atlas::flush()` compresses and writes every modified block across every cached array file, then rewrites `atlas.json` atomically (a single `PUT`). The write path scales with the number of modified chunks, not the number of datasets, and N consecutive `add_xarray_dataset` / `create_dataset` calls amortise to a single flush.

### Compaction

After deleting arrays or datasets, the underlying `.af` files may retain dead space. Calling `Atlas::compact()` rewrites every cached file in place, reclaiming storage.

---

## Benchmarks

A reproducible comparison against NetCDF (`netCDF4`) and Zarr v3 lives in [`pyatlas/benchmarks/`](pyatlas/benchmarks/). The harness writes the **same** deterministic data through each backend, then measures three things:

1. **Write** — total wall time to populate the collection.
2. **Read slice** — read a fixed slice of every variable from every dataset, summed.
3. **Storage** — total bytes on disk for the backend's directory.

Each backend uses its **canonical "many datasets" layout** rather than a forced apples-to-apples one — `atlas` uses one store with N datasets, `netcdf` uses N separate `.nc` files (the standard CMIP layout, read via `xr.open_mfdataset(parallel=True)`), `zarr` uses N separate `.zarr` stores (also read via `open_mfdataset`). The point is to compare "what's the fastest each library can do for this workload using its idiomatic pattern," not to handicap any of them.

### Workloads

The harness ships two named workloads (`--case`):

| Case | Variables | Per-dataset shape | Typical use |
|---|---|---|---|
| `sensors` (default) | `temperature`, `pressure`, `humidity` (f32/f64/f32) | `(24,)` on `(time,)` | Hourly weather-station fleet — tiny per-dataset I/O, per-dataset overhead dominates |
| `gridded` | same three vars | `(100, 100, 48)` on `(lon, lat, time)` | Geophysical-style grid — actual decompression dominates |

Both workloads run N=1000 datasets by default. Read slice is the first 25% of each dimension (`--slice-fraction 0.25`).

### Results

Numbers are wall-clock seconds on a single laptop (Apple Silicon, AC power, single run — treat as relative not absolute):

#### `--case gridded --datasets 1000` (1.8 GB raw — decompression dominates)

`(100, 100, 48)` per variable × 3 variables × 1000 datasets, chunk
shape `(50, 50, 24)` (8 chunks per variable). All three backends push
the `(0:25, 0:25, 0:12)` slice down to chunk-level reads.

| Backend | Layout / pattern | Read slice (s) | Write (s) | Storage (MiB) |
|---|---|---:|---:|---:|
| **atlas-bulk** | 1 store, `read_array_across_stacked` with slice push-down | **2.12** | 59 | 6387 |
| **atlas + `--use-dask`** | 1 store, dask-threaded per-dataset `view.read_arrays(...)` | **3.21** | 60 | 6387 |
| zarr | 1000 separate stores, `xr.open_mfdataset(parallel=True).isel(...)` | 5.99 | 38 | 6392 |
| atlas (default) | 1 store, serial per-dataset `open_as_xarray_dataset(...).isel(...).load()` | 10.23 | 51 | 6387 |
| netcdf | 1000 `.nc` files, `xr.open_mfdataset(parallel=True).isel(...)` | 13.91 | 122 | 5596 |

#### `--case profile --datasets 1000` (~67 MB raw — per-dataset overhead dominates)

2 variables on `(depth=50, time=168)` — oceanographic-style time × depth cast.

| Backend | Read slice (s) | Write (s) | Storage (MiB) |
|---|---:|---:|---:|
| **atlas-bulk** | **0.08** | 0.77 | 55.5 |
| **atlas (default)** | **0.32** | 0.91 | 55.5 |
| atlas + `--use-dask` | 2.03 | 2.98 | 55.6 |
| netcdf | 4.27 | 2.98 | 62.3 |
| zarr | 4.07 | 11.43 | 61.6 |

This is the workload atlas was structurally designed for. The shared `atlas.msgpack.zst` + one `.af` file per variable means a 1000-dataset open is essentially free, and the read path collapses to "decompress 1000 small blocks from 2 sequential files." Both zarr's 1000-store mfdataset and netcdf's 1000-file mfdataset pay metadata overhead 1000× over.

Note `--use-dask` is *slower* than default atlas here: when per-dataset I/O is tiny, dask scheduler overhead dominates. The rule of thumb flips by workload — dask helps when per-dataset decompression is the bottleneck (gridded), hurts when per-dataset overhead is the bottleneck (profile).

#### `--case sensors --datasets 1000` (tiny per-dataset shapes — atlas's design wins)

| Backend | Read slice (s) | Write (s) | Storage (KiB) |
|---|---:|---:|---:|
| atlas | ~0.02 | ~0.1 | ~80 |
| zarr | ~0.6 | ~2.0 | ~370 |
| netcdf | ~0.25 | ~0.16 | ~730 |

For tiny-shape datasets the comparison is dominated by per-dataset overhead, which is exactly atlas's structural advantage (one store, one metadata file, one physical file per array name).

### What the numbers say

- **Reads on gridded** (decompression-dominated, with realistic chunking + slice push-down): `atlas-bulk` reads in **2.12s vs zarr's 6.00s — 2.8× faster**. `atlas + --use-dask` (using the new `view.read_arrays(...)` fast path) hits **3.21s — 1.9× faster than zarr**. The win comes from atlas's structural advantage (one open of one file per variable, in-memory metadata) combined with APIs that bypass `open_as_xarray_dataset`'s xr.Dataset + per-chunk dask graph overhead. zarr and netcdf both push the slice down through `open_mfdataset(...).isel(...)` via dask graph optimization — they're not penalised by the chunking; atlas just wins by amortising metadata and skipping per-dataset Python overhead.
- **Reads on profile** (overhead-dominated): atlas wins by an order of magnitude. `atlas-bulk` is **~50× faster than zarr** (0.08s vs 4.07s) because the per-dataset I/O is small enough that everything is overhead — and atlas's structural design is built for exactly that case. Default serial `atlas` is still ~12× faster than zarr.
- **Default `atlas.open_as_xarray_dataset` iteration is currently SLOW on chunked storage** (10.23s vs zarr's 6.00s). It goes through `open_as_xarray_dataset(name)` which returns dask-backed arrays per chunk (8 chunks/var × 3 vars × 1000 datasets = 24,000 dask delayed tasks just to build the graph). Dask graph overhead exceeds the parallelism win. **The fix is to use `view.read_arrays(vars, start, shape)` for per-dataset slice reads** — that's what `atlas + --use-dask` does internally and why it hits 3.21s. For cross-dataset reads of the *same* slice from many datasets, prefer `Atlas.open_as_many_xarray_dataset` / `read_array_across_stacked` (the `atlas-bulk` row).
- **Workload sensitivity**: `--use-dask` helps when per-dataset decompression is the bottleneck and *hurts* when per-dataset overhead is the bottleneck (profile: default `atlas` 0.32s → dask 2.03s). Picking the right API for the workload matters more than picking dask vs serial.
- **Writes on gridded**: zarr is fastest (~31s vs atlas's ~50s) — each `to_zarr(...)` just dumps bytes into a per-dataset subtree, while atlas does more bookkeeping per call (schema registration, per-dataset attrs into the shared `atlas.msgpack.zst`, block addressing in the shared array file). Atlas still beats netcdf decisively.
- **Writes on profile**: atlas wins ~12× (0.91s vs zarr's 11.43s, netcdf's 2.98s). At small per-dataset sizes the metadata write dominates, and atlas's amortised single-flush model wins.
- **Storage**: atlas and zarr are essentially tied (both zstd); netcdf wins on the gridded case because zlib happens to compress this particular workload tighter, but loses on profile/sensors because the per-file overhead dominates.

### Caveats

- **Single laptop, single run** — variance can be 10–20%. The benchmark deliberately doesn't drop OS caches between runs (it measures warm-cache repeat-query performance, which is what real analytic workloads see).
- **Local filesystem only** — atlas's biggest potential read win is over a high-latency object store where the metadata-shape difference dominates; that's a different benchmark.
- **`netCDF4` vs `h5netcdf`** — using the C-library binding (the more common production choice).
- Run it yourself: `pip install -e "pyatlas[bench]"` then `python pyatlas/benchmarks/bench_collection.py --case gridded --datasets 1000`. See [`pyatlas/benchmarks/README.md`](pyatlas/benchmarks/README.md) for all flags.

---

## Thread safety

`Atlas` and `DatasetView` are `Send + Sync` and work with the default multi-thread Tokio runtime.

Each physical array file is guarded by a `tokio::sync::RwLock`:

| Operation | Lock held |
| --- | --- |
| `read_array`, `array_stats` | Shared read lock — multiple callers on the same file proceed in parallel |
| `write_array`, `define_array`, `delete_array`, `Atlas::flush`, `Atlas::compact` | Exclusive write lock — serialised against all other access |

The store-level array-file cache map uses a `parking_lot::RwLock` that is never held across an `await` point. The shared `StoreMeta` uses a separate `parking_lot::Mutex` and is only mutated through short, lock-release-await-free critical sections (so `list_datasets`, `dataset_exists`, etc. remain synchronous).

---

## License

See [LICENSE](LICENSE).
