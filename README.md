# array-store

A directory-based store for thousands of named datasets, each holding N-dimensional typed arrays. Built on top of the [`array-format`](https://github.com/robinskil/array-format) (`.af`) binary format, with Zstd compression, chunked I/O, and an [`object_store`](https://crates.io/crates/object_store) backend that works on local disk, S3, GCS, Azure Blob, and in-memory.

---

## What it does

`array-store` is designed for workloads where you have a large collection of similarly-shaped datasets — such as one dataset per time step, sensor station, or simulation run — and you want to query a single variable (e.g. `temperature`) across all of them efficiently.

Each dataset is a named group of N-dimensional arrays with typed per-dataset attributes. Datasets that share an array name (e.g. every `jan_2024`, `feb_2024`, … all have `temperature`) are stored together in the same physical file, keyed by dataset name inside the file.

```text
my_store/
├── array_store.json          ← dataset registry and per-dataset attributes (JSON)
├── temperature/
│   └── data.af         ← one ArrayFile holding temperature for every dataset
├── pressure/
│   └── data.af
└── time/
    └── data.af
```

---

## File format

### `array_store.json`

The registry is a plain JSON file written on every `flush()`. It stores:

- **Store version** — for future format upgrades.
- **Dataset names** — the complete list of datasets in the store.
- **Per-dataset attributes** — typed key-value pairs (bool, int8/16/32/64, uint8/16/32/64, float32/64, string).
- **Array schemas** — per array: dtype, shape, chunk shape, and named dimensions.

Because `array_store.json` is human-readable and self-describing, you can inspect or audit the store contents with any JSON tool without needing the library.

### `<array_name>/data.af`

Each array variable gets its own subdirectory with a single `data.af` binary file. The `.af` format (from the `array-format` crate) is a columnar, chunk-oriented binary format:

- **Multiple datasets in one file** — every dataset that owns this variable is stored as a named entry inside the same file.
- **Chunked layout** — arrays are split into chunks of a user-specified shape, so partial reads and writes touch only the relevant blocks.
- **Zstd compression** — each block is compressed with Zstd (default codec). Block target size is 8 MiB.
- **Persisted statistics** — on `flush()`, min, max, null count, and row count are computed per array per dataset and stored alongside the data. Statistics survive store reopening.
- **In-memory caches** — a 256 MiB decoded block cache and a 64 MiB raw I/O cache sit in front of the object store for repeated reads.

---

## Quick start

```rust
use std::sync::Arc;
use array_store::{ArrayStore, Attr};
use ndarray::Array2;
use object_store::{local::LocalFileSystem, path::Path};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let store: Arc<dyn object_store::ObjectStore> = Arc::new(LocalFileSystem::new());
    let prefix = Path::from_absolute_path("/tmp/my_store")?;

    // Create a new store
    let mut s = ArrayStore::create(store.clone(), prefix.clone()).await?;

    // Create a dataset and write arrays
    let mut ds = s.create_dataset("jan_2024").await?;
    ds.define_array::<f32>(
        "temperature",
        vec!["lat".into(), "lon".into()],
        vec![8, 16],
        Some(vec![4, 8]),  // chunk shape
        None,
    ).await?;

    let data = Array2::<f32>::from_elem([8, 16], 20.0).into_dyn();
    ds.write_array("temperature", vec![0, 0], data.view()).await?;

    ds.set_attribute("month", Attr::UInt32(1));
    ds.set_attribute("station", Attr::String("KNMI".into()));
    ds.flush().await?;

    // Reopen and read back
    let s2 = ArrayStore::open(store, prefix).await?;
    let ds2 = s2.open_dataset("jan_2024").await?;

    // Full read
    let temp = ds2.read_array::<f32>("temperature", vec![], vec![]).await?.unwrap();

    // Partial read — one chunk region
    let chunk = ds2.read_array::<f32>("temperature", vec![0, 0], vec![4, 8]).await?.unwrap();

    // Query persisted statistics
    let stats = ds2.array_stats("temperature").await?.unwrap();
    println!("rows={} min={:?} max={:?}", stats.row_count, stats.min, stats.max);

    Ok(())
}
```

---

## Key concepts

| Concept | Description |
| --- | --- |
| **Store** | The root directory, managed by `ArrayStore`. |
| **Dataset** | A named group of arrays + typed attributes, accessed via `DatasetView`. |
| **Array** | An N-dimensional typed array with named dimensions and an optional chunk shape. |
| **Attribute** | A typed scalar attached to a dataset (metadata, not array data). |
| **Array file** | One `data.af` file per variable name, shared across all datasets that define that variable. |
| **Flush** | Persists all pending writes and recomputes statistics. Must be called explicitly. |
| **Compact** | Rewrites the `.af` file to reclaim space after deletes. |

---

## Supported dtypes

`bool`, `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`, `f32`, `f64`, `String`, `Binary`, `List<T>`, `FixedSizeList<T, N>`

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

**array-store** uses a *variable-first* layout:

```text
array_store/
├── temperature/
│   └── data.af        ← temperature for ALL datasets in one file
└── pressure/
    └── data.af
```

Reading `temperature` for 1 000 datasets means opening exactly **one file**. This is the primary design goal of this format.

### Feature comparison

| Feature | NetCDF-4 | Zarr v3 | array-store |
| --- | --- | --- | --- |
| Layout | Dataset-first | Dataset-first | Variable-first |
| Compression | Deflate / Zstd / … | Any codec plugin | Zstd |
| Chunking | Yes | Yes | Yes |
| Cloud object store | No (needs FUSE/etc) | Yes (native) | Yes (via `object_store`) |
| Multiple datasets in one file | No | No | Yes (all datasets per variable) |
| Metadata format | Binary (HDF5) | JSON | JSON |
| Cross-dataset column scan | Slow (N file opens) | Slow (N directory opens) | Fast (1 file open) |
| Partial reads | Yes | Yes | Yes |
| Statistics (min/max/nulls) | No | No | Yes (persisted on flush) |
| Self-describing metadata | Yes | Yes | Yes (`array_store.json`) |
| Language support | C/Python/Julia/… | Python/Java/… | Rust |
| Mutable after write | Limited | Yes | Yes (chunked overwrites + compact) |

### When to choose array-store

- You have many homogeneous datasets (same variable schema, different instances — time steps, stations, runs).
- Your primary query is "give me variable X across all datasets" — a column scan across the dataset dimension.
- You want a simple on-disk layout with no special runtime dependencies and human-readable metadata.
- You are working in Rust with an async runtime and an `object_store`-compatible backend.

### When to choose Zarr or NetCDF

- You need wide ecosystem support (Python, Julia, C libraries, GIS tools).
- Your primary query pattern is "give me all variables for one dataset" (dataset-first access).
- You need hierarchical group nesting beyond two levels.
- You need codec flexibility beyond Zstd.

---

## Performance characteristics

### Cross-dataset scans (where this format excels)

Because all datasets share a single `.af` file per variable, scanning `temperature` across N datasets costs:

- **1 file open** regardless of N.
- Sequential reads within one file, which are friendly to OS read-ahead and object-store range requests.
- Zstd decompression only for the blocks actually read.

In a dataset-first format, the same scan requires N file or directory opens, which is bounded by metadata latency, not throughput — especially painful on object stores where each `HEAD`/`GET` has ~10 ms overhead.

### Chunked I/O

Chunk shapes are set per-array at definition time. A chunk shape equal to the full array shape (the default when `chunk_shape` is omitted) gives a single compressed block per dataset entry, minimising overhead for small arrays. Smaller chunks allow fine-grained partial reads and updates at the cost of more blocks.

### In-memory caches

Two caches sit in front of the object store:

| Cache | Default size | What it holds |
| --- | --- | --- |
| Decoded block cache | 256 MiB | Decompressed array chunks, ready for use |
| I/O cache | 64 MiB | Raw compressed bytes from the object store |

The decoded cache means repeated reads of the same chunk cost only a hash-map lookup. Both caches are shared across all `DatasetView`s that open the same `ArrayStore`.

### Persisted statistics

Min, max, null count, and row count are computed and persisted on every `flush()`. Downstream systems can read these statistics from the opened `DatasetView` without touching array data at all — useful for query planning, dashboards, or data-quality checks.

### Write path

Writes are buffered in-memory. Calling `flush()` compresses and writes all pending blocks and updates `array_store.json` atomically (a single `PUT`). This means the write path scales with the number of modified chunks, not the number of datasets.

### Compaction

After deleting arrays or datasets, the underlying `.af` files may retain dead space. Calling `compact()` on a `DatasetView` rewrites the file in-place, reclaiming storage.

---

## Thread safety

`ArrayStore` and `DatasetView` are `Send + Sync` and work with the default multi-thread Tokio runtime.

Each physical array file is guarded by a `tokio::sync::RwLock`:

| Operation | Lock held |
| --- | --- |
| `read_array`, `array_stats` | Shared read lock — multiple callers on the same file proceed in parallel |
| `write_array`, `define_array`, `delete_array`, `flush`, `compact` | Exclusive write lock — serialised against all other access |

The store-level cache map uses a `parking_lot::RwLock` that is never held across an `await` point.

---

## License

See [LICENSE](LICENSE).
