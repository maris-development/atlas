# ATLAS: Aggregated Tensor Large Array Store

[![CI](https://github.com/maris-development/atlas/actions/workflows/ci.yaml/badge.svg)](https://github.com/maris-development/atlas/actions/workflows/ci.yaml)
[![crates.io](https://img.shields.io/crates/v/atlas-rust.svg?logo=rust)](https://crates.io/crates/atlas-rust)
[![docs.rs](https://img.shields.io/docsrs/atlas-rust?logo=docsdotrs&label=docs.rs)](https://docs.rs/atlas-rust)
[![PyPI](https://img.shields.io/pypi/v/atlas-python.svg?logo=pypi&logoColor=white)](https://pypi.org/project/atlas-python/)
[![Python docs](https://img.shields.io/badge/docs-atlas--python-blue?logo=materialformkdocs&logoColor=white)](https://maris-development.github.io/atlas/)
[![License](https://img.shields.io/crates/l/atlas-rust.svg)](LICENSE)

**Thousands of N-dimensional datasets in one immutable file.**

A collection is one write-once file. A dataset is a set of named N-dimensional
arrays with attributes. It has the shape of a NetCDF file or an
`xarray.Dataset`. Each dataset occupies one contiguous byte range inside the
file. A footer at the end records where each one lives, with its schema and its
attributes.

```text
my_collection/
├── data.atlas      ATLS │ segment │ segment │ … │ footer │ trailer
└── deleted.mask    optional: ordinals of deleted datasets
```

Two results follow, and they are most of the point:

- **Metadata is one read.** An open of a collection fetches the footer and
  nothing else. To list the datasets, to inspect a schema, and to read an
  attribute are then free. Ten datasets and a million datasets cost the same.
- **Data arrives chunk by chunk.** A read of a region of an array fetches only
  the chunks that region overlaps.

Atlas builds on [`array-format`](https://github.com/robinskil/array-format) for
the chunked-array encoding, and on
[`object_store`](https://crates.io/crates/object_store) for I/O. A collection
therefore behaves the same on local disk, S3, GCS, Azure, and in memory.

> **Python?** `pip install atlas-python` gives the `atlas` command. Run
> `atlas create` on a directory of NetCDF files, then `ls`, `show`, `info`, and
> `rm`. Each works on a local path and on a bucket. The Rust API reads array
> data. See [`atlas-python/`](atlas-python/) and the
> [documentation site](https://maris-development.github.io/atlas/).
>
> **Architecture?** [`docs/`](docs/) walks through it:
> [architecture](docs/architecture.md), [data model](docs/data-model.md),
> [the format](docs/format.md), [write path](docs/write-path.md),
> [read path](docs/read-path.md), and [the Python package](docs/python.md).

---

## Quick start

```rust
use atlas::{Atlas, AtlasWriter, Attr, WriterConfig};
use ndarray::Array2;

# async fn run() -> atlas::Result<()> {
// Build. Nothing is readable until finish().
let w = AtlasWriter::create_path("/data/weather", WriterConfig::default()).await?;
{
    let mut ds = w.add_dataset("jan_2024").await?;
    ds.define_array::<f32>(
        "temperature",
        vec!["lat".into(), "lon".into()],
        vec![4, 8],
        Some(vec![2, 4]),   // chunk shape
        None,               // fill value
    ).await?;
    ds.write_array("temperature", vec![0, 0],
                   Array2::<f32>::from_elem([4, 8], 20.0).into_dyn().view()).await?;
    ds.set_attribute("month", Attr::Int64(1));
    ds.finish().await?;
}
w.finish().await?;

// Read. Opening touches only the footer.
let atlas = Atlas::open_path("/data/weather").await?;
assert_eq!(atlas.list_datasets(), vec!["jan_2024".to_string()]);

let ds = atlas.dataset("jan_2024")?;
assert_eq!(ds.array_meta("temperature").unwrap().shape, vec![4, 8]);
assert_eq!(ds.get_attribute("month"), Some(Attr::Int64(1)));

// Only this line fetches array bytes, and only the chunks it needs.
let window = ds.read_array::<f32>("temperature", vec![0, 0], vec![2, 4]).await?;
# Ok(())
# }
```

```bash
cargo add atlas-rust
```

---

## Immutability

A collection cannot change after a write. There is no append, no in-place
update, and no compaction. To change a dataset, rewrite the whole collection.

That constraint keeps the format simple. It removes the delta layers a read
must resolve. It removes the tombstones between the data. It removes the
ordinal that moves under a reader. It removes the durability boundary. The file
has a trailer, or it is no collection.

**A delete** is the one exception. It adds an ordinal to a small
`deleted.mask` file beside the container, and never touches the container. Each
ordinal stays put, and this reclaims no space. `Atlas::delete_datasets` takes
any number of names, and still writes the mask once.

| Not available | Instead |
|---|---|
| Append to a finished collection | Rewrite it |
| Change an array | Rewrite it |
| `flush` or `compact` | Nothing to flush, and no layer to compact |
| Reclaim the space of a deleted dataset | Rewrite it |

---

## What is in the file

### Container

```text
offset 0     b"ATLS"                     4 B   leading magic
offset 4     format_version u32 LE = 1   4 B
offset 8     segment[0]                        a complete array-format file
             segment[1] …                      back to back, no padding
             footer_bytes                      zstd(msgpack(CollectionFooter))
end - 16     footer_size u64 LE          8 B  ┐
end - 8      format_version u32 LE = 1   4 B  ├ trailer
end - 4      b"ATLS"                     4 B  ┘
```

An open reads the last 64 KiB and checks the trailer. The footer usually
arrives in the same request. The magic appears at both ends. A reader checks
the trailing copy. The leading copy lets `file` and `xxd` name the file.

### Segments

One per dataset. Each is a complete `array-format` file that describes itself.
Cut one out, and it opens on its own:

```bash
# offsets from DatasetView::segment_range()
dd if=data.atlas of=jan.af bs=1 skip=8 count=1438
```

Inside, each array keys on its real name, and each block records its own codec.
Nothing therefore needs to tell a reader how a collection compresses.

### Footer

MessagePack, then zstd. It holds every dataset name, segment byte range,
schema, and attribute. Two pools keep it small. Schemas intern by content hash,
so a fleet of a thousand datasets of one shape stores one copy. Attribute keys
intern as strings.

Attribute **values** live here too, and not in the segments. So do the minimum,
the maximum, and the null count of each array. The staging step computes those,
because the writer walks the data anyway. A metadata-only open therefore
answers every question about a collection with no further I/O.

Full byte-level detail in [`docs/format.md`](docs/format.md).

---

## Reading

```rust
let atlas = Atlas::open_path(path).await?;   // 1–2 requests, any collection size

atlas.list_datasets();
atlas.list_arrays();
atlas.array_stats("temperature");            // every live dataset, combined
atlas.array_stats_by_dataset("temperature"); // the same, split per dataset
let ds = atlas.dataset("jan_2024")?;         // no I/O
ds.schema();
ds.array_meta("temperature");
ds.attributes();
ds.array_fill_value("temperature");
ds.array_stats("temperature");               // min, max, null count, row count
```

None of that touches the store after the open. `tests/integration.rs` proves it
with a request-counting `ObjectStore`.

An array read is lazy and partial:

```rust
let all    = ds.read_array::<f32>("temperature", vec![], vec![]).await?;
let window = ds.read_array::<f32>("temperature", vec![1, 3], vec![2, 2]).await?;
```

The first read on a dataset opens its segment, in two small range reads. The
handle then stays in the cache. The read fetches the overlapping chunks, and no
more. Every cell nobody wrote comes from the fill value, and costs no I/O.

---

## Writing

`AtlasWriter` streams one object. Each dataset stages as a complete
`array-format` file on local scratch. The writer then copies that file into the
stream, byte for byte:

```text
add_dataset("jan")  ──▶ scratch/1/data.af      define, write, define, write, …
  finish()          ──▶ flush, compact, copy   ──▶ container[8 .. 4_100]
AtlasWriter::finish ──▶ footer, trailer, done
```

Memory stays bounded, whatever the dataset size. `array-format` spills each
compressed chunk to a temporary file, and the copy streams in 8 MiB pieces.

`add_dataset` returns an owned writer, so several datasets can stage at once.
Each takes the writer's lock for its append alone. The segments therefore land
in finish order, and never interleave.

---

## Types

Scalars: `Bool`, `Int8`…`Int64`, `UInt8`…`UInt64`, `Float32`, `Float64`,
`String`, `Binary`, `TimestampNs`. Nested: `List<T>`, `FixedSizeList<T, n>`.

Two datasets can declare one array name with unrelated types. Atlas stores each
as declared. There is no merged schema, and no type widening. One segment per
dataset leaves nothing to reconcile. `Atlas::array_stats` leaves such a dataset
out, because two dtypes do not compare.

An attribute takes any scalar type, or a list of one type. It sits at dataset
scope or at array scope. A timestamp has its own wire tag, so a string that
looks like a date stays a string.

---

## Compression

`WriterConfig { codec }` is `Zstd`, `Lz4`, or `Uncompressed`. `Zstd` is the
default. `block_target_size` defaults to 8 MiB. Each block describes itself, so
`Atlas::open` takes no codec argument.

---

## Thread safety

A read needs no lock. The data is immutable, so `Atlas` and `DatasetView` are
`Send + Sync`, and two reads never contend. A `OnceCell` opens each segment
handle once. One block cache serves a whole collection.

A write shares one `tokio::sync::Mutex` over the output stream and the footer.
A dataset takes that lock for its append alone.

---

## Testing

```bash
cargo test -p atlas-rust
```

This covers the format framing, the footer and mask codecs, the segment-store
adapter, and the whole lifecycle. Two committed fixtures pin compatibility.
`tests/fixtures/golden_v1/` is a v1 container, read back with every value
asserted. The Python xarray layer writes `tests/fixtures/from_python/`, and
Rust checks it. That keeps the two in agreement, because Python reads no array.

---

## License

See [LICENSE](LICENSE).
