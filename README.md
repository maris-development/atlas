# ATLAS — Aggregated Tensor Large Array Store

[![CI](https://github.com/maris-development/atlas/actions/workflows/ci.yaml/badge.svg)](https://github.com/maris-development/atlas/actions/workflows/ci.yaml)
[![crates.io](https://img.shields.io/crates/v/atlas-rust.svg?logo=rust)](https://crates.io/crates/atlas-rust)
[![docs.rs](https://img.shields.io/docsrs/atlas-rust?logo=docsdotrs&label=docs.rs)](https://docs.rs/atlas-rust)
[![PyPI](https://img.shields.io/pypi/v/atlas-python.svg?logo=pypi&logoColor=white)](https://pypi.org/project/atlas-python/)
[![Python docs](https://img.shields.io/badge/docs-atlas--python-blue?logo=materialformkdocs&logoColor=white)](https://maris-development.github.io/atlas/)
[![License](https://img.shields.io/crates/l/atlas-rust.svg)](LICENSE)

**Thousands of N-dimensional datasets in one immutable file.**

A collection is a single write-once file. Every dataset — a set of named
N-dimensional arrays with attributes, the shape a NetCDF file or an
`xarray.Dataset` has — occupies a contiguous byte range inside it, and a footer
at the end records where each one lives along with its schema and attributes.

```text
my_collection/
├── data.atlas      ATLS │ segment │ segment │ … │ footer │ trailer
└── deleted.mask    optional: ordinals of deleted datasets
```

Two things follow, and they are most of the point:

- **Metadata is one read.** Opening a collection fetches the footer and nothing
  else. Listing datasets, inspecting schemas, and reading attributes are then
  free — for ten datasets or a million.
- **Data is fetched by the chunk.** Reading a region of an array fetches only
  the chunks that region overlaps.

Built on [`array-format`](https://github.com/robinskil/array-format) for the
chunked-array encoding and [`object_store`](https://crates.io/crates/object_store)
for I/O, so a collection works identically on local disk, S3, GCS, Azure, or
in memory.

> **Python?** `pip install atlas-python`, then `import atlas`. Python builds
> collections from xarray and reads their metadata; array data is read from
> Rust. See [`atlas-python/`](atlas-python/) and the
> [documentation site](https://maris-development.github.io/atlas/).
>
> **Architecture?** [`docs/`](docs/) walks through it —
> [architecture](docs/architecture.md), [data model](docs/data-model.md),
> [the format](docs/format.md), [write path](docs/write-path.md),
> [read path](docs/read-path.md), and [Python/xarray](docs/python-xarray.md).

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

A collection cannot be changed once written. There is no append, no in-place
update, and no compaction: to change a dataset you rewrite the whole collection.

That constraint is what keeps the format simple. It removes delta layers to
resolve on read, tombstones interleaved with data, ordinals shifting under a
concurrent reader, and any durability boundary to reason about — the file either
has a trailer or it is not a collection.

The one exception is **deleting a dataset**, which appends an ordinal to a small
`deleted.mask` file beside the container and never touches it. Ordinals stay
stable, and no space is reclaimed.

| Not available | Instead |
|---|---|
| Append to a finished collection | Rewrite it |
| Modify an array | Rewrite it |
| `flush` / `compact` | Nothing to flush; no layers to compact |
| Reclaim space from a deleted dataset | Rewrite it |

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

Opening reads the last 64 KiB, validates the trailer, and usually has the footer
in the same request. The magic appears at both ends: the trailing copy is what a
reader checks, the leading copy is so `file` and `xxd` can identify it.

### Segments

One per dataset. Each is a complete, self-describing `array-format` file — you
can cut one out and open it standalone:

```bash
# offsets from DatasetView::segment_range()
dd if=data.atlas of=jan.af bs=1 skip=8 count=1438
```

Inside, arrays are keyed by their real names and every block records its own
codec, which is why a reader never needs to be told how a collection was
compressed.

### Footer

MessagePack, zstd-compressed, holding every dataset's name, segment byte range,
schema, and attributes. Two pools keep it small: schemas are interned by content
hash, so a fleet of a thousand identically-shaped datasets stores one copy; and
attribute keys are interned as strings.

Attribute **values** live here too, not in the segments — which is what makes a
metadata-only open answer every attribute question with no further I/O.

Full byte-level detail in [`docs/format.md`](docs/format.md).

---

## Reading

```rust
let atlas = Atlas::open_path(path).await?;   // 1–2 requests, any collection size

atlas.list_datasets();
atlas.list_arrays();
let ds = atlas.dataset("jan_2024")?;         // no I/O
ds.schema();
ds.array_meta("temperature");
ds.attributes();
ds.array_fill_value("temperature");
```

None of that touches the store after the open. `tests/integration.rs` proves it
with a request-counting `ObjectStore`.

Array reads are lazy and partial:

```rust
let all    = ds.read_array::<f32>("temperature", vec![], vec![]).await?;
let window = ds.read_array::<f32>("temperature", vec![1, 3], vec![2, 2]).await?;
```

The first read on a dataset opens its segment (two small range reads, cached
thereafter); the read itself fetches only the overlapping chunks. Cells never
written come from the fill value at no I/O cost.

---

## Writing

`AtlasWriter` streams one object. Each dataset is staged as a complete
`array-format` file on local scratch, then copied verbatim into the stream:

```text
add_dataset("jan")  ──▶ scratch/1/data.af      define, write, define, write, …
  finish()          ──▶ flush, compact, copy   ──▶ container[8 .. 4_100]
AtlasWriter::finish ──▶ footer, trailer, done
```

Memory stays bounded whatever the dataset size: `array-format` spills compressed
chunks to a temp file, and the copy streams in 8 MiB pieces.

`add_dataset` returns an owned writer, so several datasets can be staged
concurrently — each takes the writer's lock only for its append, so segments
land in finish order and never interleave.

---

## Types

Scalars: `Bool`, `Int8`…`Int64`, `UInt8`…`UInt64`, `Float32`, `Float64`,
`String`, `Binary`, `TimestampNs`. Nested: `List<T>`, `FixedSizeList<T, n>`.

Two datasets may declare the same array name with unrelated types; each is
stored as declared. There is no merged schema and no type widening — with one
segment per dataset there is nothing to reconcile.

Attributes take any scalar type or a homogeneous list, at dataset or array
scope. Timestamps have their own wire tag, so a string that happens to look like
a date stays a string.

---

## Compression

`WriterConfig { codec }` is `Zstd` (default), `Lz4`, or `Uncompressed`, with a
`block_target_size` defaulting to 8 MiB. Blocks are self-describing, so
`Atlas::open` takes no codec argument.

---

## Thread safety

Reads need no locks: the data is immutable, so `Atlas` and `DatasetView` are
`Send + Sync` and concurrent reads never contend. Segment handles open once
through a `OnceCell`, and one block cache is shared across a collection.

Writing shares one `tokio::sync::Mutex` over the output stream and the footer
under construction, taken only for the duration of a dataset's append.

---

## Testing

```bash
cargo test -p atlas-rust
```

Covers the format framing, the footer and mask codecs, the segment-store
adapter, and the full lifecycle end to end. Two committed fixtures pin
compatibility: `tests/fixtures/golden_v1/` is a v1 container read back with
every value asserted, and `tests/fixtures/from_python/` is written by the Python
xarray layer and verified from Rust — which is what keeps the two honest now
that Python cannot read arrays.

---

## License

See [LICENSE](LICENSE).
