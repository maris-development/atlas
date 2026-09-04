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
`xarray.Dataset`.

The file stores one segment per **variable**, not one per dataset. A segment
holds one array name across the whole collection, and each dataset's copy sits
inside it under the dataset's own name. A footer at the end records where each
variable lives, and what each dataset declares.

```text
my_collection/
├── data.atlas      ATLS │ temperature │ salinity │ … │ footer │ trailer
└── deleted.mask    optional: ordinals of deleted datasets
```

Three results follow, and they are most of the point:

- **The catalogue is one read.** An open of a collection fetches the footer
  and nothing else. To list the datasets and to inspect what each declares are
  then free. Ten datasets and a million cost the same. The footer repeats
  nothing a segment holds, so a value costs one segment open, and one open
  serves the whole collection.
- **One variable is one file.** To read `temperature` across every dataset
  opens one segment and walks it. The old shape, one segment per dataset, had
  to open every dataset and discard the other variables in each block.
- **Data arrives block by block.** A read fetches only the blocks the region
  overlaps, and a block holds one dtype for a run of neighbouring datasets. It
  therefore compresses far better than a mix.

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
assert_eq!(*ds.array_meta("temperature").unwrap().dtype(), DType::Float32);
assert_eq!(ds.get_attribute("month").await?, Some(Attr::Int64(1)));

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
offset 4     format_version u32 LE = 8   4 B
offset 8     segment[0]                        one variable, array-format
             segment[1] …                      back to back, no padding
             footer_bytes                      zstd(msgpack(CollectionFooter))
end - 16     footer_size u64 LE          8 B  ┐
end - 8      format_version u32 LE = 8   4 B  ├ trailer
end - 4      b"ATLS"                     4 B  ┘
```

An open reads the last 64 KiB and checks the trailer. The footer usually
arrives in the same request. The magic appears at both ends. A reader checks
the trailing copy. The leading copy lets `file` and `xxd` name the file.

### Segments

One per variable, each a complete `array-format` file that describes itself.
Inside, an array keys on the **dataset** name, so `temperature` holds
`jan_2024`, `feb_2024`, and every other dataset that declares it.

That is what makes a scan cheap. `array-format` packs neighbouring chunks into
one block, so a block holds `temperature` for a run of datasets. One fetch
serves them all, and the block holds one dtype, which compresses far better
than a mix. Each block records its own codec, so nothing needs to tell a reader
how a collection compresses.

It also puts the shape, the chunking, the attribute values and the statistics
in the segment, where the data is. The footer repeats none of them.

### Footer

MessagePack, then zstd. It holds every dataset name, every variable's byte
range, and what each dataset declares. **Nothing a segment already holds.**

Three pools keep it small. Every string interns once, whether it names an array
or an attribute. So does every dtype. Every distinct schema interns once.

A schema names things: array names with their element types, and attribute keys
with theirs. No shape, no chunking, no attribute value, no statistic. So
datasets whose arrays differ only in length share one entry, and a directory of
ten thousand files of one convention interns to one.

A dataset is then one `u32` into that pool. The datasets are an `IndexMap` keyed
by name, so a name and its ordinal are one structure and a lookup is one hash.

Everything else sits on the array it belongs to, inside that variable's
segment. Attribute values, because `array-format` attaches an attribute to an
array. Statistics, because it computes them while writing and stores them in
the segment's footer. A segment interns each attribute key and value once, so
`units = "celsius"` across ten thousand datasets is stored once.

A dataset-level attribute has no array of its own, so the reserved `_datasets`
segment gives it one: a rank-0 array per dataset carrying its global
attributes.

Full byte-level detail in [`docs/format.md`](docs/format.md).

---

## Reading

```rust
let atlas = Atlas::open_path(path).await?;   // 1–2 requests, any collection size

// From the footer. No I/O, whatever the collection size.
atlas.list_datasets();
atlas.list_arrays();
let ds = atlas.dataset("jan_2024")?;         // one hash lookup
ds.name();
ds.schema();                                 // array and attribute names, types
ds.array_meta("temperature");                // name and dtype

// From a segment. Everything the footer does not repeat.
ds.array_layout("temperature").await?;       // shape, chunks, dims, fill value
ds.array_stats("temperature").await?;        // min, max, null count, row count
ds.attributes().await?;                      // dataset-level values
ds.array_attributes("temperature").await?;   // one array's values
atlas.array_stats("temperature").await?;     // every live dataset, combined
atlas.array_stats_by_dataset("temperature").await?;   // keyed by dataset name
atlas.attributes_by_dataset(None, "month").await?;   // one key, every dataset
```

`schema()` and `array_meta()` borrow the footer and copy no name. Call
`to_owned_schema()` on either one for an owned copy.

The reading calls open one segment, once for the whole collection. A variable's
statistics and its layout come out of the same open. A dataset that declares no
attribute key costs nothing at all, because the schema settles it first.

None of that touches the store after the open. `tests/integration.rs` proves it
with a request-counting `ObjectStore`.

An array read is lazy and partial:

```rust
let all    = ds.read_array::<f32>("temperature", vec![], vec![]).await?;
let window = ds.read_array::<f32>("temperature", vec![1, 3], vec![2, 2]).await?;
```

The first read of a variable opens its segment, in two small range reads. The
handle then stays in the cache, and every other dataset's copy of that variable
uses it. The read fetches the overlapping blocks, and no more. Every cell
nobody wrote comes from the fill value, and costs no I/O.

---

## Writing

`AtlasWriter` streams one object. Each **variable** builds in an
`array-format` writer, and every dataset writes into it under its own name:

```text
add_dataset("jan")  ──▶ writer[temperature]    temperature/jan
                        writer[salinity]       salinity/jan
add_dataset("feb")  ──▶ writer[temperature]    temperature/feb
                        writer[salinity]       salinity/feb
AtlasWriter::finish ──▶ finish each, copy in, footer, trailer, done
```

A variable's segment is complete only when every dataset has contributed, so
nothing reaches the container until `AtlasWriter::finish`.

Memory stays bounded. The writer packs each chunk into a compressed block as it
arrives, and spills every full block to a temporary file. At finish each
variable lands on local scratch, and the copy streams in 8 MiB pieces.

A `DatasetWriter` takes the writer's lock for each define and each write,
because the variable writers are shared. Ordinals do not depend on that order.
Each dataset carries the number of its `add_dataset` call, and the footer sorts
on it.

---

## Types

Scalars: `Bool`, `Int8`…`Int64`, `UInt8`…`UInt64`, `Float32`, `Float64`,
`String`, `Binary`, `TimestampNs`. Nested: `List<T>`, `FixedSizeList<T, n>`.

Two datasets can declare one array name with unrelated types. Atlas stores each
as declared, as its own array inside that variable's segment. There is no
merged schema, and no type widening. `Atlas::array_stats` leaves such a dataset
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

A write shares one `tokio::sync::Mutex` over the output stream, the variable
writers, and the footer. A dataset takes that lock for each define and each
write.

---

## Testing

```bash
cargo test -p atlas-rust
```

This covers the format framing, the footer and mask codecs, the segment-store
adapter, and the whole lifecycle. Two committed fixtures pin compatibility.
`tests/fixtures/golden_v8/` is a v8 container, read back with every value
asserted. The Python xarray layer writes `tests/fixtures/from_python/`, and
Rust checks it. That keeps the two in agreement, because Python reads no array.

---

## License

See [LICENSE](LICENSE).
