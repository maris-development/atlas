# Architecture

## The layers

```text
┌─────────────────────────────────────────────────────────────┐
│ atlas-python/python/atlas/                                  │
│   five operations, the `atlas` CLI, the xarray mapping      │  Python
├─────────────────────────────────────────────────────────────┤
│ atlas-python/src/                                           │
│   PyO3 bindings: numpy ⇄ ndarray, Python ⇄ Attr, errors     │  Rust (glue)
├─────────────────────────────────────────────────────────────┤
│ atlas-rust/src/                                             │
│   format/  the container: framing, footer, mask, segments   │
│   writer/  builds a collection, once                        │  Rust (core)
│   reader/  opens one, reads lazily                          │
├─────────────────────────────────────────────────────────────┤
│ array-format (crates.io, pinned =0.12.0)                    │
│   one segment: chunked arrays, blocks, codecs, fill values  │  Rust (dep)
├─────────────────────────────────────────────────────────────┤
│ object_store                                                │
│   local filesystem, S3, GCS, Azure, in-memory               │
└─────────────────────────────────────────────────────────────┘
```

## Who owns what

| Concern | Owner |
|---|---|
| Container framing, footer, deletion mask | `atlas-rust`, `src/format/` |
| Building a collection | `atlas-rust`, `src/writer/` |
| Opening one, and lazy reads | `atlas-rust`, `src/reader/` |
| Chunk layout, blocks, compression, fill values | `array-format` |
| Byte-range I/O against any backend | `object_store` |
| numpy ⇄ Rust arrays, Python exceptions | `atlas-python/src/` |
| NetCDF ingest, the CLI, xarray mapping | `atlas-python/python/` |

The **file format is Rust only**. `atlas-python` holds no format knowledge. It
builds no header, no footer, and no mask. Grep it for `ATLS`, and you get
nothing. That is deliberate. One implementation of the bytes gives a bug one
place to live.

## The central types

```text
AtlasWriter ──add_dataset()──▶ DatasetWriter ──write_array()──┐
     │                              │                          │
     │                              │ writes into the          │ one staging
     │                              │ variable's file, under   │ file per
     │                              │ the dataset name         │ variable
     └──finish()──▶ compact each, append, footer + trailer ◀───┘

Atlas ──dataset()──▶ DatasetView ──read_array()──▶ ndarray
  │                       │
  │ footer, read once     │ opens the variable's segment on first use,
  │ at open               │ through SegmentStore
```

- **`AtlasWriter`** owns the output stream, the interner, the scratch area, and
  one staging file per variable.
- **`DatasetWriter`** records what one dataset declares. Each define and each
  write takes the shared lock, because the variable files are shared.
- **`Atlas`** holds the decoded footer and the deletion mask.
- **`DatasetView`** answers names, types, and statistics from the footer with
  no I/O. It opens a variable's segment for `read_array`, for `array_layout`,
  and for an attribute value.
- **`SegmentStore`** presents one byte range of the container to
  `array-format` as one standalone object.

## Why segments are `array-format` files

The bytes of each variable are a complete `array-format` file, held word for
word. The other option is a private chunk table in the atlas footer. That means
a second implementation of block allocation, per-block codecs, variable-length
encoding, fill values, and partial-region assembly. To embed the file reuses
all of it, and costs one adapter.

It also decides the layout. `array-format` packs neighbouring chunks into one
block, and walks its arrays in order. One file per variable therefore fills a
block with one dtype for a run of datasets. That compresses far better than a
mix, and one fetch then serves many datasets.

It holds the shape, the chunking, the dimension names, and the fill value too.
The footer repeats none of them.

The cost is one indirection. A read opens the variable's segment footer first,
in two small range reads, and then fetches blocks. Each collection handle
caches those reads. They happen once per variable, not once per dataset and not
once per read.

## Concurrency

**A read** needs no lock. The data is immutable, so `Atlas` and `DatasetView`
are `Send + Sync`, and two reads never contend. A `tokio::sync::OnceCell` opens
each segment handle once. Every segment in a collection shares the block cache,
which keys on `(segment path, block id)`. The virtual path carries the variable
index, so the blocks of one variable can never answer the read of another.

**A write** shares one `tokio::sync::Mutex` over the output stream, the staging
files, and the footer. A `DatasetWriter` takes that lock for each define and
each write, because every dataset writes into the same per-variable files.
Ordinals do not depend on that order. Each dataset carries the number of its
`add_dataset` call, and the footer sorts on it.

The deletion mask is the one part of a finished collection that changes. A
`parking_lot::RwLock` guards it in memory, and a write replaces the whole file
on disk. Concurrent deletes are last-writer-wins. See
[format.md](format.md#deletion-mask).
