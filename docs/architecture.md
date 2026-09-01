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
AtlasWriter ──add_dataset()──▶ DatasetWriter ──finish()──┐
     │                              │                    │
     │                       stages to a local           │ appends the
     │                       array-format file           │ segment, records
     │                                                   │ the footer entry
     └──finish()──▶ footer + trailer ◀───────────────────┘

Atlas ──dataset()──▶ DatasetView ──read_array()──▶ ndarray
  │                       │
  │ footer, read once     │ opens the segment on first use,
  │ at open               │ through SegmentStore
```

- **`AtlasWriter`** owns the output stream, the interner, and the scratch area.
- **`DatasetWriter`** stages one dataset. It touches the shared output only in
  `finish()`, under one lock. Several can therefore stage at once.
- **`Atlas`** holds the decoded footer and the deletion mask.
- **`DatasetView`** answers metadata from the footer with no I/O. It opens its
  segment on demand.
- **`SegmentStore`** presents one byte range of the container to
  `array-format` as one standalone object.

## Why segments are `array-format` files

The bytes of each dataset are a complete `array-format` file, held word for
word. The other option is a private chunk table in the atlas footer. That means
a second implementation of block allocation, per-block codecs, variable-length
encoding, fill values, and partial-region assembly. To embed the file reuses
all of it, and costs one adapter.

It also makes the format open to inspection. `DatasetView::segment_range()`
gives the byte offsets. `dd` those out, and `array-format` opens the result
with no atlas in the way. `tests/integration.rs` asserts that.

The cost is one indirection. A read of a dataset opens its segment footer
first, in two small range reads, and then fetches chunks. Each collection
handle caches those reads. They happen once per dataset, not once per read.

## Concurrency

**A read** needs no lock. The data is immutable, so `Atlas` and `DatasetView`
are `Send + Sync`, and two reads never contend. A `tokio::sync::OnceCell` opens
each segment handle once. Every segment in a collection shares the block cache,
which keys on `(segment path, block id)`. The virtual path carries the dataset
ordinal, so the blocks of one dataset can never answer the read of another.

**A write** shares one `tokio::sync::Mutex` over the output stream and the
footer. A `DatasetWriter` takes that lock only in `finish()`, for its append.
Staging is therefore parallel, and only the append serializes. Datasets land in
finish order.

The deletion mask is the one part of a finished collection that changes. A
`parking_lot::RwLock` guards it in memory, and a write replaces the whole file
on disk. Concurrent deletes are last-writer-wins. See
[format.md](format.md#deletion-mask).
