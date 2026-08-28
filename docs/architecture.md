# Architecture

## The layers

```text
┌─────────────────────────────────────────────────────────────┐
│ atlas-python/python/atlas/                                  │
│   xarray conventions, attribute encoding, the facade        │  Python
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
| Container framing, footer, deletion mask | `atlas-rust` — `src/format/` |
| Building a collection | `atlas-rust` — `src/writer/` |
| Opening one, lazy reads | `atlas-rust` — `src/reader/` |
| Chunk layout, blocks, compression, fill values | `array-format` |
| Byte-range I/O against any backend | `object_store` |
| numpy ⇄ Rust arrays, Python exceptions | `atlas-python/src/` |
| xarray mapping, attribute encoding | `atlas-python/python/` |

The **file format is Rust only**. `atlas-python` holds no format knowledge: it
cannot construct a header, a footer, or a mask. Grep it for `ATLS` and you get
nothing. That is deliberate — one implementation of the bytes, one place for a
bug to live.

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
  `finish()`, under one lock, so several may be staged at once.
- **`Atlas`** holds the decoded footer and the deletion mask.
- **`DatasetView`** answers metadata from the footer with no I/O, and opens its
  segment lazily.
- **`SegmentStore`** presents one byte range of the container to `array-format`
  as if it were a standalone object.

## Why segments are `array-format` files

Each dataset's bytes are a complete `array-format` file, embedded verbatim. The
alternative — a bespoke chunk table in the atlas footer — would mean
reimplementing block allocation, per-block codecs, variable-length encoding,
fill values, and partial-region assembly. Embedding reuses all of it for the
cost of one adapter.

It also makes the format inspectable. `DatasetView::segment_range()` gives you
byte offsets; `dd` those out and `array-format` opens the result with no atlas
involved. `tests/integration.rs` asserts exactly that.

The cost is one indirection: reading a dataset opens its segment footer first
(two small range reads), then fetches chunks. Those reads are cached per
collection handle, so it is once per dataset, not once per read.

## Concurrency

**Reading** needs no locks. The data is immutable, so `Atlas` and `DatasetView`
are `Send + Sync` and concurrent reads never contend. Segment handles open once
through a `tokio::sync::OnceCell`; the block cache is shared across every
segment in a collection and keyed by `(segment path, block id)` — the virtual
path carries the dataset ordinal, so one dataset's blocks can never answer
another's read.

**Writing** shares one `tokio::sync::Mutex` over the output stream and the
footer under construction. A `DatasetWriter` takes it only in `finish()`, for
the duration of its append, so staging is fully parallel and only the append is
serialized. Datasets land in finish order.

The one mutable thing in a finished collection is the deletion mask, guarded by
a `parking_lot::RwLock` in memory and rewritten whole on disk. Concurrent
deletes are last-writer-wins; see [format.md](format.md#deletion-mask).
