# Atlas architecture

Atlas keeps thousands of named datasets in **one immutable file**. This
directory explains how, from the top down.

Read these in order:

| # | Document | What it covers |
|---|---|---|
| 1 | [architecture.md](architecture.md) | The layers, and who owns what |
| 2 | [data-model.md](data-model.md) | Collections, datasets, arrays, attributes |
| 3 | [format.md](format.md) | The on-disk format, byte for byte |
| 4 | [write-path.md](write-path.md) | How a collection is built |
| 5 | [read-path.md](read-path.md) | How one is read, and what it costs |
| 6 | [python.md](python.md) | The Python package. Five operations and a CLI |

## The one idea

A collection is one write-once file. Every dataset occupies one contiguous byte
range inside it. A footer at the end records where each one lives, with its
schema and its attributes.

```text
my_collection/
├── data.atlas      ATLS │ segment │ segment │ … │ footer │ trailer
└── deleted.mask    optional: ordinals of deleted datasets
```

Two results follow from that, and they are most of the value of atlas:

**Metadata is one read.** An open of a collection fetches the footer and
nothing else. To list the datasets, to inspect a schema, and to read an
attribute are then free. Ten datasets and a million datasets cost the same.

**Data arrives chunk by chunk.** A segment is a complete `array-format` file
that describes itself. A read of a region of an array fetches only the chunks
that region overlaps.

## What immutability buys

A collection cannot change after a write. There is no append, no in-place
update, and no compaction. To change a dataset, rewrite the whole collection.

That is a real constraint, and it is the point. What it removes:

- no delta layers to resolve on read
- no tombstones interleaved with data
- no ordinal that shifts under a concurrent reader
- no durability boundary to reason about. The file has a trailer, or it does
  not exist

The one exception is deletion, which writes a small mask beside the container
and never touches it. See [format.md](format.md#deletion-mask).

## Where the code lives

The **file format is Rust only**. `src/format/` defines the framing, the
footer, and the deletion mask. Nothing outside the `atlas-rust` crate produces
or parses a byte of a container.

The Python package is a binding layer over that, and five operations. Build a
collection from NetCDF files, remove datasets, list them, describe one, and
summarize the whole. Both a library and the `atlas` command offer them. See
[python.md](python.md).
