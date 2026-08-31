# Atlas architecture

Atlas keeps thousands of named datasets in **one immutable file**. This
directory explains how, from the top down.

Read in this order:

| # | Document | What it covers |
|---|---|---|
| 1 | [architecture.md](architecture.md) | The layers, and who owns what |
| 2 | [data-model.md](data-model.md) | Collections, datasets, arrays, attributes |
| 3 | [format.md](format.md) | The on-disk format, byte for byte |
| 4 | [write-path.md](write-path.md) | How a collection is built |
| 5 | [read-path.md](read-path.md) | How one is read, and what it costs |
| 6 | [python.md](python.md) | The Python package: five operations and a CLI |

## The one idea

A collection is a single write-once file. Every dataset occupies a contiguous
byte range inside it, and a footer at the end records where each one lives
along with its schema and attributes.

```text
my_collection/
├── data.atlas      ATLS │ segment │ segment │ … │ footer │ trailer
└── deleted.mask    optional: ordinals of deleted datasets
```

Two consequences follow from that, and they are most of what makes atlas
worth using:

**Metadata is one read.** Opening a collection fetches the footer and nothing
else. Listing datasets, inspecting schemas, and reading attributes are then
free, whether the collection holds ten datasets or a million.

**Data is fetched by the chunk.** A segment is a complete, self-describing
`array-format` file. Reading a region of an array fetches only the chunks that
region overlaps.

## What immutability buys

A collection cannot be modified after it is written. There is no append, no
in-place update, and no compaction. To change a dataset you rewrite the whole
collection.

That is a real constraint, and it is the point. What it removes:

- no delta layers to resolve on read
- no tombstones interleaved with data
- no ordinal that shifts under a concurrent reader
- no durability boundary to reason about — the file either has a trailer or it
  does not exist

The one exception is deletion, which writes a small mask beside the container
and never touches it. See [format.md](format.md#deletion-mask).

## Where the code lives

The **file format is entirely Rust**. `src/format/` defines the framing, the
footer, and the deletion mask; nothing outside the `atlas-rust` crate can
produce or parse a byte of a container.

The Python package is a binding layer over that plus five operations — build a
collection from NetCDF files, remove datasets, list them, describe one, and
summarise the lot — available as a library and as the `atlas` command. See
[python.md](python.md).
