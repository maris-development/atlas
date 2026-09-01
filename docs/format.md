# The on-disk format

A collection is a store prefix holding one required object and one optional
one:

```text
my_collection/
├── data.atlas      the container: write-once, never modified
└── deleted.mask    optional: which datasets are hidden
```

Everything below is defined in `src/format/`. Nothing outside the `atlas-rust`
crate produces or parses these bytes.

## Container

```text
offset 0     b"ATLS"                     4 B   leading magic
offset 4     format_version u32 LE = 1   4 B
offset 8     segment[0]                        a complete array-format file
             segment[1]                        back to back, no padding
             …
             footer_bytes                      zstd(msgpack(CollectionFooter))
end - 16     footer_size u64 LE          8 B  ┐
end - 8      format_version u32 LE = 1   4 B  ├ trailer
end - 4      b"ATLS"                     4 B  ┘
```

The magic appears twice on purpose. A reader checks the trailing copy, which
arrives with the footer in one range read. The leading copy serves `file`,
`libmagic`, and a person with `xxd`. They all name the file from its first
bytes.

### Opening

1. Read the last `min(file_size, 64 KiB)`.
2. Validate the trailer: magic, then version. A mismatch is
   `NotAnAtlasCollection` or `UnsupportedVersion`.
3. The footer is already in hand when `footer_size + 16` fits in that read.
   One request then covers it. Otherwise issue a second range read.
4. Read `deleted.mask`, if present.

An open of any collection therefore costs one or two requests, whatever its
size. The check on the leading magic runs only when the probe already covered
it, which happens for a small container. A round trip to prove again what the
trailer proved is waste.

### Segments

One per dataset, in write order, packed with no alignment or padding. Byte
ranges come from the footer, so nothing has to be scanned.

Each segment is a complete `array-format` file. That is a data region of
compressed blocks, then its own rkyv footer and `ARRF` trailer. It stands
alone:

```bash
# byte offsets from DatasetView::segment_range()
dd if=data.atlas of=one_dataset.af bs=1 skip=8 count=1438
```

`array-format` then opens the result directly. Inside a segment, each array
keys on its real name, such as `temperature` or `lat`. Each block records its
own codec. Nothing therefore needs to tell a reader how a collection
compresses.

## Footer

MessagePack in compact (positional) form, then zstd. Compact form omits field
names, so the layout is pinned by `format_version`: changing any field below is
a format change.

```rust
struct CollectionFooter {
    version: u32,                    // = 1, re-checked after decode
    segment_format: u32,             // = 5, the array-format footer version
    codec: Codec,                    // for information. Each block names its own
    created_unix_ms: i64,
    schema_pool: Vec<DatasetSchema>, // interned by content hash
    attr_key_pool: Vec<String>,      // interned attribute keys
    datasets: Vec<DatasetEntry>,     // position == ordinal
}

struct DatasetEntry {
    name: String,
    schema: u32,                          // index into schema_pool
    seg_offset: u64,
    seg_len: u64,
    global_attrs: Vec<(u32, AttrS)>,      // (key_pool index, value)
    array_attrs: Vec<(u32, Vec<(u32, AttrS)>)>,   // (array position, attributes)
    array_stats: Vec<(u32, ArrayStatsS)>,         // (array position, statistics)
}

struct ArrayStatsS {
    min: Option<StatValueS>,   // None for a dtype with no ordering
    max: Option<StatValueS>,
    null_count: u64,           // elements equal to the fill value
    row_count: u64,            // total elements across every chunk
}
```

`DatasetSchema` is just `arrays: IndexMap<String, ArraySchema>`, and
`ArraySchema` carries dtype, shape, chunk shape, dimension names, and fill
value.

### Interning

Two pools keep the footer small when a collection holds many similar datasets.

**Schemas** intern by content hash. A fleet of a thousand sensors that declares
the same two arrays stores that schema once. Each entry is a `u32` that points
at it. The schema holds no attribute *value*, and that is deliberate. Two
datasets that differ only in their attributes therefore still share one schema.

**Attribute keys** are interned as strings, so a key repeated across every
dataset costs four bytes per use rather than its length.

### Statistics live here too

`array-format` computes a minimum, a maximum, and a null count for every array
while the dataset stages. It walks the data anyway, to write it. To record the
result in the footer therefore costs nothing at write time, and makes it free
at read time.

`null_count` counts the elements equal to the fill value. That is how the
format stores a cell nobody wrote. An array somebody declared and never wrote
reports `row_count == null_count`. Every element is a hole. `min` and `max` are
`None` for a dtype with no order. For a string they are raw bytes, in
lexicographic order.

`atlas show` prints this under each variable. To print it needs no more I/O
than the dataset list needed.

The footer stores one entry per dataset. It stores nothing for the collection
as a whole. `Atlas::array_stats` and `atlas info` combine the entries as they
read them. The format needs no second copy.

The deletion mask applies to those entries, as it does to everything else the
footer holds. A hidden dataset counts toward no statistic.
`Atlas::array_stats_by_dataset` hands back the entries one by one, with the
same mask applied.

### Attributes live here, not in the segments

The footer holds every attribute value, both dataset-level and per-array. A
segment carries none. A metadata-only open therefore answers every attribute
question with no further I/O. The Python reader does exactly that.

A timestamp also keeps its own tag. `AttrS` has a `TimestampNanoseconds`
variant, so nothing guesses at a date-shaped string. An RFC 3339 string is a
string, and a timestamp is a timestamp.

### Schema is recorded twice

The footer describes an array. The segment addresses its chunks. Both need the
dtype and the shape, so both store them. The reader compares the two on the
first data read. A mismatch raises `CorruptCollection`.

### Validation on decode

A decoded footer passes a check before use. Every dataset schema index must
resolve. Every attribute key index must sit in the pool. Every annotated array
position must exist. No segment may be empty. No dataset name may repeat.

One check here costs less than a dangling index at every use site. It also
turns a corrupt file into one clear error, and not a panic deep inside a read.

The name check matters because a lookup by name resolves to the first dataset
that carries it. A repeat would hide the second one, while `dataset_count`
still counted it. The writer keeps a hash set of the names it handed out, and
refuses a repeat. A duplicate on disk therefore means a damaged or foreign
footer.

## Deletion mask

The container is write-once, so a delete cannot touch it. A sidecar lists the
ordinals of the deleted datasets instead, and the reader hides them.

```text
b"ATLM"          4 B   magic
version u32 LE   4 B   = 1
count   u32 LE   4 B
count × u32 LE         ordinals, strictly increasing
```

- **Absent means nothing is deleted.** The writer never creates one.
- **A delete** reads the mask, adds the ordinals, and writes the whole file
  back in one atomic PUT. An object store has no partial write, so the whole
  file is the one option. One PUT covers a batch of any size.
- **No ordinal ever moves.** A dataset's position in `footer.datasets` holds
  for the life of the container. A stored ordinal therefore stays valid, and no
  reader sees a renumbering.
- **This reclaims no space.** The bytes of the deleted dataset stay where they
  are. Rewrite the collection to get them back.

### Tolerance

An ordinal past the end of the footer draws a warning, and the reader drops it.
A mask from a different container therefore cannot block an open. A truncated
body keeps every entry that survives. Only a wrong magic is an error. A foreign
file at that path is a mistake, and deserves a report instead of a silent
"nothing deleted".

### Concurrency

Two deletes that race on one collection are last-writer-wins. One can be lost.
Serialize deletes if that matters. A compare-and-swap on the backend etag fixes
this where a backend offers one. Version 1 does not do that.

## Constants

| Name | Value |
|---|---|
| Container magic | `ATLS`, at both ends |
| Container version | 1 |
| Header size | 8 bytes |
| Trailer size | 16 bytes |
| Tail probe | 64 KiB |
| Embedded segment format | `array-format` footer v5 |
| Mask magic | `ATLM` |
| Mask version | 1 |

`array-format` is pinned to exactly `0.12.0`. The container embeds its files
verbatim, so a change to its bytes would be a change to the atlas format.
