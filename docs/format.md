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

The magic appears twice on purpose. The trailing copy is what a reader checks —
it arrives with the footer in the same range read. The leading copy is there so
`file`, `libmagic`, and a human with `xxd` can identify the file from its first
bytes.

### Opening

1. Read the last `min(file_size, 64 KiB)`.
2. Validate the trailer: magic, then version. A mismatch is
   `NotAnAtlasCollection` or `UnsupportedVersion`.
3. If `footer_size + 16` fits in what was read, the footer is already in hand —
   one request total. Otherwise issue a second range read for it.
4. Read `deleted.mask`, if present.

So opening any collection costs one or two requests, whatever its size. The
leading magic is checked only when the probe happened to cover it (a small
container); spending a round trip to re-verify what the trailer already proved
would be waste.

### Segments

One per dataset, in write order, packed with no alignment or padding. Byte
ranges come from the footer, so nothing has to be scanned.

Each segment is a complete `array-format` file — a data region of compressed
blocks followed by its own rkyv footer and `ARRF` trailer. It is standalone:

```bash
# byte offsets from DatasetView::segment_range()
dd if=data.atlas of=one_dataset.af bs=1 skip=8 count=1438
```

and `array-format` opens the result directly. Inside a segment, each array is
keyed by its real name (`temperature`, `lat`), and blocks record their own
codec — which is why a reader never needs to be told how a collection was
compressed.

## Footer

MessagePack in compact (positional) form, then zstd. Compact form omits field
names, so the layout is pinned by `format_version`: changing any field below is
a format change.

```rust
struct CollectionFooter {
    version: u32,                    // = 1, re-checked after decode
    segment_format: u32,             // = 5, the array-format footer version
    codec: Codec,                    // informational; blocks are self-describing
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
    array_attrs: Vec<(u32, Vec<(u32, AttrS)>)>,  // (array position, attributes)
}
```

`DatasetSchema` is just `arrays: IndexMap<String, ArraySchema>`, and
`ArraySchema` carries dtype, shape, chunk shape, dimension names, and fill
value.

### Interning

Two pools keep the footer small when a collection holds many similar datasets.

**Schemas** are interned by content hash. A fleet of a thousand sensors that all
declare the same two arrays stores that schema once; each entry is a `u32`
pointing at it. Attribute *values* are deliberately not part of the schema, so
two datasets that differ only in their annotations still share one.

**Attribute keys** are interned as strings, so a key repeated across every
dataset costs four bytes per use rather than its length.

### Attributes live here, not in the segments

Every attribute value — dataset-level and per-array — is in the footer.
Segments carry none. That means a metadata-only open answers every attribute
question with zero further I/O, which is exactly what the Python reader does.

It also means timestamps keep their own tag. The 0.14 store had no timestamp
attribute type and encoded them as RFC 3339 strings, then guessed on read;
a string that happened to look like a date came back as a timestamp. `AttrS`
has a `TimestampNanoseconds` variant, so that guess is gone.

### Schema is recorded twice

The footer describes an array; the segment addresses its chunks. Both need the
dtype and shape, so both store them. The reader cross-checks on the first data
read and raises `CorruptCollection` if they disagree.

### Validation on decode

A decoded footer is checked before use: every dataset's schema index must
resolve, every attribute key index must be in the pool, every annotated array
position must exist, and no segment may be empty. Checking once here is cheaper
than handling a dangling index at each use site, and it turns a corrupt file
into one clear error instead of a panic deep in a read.

## Deletion mask

The container is write-once, so deleting a dataset cannot touch it. Instead a
sidecar lists the ordinals of deleted datasets, and the reader hides them.

```text
b"ATLM"          4 B   magic
version u32 LE   4 B   = 1
count   u32 LE   4 B
count × u32 LE         ordinals, strictly increasing
```

- **Absent means nothing is deleted.** The writer never creates one.
- **Deleting** reads the mask, adds an ordinal, and writes the whole file back
  with a single atomic PUT. Object stores have no partial write, so whole-file
  is the only option.
- **Ordinals never move.** A dataset's position in `footer.datasets` is fixed
  for the life of the container, so a stored ordinal stays valid and no
  concurrent reader sees a renumbering.
- **Space is not reclaimed.** The deleted dataset's bytes stay exactly where
  they are. Rewrite the collection to get them back.

### Tolerance

An ordinal past the end of the footer is ignored with a warning, so a mask left
over from a different container cannot stop a collection from opening. A
truncated body keeps whatever entries are intact. Only a wrong magic is an
error — a foreign file at that path is a mistake worth reporting rather than
silently treating as "nothing deleted".

### Concurrency

Two deletions racing on the same collection are last-writer-wins: one can be
lost. Serialize deletes if that matters. A compare-and-swap using the backend's
etag would fix it where the backend supports one; that is not in v1.

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
