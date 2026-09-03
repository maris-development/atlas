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
offset 4     format_version u32 LE = 6   4 B
offset 8     segment[0]                        one variable, array-format
             segment[0].stats                  its statistics sidecar
             segment[1]                        back to back, no padding
             …
             footer_bytes                      zstd(msgpack(CollectionFooter))
end - 16     footer_size u64 LE          8 B  ┐
end - 8      format_version u32 LE = 6   4 B  ├ trailer
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

One per **variable**, in the order the writer first saw each array name, packed
with no alignment or padding. Byte ranges come from the footer, so nothing has
to be scanned.

Each segment is a complete `array-format` file. That is a data region of
compressed blocks, then its own rkyv footer and `ARRF` trailer.

Inside a segment, each array keys on the **dataset** name. So `temperature`
holds one array called `jan_2024`, one called `feb_2024`, and so on:

```text
segment "temperature"
├── array "jan_2024"    f32, [4, 8], chunked [2, 4]
├── array "feb_2024"    f32, [4, 8], chunked [2, 4]
└── array "mar_2024"    f32, [4, 8], chunked [2, 4]
```

That layout is the point. `array-format` fills a block up to 8 MiB from
neighbouring chunks, and walks its arrays in order. A block therefore holds one
dtype for a run of datasets. One fetch serves them all, and one dtype
compresses far better than a mix of `f32`, `u8`, and text.

Each block records its own codec. Nothing therefore needs to tell a reader how
a collection compresses.

The segment also records the shape, the chunk shape, the dimension names, and
the fill value of every array. The footer repeats none of them.

## Footer

MessagePack in compact (positional) form, then zstd. Compact form omits field
names, so the layout is pinned by `format_version`: changing any field below is
a format change.

```rust
struct CollectionFooter {
    version: u32,                    // = 6, re-checked after decode
    segment_format: u32,             // = 5, the array-format footer version
    codec: Codec,                    // for information. Each block names its own
    created_unix_ms: i64,
    string_pool: Vec<String>,        // array names and attribute keys
    dtype_pool: Vec<DType>,          // element and attribute types
    schema_pool: Vec<InternedSchema>,
    variables: Vec<VariableEntry>,   // one segment per array name
    datasets: IndexMap<SmolStr, DatasetEntry>,   // position == ordinal
}

struct InternedSchema {
    arrays: Vec<(u32, u32)>,         // (array name, element dtype)
    attrs:  Vec<(u32, u32)>,         // (attribute key, value dtype)
    array_attrs: Vec<(u32, Vec<(u32, u32)>)>,  // (array position, keys)
}

struct VariableEntry {
    name: u32,                       // index into string_pool
    seg_offset: u64,                 // the array-format file
    seg_len: u64,
    stats_offset: u64,               // its statistics sidecar
    stats_len: u64,                  // 0 when the segment carries none
}
```

A dataset is one `u32`. Its name is the key that finds it, its position is its
ordinal, and everything else about it lives in the segments.

### The dataset map carries the ordinal

`datasets` is an `IndexMap`, so a name and its position live in one structure.
The position is the ordinal, which the deletion mask names, and a lookup by
name is one hash instead of a scan over every dataset.

On the wire it is a **sequence** of `(name, entry)` pairs, not a map. As a map,
a repeated name would collapse two datasets into one on decode and shift every
ordinal after it, while the mask still named the old ones. The decode rejects a
repeat instead. `the_wire_form_of_the_dataset_map_is_a_sequence` pins the
encoding.

A reader hands out `SchemaView` and `ArrayMeta` over the schema. Both borrow
the footer and resolve an index on demand, so no name is copied.
`DatasetSchema` is the owned form, built by `to_owned_schema`. `ArrayLayout`
carries what the segment holds, and `DatasetView::array_layout` reads it.

### A segment is two objects

`array-format` keeps an array's statistics in a sidecar beside its file, not
inside it, and reads that file once at open. So a variable contributes two byte
ranges to the container: the `.af` file and its `.stats`.

`SegmentStore` serves both. It presents `seg<n>.af` and `seg<n>.stats`, which
are the names `array-format` derives, each mapped to its own range. Without the
sidecar the crate finds no statistics and reports none, which is why the footer
records where it is.

### A schema names things and nothing more

An `InternedSchema` holds array names with their element types, and attribute
keys with their value types. It holds no shape, no chunk shape, no dimension
name, and no fill value. Those describe the data, and the segment that holds
the data records them already.

That is what keeps the pool small. Datasets whose arrays differ only in length
share one schema, because length is not in it. A directory of ten thousand
files of one convention interns to one entry.

The schema names the attribute *keys*, and the dataset carries the *values*
positionally. A key set repeated across a collection therefore costs nothing
per dataset. The trade is that two datasets whose key sets differ get two
schemas, even when they declare the same arrays.

### Three pools

**Strings** intern once each. Array names and attribute keys share one pool. A
dataset name does not enter it, because each occurs once, and because it is
already the array name inside every segment.

**Dtypes** intern once each. A collection holds a handful, so a scan finds one.

**Schemas** intern by content. Every field is a pool index, so the whole struct
is `Hash` and `Eq`, and one map lookup settles a schema. There is no content
hash and no collision fallback.

### The decoded form is not the wire form

`SmolStr` and `SmallVec` serialize as a string and as a sequence, so the bytes
above do not change with them. They only change what a decode allocates.

**A name is a `SmolStr`.** One occurs per dataset, which makes it the footer's
most repeated allocation. A name of 22 bytes or fewer therefore sits inline,
and `SmolStr` is the same 24 bytes as a `String`. Measured over 100 000
datasets, a decode went from 97.2 ms to 90.3 ms on identical bytes. A name
longer than that allocates as before.

**A schema holds its index lists inline.** The pool holds a handful of
schemas, so the bytes cost nothing.

**A dataset entry does not.** `global_attrs` and `array_attrs` stay `Vec`.
`AttrS` is 32 bytes, so four inline would add 128 to every dataset entry. That
was measured too: it grew `DatasetEntry` from 104 bytes to 336, and the memory
traffic of the larger entry cost more than the allocation it saved. A decode
went to 105.1 ms, slower than the `Vec` it replaced.

The rule that follows: inline storage pays where a structure is rare and its
contents are small. It loses where the structure exists once per dataset.

### Statistics live in the segments too

`array-format` computes a minimum, a maximum, and a null count for every array
while the dataset stages. It walks the data anyway, to write it. It then stores
them in the sidecar beside that segment, which is where a reader takes them
from.

`null_count` counts the elements equal to the fill value. That is how the
format stores a cell nobody wrote. An array somebody declared and never wrote
reports `row_count == null_count`. Every element is a hole. `min` and `max` are
`None` for a dtype with no order. For a string they are raw bytes, in
lexicographic order.

`Atlas::array_stats` folds one array over the collection. One open of that
variable's segment covers every dataset, and one pass over its table indexes
them. `array_stats` on the segment scans that table per call, so a lookup per
dataset would be quadratic. `array_stats_by_dataset` hands the entries back one
by one, and `DatasetView::array_stats` reports one dataset.

The deletion mask applies to all three, as it does to everything else. A hidden
dataset counts toward no statistic. So does a dataset that declares the name
with another dtype, because two dtypes do not compare.

### Attributes live in the segments, not here

The footer holds no attribute **value**. `array-format` attaches an attribute
to an array, and each of a dataset's arrays already sits in a segment, so that
is where the value belongs. The segment interns each key and each value once
per file, so `units = "celsius"` on ten thousand datasets is stored once.

The footer still names every key and its type, in the schema. That is what
lets a reader ask for one key without opening anything to find out whether it
exists.

**A per-array value** goes on the array's own entry: dataset `jan`'s
`temperature` attributes sit on the array `jan` inside `temperature`.

**A dataset-level value** has no array of its own, so it gets one. The
reserved `_datasets` segment holds a rank-0 array per dataset, named after the
dataset, carrying that dataset's global attributes and no data. It appears only
when some dataset has one. `validate_name` refuses a leading underscore, so no
user array can take that name.

**A timestamp needs the schema.** `AttributeValue` has no timestamp variant, so
a timestamp stores as its `i64`. The schema records the key as `TimestampNs`,
and the reader reads the tag back from there. That is the one case where the
schema is load-bearing for decoding, not just describing. Without it a
timestamp would come back as an integer.

What this costs: `DatasetView::attributes`, `get_attribute`,
`array_attributes`, `get_array_attribute` and `Atlas::attribute_by_dataset` are
async, and open one segment. A dataset or array that declares no key still
costs nothing, because the schema settles it. One open then covers every
dataset, because a segment spans the collection.

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
| Container version | 6 |
| Header size | 8 bytes |
| Trailer size | 16 bytes |
| Tail probe | 64 KiB |
| Embedded segment format | `array-format` footer v5 |
| Mask magic | `ATLM` |
| Mask version | 1 |

`array-format` is pinned to exactly `0.12.0`. The container embeds its files
verbatim, so a change to its bytes would be a change to the atlas format.
