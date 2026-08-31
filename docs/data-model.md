# Data model

## Collection, dataset, array

```text
collection                      one data.atlas file
├── dataset "jan_2024"          one segment inside it
│   ├── array "temperature"     f32, [4, 8], chunked [2, 4]
│   ├── array "lat"             f64, [4]
│   └── attributes              month=1, source="buoy"
└── dataset "feb_2024"
    ├── array "temperature"
    └── …
```

A **dataset** is what a NetCDF file or an `xarray.Dataset` holds: a set of named
N-dimensional arrays that share dimensions, plus attributes. A **collection**
holds many datasets, and is the unit of a file.

An **array** is typed, shaped, chunked, and belongs to exactly one dataset. Two
datasets may both declare `temperature`; those are separate arrays that happen
to share a name. Nothing is shared between them — not bytes, not schema
identity beyond interning, not dimensions.

## Ordinals

A dataset's position in the footer is its **ordinal**, assigned in write order
and fixed for the life of the container. It is the identity the deletion mask
refers to.

Ordinals never shift. Deleting a dataset does not renumber the others, because
there is no operation that could — the footer is immutable. An ordinal you
recorded a year ago still names the same dataset.

## Arrays

| Property | Meaning |
|---|---|
| `dtype` | Element type |
| `shape` | Logical shape, one entry per axis |
| `chunk_shape` | Storage granularity. Equal to `shape` means one chunk |
| `dimension_names` | One per axis, in `shape` order |
| `fill_value` | What a read returns for elements never written |

### Chunking

Chunk shape is what makes a partial read cheap: reading a region fetches only
the chunks it overlaps. An array stored as one chunk is read whole or not at
all.

Declaring an array does not allocate anything. Write into whatever part of it
you like, in any order and any number of slabs, aligned or not — partially
covered chunks are handled for you. Regions never written cost no bytes and read
back as the fill value.

### Types

Scalars: `Bool`, `Int8`…`Int64`, `UInt8`…`UInt64`, `Float32`, `Float64`,
`String`, `Binary`, `TimestampNs`.
Nested: `List<T>`, `FixedSizeList<T, n>`.

Strings and lists are variable length: an offsets buffer plus concatenated
values. `TimestampNs` is nanoseconds since the Unix epoch, stored as `i64`.

Not every type is reachable from Python. `Bool` arrays are not supported by
`array-format`; `Binary` and the nested types are not yet exposed in the
bindings. All of them round-trip in the footer as *attribute* values.

### No type reconciliation

Two datasets may declare the same array name with unrelated types — `int32` in
one, `string` in another — and both are stored as declared. There is no merged
schema, no widening, and no mismatch policy: with one segment per dataset there
is nothing to reconcile. A reader that wants a collection-wide view builds it
from the per-dataset schemas, which are all in the footer.

## Attributes

Two scopes:

- **Dataset-level** — annotates the whole dataset. `title`, `month`, `source`.
- **Per-array** — annotates one array. `units`, `long_name`.

Both hold a typed value: any scalar type above, or a homogeneous list of one.
Both preserve the order they were set in. Setting a key twice replaces the
value.

Values live in the footer, not in the segments, so reading them costs nothing
beyond the open. See [format.md](format.md#attributes-live-here-not-in-the-segments).

### Timestamps are a real type

`Attr::TimestampNanoseconds` has its own wire tag, so nothing has to guess at a
date-shaped string. An RFC 3339 string round-trips as a string; a timestamp
round-trips as a timestamp.

## What a collection cannot do

Worth stating plainly:

| Not available | Instead |
|---|---|
| Append a dataset to a finished collection | Rewrite it |
| Modify an array | Rewrite the collection |
| Add an array to an existing dataset | Rewrite the collection |
| Reclaim space from a deleted dataset | Rewrite the collection |
| Compact | Nothing to compact — there are no layers |
| Flush | No durability boundary; the file exists or it does not |

Deleting a dataset is the one post-write operation, and it only writes the mask.
