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

A **dataset** holds what a NetCDF file or an `xarray.Dataset` holds. That is a
set of named N-dimensional arrays that share dimensions, and attributes. A
**collection** holds many datasets. One collection is one file.

An **array** has a type, a shape, and a chunking. It belongs to one dataset.
Two datasets can both declare `temperature`. Those are two arrays with one
name. They share no bytes, no dimensions, and no schema identity beyond the
interning.

## Ordinals

A dataset's position in the footer is its **ordinal**. The write order assigns
it, and it holds for the life of the container. The deletion mask names it.

No ordinal ever moves. A delete does not renumber the others, because no
operation can. The footer is immutable. An ordinal from a year ago still names
the same dataset.

## Arrays

| Property | Meaning |
|---|---|
| `dtype` | Element type |
| `shape` | Logical shape, one entry per axis |
| `chunk_shape` | Storage granularity. Equal to `shape` means one chunk |
| `dimension_names` | One per axis, in `shape` order |
| `fill_value` | What a read returns for elements never written |

### Chunking

The chunk shape makes a partial read cheap. A read of a region fetches only
the chunks it overlaps. An array in one chunk reads whole, or not at all.

To declare an array allocates nothing. Write into any part of it, in any order,
and in any number of slabs. Alignment does not matter, because atlas handles a
part-covered chunk. A region nobody writes costs no bytes, and reads back as
the fill value.

### Types

Scalars: `Bool`, `Int8`…`Int64`, `UInt8`…`UInt64`, `Float32`, `Float64`,
`String`, `Binary`, `TimestampNs`.
Nested: `List<T>`, `FixedSizeList<T, n>`.

A string and a list have a variable length. Each holds an offsets buffer and
the values behind it. `TimestampNs` is nanoseconds from the Unix epoch, in an
`i64`.

Python does not reach every type. `array-format` supports no `Bool` array. The
bindings expose neither `Binary` nor the nested types. Every one of them still
round-trips in the footer as an *attribute* value.

### No type reconciliation

Two datasets can declare one array name with unrelated types, such as `int32`
in one and `string` in another. Atlas stores each as declared. There is no
merged schema, no widening, and no mismatch policy. One segment per dataset
leaves nothing to reconcile. A reader that wants a collection-wide view builds
it from the per-dataset schemas, which all sit in the footer.

## Attributes

Two scopes:

- **Dataset-level.** It annotates the whole dataset. `title`, `month`,
  `source`.
- **Per-array.** It annotates one array. `units`, `long_name`.

Both hold a typed value. That is any scalar type above, or a list of one type.
Both keep the order somebody set them in. A second write to one key replaces
the value.

The values live in the footer, and not in the segments. To read them therefore
costs nothing beyond the open. See
[format.md](format.md#attributes-live-here-not-in-the-segments).

### Timestamps are a real type

`Attr::TimestampNanoseconds` has its own wire tag, so nothing guesses at a
date-shaped string. An RFC 3339 string round-trips as a string. A timestamp
round-trips as a timestamp.

## What a collection cannot do

State it plainly:

| Not available | Instead |
|---|---|
| Append a dataset to a finished collection | Rewrite it |
| Change an array | Rewrite the collection |
| Add an array to an existing dataset | Rewrite the collection |
| Reclaim the space of a deleted dataset | Rewrite the collection |
| Compact | Nothing to compact. There are no layers |
| Flush | No durability boundary. The file exists, or it does not |

A delete is the one operation after a write. It writes the mask, and no more.
One write covers any number of datasets.
