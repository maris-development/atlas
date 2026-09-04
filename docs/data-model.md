# Data model

## Collection, dataset, array

```text
collection                      one data.atlas file
├── dataset "jan_2024"          a footer entry, no bytes of its own
│   ├── array "temperature"     f32, [4, 8], chunked [2, 4]
│   ├── array "lat"             f64, [4]
│   └── attributes              month=1, source="buoy"
└── dataset "feb_2024"
    ├── array "temperature"
    └── …
```

The file is laid out the other way round. One segment per **variable**, and
every dataset's copy of that array sits inside it under the dataset's name:

```text
data.atlas
├── segment "temperature"       jan_2024, feb_2024, …
├── segment "lat"               jan_2024, feb_2024, …
└── footer                      which datasets exist, what each declares
```

A **dataset** holds what a NetCDF file or an `xarray.Dataset` holds. That is a
set of named N-dimensional arrays that share dimensions, and attributes. A
**collection** holds many datasets. One collection is one file.

An **array** has a type, a shape, and a chunking. It belongs to one dataset.
Two datasets can both declare `temperature`. Those are two arrays with one
name. They share a segment and a pooled name, and nothing else. No bytes, no
dimensions, and no schema identity beyond the interning.

The type sits in the footer. The shape and the chunking do not. The segment
records those, and `DatasetView::array_layout` reads them from there. That is
what lets datasets of unequal length share one interned schema.

## Ordinals

A dataset's position in the footer is its **ordinal**. The write order assigns
it, and it holds for the life of the container. The deletion mask names it.

The footer keys its datasets by name in an `IndexMap`, so the name and the
ordinal are one structure. Nothing stores the ordinal separately, and a lookup
by name costs no scan.

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
in one and `string` in another. Atlas stores each as declared, as its own array
inside that variable's segment. There is no merged schema, no widening, and no
mismatch policy. A reader that wants a collection-wide view builds it from the
per-dataset schemas, which all sit in the footer.

## Attributes

Two scopes:

- **Dataset-level.** It annotates the whole dataset. `title`, `month`,
  `source`.
- **Per-array.** It annotates one array. `units`, `long_name`.

Both hold a typed value. That is any scalar type above, or a list of one type.
Both keep the order somebody set them in. A second write to one key replaces
the value.

The values live in the segments, on the array they annotate. To read one
therefore opens a segment, and one open then serves every dataset. The keys and
their types stay in the footer, so a key nobody declared costs nothing to ask
for. See [format.md](format.md#attributes-live-in-the-segments-not-here).

A dataset-level value has no array of its own, so the reserved `_datasets`
segment gives it one: a rank-0 array per dataset, carrying that dataset's
global attributes and no data.

### An attribute carries its own type

Every `Attr` variant writes its tag and reads it back, so a value round-trips
to the same variant and a read consults no schema. A `u16` stays a `u16`. An
RFC 3339 string stays a string, and nothing guesses at a date-shaped one.

There is no timestamp attribute. `array-format` stores none, so one would have
to go in as an `i64` and could not come back. Store the nanoseconds as
`Attr::Int64` and name the unit in a second attribute. This is about attributes
alone: an *array* element type still has `DType::TimestampNs`, and a time axis
keeps it.

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
