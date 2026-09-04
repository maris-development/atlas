# Read path

## What opening costs

```rust
let atlas = Atlas::open_path(dir).await?;
```

One range read of the container's tail, plus one for the deletion mask if it
exists. That is the whole open, for a collection of ten datasets or a million.

Everything below is then answered from memory, with no further I/O:

```rust
atlas.list_datasets();
atlas.list_arrays();
atlas.dataset_count();
atlas.dataset_count();
atlas.interned_schemas();

let ds = atlas.dataset("jan_2024")?;   // no I/O, one hash lookup
ds.name();
ds.ordinal();
ds.schema();                           // array and attribute names, types
ds.array_meta("temperature");          // name and dtype
```

That is the whole of it: which datasets exist, and what each one declares. The
footer holds nothing a segment already holds, so everything else reads one:

```rust
ds.array_layout("temperature").await?;          // shape, chunks, dims, fill
ds.attributes().await?;                         // dataset-level values
ds.array_attributes("temperature").await?;      // one array's values
ds.array_stats("temperature").await?;           // min, max, nulls, rows
atlas.array_stats("temperature").await?;        // the same, over every dataset
atlas.array_stats_by_dataset("temperature").await?;
atlas.attributes_by_dataset(None, "month").await?;  // one key, every dataset
```

Each opens one segment, and one open serves every dataset in the collection.
The statistics and the layout of a variable come out of the same open, so
asking for both costs one. A dataset that declares no attribute key costs
nothing at all, because the schema settles it before anything is read.

## A schema names things

`DatasetView::schema` returns a `SchemaView`, and `array_meta` an `ArrayMeta`.
Both borrow the footer, and cost nothing:

```rust
let schema = ds.schema();
schema.len();                          // how many arrays
schema.names();                        // an iterator of &str
schema.index_of("temperature");        // position in definition order
schema.attribute_names();              // dataset-level keys
schema.attribute_dtype("month");       // Option<&DType>

let meta = schema.get("temperature").unwrap();
meta.name();
meta.dtype();                          // &DType
meta.attribute_names();                // this array's keys
```

That is the whole schema: array names with their element types, and attribute
keys with theirs. It holds no shape, no chunk shape, no dimension name, and no
fill value. The segment records those, so the footer does not repeat them.

The footer stores a schema as pairs of indices into a string pool and a dtype
pool. A view resolves an index when you ask, so no name is copied. Call
`to_owned_schema` for the owned `DatasetSchema`.

Datasets that declare the same things share one interned schema. To find two of
them equal therefore costs no compare:

```rust
assert_eq!(atlas.dataset("jan")?.schema(), atlas.dataset("feb")?.schema());
```

Shape is not in the schema, so two datasets of unequal length still share one.
The schema does name the attribute keys, so two datasets whose key sets differ
do not.

## Layout costs one open per variable

```rust
let layout = ds.array_layout("temperature").await?;
layout.shape();                        // &[usize]
layout.chunk_shape();
layout.dimension_names();              // Vec<&str>
layout.fill_value();                   // Option<&FillValue>
layout.element_count();
```

This is the one metadata call that reads. It opens the segment that holds
`temperature`, which holds that array for every dataset in the collection. The
handle then stays cached, so a sweep of every dataset's layout costs one open
per variable, not one per dataset.

## Collection-wide statistics

`Atlas::array_stats` combines the statistics of one array over the whole
collection. It opens that variable's segment and reads its statistics table in
one pass. The counts add up. The minimum is the smallest of the minimums. The
maximum is the largest of the maximums. The call skips deleted datasets.

`Atlas::array_stats_by_dataset` returns the same numbers, split per dataset:

```rust
for stats in atlas.array_stats_by_dataset("temperature").await? {
    println!("{}: {:?}..{:?}", stats.name, stats.min, stats.max);
}
```

One entry per live dataset that holds statistics for the array, in write order.
Every entry names its own dataset in `ArrayStats::name`, so a row identifies
itself and needs no second list to read. The deletion mask applies here too, so
a hidden dataset never appears. A dataset that does not declare the array does
not appear either. Both calls share the one open, so asking for the combined
and the split view costs the same as either.

`DatasetView::array_stats` reports one dataset on its own. It names the array
instead, because there the dataset is already known.

## A table of what the collection holds

`Atlas::attributes_by_dataset` reads one attribute over the live datasets. The
first argument names the array the attribute annotates, and `None` reads a
dataset-level attribute out of the reserved `_datasets` segment.

It returns `IndexMap<String, Attr>`, keyed by dataset name and held in write
order. A dataset that does not carry the key has no entry, and the map holds
no placeholder for it. A column is therefore shorter than `list_datasets`
whenever some dataset lacks the key, and the two do not line up position for
position. `list_datasets` gives the rows. Every column is a lookup on the
name.

Call it once per key and join the columns by name:

```rust
let months = atlas.attributes_by_dataset(None, "month").await?;
let sources = atlas.attributes_by_dataset(None, "source").await?;
let temperature: HashMap<String, ArrayStats> = atlas
    .array_stats_by_dataset("temperature")
    .await?
    .into_iter()
    .map(|stats| (stats.name.clone(), stats))
    .collect();

for name in atlas.list_datasets() {
    println!(
        "{name} {:?} {:?} {:?}",
        months.get(&name),
        sources.get(&name),
        temperature.get(&name).and_then(|s| s.max.clone()),
    );
}
```

```text
dataset       month  source   t.min   t.max   rows
f0000.nc          1    test     1.0    24.0     24
f0001.nc          2    test     2.0    25.0     24
```

Every column keys on the dataset name, so the join is a lookup and never an
alignment by position. A dataset missing from a column simply has no entry, and
`get` answers `None` for it. Columns differ in length, because the keys a
dataset carries differ from dataset to dataset.

The first call opens the `_datasets` segment, and every later one reuses the
handle. A table over ten thousand datasets therefore costs one open, whatever
the number of keys. Each call walks the collection once.

`tests/integration.rs` asserts this with a request-counting `ObjectStore`. An
open of a collection above the 64 KiB tail probe issues at most two reads, and
moves at most 64 KiB. The metadata calls above issue none.

The whole format serves this one property. It is also why the Python bindings
give metadata and not data. To serve a catalogue of what a collection holds
needs no array byte.

## What reading data costs

```rust
let temp = ds.read_array::<f32>("temperature", vec![10, 20], vec![2, 2]).await?;
```

The first read of a variable opens its segment. That takes two small range
reads, for the `array-format` trailer and footer. Each collection caches the
handle. The open therefore happens once per **variable**, whatever the number
of datasets you then read it for.

The read itself fetches only the blocks the requested region overlaps. A 2×2
window out of a 32×64 array chunked 8×16 fetches one chunk, not sixteen. Pass
empty `start` and `shape` to read the whole array.

A block holds a run of neighbouring datasets, all of one variable. So a sweep
of `temperature` across the collection reads that segment once and walks it,
and every block it fetches is `temperature`.

Every element nobody wrote comes from the fill value, and costs no I/O. An
array somebody declared and never wrote reads back as fill, and never touches
the container.

## SegmentStore

`array-format` opens a file through an `ObjectStore` and a path. A segment is a
byte range inside a larger file. `SegmentStore` closes that gap. It implements
`ObjectStore` over one virtual object, mapped to
`container[offset .. offset + len]`.

It translates each range request and forwards it. It buffers nothing.

A segment is **two** objects. `array-format` keeps an array's statistics in a
sidecar beside its file, so the store serves `seg<n>.af` and `seg<n>.stats`,
each mapped to its own range of the container. Those are the names that crate
derives.

Three behaviours matter as much as the translation:

- **`list` returns nothing.** Sidecar discovery therefore finds no delta layer.
  A segment is always one compacted base.
- **Any other path is `NotFound`.** So is `seg<n>.stats` for a segment that
  carries no sidecar, which `array-format` tolerates.
- **A range that ends past the object clamps.** A range that starts past it is
  an error. Without the clamp, a read reaches the bytes of the next object.

Every write method returns `NotSupported`. A collection is immutable, so
nothing must try.

### The virtual name carries the variable index

A segment is named `seg0.af`, `seg1.af`, and so on. That is not cosmetic. The
block cache keys on `(path, block_id)`, and every segment in a collection
shares it. One name across two segments lets block 0 of one variable answer the
read of block 0 of another. The tests catch that.

## Caching

One `DeltaCache` serves each `Atlas` handle, and every segment shares it. It
holds 256 MiB of decompressed blocks and 64 MiB of raw I/O slabs. The second
dataset to read a variable with an open segment costs fewer requests than the
first. `tests/integration.rs` asserts that too.

## Type safety

`read_array::<T>` checks `T` against the dtype the footer records for that
dataset's array. A request for the wrong type is an error, and never a second
reading of the bytes.

## Deletion

`delete_dataset` and `delete_datasets` are the only operations on an open
collection that write anything. They write the mask, and no more:

1. Resolve each name to an ordinal, or return `DatasetNotFound`.
2. Read the mask from the store again. That keeps a delete another handle made
   since this one opened.
3. Insert the ordinals, and write the whole mask back.
4. Update the in-memory set, so this handle sees the change at once.

`delete_dataset` calls `delete_datasets` with one name, so the two share a
path. The cost is two requests, whatever the number of names:

```rust
atlas.delete_datasets(&names).await?;   // 10_000 names, 1 GET + 1 PUT
```

Step 1 runs one pass over the footer for the whole batch. A name at a time
would be one pass each, which matters on a collection of a million datasets.
The batch stands or falls together. One absent name returns `DatasetNotFound`
and writes nothing.

Step 2 lets two handles each delete a different dataset, and keeps both
deletes. It does not make concurrent deletes safe. Two deletes that interleave
between the read and the write still lose one. See
[format.md](format.md#deletion-mask).

## Reading from Python

You cannot. `Atlas` in Python lists datasets, reports schemas, and reads
attributes. There is no `read_array`. The Rust API reads array data.

The split is deliberate. Python builds collections from xarray, and serves
their metadata. Array bytes through the GIL were never the fast path.
`tests/cross_fixture.rs` keeps the two in agreement. Python writes a
collection, and Rust reads it back and checks every value.
