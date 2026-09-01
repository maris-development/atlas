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
atlas.array_stats("temperature");      // every live dataset, combined
atlas.array_stats_by_dataset("temperature");   // the same, split per dataset

let ds = atlas.dataset("jan_2024")?;   // no I/O
ds.schema();
ds.array_meta("temperature");
ds.attributes();
ds.array_attributes("temperature");
ds.array_fill_value("temperature");
ds.array_stats("temperature");         // min, max, null count, row count
```

`Atlas::array_stats` combines the statistics of one array over the whole
collection. The counts add up. The minimum is the smallest of the minimums.
The maximum is the largest of the maximums. The call skips deleted datasets. It
also skips a dataset that declares the same name with a different dtype,
because two dtypes do not compare.

`Atlas::array_stats_by_dataset` returns the same numbers, split per dataset:

```rust
for (dataset, stats) in atlas.array_stats_by_dataset("temperature") {
    println!("{dataset}: {:?}..{:?}", stats.min, stats.max);
}
```

One entry per live dataset that holds statistics for the array, in write order.
The deletion mask applies here too, so a hidden dataset never appears. A
dataset that does not declare the array does not appear either.

Nothing merges in that call, so two dtypes never have to compare. It therefore
keeps a dataset the combined call skips.

`DatasetView::array_stats` reports one dataset on its own.

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

The first data read on a dataset opens its segment. That takes two small range
reads, for the `array-format` trailer and footer. Each collection caches the
handle. The open therefore happens once per dataset, whatever the number of
arrays you then read.

The read itself fetches only the chunks the requested region overlaps. A 2×2
window out of a 32×64 array chunked 8×16 fetches one chunk, not sixteen. Pass
empty `start` and `shape` to read the whole array.

Every element nobody wrote comes from the fill value, and costs no I/O. An
array somebody declared and never wrote reads back as fill, and never touches
the container.

## SegmentStore

`array-format` opens a file through an `ObjectStore` and a path. A segment is a
byte range inside a larger file. `SegmentStore` closes that gap. It implements
`ObjectStore` over one virtual object, mapped to
`container[offset .. offset + len]`.

It translates each range request and forwards it. It buffers nothing. Three
behaviours matter as much as the translation:

- **`list` returns nothing.** Sidecar discovery therefore finds no delta layer.
  A segment is always one compacted base.
- **Any other path is `NotFound`.** The statistics probe for `seg<n>.stats`
  therefore comes back empty, and does not fail the open.
- **A range that ends past the segment clamps.** A range that starts past it is
  an error. Without the clamp, a read reaches the bytes of the next dataset.

Every write method returns `NotSupported`. A collection is immutable, so
nothing must try.

### The virtual name carries the ordinal

A segment is named `seg0.af`, `seg1.af`, and so on. That is not cosmetic. The
block cache keys on `(path, block_id)`, and every segment in a collection
shares it. One name across two segments lets block 0 of one dataset answer the
read of block 0 of another. The tests catch that.

## Caching

One `DeltaCache` serves each `Atlas` handle, and every segment shares it. It
holds 256 MiB of decompressed blocks and 64 MiB of raw I/O slabs. A second
array from a dataset with an open segment costs fewer requests than the first.
`tests/integration.rs` asserts that too.

## Type safety

`read_array::<T>` checks `T` against the dtype in the footer, and against the
dtype in the segment. A mismatch between those two raises
`CorruptCollection`. A request for the wrong type is an error, and never a
second reading of the bytes.

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
