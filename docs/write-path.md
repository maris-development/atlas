# Write path

## The shape of a write

```rust
let w = AtlasWriter::create_path(dir, WriterConfig::default()).await?;

let mut ds = w.add_dataset("jan_2024").await?;
ds.define_array::<f32>("temperature", dims, shape, chunk_shape, fill).await?;
ds.write_array("temperature", vec![0, 0], data.view()).await?;
ds.set_attribute("month", Attr::Int64(1));
ds.finish().await?;          // the dataset enters the container here

w.finish().await?;           // the collection becomes readable here
```

There are two commit points, and nothing is visible before them. A
`DatasetWriter` you drop without `finish()` never enters the container. An
`AtlasWriter` you drop without `finish()` leaves no trailer. Nothing at the
target then opens as a collection.

## Staging

Each **variable** builds in an `array-format` writer. Every dataset writes into
it under the dataset's own name:

```text
add_dataset("jan")  ──▶ writer[temperature]    temperature/jan
                        writer[salinity]       salinity/jan

add_dataset("feb")  ──▶ writer[temperature]    temperature/feb
                        writer[salinity]       salinity/feb

AtlasWriter::finish ──▶ finish → copy, per variable
                    ──▶ footer, trailer, done
```

A variable's segment is complete only when every dataset has contributed, so
nothing reaches the container until `AtlasWriter::finish`.

Memory stays bounded. The writer packs each chunk into a compressed block as it
arrives, and spills every full block to a temporary file. It keeps one open
block per variable in memory, and the chunk table beside it. The writer's
memory therefore does not grow with the number of datasets.

### finish, then copy

`ArrayWriter::finish` writes one self-contained file: the blocks, then a footer
that holds every array's chunk table, attributes, and statistics. That is what
a segment must be.

The writer writes a whole object, and a segment is a byte range of one. Each
variable therefore lands in a scratch directory first, and the copy into the
container streams it in 8 MiB pieces. The scratch copy goes as soon as its
segment lands, so local disk holds each variable twice for a moment, and the
whole collection once.

The statistics cost no second pass. The writer computes a partial per chunk as
it packs the chunk, and merges them at finish.

## Streaming to the container

The output goes through `object_store::buffered::BufWriter`. It holds a small
collection in one atomic PUT. It moves to a multipart upload once the data
passes its capacity. The footer sits at the end, which makes this one forward
pass. Nothing needs a rewrite.

The writer keeps a running byte offset. Each segment records
`(seg_offset, seg_len)` in its footer entry. A segment therefore needs no
alignment, no padding, and no separator.

## Concurrent datasets

`add_dataset` returns an owned `DatasetWriter`. Several can therefore stay open
at once:

```rust
let w = Arc::new(AtlasWriter::create_path(dir, cfg).await?);
for path in files {
    let w = Arc::clone(&w);
    tasks.push(tokio::spawn(async move {
        let mut ds = w.add_dataset(&name).await?;
        // … stage it …
        ds.finish().await
    }));
}
```

A `DatasetWriter` takes the writer's lock for each define and each write,
because every dataset writes into the same per-variable writers. That is the
price of the layout: the writes serialize, and only the work outside the lock
runs in parallel.

**Ordinals do not follow that order.** Each dataset carries the number of the
`add_dataset` call that opened it. `AtlasWriter::finish` then sorts the footer
entries on that number. Stage a directory twice and every dataset lands at the
same ordinal, however many threads did the work. No dataset holds a byte range,
so nothing on disk has to match that order.

## Failure

| What happens | Result |
|---|---|
| A drop or an abort of a `DatasetWriter` | That dataset never appears. The others stay |
| A repeated dataset name | `DatasetAlreadyExists`. The name is reserved from `add_dataset` |
| A failure in `define_array` or `write_array` | The same. Abandon that dataset, and keep the others |
| A drop of the `AtlasWriter` before `finish` | No trailer. Nothing at the target opens |
| A dead process during a write | The same. No trailer, and no collection |

There is no half-written collection to find, and none to clean up. A container
without a trailer is no collection. The Python layer depends on this. A failed
`add_xarray_dataset` aborts its `DatasetWriter`, and the collection continues.

The backend decides whether a partial object stays after a dropped writer. A
lifecycle rule clears an incomplete S3 multipart upload. This is a question of
hygiene, not of correctness.

## Attributes go into the segments

`DatasetWriter::finish` writes them into the variable writers.

A per-array value goes on the array's own entry in that variable's writer,
under this dataset's name. A dataset-level value has no array to sit on, so the
reserved `_datasets` writer gets a rank-0 array per dataset to carry it. That
segment appears only when some dataset has a global attribute.

`array-format` interns each key and each value per file, so `units =
"celsius"` across ten thousand datasets is stored once per segment.

A timestamp is the one lossy step. `AttributeValue` has no timestamp variant,
so it stores as its `i64`, and the schema records the key as `TimestampNs` so a
reader can tell it from an integer.

## Interning as you write

The writer holds an `Interner`. Each finished dataset hands it its array names
with their element types, and its attribute keys with theirs, and gets back a
`u32`. The attribute **values** do not go to the interner. They go into the
segments, and the key with its type is what the footer keeps.

The interner works in two steps. Every name and every dtype goes to a pool
first. What is left is pairs of `u32`, which are `Hash` and `Eq`. One map
lookup then settles the whole schema. Two equal schemas land on one pool entry.
There is no content hash and no collision fallback.

Nothing about the data enters the schema. No shape, no chunk shape, no
dimension name, and no fill value. Those go into the variable's segment, which
records them anyway. That is what lets a directory of files of unequal length
share one pool entry.

The pool holds `SmolStr`, not `String`. A name of 22 bytes or fewer sits
inline. See
[format.md](format.md#the-decoded-form-is-not-the-wire-form) for what that buys
and where it does not.
