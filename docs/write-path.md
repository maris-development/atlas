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

Each **variable** builds as a complete `array-format` file, in a local scratch
directory. Every dataset writes into it under the dataset's own name:

```text
add_dataset("jan")  ──▶ scratch/v0/data.af     temperature/jan
                        scratch/v1/data.af     salinity/jan

add_dataset("feb")  ──▶ scratch/v0/data.af     temperature/feb
                        scratch/v1/data.af     salinity/feb

AtlasWriter::finish ──▶ flush → compact → copy, per variable
                    ──▶ footer, trailer, done
```

A variable's segment is complete only when every dataset has contributed, so
nothing reaches the container until `AtlasWriter::finish`. Local scratch
therefore holds the whole collection once. The scratch directory of each
variable goes as soon as its segment lands.

Memory stays bounded. `array-format` keeps a pending write in memory until
`flush`, and each flush seals a sidecar layer that `compact` must later merge.
A variable past `STAGING_FLUSH_BUDGET`, which is 64 MiB, therefore flushes. A
small budget costs layers, and a large one costs memory. Per-dataset flushing
is the wrong end of that trade: it made a 800-dataset write nine times slower,
and its cost grew faster than the dataset count.

The copy into the container streams in 8 MiB pieces.

### flush, then compact

Run both, in that order. The order matters.

`flush()` commits the buffered writes into a sidecar layer. `compact()` merges
every layer into one base file. A `compact()` without a `flush()` first leaves
the buffered writes behind. It can also produce a dangling attribute index,
because `compact` builds its attribute dictionary from committed layers only.
The attribute values matter here now: `DatasetWriter::finish` writes them into
the staging files, and `AtlasWriter::finish` flushes before it compacts.

The result is one self-contained file, which is what a segment must be.

The flush also computes the minimum, the maximum, and the null count of each
array, and writes them to the `{stem}.stats` sidecar beside the file. The
container embeds that sidecar next to the segment, and records where it is. A
reader then finds the statistics where `array-format` looks for them. See
[format.md](format.md#a-segment-is-two-objects).

> **Cost.** This pass reads, decompresses, and compresses every chunk again. It
> also computes the statistics twice. All of it happens on local scratch.
> Ingest therefore spends about twice the compression CPU of a one-shot
> builder. The work sits in `create_staging_file` and `DatasetWriter::finish`.
> An `array-format` API that writes a base directly would replace it as it
> stands.

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
because every dataset writes into the same per-variable files. That is the
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

`DatasetWriter::finish` writes them, before anything is compacted.

A per-array value goes on the array's own entry in that variable's staging
file, under this dataset's name. A dataset-level value has no array to sit on,
so the reserved `_datasets` staging file gets a rank-0 array per dataset to
carry it. That file appears only when some dataset has a global attribute.

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
