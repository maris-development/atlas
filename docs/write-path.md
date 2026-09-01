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

Each dataset builds as a complete `array-format` file, in a local scratch
directory. The writer then copies that file into the output stream, byte for
byte:

```text
add_dataset("jan")  ──▶ scratch/1/data.af      define, write, define, write, …
  finish()          ──▶ flush → compact → copy ──▶ container[8 .. 4_100]

add_dataset("feb")  ──▶ scratch/2/data.af
  finish()          ──▶ flush → compact → copy ──▶ container[4_100 .. 9_002]

AtlasWriter::finish ──▶ footer, trailer, done
```

Local staging bounds the memory. `array-format` spills each compressed chunk to
a temporary file on arrival. The copy into the container streams in 8 MiB
pieces. A dataset far larger than RAM therefore writes without trouble. The
scratch directory goes as soon as its segment lands.

### flush, then compact

Run both, in that order. The order matters.

`flush()` commits the buffered writes into a sidecar layer. `compact()` merges
every layer into one base file. A `compact()` without a `flush()` first leaves
the buffered writes behind. It can also produce a dangling attribute index,
because `compact` builds its attribute dictionary from committed layers only.

The result is one self-contained file, which is what a segment must be.

The flush also computes the minimum, the maximum, and the null count of each
array. Those reach the footer entry before the staged file closes. A reader
therefore gets them without the segment. See
[format.md](format.md#statistics-live-here-too).

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

Staging is parallel. Only the append serializes. A `DatasetWriter` takes the
writer's lock once, in `finish()`, for its append and its footer entry.
Concurrent datasets therefore land in finish order, and never interleave their
bytes. `tests/integration.rs` asserts that the segments still tile the
container without a gap under concurrent staging.

**Ordinals do not follow that order.** Each dataset carries the number of the
`add_dataset` call that opened it. `AtlasWriter::finish` then sorts the footer
entries on that number. Stage a directory twice and every dataset lands at the same
ordinal, however many threads did the work. Each entry holds its own byte
range, so the segments need no matching order on disk.

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

## Interning as you write

The writer holds an `Interner`. Each finished dataset hands it a
`DatasetSchema`, and gets back a `u32`. Two equal schemas land on one pool
entry. A content hash resolves them, and `PartialEq` settles a hash collision.
Attribute keys intern the same way.

One detail matters. `FillValueS` compares floats by bit pattern, so one NaN
fill equals another. Without that, every float array with the default NaN fill
takes its own pool entry, and interning never fires on the common case.
