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

Two commit points, and nothing is visible before them. A `DatasetWriter` dropped
without `finish()` never enters the container. An `AtlasWriter` dropped without
`finish()` leaves no trailer, so nothing at the target opens as a collection.

## Staging

Each dataset is built as a complete `array-format` file in a local scratch
directory, then copied verbatim into the output stream:

```text
add_dataset("jan")  ──▶ scratch/1/data.af      define, write, define, write, …
  finish()          ──▶ flush → compact → copy ──▶ container[8 .. 4_100]

add_dataset("feb")  ──▶ scratch/2/data.af
  finish()          ──▶ flush → compact → copy ──▶ container[4_100 .. 9_002]

AtlasWriter::finish ──▶ footer, trailer, done
```

Staging on local disk is what keeps memory bounded. `array-format` spills
compressed chunks to a temporary file as they arrive, and the copy into the
container streams in 8 MiB pieces, so a dataset far larger than RAM writes
without trouble. The scratch directory is removed as soon as its segment is
appended.

### flush, then compact

Both, in that order, and the order matters.

`flush()` commits buffered writes into a sidecar layer. `compact()` merges every
layer into a single base file. Compacting without flushing first would leave the
buffered writes behind, and — because `compact` builds its attribute dictionary
from committed layers only — could produce dangling attribute indices.

The result is one self-contained file, which is what a segment has to be.

> **Cost.** This pass re-reads, decompresses, and recompresses every chunk, and
> computes statistics twice, all on local scratch. Ingest therefore pays roughly
> double the compression CPU of a hypothetical one-shot builder. It is isolated
> in `create_staging_file` / `DatasetWriter::finish`, so an `array-format` API
> that writes a base directly would be a drop-in replacement.

## Streaming to the container

Output goes through `object_store::buffered::BufWriter`, which buffers small
collections into a single atomic PUT and switches to a multipart upload once it
outgrows its capacity. Footer-at-end is what makes this a single forward pass:
nothing has to be rewritten once written.

The writer tracks a running byte offset. Each appended segment records
`(seg_offset, seg_len)` in its footer entry — which is why segments need no
alignment, padding, or separator.

## Concurrent datasets

`add_dataset` returns an owned `DatasetWriter`, so several may be open at once:

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

Staging is fully parallel; only the append is serialized. A `DatasetWriter`
takes the writer's lock once, in `finish()`, for the duration of its append and
footer entry — so concurrent datasets land in finish order and can never
interleave their bytes. `tests/integration.rs` asserts that segments still tile
the container contiguously under concurrent staging.

## Failure

| What happens | Result |
|---|---|
| `DatasetWriter` dropped or aborted | That dataset never appears. Others unaffected |
| `define_array` / `write_array` fails | Same — abandon the dataset, keep writing others |
| `AtlasWriter` dropped without `finish` | No trailer. Nothing at the target opens |
| Process dies mid-write | Same: no trailer, no collection |

There is no half-written collection to detect or clean up, because a container
without a trailer is not a collection. The Python layer leans on this: a failed
`add_xarray_dataset` aborts its `DatasetWriter` and the collection carries on.

Whether a partial object lingers on the backend after a dropped writer is the
backend's business — an S3 multipart upload left incomplete is cleaned by a
lifecycle rule. It is a hygiene concern, not a correctness one.

## Interning as you write

The writer holds an `Interner`. Each finished dataset hands it a `DatasetSchema`
and gets back a `u32`; identical schemas collide onto one pool entry, resolved
by content hash with a `PartialEq` fallback for hash collisions. Attribute keys
intern the same way.

One subtlety: `FillValueS` compares floats by bit pattern, so a NaN fill equals
a NaN fill. Without that, every float array with the default NaN fill would get
its own pool entry and interning would never fire on the most common case.
