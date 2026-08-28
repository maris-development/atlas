# Cloud storage (S3, GCS, Azure)

[`AtlasWriter.create`](../reference/atlas-writer.md) and
[`Atlas.open`](../reference/atlas.md) both accept an
[obstore](https://github.com/developmentseed/obstore) store handle in place of a
filesystem path. obstore is a thin Python binding around the Rust
[`object_store`](https://docs.rs/object_store) crate, so any backend it supports
— S3, GCS, Azure Blob, HTTP, local — works with atlas. Atlas never sees the
credentials.

The single-file format suits object storage particularly well. Writing is one
multipart upload; opening is one range read of the tail, however many datasets
the collection holds.

## Install

```bash
pip install "atlas-python[cloud]"
```

Equivalent to `pip install atlas-python obstore`. Without it, atlas works
against local filesystem paths as usual.

## Quickstart — S3

```python
import numpy as np
import obstore as obs
import atlas

# Credentials come from the standard AWS env vars or ~/.aws/credentials
# unless you pass them explicitly.
store = obs.store.S3Store(
    "my-bucket",
    prefix="collections/2024",
    region="us-east-1",
)

with atlas.AtlasWriter.create(store, codec="zstd") as w:
    ds = w.add_dataset("jan_2024")
    ds.define_array("temperature", dtype="float32", dims=["lat", "lon"],
                    shape=[8, 16], chunk_shape=[4, 8])
    ds.write_array("temperature", start=[0, 0],
                   data=np.full((8, 16), 20.0, dtype=np.float32))
    ds.set_attribute("month", 1)
    ds.finish()

collection = atlas.Atlas.open(store)
collection.list_datasets()
collection.dataset("jan_2024").array_meta("temperature")
```

The objects land under the store's prefix:

```text
s3://my-bucket/collections/2024/data.atlas
s3://my-bucket/collections/2024/deleted.mask     (only after a delete)
```

## Other backends

```python
store = obs.store.GCSStore("my-bucket", prefix="collections/2024")
store = obs.store.AzureStore("my-container", prefix="collections/2024")
store = obs.store.HTTPStore("https://example.org/data")   # read-only
store = obs.store.LocalStore("/data")
store = obs.store.MemoryStore()
```

Nothing else in your code changes.

## What it costs

| Operation | Requests |
|---|---|
| `Atlas.open` | 1 range read (tail), plus 1 for the mask if it exists |
| `list_datasets`, schemas, attributes | 0 — answered from the footer |
| `delete_dataset` | 1 GET + 1 PUT of the small mask |
| Writing a collection | 1 multipart upload |
| Reading one array region (Rust) | 2 to open the segment, then 1 per chunk touched |

Opening scales with nothing: a collection of ten datasets and one of a million
both cost a single request, because the footer is one object range.

## Writing

Output streams through a buffered writer that becomes a multipart upload once it
outgrows its buffer. Footer-at-end is what makes this a single forward pass —
nothing is ever rewritten.

Nothing at the target is readable until the writer finishes. If the process dies
mid-write, no trailer is written and the path does not open as a collection; an
incomplete multipart upload is cleaned up by the bucket's lifecycle rule, as
usual.

## Deleting

`delete_dataset` re-reads the mask before writing it, so two handles deleting
different datasets both survive. Two deletes that interleave between the read
and the write still lose one — object stores have no compare-and-swap here.
Serialize deletes against a collection if that matters.

## HTTP and read-only backends

A read-only store works for opening a collection and reading metadata. A delete
will fail, since it needs a PUT.

## Version note

The obstore handle crosses into atlas either directly, when the two were built
against the same `pyo3-object_store`, or by being reconstructed from its
configuration when they were not. Reconstruction emits a `RuntimeWarning` about
connection pooling and works for any store with a URL or path to rebuild from.
A `MemoryStore` has nothing to reconstruct, so it needs matching builds.
