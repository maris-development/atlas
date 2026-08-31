# Cloud storage (S3, GCS, Azure)

Every operation accepts a URL, or an
[obstore](https://github.com/developmentseed/obstore) store handle, in place of
a filesystem path. obstore is a thin Python binding around the Rust
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

The simplest form is a URL. Credentials come from the usual environment
variables or `~/.aws/credentials`:

```bash
atlas create /data/nc s3://my-bucket/collections/2024 --region eu-west-1
atlas ls   s3://my-bucket/collections/2024 --region eu-west-1
atlas info s3://my-bucket/collections/2024 --region eu-west-1
```

```python
import atlas

atlas.create("/data/nc", "s3://my-bucket/collections/2024", region="eu-west-1")
atlas.list_datasets("s3://my-bucket/collections/2024", region="eu-west-1")
```

For anything the URL cannot express — a custom credential provider, a retry
policy — build the handle yourself and pass it in:

```python
import obstore as obs

store = obs.store.S3Store(
    "my-bucket",
    prefix="collections/2024",
    region="eu-west-1",
)

atlas.create("/data/nc", store)
atlas.list_datasets(store)
atlas.describe(store, "2024-01")
atlas.remove(store, ["2024-02"])
atlas.info(store)
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
| `list_datasets`, `describe`, `info` | 1 head + 1 range read of the tail, plus 1 for the mask if present |
| `remove` | The above, plus 1 GET and 1 PUT of the small mask |
| `create` | 1 multipart upload |
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

## Removing

`remove` re-reads the mask before writing it, so two processes removing
different datasets both survive. Two removals that interleave between the read
and the write still lose one — object stores have no compare-and-swap here.
Serialize removals against a collection if that matters.

## HTTP and read-only backends

A read-only store serves `ls`, `show`, and `info`. `create` and `rm` need a
PUT and will fail.

## Version note

The obstore handle crosses into atlas either directly, when the two were built
against the same `pyo3-object_store`, or by being reconstructed from its
configuration when they were not. Reconstruction emits a `RuntimeWarning` about
connection pooling and works for any store with a URL or path to rebuild from.
A `MemoryStore` has nothing to reconstruct, so it needs matching builds.
