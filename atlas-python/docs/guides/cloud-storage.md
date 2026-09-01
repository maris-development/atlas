# Cloud storage (S3, GCS, Azure)

Every operation takes a URL, or an
[obstore](https://github.com/developmentseed/obstore) store handle, in place of
a filesystem path. obstore is a thin Python binding around the Rust
[`object_store`](https://docs.rs/object_store) crate. Every backend it supports
therefore works with atlas: S3, GCS, Azure Blob, HTTP, and local. Atlas never
sees a credential.

The single-file format suits object storage well. A write is one multipart
upload. An open is one range read of the tail, whatever the number of datasets.

## Install

```bash
pip install "atlas-python[cloud]"
```

That equals `pip install atlas-python obstore`. Without it, atlas still works
against a local filesystem path.

## Quickstart on S3

A URL is the simplest form. The credentials come from the usual environment
variables, or from `~/.aws/credentials`:

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

Build the handle yourself for anything a URL cannot say, such as a custom
credential provider or a retry policy. Then pass it in:

```python
import obstore as obs

store = obs.store.S3Store(
    "my-bucket",
    prefix="collections/2024",
    region="eu-west-1",
)

atlas.create("/data/nc", store)
atlas.list_datasets(store)
atlas.describe(store, "2024-01.nc")
atlas.remove(store, ["2024-02.nc"])
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

An open scales with nothing. Ten datasets and a million cost one request each,
because the footer is one object range.

## Writing

The output streams through a buffered writer. That writer becomes a multipart
upload once the data passes its buffer. The footer sits at the end, which makes
this one forward pass. Nothing needs a rewrite.

Nothing at the target is readable until the writer finishes. A process that
dies during a write leaves no trailer, and the path then opens as no
collection. The lifecycle rule of the bucket clears an incomplete multipart
upload, as usual.

## Removing

`remove` reads the mask again before it writes. Two processes that remove
different datasets therefore both survive. Two removes that interleave between
the read and the write still lose one. An object store has no compare-and-swap
here. Serialize the removes against one collection if that matters.

## HTTP and read-only backends

A read-only store serves `ls`, `show`, and `info`. `create` and `rm` need a
PUT, so they fail.

## Version note

An obstore handle reaches atlas in one of two ways. It passes straight through
when both sides build against the same `pyo3-object_store`. Otherwise atlas
rebuilds it from its configuration. A rebuild raises a `RuntimeWarning` about
connection pooling. It works for any store with a URL or a path to rebuild
from. A `MemoryStore` has nothing to rebuild, so it needs two matching builds.
