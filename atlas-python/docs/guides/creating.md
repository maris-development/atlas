# Creating a collection

One call turns a directory of NetCDF files into one collection.

```python
import atlas

atlas.create("/data/nc", "/data/collection")
```

```bash
atlas create /data/nc /data/collection
```

## What happens

Files matching `.nc`, `.nc4`, `.cdf`, or `.netcdf` are collected, **sorted**,
and written one dataset each, named after the file stem. `2024-01.nc` becomes
the dataset `2024-01`.

Sorting matters: ordinals are handed out in write order, so a sorted ingest
makes them reproducible. Rebuild the same directory and every dataset lands at
the same position.

Check what will be picked up before committing to it:

```python
atlas.find_netcdf_files("/data/nc")             # sorted list of paths
atlas.find_netcdf_files("/data/nc", recursive=True)
```

## All or nothing

Nothing at the destination is readable until every file has been written and
the footer lands. If the process dies at file 900 of 1000, there is no
collection at the destination — not a partial one.

That is usually what you want. When it is not:

```python
result = atlas.create("/data/nc", dest, on_error="skip")
result["written"]   # ['2024-01', '2024-03']
result["skipped"]   # [{'file': '.../2024-02.nc', 'error': '...'}]
```

```bash
atlas create /data/nc /data/collection --skip-errors
```

A skipped file leaves no trace in the collection; the writer moves to the next.
The CLI exits `1` when anything was skipped, so a pipeline notices, while still
writing the collection.

## Chunking

Chunk shape is the granularity at which a reader later fetches: a region read
pulls only the chunks it overlaps. It is the one decision worth making
deliberately.

By default an array takes the source file's dask chunking, or one full-shape
chunk if it has none. Override per variable:

```python
atlas.create("/data/nc", dest, chunks={"temperature": [64, 64]})
```

```bash
atlas create /data/nc dest --chunks '{"temperature": [64, 64]}'
```

Chunk the large variables you expect to slice; leave small coordinate vectors
alone. A one-chunk array is read whole or not at all.

Confirm what landed:

```python
arrays = {a["name"]: a for a in atlas.describe(dest, "2024-01")["arrays"]}
arrays["temperature"]["chunk_shape"]
```

## Memory

Dask-backed variables stream one block at a time, so peak memory is one block
per variable rather than the whole array. A file far larger than RAM ingests
without trouble, provided xarray opened it lazily:

```python
# xr.open_dataset(path, chunks=...) inside your own pipeline; `create` opens
# files itself and streams whatever chunking they carry.
```

Staging happens on local disk — `array-format` spills compressed chunks to a
temporary file — so the writer's memory does not grow with the collection
either.

## Compression

```python
atlas.create("/data/nc", dest, codec="lz4")
```

| Codec | When |
|---|---|
| `zstd` *(default)* | Best ratio at moderate CPU. Pick this unless you have a reason not to |
| `lz4` | Larger files, faster to decompress. For scan-heavy reading |
| `none` | Fastest write, no size reduction |

Blocks record their own codec, so nothing has to be told which was used when
reading.

## Progress

```python
atlas.create("/data/nc", dest, progress=lambda name: print(name))
```

The CLI does this by default, to stderr, so stdout stays pipeable. `-q` turns
it off.

## What can go wrong

| Situation | Result |
|---|---|
| Directory holds no NetCDF files | `AtlasError` |
| Two files share a stem | `AtlasError` — dataset names must be unique |
| A variable has a dtype atlas cannot store | `AtlasError` (or skipped) |
| Destination URL cannot be resolved | `SourceError` |

Two files sharing a stem is the one that surprises people: `a/jan.nc` and
`b/jan.nc` both want to be `jan`. Rename, or ingest them into separate
collections.

For the dtype rules, see [Supported dtypes](dtypes.md).

## Changing a collection

You cannot. A collection is written once; there is no append and no in-place
update. To change a dataset, rebuild the collection from its sources.

The one exception is removing datasets — see [Removing datasets](removing.md).
