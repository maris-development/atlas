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

## Chunking and memory

These are one decision, because the blocks a file is *read* in are the chunks
it is *stored* in.

Files are opened with dask chunking. By default `open_chunks="auto"` lets dask
pick blocks sized to `chunk_size` (128 MiB), so:

- a file far larger than memory streams block by block rather than being read
  whole;
- a small variable still comes out as one chunk, exactly as if nothing were
  chunked at all;
- a large variable is stored in blocks of roughly `chunk_size`, which is a
  sensible read granularity.

```bash
atlas create /data/nc dest --chunk-size 64MiB
```

```python
atlas.create("/data/nc", dest, chunk_size="64MiB")
```

`chunk_size` is roughly the memory ceiling per variable. Lower it on a small
machine; raise it when you want bigger stored chunks.

### How files are opened

| `open_chunks` | Reads | Stored chunk shape |
|---|---|---|
| `"auto"` *(default)* | blocks sized to `chunk_size` | those blocks |
| `"native"` | the file's own chunk encoding | that encoding |
| `None` | each variable whole | one full-shape chunk |
| `{"time": 100}` | as given, per dimension | as given |

`"native"` avoids read amplification during ingest, since dask blocks line up
with the NetCDF chunks exactly. The catch is that a netCDF4 file often has very
small chunks — which then become very small atlas chunks — and a netCDF3 file
has no chunking at all, so `"native"` reads it whole.

`None` is only for files you know are small; it is the fastest path when a
whole variable fits comfortably.

### Overriding the stored shape

`chunks` sets the stored chunk shape directly, whatever the file was read in:

```bash
atlas create /data/nc dest --chunks '{"temperature": [64, 64]}'
```

Use it when the read granularity you want on disk differs from the one that
suits ingest. Note the cost: source blocks no longer align with stored chunks,
so writes become read-modify-write. Correct, but slower.

### Choosing

Chunk the large variables you expect to slice; leave small coordinate vectors
alone. A one-chunk array is read whole or not at all.

Confirm what landed:

```python
arrays = {a["name"]: a for a in atlas.describe(dest, "2024-01")["arrays"]}
arrays["temperature"]["chunk_shape"]
```

```bash
atlas show dest 2024-01 | grep _ChunkShape
```

### The writer's own memory

Staging happens on local disk — `array-format` spills compressed chunks to a
temporary file — so the writer's memory does not grow with the number of
datasets either. A thousand-file ingest costs the same as a one-file ingest.

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
