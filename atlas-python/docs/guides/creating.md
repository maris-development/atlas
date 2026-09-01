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

`create` collects every file that matches `.nc`, `.nc4`, `.cdf`, or `.netcdf`.
It **sorts** them, and writes one dataset per file, named after the file stem.
`2024-01.nc` becomes the dataset `2024-01`.

The sort matters. An ordinal comes from the write order, so a sorted ingest
makes it reproducible. Rebuild the same directory, and every dataset lands at
the same position.

Check what the call picks up before you run it:

```python
atlas.find_netcdf_files("/data/nc")             # sorted list of paths
atlas.find_netcdf_files("/data/nc", recursive=True)
```

## All or nothing

Nothing at the destination is readable until every file lands, with the footer.
A process that dies at file 900 of 1000 leaves no collection there, and not a
partial one.

That is what you usually want. When it is not:

```python
result = atlas.create("/data/nc", dest, on_error="skip")
result["written"]   # ['2024-01', '2024-03']
result["skipped"]   # [{'file': '.../2024-02.nc', 'error': '...'}]
```

```bash
atlas create /data/nc /data/collection --skip-errors
```

A skipped file leaves no trace in the collection. The writer moves to the next
one. The CLI exits `1` when it skipped anything, so a pipeline sees that. It
still writes the collection.

## Chunking and memory

These are one decision. The blocks a file *reads* in are the chunks it
*stores* in.

Each file opens with dask chunking. `open_chunks="auto"` is the default. dask
then sizes the blocks to `chunk_size`, which is 128 MiB. Three results follow:

- A file far larger than memory streams block by block, and does not read
  whole.
- A small variable still comes out as one chunk, as it would with no chunking.
- A large variable stores in blocks of about `chunk_size`. That is a sensible
  read size.

```bash
atlas create /data/nc dest --chunk-size 64MiB
```

```python
atlas.create("/data/nc", dest, chunk_size="64MiB")
```

`chunk_size` is about the memory ceiling per variable. Lower it on a small
machine. Raise it for larger stored chunks.

### How files are opened

| `open_chunks` | Reads | Stored chunk shape |
|---|---|---|
| `"auto"` *(default)* | blocks sized to `chunk_size` | those blocks |
| `"native"` | the file's own chunk encoding | that encoding |
| `None` | each variable whole | one full-shape chunk |
| `{"time": 100}` | as given, per dimension | as given |

`"native"` reads no extra bytes during ingest, because the dask blocks match
the NetCDF chunks. There is a catch. A netCDF4 file often has very small
chunks, and those become very small atlas chunks. A netCDF3 file has no
chunking, so `"native"` reads it whole.

Use `None` only for a file you know is small. It is the fastest path when a
whole variable fits with room to spare.

### Overriding the stored shape

`chunks` sets the stored chunk shape directly, whatever the read used:

```bash
atlas create /data/nc dest --chunks '{"temperature": [64, 64]}'
```

Use it when the read size you want on disk differs from the one that suits
ingest. Note the cost. The source blocks no longer align with the stored
chunks, so each write becomes a read-modify-write. That is correct, and
slower.

### Choosing

Chunk the large variables you expect to slice. Leave a small coordinate vector
alone. A one-chunk array reads whole, or not at all.

Confirm what landed:

```python
arrays = {a["name"]: a for a in atlas.describe(dest, "2024-01")["arrays"]}
arrays["temperature"]["chunk_shape"]
```

```bash
atlas show dest 2024-01 | grep _ChunkShape
```

### The writer's own memory

Staging runs on local disk. `array-format` spills each compressed chunk to a
temporary file. The memory of the writer therefore does not grow with the
number of datasets. A thousand-file ingest costs what a one-file ingest costs.

## Compression

```python
atlas.create("/data/nc", dest, codec="lz4")
```

| Codec | When |
|---|---|
| `zstd` *(default)* | The best ratio at moderate CPU. Pick this without a reason to do otherwise |
| `lz4` | Larger files, and faster to decompress. For a read-heavy scan |
| `none` | The fastest write. It makes the file no smaller |

Each block records its own codec, so nothing tells a reader which one the write
used.

## Progress

```python
atlas.create("/data/nc", dest, progress=lambda name: print(name))
```

The CLI does this by default, to stderr, so a pipe still reads stdout. `-q`
turns it off.

## What can go wrong

| Situation | Result |
|---|---|
| Directory holds no NetCDF files | `AtlasError` |
| Two files share a stem | `AtlasError`. A dataset name must be unique |
| A variable has a dtype atlas cannot store | `AtlasError` (or skipped) |
| Destination URL cannot be resolved | `SourceError` |

Two files with one stem surprise people. `a/jan.nc` and `b/jan.nc` both want
the name `jan`. Rename one, or ingest them into two collections.

For the dtype rules, see [Supported dtypes](dtypes.md).

## Changing a collection

You cannot. One write builds a collection. There is no append, and no in-place
update. To change a dataset, rebuild the collection from its sources.

A remove is the one exception. See [Removing datasets](removing.md).
