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
The scan **descends into every subdirectory**. It **sorts** the result, and
writes one dataset per file, named after the file. `2024-01.nc` becomes the
dataset `2024-01.nc`. The suffix is part of the name, so `jan.nc` and
`jan.nc4` are two datasets.

Pass `recursive=False`, or `--no-recursive`, to scan the top directory alone:

```python
atlas.create("/data/nc", dest, recursive=False)
```

```bash
atlas create /data/nc /data/collection --no-recursive
```

The sort matters. An ordinal comes from the write order, so a sorted ingest
makes it reproducible. Rebuild the same directory, and every dataset lands at
the same position.

Check what the call picks up before you run it:

```python
atlas.find_netcdf_files("/data/nc")                  # sorted, and recursive
atlas.find_netcdf_files("/data/nc", recursive=False)  # the top level alone
```

## All or nothing

Nothing at the destination is readable until every file lands, with the footer.
A process that dies at file 900 of 1000 leaves no collection there, and not a
partial one.

That is what you usually want. When it is not:

```python
result = atlas.create("/data/nc", dest, on_error="skip")
result["written"]   # ['2024-01.nc', '2024-03.nc']
result["skipped"]   # [{'file': '.../2024-02.nc', 'error': '...'}]
```

```bash
atlas create /data/nc /data/collection --skip-errors
```

A skipped file leaves no trace in the collection. The writer moves to the next
one. The CLI exits `1` when it skipped anything, so a pipeline sees that. It
still writes the collection.

## One bad array, not one bad file

`on_error` works at the granularity of a file. One variable of an unsupported
dtype therefore costs the whole dataset. `on_unsupported="skip"` narrows that
to the array:

```python
result = atlas.create("/data/nc", dest, on_unsupported="skip")
result["skipped_arrays"]
# [{'array': 'flag', 'dtype': 'bool', 'error': '...', 'dataset': '2024-01.nc'}]
```

```bash
atlas create /data/nc /data/collection --skip-unsupported
```

The rest of the dataset lands as usual: every other array, every attribute,
and the dataset itself. The skipped name is absent from the schema, so no
empty array stands in for it.

Atlas resolves every dtype before it defines the first array. A skip therefore
never leaves a half-written array behind.

The two settings compose. `--skip-unsupported` handles the array atlas cannot
store. `--skip-errors` handles the file that fails for any other reason.

See [Supported dtypes](dtypes.md) for what atlas can store.

## The log file

Both kinds of skip go to a log file, with the reason:

```bash
atlas create /data/nc /data/collection --skip-unsupported --log-file ingest.log
```

```text
2026-09-01 14:30:41 INFO    atlas.ops: ingesting 2 file(s) into /data/collection
2026-09-01 14:30:41 WARNING atlas.ops: /data/nc/broken.nc: ValueError: did not find a match ...
2026-09-01 14:30:41 WARNING atlas.ops: /data/nc/buoy.nc: skipped array 'flag' of dtype bool: numpy dtype dtype('bool') is not supported by atlas (supported: ...)
2026-09-01 14:30:41 INFO    atlas.ops: wrote 1 dataset(s); skipped 1 file(s) and 1 array(s)
```

Each line names the file, so a thousand-file ingest stays readable. The file
opens in append mode, so a repeat run adds to it.

From the library:

```python
import atlas

atlas.log_to_file("ingest.log")
atlas.create("/data/nc", dest, on_unsupported="skip")
```

Atlas logs to the `atlas` logger and attaches no handler of its own, so attach
your own to send the records somewhere else:

```python
import logging

logging.getLogger("atlas").addHandler(logging.StreamHandler())
logging.getLogger("atlas").setLevel(logging.INFO)
```

`log_to_file` also captures Python warnings, such as the one about missing
string cells. That moves them off stderr, because `logging.captureWarnings` is
process-wide.

The Rust core logs separately, through `tracing`. `atlas.init_tracing()` sends
that stream to stderr. See [Installation](../installation.md).

## Chunking and memory

These are one decision. The blocks a file *reads* in are the chunks it
*stores* in.

`open_chunks="auto"` is the default, and it picks a strategy per file. A file
well inside the block budget opens whole. The dask graph costs about 40 ms
per file, and a small variable lands as one chunk either way. A larger
file streams through dask, so memory stays bounded. Three results follow:

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
machine. Raise it for larger stored chunks. It also moves the threshold at
which `"auto"` stops opening a file whole.

### How files are opened

| `open_chunks` | Reads | Stored chunk shape |
|---|---|---|
| `"auto"` *(default)* | whole when small, else blocks sized to `chunk_size` | those blocks |
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
arrays = {a["name"]: a for a in atlas.describe(dest, "2024-01.nc")["arrays"]}
arrays["temperature"]["chunk_shape"]
```

```bash
atlas show dest 2024-01.nc | grep _ChunkShape
```

### Speed on many small files

A directory of small files spends most of its time per file, not per byte. On
213 KiB profile files the default runs at about 30 files per second, or five
minutes for ten thousand.

**`--workers N` stages N files at once.** It is the largest single win:

```bash
atlas create /data/nc /data/collection --workers 4
```

```text
workers=1 :  31 files/s   1.00x
workers=2 :  53 files/s   1.71x
workers=4 :  91 files/s   2.93x
workers=8 :  88 files/s   2.81x   <- plateau
```

The costly part of an ingest is the flush. It holds no lock and releases the
GIL, so it overlaps. The rest, the netCDF read and the append, does not, which
is why the curve flattens near four.

Nothing else changes. `add_dataset` runs on one thread in file order, so every
ordinal matches a sequential build. The summary sorts back into file order
too. Only `progress` reports in completion order.

Two more settings matter when that is still too slow:

- **`--open-chunks native`** forces dask on every file. That costs about twice
  as long on a small file. Use it only when the file's own chunking is what
  you want stored.
- **`--codec none`** trades size for speed. It saved about 20 percent on the
  same files.

Past the plateau, run several `atlas create` commands at once, one per part of
the tree. Each has its own writer, so nothing serialises between them. That
reached 3.7x on eight processes where threads reached 2.9x. The cost is one
collection per part.

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

The CLI does this by default, to stderr, so a pipe still reads stdout. Each
line counts the files and says how many remain:

```text
  [ 12/10000] 000043_CFPOINT_3593_V0.nc  (9988 left)
```

`-q` turns the per-file lines off. Pass `--log-file PATH` to keep the same
counter in a file. The command prints the absolute path it opened.

## What can go wrong

| Situation | Result |
|---|---|
| Directory holds no NetCDF files | `AtlasError` |
| Two files share a name | `AtlasError`. A dataset name must be unique |
| A variable has a dtype atlas cannot store | `AtlasError`, or one skipped array under `--skip-unsupported` |
| Any other bad file | `AtlasError`, or one skipped file under `--skip-errors` |
| Destination URL cannot be resolved | `SourceError` |

Two files with one name surprise people. `a/jan.nc` and `b/jan.nc` both want
the name `jan.nc`. The scan descends by default, so a tree of monthly
directories hits this often. A dataset name carries no directory, because a
name may hold no `/`.

Three ways out. Rename the files. Ingest each subdirectory into its own
collection. Or pass `on_error="skip"`, which keeps the first file and reports
the second.

For the dtype rules, see [Supported dtypes](dtypes.md).

## Changing a collection

You cannot. One write builds a collection. There is no append, and no in-place
update. To change a dataset, rebuild the collection from its sources.

A remove is the one exception. See [Removing datasets](removing.md).
