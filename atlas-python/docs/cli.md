# The `atlas` command

`pip install atlas-python` puts `atlas` on your PATH. Five subcommands, one per
operation.

`python -m atlas` runs the same command without a PATH lookup. Use it in a
container, in a CI job, or when the shell cannot find `atlas`. See
[Installation](installation.md#the-atlas-command-is-not-found).

```text
atlas create <netcdf-dir> <collection>   build a collection
atlas rm     <collection> <name>...      remove datasets
atlas ls     <collection>                list datasets
atlas show   <collection> <name>         one dataset, ncdump style
atlas info   <collection>                the whole collection
```

A `<collection>` is a local path or a URL: `s3://bucket/prefix`, `gs://...`,
`az://...`, or `https://...`. A remote source needs
`pip install "atlas-python[cloud]"`. See
[Cloud storage](guides/cloud-storage.md).

Every subcommand takes `--json` and `--log-file PATH`, plus the remote flags
`--region`, `--endpoint`, and `--anonymous`.

## Logging

`--log-file PATH` appends to the file you name. There is no default location,
and no log at all without the flag. The command prints the absolute path it
opened, so it is never a guess:

```text
atlas: logging to /home/you/ingest.log
```

The file gets every error and warning, with the reason:

```bash
$ atlas create /data/nc /data/collection --skip-unsupported --log-file ingest.log
$ cat ingest.log
2026-09-01 14:30:41 INFO    atlas.cli: atlas 0.16.4: create /data/nc ...
2026-09-01 14:30:41 INFO    atlas.ops: ingesting 2 file(s) into /data/collection
2026-09-01 14:30:41 WARNING atlas.ops: /data/nc/buoy.nc: skipped array 'flag' of dtype bool: numpy dtype dtype('bool') is not supported by atlas (supported: ...)
2026-09-01 14:30:41 INFO    atlas.ops: wrote 1 dataset(s); skipped 0 file(s) and 1 array(s)
```

The file opens in append mode. Each line names the file it came from, so a
large ingest stays readable.

This is separate from `ATLAS_LOG`, which turns on the Rust `tracing` stream to
stderr.

## create

```bash
atlas create /data/nc /data/collection
```

The scan descends into every subdirectory. Each NetCDF file becomes one
dataset, named after the file. `2024-01.nc` becomes `2024-01.nc`, suffix and
all. A name carries no directory, so two files of one name in two
subdirectories collide. The files land in sorted order, which makes the
ordinals of a collection reproducible.

Nothing at the destination is readable until every file lands, with the footer.
A failure part-way leaves no collection, and not a partial one.

| Flag | Effect |
|---|---|
| `--no-recursive` | Scan the top directory alone. The scan descends by default |
| `-r`, `--recursive` | Accepted for compatibility. The scan already descends |
| `--codec {zstd,lz4,none}` | Block compression. Default `zstd` |
| `--chunk-size SIZE` | Block size to aim for. Default `128MiB` |
| `--open-chunks MODE` | How files are read: `auto`, `native`, `none`, or a JSON dict |
| `--chunks JSON` | Override the stored chunk shape, `'{"temperature": [64, 64]}'` |
| `--skip-errors` | Skip files that fail instead of abandoning the collection |
| `-j`, `--workers N` | Stage N files at once. About 3x on a many-core machine. Ordinals do not move |
| `--convert-calendar` | Turn a cftime axis into exact Gregorian timestamps, each keeping its instant |
| `--no-decode-times` | Keep a time axis as raw numbers, for a calendar that decodes to cftime |
| `--skip-unsupported` | Leave out an array of an unsupported dtype, and keep the rest of the dataset |
| `-q`, `--quiet` | Do not list a file as it lands |

Progress goes to stderr, so a pipe still reads stdout. Each line counts the
files and says how many remain:

```text
Writing /data/collection from 3 file(s)
  [1/3] 2024-01.nc  (2 left)
  [2/3] 2024-02.nc  (1 left)
  [3/3] 2024-03.nc  (0 left)
3 dataset(s) written to /data/collection
```

`-q` turns the per-file lines off. The same counter goes to `--log-file`, for
a run nobody watched.

### Unsupported dtypes

Atlas cannot store every numpy dtype. `bool` is the common case. By default one
such variable fails the whole file:

```bash
$ atlas create /data/nc /data/collection
atlas: /data/nc/buoy.nc: numpy dtype dtype('bool') is not supported by atlas ...
```

`--skip-unsupported` narrows that to the one array:

```bash
$ atlas create /data/nc /data/collection --skip-unsupported
  skipped array buoy/flag (bool)
1 dataset(s) written to /data/collection
```

Every other array of that dataset lands, with its attributes. `--json` reports
the skipped arrays under `skipped_arrays`, with the dataset, the dtype, and the
reason. See [Supported dtypes](guides/dtypes.md).

### Large files

Each file reads in dask blocks. A file far larger than memory therefore
streams, and does not load whole. `--chunk-size` sets the block size. It is
about the memory ceiling per variable:

```bash
# A machine with little RAM
atlas create /data/nc /data/collection --chunk-size 32MiB
```

Those blocks also become the stored chunk shape. A reader later fetches at that
size. `--open-chunks` picks another strategy:

| Mode | Reads | Stored chunk shape |
|---|---|---|
| `auto` *(default)* | whole when the file is small, else blocks sized to `--chunk-size` | those blocks |
| `native` | the file's own chunk encoding | that encoding |
| `none` | each variable whole | one full-shape chunk |
| JSON dict | as given, per dimension | as given |

```bash
# Match the NetCDF file's own chunking exactly
atlas create /data/nc /data/collection --open-chunks native

# Per-dimension, explicitly
atlas create /data/nc /data/collection --open-chunks '{"time": 100, "lat": -1}'
```

`--chunks` overrides the *stored* shape, and does not change how the file
reads. Each misaligned block then costs a read-modify-write. Use
`--open-chunks` when it can say what you want.

```bash
# One collection from a tree of monthly directories, tolerating bad files
atlas create /data/nc /data/collection --skip-errors

# A big grid, chunked for selective reads, straight to a bucket
atlas create /data/nc s3://bucket/2024 --chunk-size 64MiB --region eu-west-1
```

## rm

```bash
atlas rm /data/collection 2024-02.nc 2024-03.nc
```

This removes several datasets in one call. A name is a dataset name, or the
NetCDF path the dataset came from. The list that built a collection can
therefore tear part of it down:

```bash
atlas rm /data/collection /data/nc/2024-02.nc
```

This writes the deletion mask beside the container. **The container does not
change.** It reclaims no space, and moves no ordinal. Rebuild the collection to
reclaim the bytes.

One mask write covers every name in the call, so a long list costs what one
name costs. A list too long for a command line belongs in `atlas.remove` from
Python.

`--missing-ok` reports a name that is absent or already removed, instead of an
error.

## ls

```bash
$ atlas ls /data/collection
2024-01.nc
2024-02.nc
2024-03.nc
```

One name per line, in write order. A removed dataset does not appear. This
costs one range read of the container tail, whatever the size of the
collection.

```bash
atlas ls s3://bucket/2024 | wc -l          # how many datasets
atlas ls /data/collection --json | jq .    # as a JSON array
```

## show

```bash
$ atlas show /data/collection 2024-01.nc
dataset 2024-01.nc {
dimensions:
	lat = 4 ;
	lon = 6 ;
variables:
	float64 lat(lat) ;  // coordinate
		lat:_FillValue = nan ;
		// stats: count=4  min=0.0  max=3.0
	float32 temperature(lat, lon) ;
		temperature:_FillValue = nan ;
		temperature:units = "celsius" ;
		// stats: count=24  min=1.0  max=24.0
	string station(lat) ;
		station:_FillValue = "" ;
		// stats: count=4  min="a"  max="d"

// global attributes:
		:month = 1 ;
		:source = "example" ;

// ordinal 0, segment bytes 8..1691
}
```

The shape follows `ncdump -h` on purpose, with two additions.

**Statistics** under each variable. `count` is the total element count.
`nulls` counts the elements equal to the fill value, and appears only above
zero. `min` and `max` are the two bounds. The write computed these, and the
footer holds them. To print them therefore needs no more I/O than `ls`.

**Segment bytes** at the end. That is the byte range this dataset occupies in
`data.atlas`. Those bytes are a complete `array-format` file. `dd` them out,
and any `array-format` reader opens the result.

`--json` prints the whole structure. Script against that form.

## info

```bash
$ atlas info /data/collection
collection /data/collection
  format version    1
  created           2026-08-31T08:32:57Z
  codec             zstd
  container size    5.4 KiB
  datasets          2
  removed           1 (of 3 written; space not reclaimed)
  interned schemas  1
  distinct arrays   4
      lat          count=8  min=0.0  max=3.0
      lon          count=12  min=0.0  max=5.0
      station      count=8  min="a"  max="d"
      temperature  count=48  min=1.0  max=26.0
```

`removed` appears only when the mask hides something, and says plainly that
those bytes are still in the file.

`interned schemas` is how many distinct schemas the datasets share between
them. A fleet of a thousand identically-shaped datasets shows `1`.

Each array line gives one set of statistics for the whole collection. The
counts add up over every live dataset that holds the array. The minimum is the
smallest of the minimums. The maximum is the largest of the maximums. A
removed dataset counts for nothing. Use `atlas show` for one dataset.

## Exit codes

`0` on success, `1` on any error, with a one-line message on stderr:

```bash
$ atlas ls /tmp/not-a-collection
atlas: not an atlas collection: no 'data.atlas' under this prefix
$ echo $?
1
```

`atlas create` also exits `1` when `--skip-errors` skipped something, so a
pipeline notices, while still writing the collection.

## Scripting

`--json` on any read command, and the output is stable:

```bash
# Every dataset whose temperature maximum exceeds 30
for name in $(atlas ls "$C" --json | jq -r '.[]'); do
  atlas show "$C" "$name" --json \
    | jq -r --arg n "$name" \
        '.arrays[] | select(.name=="temperature" and .stats.max > 30) | $n'
done

# Total elements across the collection, in one call
atlas info "$C" --json | jq '[.array_stats[].row_count] | add'

# The temperature range over every live dataset
atlas info "$C" --json | jq '.array_stats.temperature | {min, max}'
```

A string statistic comes back as text in JSON, and as `bytes` from the library.
