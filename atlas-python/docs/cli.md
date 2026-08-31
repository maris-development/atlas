# The `atlas` command

`pip install atlas-python` puts `atlas` on your PATH. Five subcommands, one per
operation.

```text
atlas create <netcdf-dir> <collection>   build a collection
atlas rm     <collection> <name>...      remove datasets
atlas ls     <collection>                list datasets
atlas show   <collection> <name>         one dataset, ncdump style
atlas info   <collection>                the whole collection
```

A `<collection>` is a local path or a URL — `s3://bucket/prefix`, `gs://...`,
`az://...`, `https://...`. Remote sources need
`pip install "atlas-python[cloud]"`. See [Cloud storage](guides/cloud-storage.md).

Every subcommand takes `--json`, and the remote flags `--region`,
`--endpoint`, and `--anonymous`.

## create

```bash
atlas create /data/nc /data/collection
```

Each NetCDF file becomes one dataset named after its stem, so `2024-01.nc`
becomes `2024-01`. Files are ingested in sorted order, which makes the ordinals
a collection hands out reproducible.

Nothing is readable at the destination until every file has been written and
the footer lands. A failure part-way leaves no collection at all, rather than a
partial one.

| Flag | Effect |
|---|---|
| `-r`, `--recursive` | Descend into subdirectories |
| `--codec {zstd,lz4,none}` | Block compression. Default `zstd` |
| `--chunk-size SIZE` | Block size to aim for. Default `128MiB` |
| `--open-chunks MODE` | How files are read: `auto`, `native`, `none`, or a JSON dict |
| `--chunks JSON` | Override the stored chunk shape, `'{"temperature": [64, 64]}'` |
| `--skip-errors` | Skip files that fail instead of abandoning the collection |
| `-q`, `--quiet` | Do not list files as they are written |

Progress goes to stderr, so stdout stays pipeable.

### Large files

Files are read in dask blocks, so a file far larger than memory streams rather
than being loaded whole. `--chunk-size` sets the block size and is roughly the
memory ceiling per variable:

```bash
# A machine with little RAM
atlas create /data/nc /data/collection --chunk-size 32MiB
```

Those blocks also become the stored chunk shape, which is the granularity a
reader later fetches at. `--open-chunks` picks a different strategy:

| Mode | Reads | Stored chunk shape |
|---|---|---|
| `auto` *(default)* | blocks sized to `--chunk-size` | those blocks |
| `native` | the file's own chunk encoding | that encoding |
| `none` | each variable whole | one full-shape chunk |
| JSON dict | as given, per dimension | as given |

```bash
# Match the NetCDF file's own chunking exactly
atlas create /data/nc /data/collection --open-chunks native

# Per-dimension, explicitly
atlas create /data/nc /data/collection --open-chunks '{"time": 100, "lat": -1}'
```

`--chunks` overrides the *stored* shape without changing how the file is read.
It costs a read-modify-write per misaligned block, so prefer `--open-chunks`
when you can express what you want that way.

```bash
# One collection from a tree of monthly directories, tolerating bad files
atlas create /data/nc /data/collection --recursive --skip-errors

# A big grid, chunked for selective reads, straight to a bucket
atlas create /data/nc s3://bucket/2024 --chunk-size 64MiB --region eu-west-1
```

## rm

```bash
atlas rm /data/collection 2024-02 2024-03
```

Removes several datasets in one call. Names may be given as dataset names or as
the NetCDF paths they came from, so the same list that built a collection can
tear part of it down:

```bash
atlas rm /data/collection /data/nc/2024-02.nc
```

This writes the deletion mask beside the container. **The container is not
touched**: no space is reclaimed, and no ordinal moves. Rebuild the collection
to reclaim the bytes.

`--missing-ok` reports names that are absent or already removed instead of
failing.

## ls

```bash
$ atlas ls /data/collection
2024-01
2024-02
2024-03
```

One name per line, in write order. Removed datasets are not listed. Costs one
range read of the container tail, whatever the collection size.

```bash
atlas ls s3://bucket/2024 | wc -l          # how many datasets
atlas ls /data/collection --json | jq .    # as a JSON array
```

## show

```bash
$ atlas show /data/collection 2024-01
dataset 2024-01 {
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

Deliberately shaped like `ncdump -h`, with two additions.

**Statistics** under each variable: `count` is the total element count,
`nulls` (shown only when non-zero) counts elements equal to the fill value, and
`min`/`max` are the extremes. These were computed when the array was written
and stored in the footer, so printing them needs no more I/O than `ls` did.

**Segment bytes** at the end: the byte range this dataset occupies in
`data.atlas`. Those bytes are a complete `array-format` file — `dd` them out
and any `array-format` reader opens the result.

`--json` emits the whole structure, which is the form to script against.

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
      lat
      lon
      station
      temperature
```

`removed` appears only when the mask hides something, and says plainly that
those bytes are still in the file.

`interned schemas` is how many distinct schemas the datasets share between
them. A fleet of a thousand identically-shaped datasets shows `1`.

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

# Total elements across the collection
atlas ls "$C" --json | jq -r '.[]' | while read -r n; do
  atlas show "$C" "$n" --json | jq '[.arrays[].stats.row_count] | add'
done | paste -sd+ | bc
```

String statistics come back as text in JSON, and as `bytes` from the library.
