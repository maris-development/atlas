# Quickstart

A directory of NetCDF files in, one collection out, then a look at what landed.

## Build

```bash
atlas create /data/nc /data/collection
```

```text
Writing /data/collection
  2024-01
  2024-02
  2024-03
3 dataset(s) written to /data/collection
```

Each file became one dataset, named after its stem. The result is a single
file:

```bash
$ ls /data/collection
data.atlas
```

Nothing was readable there until the last file was written — a failure part-way
would have left no collection at all, rather than a partial one.

From Python:

```python
import atlas

atlas.create("/data/nc", "/data/collection")
```

## Look

```bash
$ atlas ls /data/collection
2024-01
2024-02
2024-03
```

```bash
$ atlas info /data/collection
collection /data/collection
  format version    1
  created           2026-08-31T08:32:57Z
  codec             zstd
  container size    5.4 KiB
  datasets          3
  interned schemas  1
  distinct arrays   4
      lat
      lon
      station
      temperature
```

Both come from one range read of the container's tail — the same cost whether
the collection holds three datasets or a million.

`interned schemas 1` means all three months declare the same arrays, so that
schema is stored once.

## Inspect one dataset

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

Shaped like `ncdump -h`, with the statistics that were computed when each array
was written. Those are in the footer too, so printing them cost nothing extra.

From Python, the same thing as a structure:

```python
detail = atlas.describe("/data/collection", "2024-01")
detail["dimensions"]                                    # {'lat': 4, 'lon': 6}
detail["coordinates"]                                   # ['lat', 'lon']
{a["name"]: a["stats"] for a in detail["arrays"]}
```

## Remove

```bash
$ atlas rm /data/collection 2024-02
removed 1: 2024-02
2 dataset(s) remain
```

That wrote a small `deleted.mask` beside the container. The container itself is
untouched, so nothing was reclaimed and no ordinal moved. Rebuild the
collection to get the space back.

## Against a bucket

Every command takes a URL in place of a path:

```bash
atlas create /data/nc s3://my-bucket/collections/2024 --region eu-west-1
atlas ls s3://my-bucket/collections/2024 --region eu-west-1
```

Needs `pip install "atlas-python[cloud]"`. See
[Cloud storage](guides/cloud-storage.md).

## Reading the data

Not from Python — array values are read through the Rust API:

```rust
let atlas = Atlas::open_path("/data/collection").await?;
let ds = atlas.dataset("2024-01")?;
let window = ds.read_array::<f32>("temperature", vec![0, 0], vec![2, 3]).await?;
```

See [Reading data](guides/reading-data.md) for why, and what to do if you need
values in Python.

## Next

- [**The `atlas` command**](cli.md) — every flag
- [**Creating a collection**](guides/creating.md) — chunking, errors, memory
- [**Inspecting a collection**](guides/inspecting.md) — what the metadata holds
- [**Removing datasets**](guides/removing.md) — the mask, and what it costs
