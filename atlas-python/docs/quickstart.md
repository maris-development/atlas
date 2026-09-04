# Quickstart

A directory of NetCDF files in, one collection out, then a look at what landed.

## Build

```bash
atlas create /data/nc /data/collection
```

```text
Writing /data/collection
  2024-01.nc
  2024-02.nc
  2024-03.nc
3 dataset(s) written to /data/collection
```

Each file becomes one dataset, named after the file. The result is one file:

```bash
$ ls /data/collection
data.atlas
```

Nothing there was readable until the last file landed. A failure part-way
leaves no collection, and not a partial one.

From Python:

```python
import atlas

atlas.create("/data/nc", "/data/collection")
```

## Look

```bash
$ atlas ls /data/collection
2024-01.nc
2024-02.nc
2024-03.nc
```

```bash
$ atlas info /data/collection
collection /data/collection
  format version    8
  created           2026-08-31T08:32:57Z
  codec             zstd
  container size    5.4 KiB
  datasets          3
  interned schemas  1
  distinct arrays   4
      lat          count=12  min=0.0  max=3.0
      lon          count=18  min=0.0  max=5.0
      station      count=12  min="a"  max="d"
      temperature  count=72  min=1.0  max=26.0
```

Both come from one range read of the container tail. Three datasets and a
million cost the same.

`interned schemas 1` means all three months declare the same arrays, so that
schema is stored once.

The statistics cover the whole collection. Each month holds 24 temperature
elements, so the three months together hold 72. The minimum comes from January.
The maximum comes from March. Removed datasets do not count.

## Inspect one dataset

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

The shape follows `ncdump -h`. It adds the statistics of each array write. The
footer holds those too, so to print them costs nothing extra.

From Python, the same thing as a structure:

```python
detail = atlas.describe("/data/collection", "2024-01.nc")
detail["dimensions"]                                    # {'lat': 4, 'lon': 6}
detail["coordinates"]                                   # ['lat', 'lon']
{a["name"]: a["stats"] for a in detail["arrays"]}
```

## Remove

```bash
$ atlas rm /data/collection 2024-02.nc
removed 1: 2024-02.nc
2 dataset(s) remain
```

That writes a small `deleted.mask` beside the container. The container does not
change, so this reclaims nothing and moves no ordinal. Rebuild the collection
to get the space back.

## Against a bucket

Every command takes a URL in place of a path:

```bash
atlas create /data/nc s3://my-bucket/collections/2024 --region eu-west-1
atlas ls s3://my-bucket/collections/2024 --region eu-west-1
```

This needs `pip install "atlas-python[cloud]"`. See
[Cloud storage](guides/cloud-storage.md).

## Reading the data

Not from Python. The Rust API reads array values:

```rust
let atlas = Atlas::open_path("/data/collection").await?;
let ds = atlas.dataset("2024-01.nc")?;
let window = ds.read_array::<f32>("temperature", vec![0, 0], vec![2, 3]).await?;
```

See [Reading data](guides/reading-data.md) for the reason, and for what to do
when you need values in Python.

## Next

- [**The `atlas` command**](cli.md). Every flag
- [**Creating a collection**](guides/creating.md). Chunking, errors, memory
- [**Inspecting a collection**](guides/inspecting.md). What the metadata holds
- [**Removing datasets**](guides/removing.md). The mask, and what it costs
