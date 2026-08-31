# vs Zarr / netCDF

Atlas lives in the same neighbourhood as [Zarr v3](https://zarr.dev/) and
[netCDF4](https://www.unidata.ucar.edu/software/netcdf/): all three store
labelled N-D arrays with chunking and compression, and all three are reachable
from xarray. They differ in **layout**, and that is what makes atlas suited to
one class of workload and not others.

## What each one is, in one line

| | Unit of storage | "Many datasets" layout |
|---|---|---|
| **netCDF4** | One self-describing `.nc` file per dataset | N files, or one file with N internal groups |
| **Zarr v3** | One directory of chunk files per array | N stores, or one store with N groups |
| **atlas** | One immutable file holding **N** datasets | N datasets in one file, by construction |

Atlas is the only one whose natural unit is *a fleet of related datasets*, which
is where the comparison gets interesting.

## The difference that matters: cataloguing

A collection's entire metadata — every dataset name, every array's dtype, shape,
chunking, and every attribute — lives in one footer at the end of one file.
Opening reads it in a single range request.

```bash
$ atlas info s3://bucket/collections/2024      # 1 request
$ atlas ls   s3://bucket/collections/2024      # 1 request
$ atlas show s3://bucket/collections/2024 jan  # 1 request
```

Each of those reads the footer and answers everything from it — dataset names,
every array's type and shape, every attribute, and the statistics recorded when
each array was written.

The same catalogue over N netCDF files means opening N files; over N Zarr stores
or groups it means fetching N sets of `zarr.json`. On object storage, where each
of those is a round trip, that is the whole cost of the operation.

This is what atlas is for: you have thousands of similar datasets, you need to
know what is in them, and you need that answer quickly and often.

## Where atlas is a poor fit

Worth being blunt, because the constraint is real:

**You need to modify data.** Atlas collections are immutable. Zarr is designed
for incremental and concurrent writes — regions updated in place, arrays resized,
new variables appended. If your workflow appends daily to a growing store, use
Zarr.

**You have one big dataset, not many.** A single 10 TB array in one collection
gets nothing from the format. Zarr's chunk-per-object layout parallelises writes
across workers in a way atlas's single-stream writer does not.

**You need the data back in Python.** Atlas reads array values from Rust only.
Zarr and netCDF hand you numpy and dask directly. If the analysis is in Python
and the storage layer needs to feed it, that is a real cost.

**Your data is not in NetCDF.** NetCDF is the only ingest route atlas offers
from Python. Zarr will take anything you can put in a numpy array.

**You need an ecosystem.** Zarr and netCDF have decades of tooling, viewers,
converters, and institutional familiarity. Atlas has this documentation.

## Where it fits well

**A published, versioned archive.** Immutability is a feature when the data is
final: one file to copy, checksum, or mirror, with no partially-written state to
detect.

**A catalogue served over the network.** Metadata for a whole collection in one
request is hard to beat when the alternative is N round trips.

**Ingest from many source files.** One pass over a directory of NetCDF files
produces one artefact, with dask streaming keeping memory flat.

**Cold object storage.** One object per collection rather than thousands means
fewer requests, no listing costs, and no small-object overhead.

## Layout, concretely

```text
netCDF                     Zarr                        atlas
──────                     ────                        ─────
jan_2024.nc                jan_2024.zarr/              collection/
feb_2024.nc                  zarr.json                   data.atlas
mar_2024.nc                  temperature/                deleted.mask
…                              c/0/0 …                   (that's all)
                           feb_2024.zarr/
1000 files                   …                         1 file
                           1000 stores,
                           many objects each
```

Atlas's one file is not a compressed archive — it is randomly addressable. A
segment is a complete, self-describing `array-format` file at a known byte
offset, so reading one dataset's chunk touches only that range.

## Compression

All three compress per chunk or per block, with similar codecs (zstd, lz4).
Expect comparable ratios on the same data; the difference between the formats
is layout and access cost, not bytes saved.

## Reading atlas data from Rust

```rust
let atlas = Atlas::open_path("/data/collection").await?;
let ds = atlas.dataset("2024-01")?;
let window = ds.read_array::<f32>("temperature", vec![1, 3], vec![2, 2]).await?;
```

Only the chunks the window overlaps are fetched — the same partial-read
behaviour Zarr gives you, at the same granularity.
