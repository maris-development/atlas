# vs Zarr / netCDF

Atlas sits near [Zarr v3](https://zarr.dev/) and
[netCDF4](https://www.unidata.ucar.edu/software/netcdf/). All three store
labelled N-D arrays with chunking and compression, and xarray reaches all
three. They differ in **layout**. That difference suits atlas to one class of
work, and not to others.

## What each one is, in one line

| | Unit of storage | "Many datasets" layout |
|---|---|---|
| **netCDF4** | One self-describing `.nc` file per dataset | N files, or one file with N internal groups |
| **Zarr v3** | One directory of chunk files per array | N stores, or one store with N groups |
| **atlas** | One immutable file holding **N** datasets | N datasets in one file, by construction |

Atlas is the one format whose natural unit is *a fleet of related datasets*.
That is where the comparison matters.

## The difference that matters is the catalogue

One footer at the end of one file holds the whole metadata of a collection.
That is every dataset name, the dtype, shape, and chunking of every array, and
every attribute. An open reads it in one range request.

```bash
$ atlas info s3://bucket/collections/2024      # 1 request
$ atlas ls   s3://bucket/collections/2024      # 1 request
$ atlas show s3://bucket/collections/2024 jan  # 1 request
```

Each of those reads the footer, and answers from it alone. The dataset names,
the type and shape of every array, every attribute, and the statistics of each
array write.

The same catalogue over N netCDF files means N file opens. Over N Zarr stores
or groups it means N fetches of `zarr.json`. On object storage each of those is
a round trip, and those trips are the whole cost.

This is what atlas is for. You hold thousands of similar datasets. You need to
know what is in them, and you need that answer fast and often.

## Where atlas is a poor fit

State this plainly, because each constraint is real:

**You must change data.** An atlas collection is immutable. Zarr serves an
incremental and concurrent write. It updates a region in place, resizes an
array, and appends a new variable. Use Zarr when your workflow appends to a
store every day.

**You hold one large dataset, not many.** One 10 TB array in one collection
gains nothing from this format. The chunk-per-object layout of Zarr spreads a
write across workers. The single-stream writer of atlas does not.

**You need the data back in Python.** Atlas reads an array value from Rust
only. Zarr and netCDF hand you numpy and dask. That is a real cost when the
analysis runs in Python, and the storage layer must feed it.

**Your data is not NetCDF.** NetCDF is the one ingest route atlas offers from
Python. Zarr takes anything a numpy array can hold.

**You need an ecosystem.** Zarr and netCDF hold decades of tools, viewers,
converters, and shared knowledge. Atlas holds this documentation.

## Where it fits well

**A published, versioned archive.** Immutability helps when the data is final.
There is one file to copy, to checksum, or to mirror, and no half-written state
to find.

**A catalogue over the network.** The metadata of a whole collection in one
request beats N round trips.

**Ingest from many source files.** One pass over a directory of NetCDF files
makes one artefact. The dask stream keeps the memory flat.

**Cold object storage.** One object per collection, instead of thousands, means
fewer requests, no listing cost, and no small-object overhead.

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

The one atlas file is no compressed archive. It addresses at random. A segment
is a complete `array-format` file that describes itself, at a known byte
offset. A read of one chunk of one dataset therefore touches that range alone.

## Compression

All three compress per chunk or per block, with the same kind of codec, such as
zstd or lz4. Expect a similar ratio on the same data. The formats differ in
layout and access cost, not in bytes saved.

## Reading atlas data from Rust

```rust
let atlas = Atlas::open_path("/data/collection").await?;
let ds = atlas.dataset("2024-01.nc")?;
let window = ds.read_array::<f32>("temperature", vec![1, 3], vec![2, 2]).await?;
```

This fetches the chunks the window overlaps, and no more. Zarr gives the same
partial read, at the same size.
