# Benchmarks: atlas vs netCDF vs zarr

A workload-focused comparison of pyatlas against the two alternatives
xarray-using folks reach for: **netCDF4** (the C-library binding) and **zarr v3**.

## Workload

- **1000 datasets**, each one a small fleet-of-weather-stations style record:
  - 3 variables: `temperature` (float32), `pressure` (float64), `humidity` (float32)
  - 1 time coordinate (24 hourly i64-ns timestamps)
  - 1 dataset-level attr (`station_id`)
- Each dataset's values are deterministic from a seeded `numpy.random.default_rng(idx)`
  so storage and read times reflect format choices, not data.

## Layout per backend

Default layouts are **N separate files/stores** for netCDF and zarr (the most
common production patterns). The `--netcdf-groups` and `--zarr-groups` flags
**add** an additional row for the single-container variant alongside the
default — they don't replace it, so one run can compare both layouts head-to-head.

| Backend | Always              | Added by `--netcdf-groups` | Added by `--zarr-groups` |
|---|---|---|---|
| atlas         | 1 store, N datasets       | —                                 | —                          |
| netcdf        | N separate `.nc` files    | 1 `.nc` file w/ N netCDF4 groups | —                          |
| netcdf-groups | —                         | (this row)                        | —                          |
| zarr          | N separate `.zarr` stores | —                                 | 1 zarr store w/ N groups   |
| zarr-groups   | —                         | —                                 | (this row)                 |

The atlas row is always "1 store, N datasets" because atlas's multi-dataset
container is built in; there's no equivalent toggle.

## Read pattern

Read a slice `[0:6]` (the first 6 hours) of all 3 variables from every one of
the N datasets. Summed wall time across the collection.

- **atlas** — `Atlas.open(path)` once, then iterate `to_xarray(name)` N times.
  Cheap because the multi-dataset container is built into the store.
- **netCDF default** (N files) — `xr.open_mfdataset(files, combine="nested",
  concat_dim="station", parallel=True)`. Stacks all N files into one lazy
  `(station=N, time=24)` Dataset; `.load()` drives parallel reads via dask.
- **netCDF `--netcdf-groups`** — iterate `xr.open_dataset(path, group=name)` N
  times. `open_mfdataset` doesn't apply to groups in a single file.
- **zarr default** (N stores) — `xr.open_mfdataset(stores, engine="zarr",
  combine="nested", concat_dim="station", parallel=True)`. Same dask fan-out
  as the netCDF default.
- **zarr `--zarr-groups`** — iterate `xr.open_zarr(store, group=name)` N
  times. `open_mfdataset` doesn't apply to groups in a single store.

## Dask (`--use-dask`)

The iteration-based read paths (atlas, netcdf-groups, zarr-groups) run
single-threaded by default — one dataset at a time. The `--use-dask` flag
opts each of these into `dask.delayed` parallelism so they're competitive
with `open_mfdataset`'s built-in dask fan-out.

When set, `--use-dask` does three things:

1. **Reads (iteration paths)** — atlas, netcdf-groups, zarr-groups: each
   per-dataset slice load is wrapped in `dask.delayed` and dispatched via
   `dask.compute(*, scheduler="threads")`.
2. **Reads (mfdataset paths)** — default netcdf, default zarr: already use
   dask internally; the flag only constrains the thread pool to
   `--dask-workers` if specified.
3. **Writes** — `generate_dataset()` returns dask-backed `xr.DataArray`s
   (2 chunks along `time`). Each backend's xarray write triggers a dask
   compute that streams chunks. Atlas's `add_xr_dataset` already does this
   chunk-by-chunk; netCDF/zarr to_* handle it transparently.

Use `--dask-workers N` to set the thread count; defaults to dask's default
(typically CPU count).

```bash
# Single-threaded reads (default).
python pyatlas/benchmarks/bench_collection.py --datasets 1000

# Same workload + dask everywhere it applies.
python pyatlas/benchmarks/bench_collection.py --datasets 1000 --use-dask --dask-workers 4
```

Note: the threaded scheduler is intentional (not processes) — pyatlas,
netCDF, and zarr handles aren't picklable, and threads are fine for I/O-bound
work because each library releases the GIL during heavy lifting.

## Compression

Matched where each ecosystem supports it:

- **atlas**: `codec="zstd"` (array blocks), `meta_format="msgpack"` +
  `meta_compression="zstd"` (metadata).
- **zarr**: `numcodecs.Zstd(level=3)` per variable, via xarray
  encoding (`compressors=(numcodecs.Zstd(level=3),)`).
- **netCDF**: `zlib=True, complevel=4` per variable. netCDF4-Python in most
  distributions doesn't ship zstd; documenting the asymmetry rather than
  working around it.

## Install

```bash
# From repo root
source .venv/bin/activate
pip install -e "pyatlas[bench]"
```

The `[bench]` extra adds `zarr>=3`, `numcodecs`, and `netCDF4`.

## Run

```bash
# Smoke run — fast.
python pyatlas/benchmarks/bench_collection.py --datasets 50

# Full run — the headline number.
python pyatlas/benchmarks/bench_collection.py --datasets 1000 --repeats 3

# Subset of backends.
python pyatlas/benchmarks/bench_collection.py --backends atlas,zarr --datasets 500

# Switch netcdf and/or zarr to their groups-in-one-container layouts.
python pyatlas/benchmarks/bench_collection.py --datasets 500 --netcdf-groups --zarr-groups

# Keep the output dir for poking around.
python pyatlas/benchmarks/bench_collection.py --datasets 50 --keep-output
```

## Sample output

```
Workload: 1000 datasets × 3 variables × 24 time elements, read slice [0:6]
────────────────────────────────────────────────────────────────────
backend       write (s)   read slice (s)    storage (MiB)
────────────────────────────────────────────────────────────────────
atlas             ...              ...              ...
netcdf            ...              ...              ...
zarr              ...              ...              ...
```

Actual numbers depend hard on hardware, filesystem, and the libraries'
patch versions — treat this as a "is atlas roughly competitive on this
workload?" check rather than a publication-quality result.

## Non-goals / honest caveats

- **No OS-cache eviction between runs.** Measures warm-cache repeat-query
  performance, which is what analytic workloads see in practice but not
  the cold-cache number.
- **No statistical tests.** `--repeats N` reports the mean; we don't compute
  confidence intervals.
- **Local filesystem only.** Atlas's biggest potential win is high-latency
  object stores where shared metadata + shared array files matter more —
  that's a different benchmark.
- **netCDF4 vs h5netcdf.** Using the C-library binding (`netCDF4`) — the more
  common production choice. `h5netcdf` would produce different numbers.
- **Read patterns differ on purpose.** Each backend uses its canonical
  "many datasets" idiom — see [Read pattern](#read-pattern) above. Forcing
  identical access patterns (e.g. per-file `open_dataset` everywhere) would
  unfairly penalize netCDF for being multi-file.
