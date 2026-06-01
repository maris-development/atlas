# Benchmarks: atlas vs netCDF vs zarr

A workload-focused comparison of atlas against the two alternatives
xarray-using folks reach for: **netCDF4** (the C-library binding) and **zarr v3**.

## Workload

Choose with `--case <name>`. Default is `sensors` (the original
sensor-fleet behavior). Each case is a self-contained spec: variables,
dtypes, shape, dim names, coord types, attrs, and optional chunk shape.

| Case      | Variables                                    | Shape (dims)                            | Chunks | ~Bytes / dataset | Notes |
|---|---|---|---|---:|---|
| `sensors` (default) | temperature/pressure/humidity (f32/f64/f32) | `(24,)` on `(time,)`             | single chunk | ~480 B | Hourly weather-station fleet |
| `gridded`           | temperature/pressure/humidity (f32/f64/f32) | `(100, 100, 48)` on `(lon, lat, time)` | `(50, 50, 24)` | ~1.8 MB | Geophysical-style grid; scale `--datasets` down |
| `profile`           | temperature/salinity (f32/f32)              | `(50, 168)` on `(depth, time)`        | single chunk | ~67 KB | Oceanographic time × depth cast |

Each dataset's values are deterministic from a seeded
`numpy.random.default_rng(idx)`, so the same `(idx, case)` pair produces
bit-identical input across backends — storage and read times reflect format
choices, not data.

### Why chunking matters for fairness

The `gridded` case writes with realistic chunks `(50, 50, 24)` — 8 chunks per
variable, ~120 KiB raw each. The default `--slice-fraction 0.25` reads
`(0:25, 0:25, 0:12)` which fits inside one chunk per dim, so all three
backends decompress 1/8 of each variable per dataset (with slice push-down)
instead of the full chunk. **Without chunking, every backend is forced to
decompress the full 1.92 MB volume**, which inflates everyone's read times
roughly uniformly but gives an unrepresentative picture.

`sensors` and `profile` are single-chunk because the per-dataset shapes are
already small enough that chunking would add more overhead than it saves.

## Read slice

`--slice-fraction F` (default `0.25`) reads the first `int(F * dim_len)`
elements of **every** dim from every variable. For `sensors` (`time=24`)
with `F=0.25` that's `time=[0:6]`. For `gridded` `(100, 100, 48)` with
`F=0.25` it's `(25, 25, 12)` per variable.

## Layout per backend

Default layouts are **N separate files/stores** for netCDF and zarr (the most
common production patterns). The `--netcdf-groups`, `--zarr-groups`, and
`--atlas-bulk` flags **add** extra rows alongside the defaults — they don't
replace them, so one run can compare multiple variants head-to-head.

| Backend | Always              | Added by `--atlas-bulk` | Added by `--netcdf-groups` | Added by `--zarr-groups` |
|---|---|---|---|---|
| atlas         | 1 store, N datasets (iterate `to_xarray` or `view.read_arrays`) | (use `read_array_across_stacked`) | — | — |
| atlas-bulk    | — | (this row) | — | — |
| netcdf        | N separate `.nc` files    | — | 1 `.nc` file w/ N netCDF4 groups | — |
| netcdf-groups | —                         | — | (this row)                       | — |
| zarr          | N separate `.zarr` stores | — | — | 1 zarr store w/ N groups |
| zarr-groups   | —                         | — | — | (this row)               |

The atlas row is always "1 store, N datasets" because atlas's multi-dataset
container is built in; there's no equivalent toggle. `--atlas-bulk` swaps the
read pattern (not the layout) — see "Read pattern" below.

## Read pattern

Read the slice from every dataset. Each backend uses its canonical
"many datasets" idiom:

- **atlas (default, no `--use-dask`)** — iterate `to_xarray(name).isel(slice).load()`.
  Returns a full xr.Dataset per dataset and slices in xarray. On chunked
  storage this pays per-chunk dask graph build overhead; slow on `gridded`.
- **atlas (with `--use-dask`)** — fast path: each dask worker calls
  `view.read_arrays(vars, start, shape)` per dataset, returning
  `dict[str, np.ndarray]` directly. Skips xr.Dataset and per-chunk dask graph
  build; the dask scheduler still parallelises across datasets.
- **atlas-bulk** (`--atlas-bulk`) — one `Atlas.read_array_across_stacked(var, names, start, shape)`
  call per variable. Returns a pre-stacked `(N, *slice_shape)` numpy array
  per variable; all per-dataset reads run on the tokio runtime with capped
  concurrency. Single PyO3 round-trip per variable for the entire 1000-dataset batch.
- **netCDF default** (N files) — `xr.open_mfdataset(files, combine="nested",
  concat_dim="station", parallel=True).isel(...).load()`. Slice push-down via
  dask graph optimization; `.load()` drives parallel chunk reads.
- **netCDF `--netcdf-groups`** — iterate `xr.open_dataset(path, group=name).isel(...).load()` N
  times. `open_mfdataset` doesn't apply to groups in a single file.
- **zarr default** (N stores) — `xr.open_mfdataset(stores, engine="zarr",
  combine="nested", concat_dim="station", parallel=True).isel(...).load()`.
  Same dask fan-out as the netCDF default.
- **zarr `--zarr-groups`** — iterate `xr.open_zarr(store, group=name).isel(...).load()` N
  times. `open_mfdataset` doesn't apply to groups in a single store.

## Dask (`--use-dask`)

The iteration-based read paths (atlas, netcdf-groups, zarr-groups) run
single-threaded by default — one dataset at a time. The `--use-dask` flag
opts each of these into `dask.delayed` parallelism so they're competitive
with `open_mfdataset`'s built-in dask fan-out.

When set, `--use-dask` does three things:

1. **Reads (iteration paths)** — atlas, netcdf-groups, zarr-groups: each
   per-dataset slice load is wrapped in `dask.delayed` and dispatched via
   `dask.compute(*, scheduler="threads")`. For atlas specifically, this
   path uses `view.read_arrays(vars, start, shape)` (the fast Rust dict
   path) rather than `to_xarray(...).isel(...).load()`, avoiding the
   xr.Dataset + dask graph overhead that dominates default `atlas`.
2. **Reads (mfdataset paths)** — default netcdf, default zarr: already use
   dask internally; the flag only constrains the thread pool to
   `--dask-workers` if specified.
3. **Writes** — `generate_dataset()` returns dask-backed `xr.DataArray`s
   (2 chunks along `time`). Each backend's xarray write triggers a dask
   compute that streams chunks. Atlas's `add_xr_dataset` already does this
   chunk-by-chunk; netCDF/zarr `to_*` handle it transparently.

Use `--dask-workers N` to set the thread count; defaults to dask's default
(typically CPU count).

```bash
# Single-threaded reads (default).
python atlas-python/benchmarks/bench_collection.py --datasets 1000

# Same workload + dask everywhere it applies.
python atlas-python/benchmarks/bench_collection.py --datasets 1000 --use-dask --dask-workers 4
```

Note: the threaded scheduler is intentional (not processes) — atlas,
netCDF, and zarr handles aren't picklable, and threads are fine for I/O-bound
work because each library releases the GIL during heavy lifting.

### `--use-dask` is workload-dependent

It helps when per-dataset decompression is the bottleneck and *hurts* when
per-dataset overhead is the bottleneck. Rough rule:

- **Big per-dataset arrays** (`gridded`): `--use-dask` is a clear win
  (default `atlas` 10s → atlas+dask 3.2s).
- **Tiny per-dataset arrays** (`profile`, `sensors`): `--use-dask` *slows
  things down* (default `atlas` 0.32s → atlas+dask 2.03s) — dask scheduler
  overhead exceeds the actual I/O work.

## Compression

Matched where each ecosystem supports it:

- **atlas**: `codec="zstd"` (array blocks), `meta_format="msgpack"` +
  `meta_compression="zstd"` (metadata).
- **zarr**: `zarr.codecs.ZstdCodec(level=3)` per variable, via xarray
  encoding.
- **netCDF**: `zlib=True, complevel=4` per variable. netCDF4-Python in most
  distributions doesn't ship zstd; documenting the asymmetry rather than
  working around it.

## Install

```bash
# From repo root
source .venv/bin/activate
pip install -e "atlas-python[bench]"
```

The `[bench]` extra adds `zarr>=3`, `numcodecs`, and `netCDF4`.

## Run

```bash
# Default sensors case, full N=1000 (atlas dominates by design on this one).
python atlas-python/benchmarks/bench_collection.py --datasets 1000

# Gridded case (100×100×48 per dataset, chunked). Add --atlas-bulk for the
# fast Rust bulk path, --use-dask for the per-dataset fast path.
python atlas-python/benchmarks/bench_collection.py \
    --case gridded --datasets 1000 --atlas-bulk --use-dask

# Profile case at full N (atlas wins on everything here).
python atlas-python/benchmarks/bench_collection.py --case profile --datasets 1000

# Tighten or loosen the read slice.
python atlas-python/benchmarks/bench_collection.py --case gridded --datasets 1000 --slice-fraction 0.1

# Subset of backends.
python atlas-python/benchmarks/bench_collection.py --backends atlas,zarr --datasets 500

# Add the groups-in-one-container variants as extra rows.
python atlas-python/benchmarks/bench_collection.py --datasets 500 --netcdf-groups --zarr-groups

# Everything together.
python atlas-python/benchmarks/bench_collection.py --case gridded --datasets 1000 \
    --atlas-bulk --netcdf-groups --zarr-groups --use-dask --dask-workers 8

# Keep the output dir for poking around.
python atlas-python/benchmarks/bench_collection.py --datasets 50 --keep-output
```

## Sample output

Apple Silicon laptop, on battery, 1000 datasets — relative not absolute:

### `--case gridded --datasets 1000 --atlas-bulk --use-dask`

```
Workload: 1000 datasets × case='gridded' × 3 vars
  shape per var : (lon=100, lat=100, time=48)
  read slice    : lon=[0:25], lat=[0:25], time=[0:12]  (fraction=0.25)
────────────────────────────────────────────────────────────────────
backend           write (s)   read slice (s)    storage (MiB)
────────────────────────────────────────────────────────────────────
atlas                60.414            3.209         6387.342
atlas-bulk           59.306            2.121         6387.342
zarr                 38.477            5.996         6391.955
netcdf              121.760           13.908         5596.219
```

### `--case profile --datasets 1000 --atlas-bulk`

```
Workload: 1000 datasets × case='profile' × 2 vars
  shape per var : (depth=50, time=168)
  read slice    : depth=[0:12], time=[0:42]  (fraction=0.25)
────────────────────────────────────────────────────────────────────
backend           write (s)   read slice (s)    storage (MiB)
────────────────────────────────────────────────────────────────────
atlas                 0.907            0.316           55.510
atlas-bulk            0.771            0.081           55.510
zarr                 11.426            4.067           61.622
netcdf                2.984            4.267           62.340
```

Actual numbers depend on hardware, filesystem, and the libraries' patch
versions. The top-level [README's Benchmarks section](../../README.md#benchmarks)
has the interpretive commentary; this README is for "what does each flag do"
reference.

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
  unfairly penalize netCDF for being multi-file. Symmetrically, the
  `--atlas-bulk` / `view.read_arrays` paths are atlas's canonical bulk
  APIs; they're the right comparison for "what's the fastest each library
  can do for this workload."
