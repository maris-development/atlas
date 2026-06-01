# Benchmarks

A reproducible comparison against `netCDF4` and Zarr v3 lives in
[`atlas-python/benchmarks/`](https://github.com/robinskil/atlas/tree/main/atlas-python/benchmarks).
The harness writes the **same** deterministic data through each backend,
then measures write time, slice-read time, and on-disk size. Each backend
uses its canonical "many datasets" layout:

- **atlas** — one store with N datasets.
- **netcdf+dask** — N separate `.nc` files, read via
  `xr.open_mfdataset(parallel=True, ...).isel(...).load()`. Dask threaded
  scheduler under the hood — the canonical xarray-on-netCDF pattern.
- **zarr+dask** — N separate `.zarr` stores, also via `open_mfdataset` —
  the canonical xarray-on-zarr pattern.

For atlas the sweep shows two rows — the two parallel paths people
actually reach for:

- **atlas+dask** — `view.read_arrays(...)` wrapped in `dask.delayed`
  across datasets (opt in with `--use-dask`).
- **atlas-bulk** — `Atlas.read_array_across_stacked(...)`: one PyO3 call
  per variable, parallelised in Rust on the tokio runtime, returns eager
  numpy (opt in with `--atlas-bulk`).

The default (serial) atlas read path — iterate
`view.to_xarray(name).isel(...).load()` — isn't in the chart; it scales
linearly with N and is fine for small fleets, but at the scales below
(100+ datasets) you should already be reaching for one of the parallel
paths. `--netcdf-groups` / `--zarr-groups` add layouts that use one file
with N internal groups, and `--netcdf-no-dask` / `--zarr-no-dask` add
serial-loop variants of the netcdf and zarr reads — useful for isolating
dask's contribution to those paths, but not part of the default sweep.

## Sweep results

Each chart sweeps **datasets ∈ {100, 500, 1000}** for one case at a fixed
variable count, run on a typical Apple Silicon laptop. Bars are grouped
by dataset count and color-coded per backend: orange = `netcdf+dask`,
red = `zarr+dask`, purple = `atlas+dask`, teal = `atlas-bulk`.

### `--case profile` (4 variables)

`(50, 168)` per variable × 4 variables, slice 25%. Overhead-dominated; the
per-dataset work is small enough that file-open / metadata cost dominates
the read path for netCDF and zarr.

![Profile read-slice sweep — bars grouped by dataset count (100 / 500 / 1000), one color per backend](assets/bench_profile.svg)

| Datasets | netcdf+dask (s) | zarr+dask (s) | atlas+dask (s) | atlas-bulk (s) |
|---:|---:|---:|---:|---:|
| 100 | 0.564 | 0.543 | 0.023 | **0.013** |
| 500 | 3.901 | 2.734 | 0.081 | **0.064** |
| 1000 | 7.409 | 5.879 | 0.148 | **0.122** |

atlas-bulk at 0.12 s wins by **~48× over zarr+dask** at 1000 datasets,
and the gap *widens* with N because per-dataset open cost compounds for
the netcdf/zarr paths. atlas+dask lands within ~20% of atlas-bulk —
per-dataset overhead is already tiny here, so the extra fan-out from
`dask.delayed` mostly just amortises the Python loop.

### `--case gridded` (3 variables)

`(100, 100, 48)` per variable × 3 variables, chunks `(50, 50, 24)`, slice
25%. Decompression-dominated; all three backends push the slice down to
chunk-level reads.

![Gridded read-slice sweep — bars grouped by dataset count (100 / 500 / 1000), one color per backend](assets/bench_gridded.svg)

| Datasets | netcdf+dask (s) | zarr+dask (s) | atlas+dask (s) | atlas-bulk (s) |
|---:|---:|---:|---:|---:|
| 100 | 0.873 | 0.506 | 0.233 | **0.220** |
| 500 | 6.025 | 2.711 | 1.410 | **1.114** |
| 1000 | 14.037 | 5.968 | 3.763 | **2.031** |

`atlas-bulk` is the only path whose lead grows with N — at 1000 datasets
it's **2.9× faster than zarr+dask** and **6.9× faster than netcdf+dask**
on slice reads. **atlas + `--use-dask`** is the right pick when you need
dask-graph composability (downstream `xr` operations stay lazy): at 1000
datasets it's still **1.6× faster than zarr+dask** and sits ~1.9× behind
atlas-bulk because per-dataset decompression is the bottleneck and the
threaded `dask.delayed` fan-out has more overhead than tokio's
`JoinSet`.

## TL;DR

- **Large per-dataset workloads (`gridded`, realistic chunking + slice
  push-down):** atlas-bulk beats `zarr+dask` (the canonical xarray
  pattern) by **2.9×** on slice reads at 1000 datasets and `netcdf+dask`
  by **6.9×** — and the lead grows with N. zarr+dask remains the
  fastest writer for chunked grids.
- **Small per-dataset workloads (`profile`):** atlas-bulk reads **~48×
  faster than `zarr+dask`** at 1000 datasets (0.12 s vs 5.88 s) and the
  gap *widens* with dataset count because per-dataset open cost
  compounds for everyone else. Per-dataset overhead is atlas's home
  court.
- **`atlas + --use-dask`** picks up most of atlas-bulk's win without
  giving up the dask graph: 1.6–4× faster than `zarr+dask` across the
  sweep, and the natural choice when downstream `xr` code needs to stay
  lazy (a `to_xarray(...).isel(...)` slice chain composes with the
  upstream graph, where `atlas-bulk` returns eager numpy). Reach for
  serial `atlas` only when the fleet is small enough that the dask
  fan-out overhead doesn't pay back.

## Reproducing these numbers

The sweep above was produced with:

```bash
# Profile, 4 variables — captures atlas-bulk, netcdf+dask, zarr+dask.
# (The `atlas` row in the output is the serial baseline, omitted from the chart.)
for n in 100 500 1000; do
    python atlas-python/benchmarks/bench_collection.py --case profile --n-vars 4 --atlas-bulk --datasets $n
done

# Gridded, case-default 3 variables.
for n in 100 500 1000; do
    python atlas-python/benchmarks/bench_collection.py --case gridded --atlas-bulk --datasets $n
done

# The atlas+dask row comes from a second pass that flips --use-dask. The
# other backends are unaffected by --use-dask on the read path, so we
# limit --backends to atlas to keep the run cheap.
for n in 100 500 1000; do
    python atlas-python/benchmarks/bench_collection.py --case profile --n-vars 4 --backends atlas --use-dask --datasets $n
    python atlas-python/benchmarks/bench_collection.py --case gridded --backends atlas --use-dask --datasets $n
done
```

To break netcdf+dask / zarr+dask down further into "what part is the
dask threading vs the file format itself", `--netcdf-no-dask` and
`--zarr-no-dask` add serial-loop rows alongside the default
`open_mfdataset` rows in a single run.

`--n-vars N` pads (or trims) the case's default variable list — pass
copies of the originals with `_2`/`_3`/… suffixes — so you can stress the
same case at any variable count without editing the case definition.

The charts above are regenerated from the recorded numbers via:

```bash
python atlas-python/benchmarks/generate_bench_charts.py    # writes docs/assets/bench_*.svg
```

Update the `RESULTS` dict in that script and re-run after a fresh sweep.

## API picker for reads (in rough order of speed)

- **Cross-dataset slice of the same vars across many datasets** →
  [`Atlas.to_xarray_many`](reference/atlas.md) /
  [`Atlas.read_array_across_stacked`](reference/atlas.md) (the
  `atlas-bulk` path; one Rust call per variable).
- **Per-dataset slice reads inside a dask worker** →
  [`view.read_arrays(vars, start, shape)`](reference/dataset-view.md)
  (returns `dict[str, np.ndarray]`; skips xr.Dataset + per-chunk dask
  graph). This is what `bench_atlas` with `--use-dask` does internally.
- **Natural xarray code** → `to_xarray(name).isel(...).load()`. Most
  ergonomic but pays per-chunk dask graph build overhead on chunked
  storage.

See [Bulk reads](guides/bulk-reads.md) for the full guide on which API
to reach for.

## Install and run

```bash
pip install -e "atlas-python[bench]"      # adds zarr + netCDF4 to your env
python atlas-python/benchmarks/bench_collection.py --case gridded --datasets 1000
```

The harness accepts a stack of flags — `--case sensors|gridded|profile`,
`--use-dask`, `--atlas-bulk`, `--netcdf-groups`, `--zarr-groups`,
`--slice-fraction`, `--dask-workers`, … — fully documented in
[`atlas-python/benchmarks/README.md`](https://github.com/robinskil/atlas/blob/main/atlas-python/benchmarks/README.md).

## Caveats

- These numbers reflect **local-filesystem** performance on a single laptop.
  On cloud object storage (where zarr is a natural fit) the picture flips
  for some workloads — atlas doesn't target that environment yet.
- The `--use-dask` row is the threaded dask scheduler. Distributed /
  multiprocessing scheduling isn't supported by atlas's lazy-read path
  this release (see [Dask streaming and lazy reads](guides/dask.md)).
- "Storage" is the directory size after a final `flush()` /
  `atlas.close()`. atlas's metadata file shows up at 0.5–2 MB on these
  workloads; switch to msgpack + zstd via
  [`meta_format` / `meta_compression`](guides/codecs-and-meta.md) if the
  metadata is a meaningful fraction of your total size.
