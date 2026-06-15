# Bulk reads

Atlas exposes four APIs that read multiple things in one PyO3 round-trip.
Each one trades a different chunk of Python-side overhead for raw Rust
throughput.

## The four APIs at a glance

| API | Returns | Use when |
|---|---|---|
| `Atlas.open_as_many_xarray_dataset(names, concat_dim, parallel=True)` | `xr.Dataset` of shape `(N, *original)` | You want xarray ergonomics on top of a fleet of identically-shaped datasets. |
| `Atlas.read_array_across(array, names, start, shape)` | `list[np.ndarray | None]` | Per-dataset numpy arrays; some datasets may not declare `array`. |
| `Atlas.read_array_across_stacked(array, names, start, shape)` | `np.ndarray` of shape `(N, *slice)` | You'll immediately stack — skip the `np.stack` copy. Errors if any listed dataset doesn't declare `array`. |
| `DatasetView.read_arrays(names, start, shape)` | `dict[str, np.ndarray | None]` | Many arrays out of **one** dataset (e.g. inside a dask worker). |

## Cross-dataset, one variable: `read_array_across` / `_stacked`

The classic "give me the same slice of `temperature` from every dataset"
query:

```python
names = atlas.list_datasets()                # ["sensor_000", "sensor_001", ...]

# Per-dataset list, with None for missing entries.
per_ds = atlas.read_array_across("temperature", names, start=[0], shape=[24])
# [np.ndarray, np.ndarray, None, np.ndarray, ...]

# Pre-stacked into one (N, *slice_shape) array; faster on the stacking path.
stacked = atlas.read_array_across_stacked("temperature", names, start=[0], shape=[24])
# stacked.shape == (len(names), 24)
```

What makes these fast:

- One PyO3 round-trip for N datasets, not N.
- One `RwLock::read` guard on the shared physical file for the whole
  batch — atlas's defining shared-array layout makes this cheap.
- N per-dataset reads dispatched concurrently on the tokio runtime via a
  `JoinSet` capped at `num_cpus` in-flight tasks. The GIL is released for
  the duration.

`read_array_across_stacked` additionally **pre-allocates** the output
buffer in Rust and writes each row in as the per-dataset task completes,
so you skip the ~N × per_dataset_size of memory copies that
`np.stack(read_array_across(...))` would do. The trade-off: it errors if
any listed dataset doesn't declare the array (there's no positional
"missing" sentinel in the stacked representation).

## Cross-dataset, all variables: `open_as_many_xarray_dataset`

This is the atlas-native equivalent of `xr.open_mfdataset(...)`:

```python
combined = atlas.open_as_many_xarray_dataset(
    ["jan_2024", "feb_2024", "mar_2024"],
    concat_dim="month",
)
# Each variable comes back shape (3, *original), eager numpy.
# Coords + dataset-level attrs are taken from the first dataset.
```

Internally `open_as_many_xarray_dataset` calls `Atlas.read_array_across` once per
variable, so you get the same shared-file-handle / tokio fan-out as the
single-variable bulk APIs but with the xarray ergonomics.

Constraints:

- Variable names and dtypes must match across every listed dataset.
- The result is **eager** numpy. Wrap with `.chunk(...)` downstream if you
  need dask laziness.
- The `parallel` parameter is accepted for API compatibility but the bulk
  path is always taken; it doesn't switch implementations.

## Per-dataset multi-array: `view.read_arrays`

Inside a dask worker (or any hot loop over one dataset), the bottleneck is
usually the per-call `open_as_xarray_dataset` + dask-graph build, not the I/O. Skip
both:

```python
view = atlas.open_dataset("jan_2024")
result = view.read_arrays(["temperature", "pressure"], start=[0, 0], shape=[4, 8])
# {"temperature": np.ndarray, "pressure": np.ndarray}
```

`read_arrays` is what `bench_collection.py --use-dask` calls inside each
delayed task — the ~3–4× speedup over `open_as_xarray_dataset(name).isel(...).load()`
on chunked storage comes from skipping the per-chunk dask graph build.

Same start/shape applies to every array; missing arrays come back as
`None` (so this is safe to call against a dataset that may or may not
declare every variable).

## API picker (in rough order of speed)

- **Cross-dataset slice of the same vars across many datasets** →
  `Atlas.open_as_many_xarray_dataset` / `Atlas.read_array_across_stacked` (the
  `atlas-bulk` benchmark path; one Rust call per variable).
- **Per-dataset slice reads inside a dask worker** →
  `view.read_arrays(vars, start, shape)` (returns `dict[str, np.ndarray]`;
  skips the xr.Dataset + per-chunk dask graph).
- **Natural xarray code, per dataset** → `open_as_xarray_dataset(name).isel(...).load()`.
  Most ergonomic, but pays per-chunk dask graph build overhead on chunked
  storage.

See [Benchmarks](../benchmarks.md) for what those differences look like
on real workloads.
