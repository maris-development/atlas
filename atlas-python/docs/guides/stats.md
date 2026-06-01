# Stats and scans

Atlas computes and persists per-array summary statistics on every flush.
You can scan thousands of datasets without reading any raw chunks.

## What's tracked

`view.array_stats(name)` returns a dict (or `None` if the array doesn't
exist in this dataset, or hasn't been flushed yet):

| Key | Meaning |
|---|---|
| `row_count` | Number of written cells (logical, after collapsing chunked writes). |
| `null_count` | Number of cells equal to the array's `fill_value`. |
| `min` | Minimum value across all written cells. |
| `max` | Maximum value across all written cells. |

Stats are populated **after** [`atlas.flush()`](durability.md). Between a
`define_array` / `write_array` and the next flush, `array_stats(name)`
returns `None`.

```python
ds.write_array("readings", start=[0], data=values)
ds.array_stats("readings")        # None — not flushed yet
atlas.flush()
ds.array_stats("readings")        # {"row_count": ..., "null_count": ..., "min": ..., "max": ...}
```

## Cross-dataset scans

Because stats are persisted alongside the schema in `atlas.json`, scanning
them across the entire store is essentially free — no chunk decompression,
no array file I/O:

```python
atlas = atlas.Atlas.open("/tmp/store")

peak_sensor, peak_max = "", -float("inf")
for name in atlas.list_datasets():
    stats = atlas.open_dataset(name).array_stats("readings")
    if stats and stats["max"] > peak_max:
        peak_sensor, peak_max = name, stats["max"]

print(peak_sensor, peak_max)
```

This pattern is what makes "find the dataset with the most extreme value"
queries scale to fleets of 10 000 datasets — the work is bounded by the
metadata-file size, which is a few MB even at that scale (see
[Codecs and metadata](codecs-and-meta.md) for msgpack / compression
options if you want it smaller).

[`examples/06_stats_scan.py`](https://github.com/maris-development/atlas/blob/main/atlas-python/examples/06_stats_scan.py)
runs exactly this pattern across 32 sensor datasets.

## How `null_count` works with `fill_value`

A "null" in atlas is **a cell whose stored value equals the array's
`fill_value`**. Both *unwritten* cells (which read back as the fill value)
and *written* cells whose value happens to match contribute to the count.

For float arrays the natural choice is `fill_value=float("nan")` — NaN
doesn't compare equal to itself by IEEE rules, but atlas special-cases
NaN, so a written NaN counts as a null.

```python
ds.define_array("temp", dtype="float32", ..., fill_value=float("nan"))
ds.write_array("temp", start=[0], data=np.array([1.0, np.nan, 3.0], dtype=np.float32))
atlas.flush()
ds.array_stats("temp")
# {"row_count": 3, "null_count": 1, "min": 1.0, "max": 3.0}
```

For integer arrays, pick a sentinel value (`-1`, `np.iinfo(dtype).min`,
etc.) — any *written* cell equal to it will also be counted as null. Pick
a value that can't appear in real data.

## What stats don't include

- **No mean, sum, stdev, or quantiles.** If you need those, read the array
  and compute them with numpy / dask. Atlas's stats are designed to be
  cheap-on-write and cheap-on-read, not a full analytics engine.
- **No per-dimension reductions.** `min` / `max` are scalars over the
  whole array, not per-row / per-column.
- **String / timestamp arrays:** `row_count` and `null_count` are
  populated; `min` / `max` may be omitted for dtypes where they don't
  have a natural meaning.
