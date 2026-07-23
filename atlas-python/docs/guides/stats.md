# Stats and scans

Atlas computes and persists per-array summary statistics on every flush.
You can scan thousands of datasets without reading any raw chunks.

## What's tracked

`view.array_stats(name)` returns a dict (or `None` if the array doesn't
exist in this dataset, or hasn't been flushed yet):

| Key | Meaning |
|---|---|
| `row_count` | The array's full logical size — the product of its shape. |
| `null_count` | Cells equal to the `fill_value`, **plus every cell never written**. |
| `min` | Minimum across written, non-fill cells. |
| `max` | Maximum across written, non-fill cells. |

`row_count` is capacity, not how much you wrote. An array declared with shape
`[10]` reports `row_count == 10` even if you only ever wrote 5 cells — the
other 5 show up in `null_count`, since they read back as the fill value:

```python
ds.define_array("t", dtype="int32", dims=["i"], shape=[10], chunk_shape=[5], fill_value=0)
ds.write_array("t", start=[0], data=np.array([1, 2, 3, 4, 5], dtype=np.int32))
atlas.flush()
ds.array_stats("t")
# {"row_count": 10, "null_count": 5, "min": 1, "max": 5}
```

So `row_count - null_count` is the count of real values.

Stats are populated **after** [`atlas.flush()`](durability.md). Between a
`define_array` / `write_array` and the next flush, `array_stats(name)`
returns `None`.

```python
ds.write_array("readings", start=[0], data=values)
ds.array_stats("readings")        # None — not flushed yet
atlas.flush()
ds.array_stats("readings")        # {"row_count": ..., "null_count": ..., "min": ..., "max": ...}
```

Stats live in a per-array `{array}/data.stats` sidecar, next to that array's
`data.af` — **not** in `atlas.json`, which holds only schemas.

## Cross-dataset scans: the pruning index

`array_stats` answers for one array in one dataset. To ask "which datasets
could possibly match this?" across a whole collection, use the **pruning
index** — a flattened, columnar view with one row per dataset:

```python
store = atlas.Atlas.open("/tmp/store")

idx = store.pruning_index(arrays=["readings"])
col = idx["columns"]["readings"]

# Vectorized: no Python loop over datasets.
ok   = col["present"] & col["stats_valid"] & idx["live"]
hits = np.where(ok & (col["max"] > 25.0))[0]
candidates = [idx["datasets"][i] for i in hits]
```

Each column is a dict of numpy arrays, all the same length as `idx["rows"]`:

| Key | Meaning |
|---|---|
| `present` | The dataset at this row declares the array/attribute. |
| `stats_valid` | `min`/`max` are meaningful here. |
| `min`, `max` | The per-dataset range. |
| `row_count`, `null_count` | As above — **both 0** where `present` is `False`. |

Row `i` is the dataset at ordinal `i` (`store.dataset_row(name)`), and
`idx["datasets"][i]` maps back to its name. Datasets that don't declare the
array are **explicit gaps** rather than missing entries, so columns stay
aligned across the collection no matter how heterogeneous it is. A dataset
without the array contributes `row_count == 0`, which is a real answer you can
sum or filter on directly.

`present` and `stats_valid` are separate on purpose: `stats_valid` is `False`
where there is no usable range even though the dataset does declare the array
— `List` dtypes, which carry no statistics at all, and rows written but not
yet flushed. Their `row_count` / `null_count` are still valid, so check
`present` for "does it have this?" and `stats_valid` before touching
`min`/`max`.

Deleted datasets keep their row and are masked by `idx["live"]`. Always `&` it
in, or a deleted dataset's values will widen your result.

### Read only what you need

`pruning.idx` is column-addressed: asking for two columns fetches two byte
ranges, not the file. On a 10 000-dataset collection with 781 columns, reading
one column takes ~2 ms.

Cheaper still, `column_summaries()` reads the **footer only** — every column's
collection-wide min/max and present count, no column data at all:

```python
summaries = store.column_summaries()      # ~2 ms for 781 columns
if summaries["readings"]["max"] < 25.0:
    ...  # no dataset can match; skip fetching the column entirely
```

The index is compressed with zstd by default
(`StoreConfig.pruning_compression`) — on that same 10 000-dataset collection
it is 0.29 MB rather than 17.4 MB. Blocks are compressed individually, so
single-column reads stay ranged whatever codec you pick, and the codec is
recorded in the index itself so readers adapt without being told.

### Types

Statistics keep the type they were computed with — nothing is cast. `min` and
`max` come back as `int64`, `uint64`, `float64`, or a list of `bytes | None`,
chosen from what the column actually holds. A column that is `int32` in some
datasets and `float64` in others promotes to `float64` at the numpy boundary.

For a column's collection-wide *declared* type, use
[`merged_schema()`](datasets-and-arrays.md) — the index itself stores no dtype.

### Attribute columns

Attribute columns currently record **presence only**: `present` tells you which
datasets carry the key, but `stats_valid` is `False` and `min`/`max` are unset.
Filtering on attribute *values* through the index isn't wired up yet — read
them per dataset with `attributes()` for now.

[`examples/06_stats_scan.py`](https://github.com/maris-development/atlas/blob/main/atlas-python/examples/06_stats_scan.py)
runs a per-dataset scan across 32 sensor datasets.

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

When ingesting via [`add_xarray_dataset`](xarray.md#fill-values-and-missing-data)
you don't set these by hand: float arrays default to a `NaN` fill, datetimes
to `NaT`, and strings to `""`, so cells masked by `mask_and_scale=True` are
counted as null automatically. Pass `fill_value=` to override.

## What stats don't include

- **No mean, sum, stdev, or quantiles.** If you need those, read the array
  and compute them with numpy / dask. Atlas's stats are designed to be
  cheap-on-write and cheap-on-read, not a full analytics engine.
- **No per-dimension reductions.** `min` / `max` are scalars over the
  whole array, not per-row / per-column.
- **No per-chunk zone maps.** Statistics are per array per dataset; there is
  no finer granularity to prune within a large array.
- **`List` / `FixedSizeList` arrays get no statistics at all** — not even
  counts.

Strings and timestamps *are* covered. String `min`/`max` come back as `bytes`
and order lexicographically; timestamps come back as integer nanoseconds since
the epoch:

```python
ds.array_stats("station")["min"]    # b"AAA01"
np.array(ds.array_stats("time")["min"]).astype("datetime64[ns]")
```
