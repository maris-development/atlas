# 6. The Pruning Index

The pruning index answers **"which datasets could possibly match this
predicate?"** without opening any dataset's data. It's a flat, columnar table:

```
                 columns = arrays / attributes
                 ┌────────── TEMP ──────────┐  ┌──── year ────┐
   row=ordinal   present  min    max  n  nulls  present  value
   ─────────────────────────────────────────────────────────────
   0  ds0         true    1.2   28.4  …    …      true   2019
   1  ds1         false    –      –   0    0      false    –
   2  ds2         true   -0.5   14.0  …    …      true   2023
   …
```

A client reads the columns it cares about and prunes:
`view.candidates(|lo, hi| hi > &StatVal::Float(25.0))` → the ordinals whose
`TEMP` range *could* exceed 25°, with deleted and absent rows already excluded.

## The key decision: it is **not persisted**

The array files **already** store per-dataset statistics — array-format computes
`min`/`max`/`row_count`/`null_count` for every entry as a byproduct of writing
(the `StatsFile`). A separate on-disk pruning index would just be a
*denormalized pivot* of data that already exists. So Atlas doesn't store one:

> **`pruning_index(cols)` rebuilds the requested columns on demand by reading
> those arrays' StatsFiles and pivoting them by dataset ordinal.**

```
   TEMP/data.af  StatsFile              pivot by ordinal          flat column
   ┌────────────────────────┐          ┌───────────────┐        ┌──────────┐
   │ "ds0" → min 1.2 max 28 │          │ name→ordinal  │        │ row 0 ✓  │
   │ "ds2" → min -.5 max 14 │  ──read──►│ ds0→0 ds2→2 …│──scatter►│ row 1 ·  │
   │ …                      │          └───────────────┘        │ row 2 ✓  │
   └────────────────────────┘                                    │ …        │
```

Why this is the right shape:

| Property | Why it holds |
|----------|--------------|
| **Zero extra write memory** | stats are a byproduct; nothing to accumulate at flush |
| **Always consistent** | the StatsFiles *are* the source of truth — no staleness, no epoch |
| **Scales to many columns** | cost is proportional to *requested* columns, not the 50K+ total |
| **No index files on disk** | a cold reopen serves it straight from the array stats |

## What a build does (`pruning_index([C1, C2, …])`)

```
   1. Snapshot from StoreMeta (one short lock, no I/O):
        rows = row_slots()          liveness mask       row→name mapping
        name→ordinal map  ← CACHED on the store (rebuilt only when the
                             dataset set changes: create / delete / compact)

   2. For each requested column, build a length-N StatColumn (in parallel):

        ColumnKey::Array(name)         one StatsFile read of name/data.af;
                                       scatter each entry into its ordinal row
        ColumnKey::ArrayAttr(a, key)   one attribute_index read of a/data.af;
                                       each dataset's scalar value → point range [v,v]
        ColumnKey::GlobalAttr(key)     same, from _global/data.af

   3. Assemble the PruningIndex: insert columns, attach liveness mask + names.
      The result is self-describing — view() hides deleted rows, dataset_name(row)
      maps back to names.
```

The output type is `StatColumn` per column — dense vectors (`present`,
`stats_valid`, `min[]`, `max[]`, `row_count[]`, `null_count[]`) of length N, so a
consumer can compare them vectorized. `ColumnView` folds in the liveness mask so
deleted/absent rows can't leak into a result.

## Three masks, one safe view

Reading a cell correctly means combining three masks. `ColumnView` does it for
you:

```
   is_present(row)  =  live[row]      ∧  present[row]
   has_stats(row)   =  is_present     ∧  stats_valid[row]
   candidates(pred) =  rows where has_stats ∧ pred(min,max)
```

`ColumnSummary` (collection-wide `min`/`max`/`present_count`) gives a cheap
"can any row match?" pre-filter via `might_match` before you fetch a column.

## Attributes as columns

An attribute is one value per dataset. `attribute_index(key)` returns every
dataset's value for a key in a single read, and each **scalar** becomes a point
range `[v, v]` — so you range-prune on `year > 2020` or match `platform == "BO"`
exactly the same way as on array data. List-valued attributes are marked present
without a range.

## Performance & parallelism

Two shape-independent optimizations keep it fast:

- **Cached `name→ordinal` map** — built once, reused across calls, invalidated
  only when the dataset set changes. Removes an O(datasets) rebuild per call.
- **Parallel column builds** — each column reads an *independent* array file, so
  builds run as bounded concurrent tasks (capped at the CPU count). Wall-clock
  for K columns collapses toward the cost of the slowest single column.

### Benchmark — 1,000,000 datasets, 4 arrays, 25% present each (250K/column)

`cargo run --release --example bench_pruning 1000000 4`

| Operation | Serial (before) | Cached map + parallel |
|-----------|-----------------|-----------------------|
| `pruning_index([arr0])` cold | 390 ms | **299 ms** |
| `pruning_index([arr0])` warm | 387 ms | **137 ms** |
| `pruning_index(all 4 cols)` | 570 ms | **147 ms** |
| `column_summaries(4 cols)` | 689 ms | **158 ms** |

The residual cost of a single cold 1M-row column is (a) the ~64 MB dense
`StatColumn` allocation and (b) the 250K-entry StatsFile deserialize — both
addressable only by changing the *representation* (a typed, ordinal-aligned,
mmap-able stats sidecar), not by more parallelism. For couple-ms cold reads at
1M, that's the next step; for a 1 s budget, the current numbers clear it easily.

## Module map

```
   pruning/
   ├── mod.rs      PruningIndex (the flat result), ColumnKey
   ├── column.rs   StatColumn (dense vectors) + ColumnView + ColumnSummary
   ├── value.rs    StatVal (typed min/max) + attribute→StatVal conversion
   └── bitmap.rs   packed present / stats_valid masks

   store/mod.rs    pruning_index(), column_summaries(), build_column (the pivot),
                   the cached ordinal map + parallel builder
```

## What "sparse vs dense across arrays" means here

Your arrays need not align. `arr0` may be declared by datasets {0,4,8,…} and
`arr1` by {1,5,9,…}; each column is scattered into the *global* ordinal space
independently via the name→ordinal map, so row `i` always means dataset `i`
across every column. (That's why the pivot uses names rather than assuming entry
`i` in a StatsFile is ordinal `i` — an assumption that only holds for arrays
declared by *every* dataset in order.)

## History

Two earlier designs were built and discarded in favor of on-demand:

1. **A single dense `pruning.idx`** (columns × rows) rebuilt at every flush —
   O(distinct-columns × datasets) memory, catastrophic when array names aren't
   shared (columns ≈ datasets → quadratic).
2. **A segmented LSM index** — one on-disk segment per flush, merged on read;
   bounded write memory but added a whole persistence/compaction subsystem.

On-demand supersedes both: the stats already live in the array files, so the
index is a read-time view over them — no dense matrix, no segments, no
persistence, no staleness. (The segmented experiment is archived on branch
`_archive/pruning-segmented-index`.)
