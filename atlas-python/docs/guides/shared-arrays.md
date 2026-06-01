# Shared physical arrays

Atlas's defining layout trick: **arrays are shared by name across all
datasets in a store.** When N datasets all `define_array("temperature",
...)` with the same schema, they all write into the *same* physical
`temperature/data.af` file, keyed by dataset name inside the file. Adding
the 1000th dataset that re-uses an existing schema is one append + one
`atlas.json` rewrite — not 1000 new directories.

## What the on-disk layout looks like

For a store with 50 sensor datasets, each declaring `temperature` and
`pressure`:

```text
/tmp/store/
├─ atlas.json                  # registry of all 50 datasets + array schemas
├─ temperature/
│  └─ data.af                  # holds 50 entries, keyed by dataset name
└─ pressure/
   └─ data.af                  # holds 50 entries, keyed by dataset name
```

`Atlas.list_arrays()` returns the **distinct** array names across the
whole store (here: `["temperature", "pressure"]`) — usually a small
constant. `Atlas.list_datasets()` returns the logical dataset names that
may run into the thousands.

[`examples/07_shared_arrays.py`](https://github.com/robinskil/atlas/blob/main/atlas-python/examples/07_shared_arrays.py)
walks this with 50 stations × 2 variables and prints the on-disk
directory listing — exactly two array directories regardless of N.

## Why this scales

1. **Fewer inodes.** A fleet of 1000 datasets that share 5 array names
   uses 5 physical files, not 5000. Filesystem walk time stays constant
   as the dataset count grows.
2. **One file handle per array.** Cross-dataset bulk reads
   ([`read_array_across_stacked`](bulk-reads.md)) share one
   `RwLock::read` guard on each array's file and dispatch N per-dataset
   reads on the tokio runtime. The kernel-level open/close cost is paid
   once per *array*, not once per *dataset*.
3. **One delta file per array per flush.** N consecutive
   `add_xr_dataset` calls write one delta segment per *touched array
   name* on the next flush, no matter how many datasets contributed. See
   [Durability and flushing](durability.md).

## When schemas don't line up

There's no requirement that every dataset declare every array. If five out
of 1000 datasets skip `humidity`:

- The `humidity/data.af` file holds 995 entries, not 1000.
- `view.array_meta("humidity")` returns `None` on the five that don't
  declare it.
- `Atlas.read_array_across("humidity", all_names, ...)` returns `None`
  in those five positions; `read_array_across_stacked` errors (because
  there's no positional "missing" sentinel in the stacked output).

So shared physical files are an optimisation that activates when schemas
*do* line up; mixed-schema stores work too, you just don't get the bulk-read
density.

## Dtype rules across a shared array

All datasets that share a physical array file must declare it with the
**same dtype**. The first `define_array(name, dtype=...)` in the store
fixes the on-disk dtype; subsequent definitions with a different dtype
raise.

```python
ds1.define_array("temperature", dtype="float32", ...)    # ok
ds2.define_array("temperature", dtype="float32", ...)    # ok, joins the shared file
ds3.define_array("temperature", dtype="float64", ...)    # raises — dtype mismatch
```

Shape and chunk shape can differ per dataset — only the dtype is global to
the physical file.

## When to *not* share

If two datasets happen to have an array named `temperature` but with
incompatible meanings (different units, different reference frames), give
them distinct names (`indoor_temperature`, `surface_temperature`). The
shared-file layout is a performance and storage win, not a semantic
contract — atlas won't enforce that two datasets' `temperature` mean the
same thing, so you should.
