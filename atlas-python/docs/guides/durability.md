# Durability and flushing

Atlas has exactly **one** durability boundary: `flush()`.

`atlas.json` (or `atlas.msgpack`) is loaded once when the store is opened or
created. Every subsequent mutation — `create_dataset`, `define_array`,
`write_array`, `set_attribute`, `delete_array`, … — updates an in-memory
`StoreMeta` only. Array writes buffer inside per-array caches. **Nothing
reaches disk until `flush()` is called.**

```python
import atlas
atlas = atlas.Atlas.create("/tmp/store")
atlas.create_dataset("a")          # in-memory only
atlas.create_dataset("b")          # in-memory only
del atlas                          # ← everything is lost
```

## The `with` pattern

Always wrap an `Atlas` in `with`, or call `close()` explicitly. The
context-manager `__exit__` calls `close()`, which is an alias for `flush()`.

```python
with atlas.Atlas.create("/tmp/store") as atlas:
    atlas.create_dataset("a").set_attribute("month", 1)
    atlas.create_dataset("b").set_attribute("month", 2)
# ← flush happens here, once, for both datasets
```

If you prefer explicit calls:

```python
atlas = atlas.Atlas.create("/tmp/store")
try:
    ...
finally:
    atlas.close()
```

## Why this is the right default

N consecutive mutations amortise to a single flush:

- One `atlas.json` rewrite, no matter how many datasets you added.
- **One delta file per touched array name**, even if you wrote to that
  array from 1000 different datasets in the same session.

This is the property that makes the "ingest 1000 NetCDF files into one
atlas" loop fast:

```python
with atlas.Atlas.create("/tmp/store") as atlas:
    for nc_path in sorted(glob.glob("*.nc")):
        atlas.add_xarray_dataset(xr.open_dataset(nc_path), Path(nc_path).stem)
# One delta file per array name across the whole batch, not one per file.
```

If `add_xarray_dataset` flushed on every call you'd pay 1000 `atlas.json`
rewrites and 1000 separate fsyncs per array. The trade-off is that an
unclean exit (`del atlas` without flush, a crash, a `KeyboardInterrupt`
inside the `with`) loses everything since the last flush.

## What `flush` actually does

1. Re-serialises the in-memory `StoreMeta` to the configured `meta_format`
   (json / msgpack), optionally compresses it, and replaces the on-disk
   `atlas.*` file atomically.
2. For each touched array file, writes one delta segment containing all
   pending blocks for that array (from every dataset that wrote to it).
3. Computes and persists `array_stats` (`min`, `max`, `row_count`,
   `null_count`) — these are only populated *after* a flush. Calling
   `array_stats(name)` between mutations and the next flush returns `None`.

## `compact()`

`delete_dataset` and `delete_array` tombstone entries inside the shared
physical files but don't reclaim the bytes. Run `atlas.compact()` to
rewrite each cached array file with the tombstoned regions dropped:

```python
atlas.delete_dataset("old")
atlas.flush()           # tombstone is now on disk
atlas.compact()         # array files shrink to live data only
```

`compact()` is safe to skip — tombstoned bytes are inert, they just sit in
the file until the next compaction. For a long-running ingest loop with
no deletions, you never need to call it.

## What to call when

| You want to … | Call |
|---|---|
| Persist pending writes and keep going | `atlas.flush()` |
| Persist and stop using the atlas | `atlas.close()` (or exit a `with` block) |
| Reclaim deleted-entry space | `atlas.compact()` after a flush |
| Discard everything since the last flush | `del atlas` |
