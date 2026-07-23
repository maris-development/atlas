# 5. The Write Path & Durability

Atlas batches aggressively: **every mutation updates in-memory state only —
nothing reaches disk until `flush()`.** This is what makes bulk ingestion of
many datasets fast, and it's the single durability boundary.

## create → define → write (all in memory)

```
   atlas.create_dataset("ds0")            registers ds0 in StoreMeta (ordinal assigned)
     └─ returns DatasetView
          view.define_array("temp", …)    records the array in the schema;
                                          opens temp/data.af lazily (ArrayCache)
          view.write_array("temp", start, data)
              └─ ArrayFile buffers into its PENDING delta layer:
                   encodes chunks, packs them into 4 MiB blocks;
                   a full block is compressed and spilled to a per-array TEMPFILE;
                   the current partial block stays in RAM.
```

Key consequence: **bulk array data does not accumulate in RAM.** It streams into
per-array tempfiles as blocks fill. What stays in memory between flushes is the
much smaller per-array metadata (the pending delta's index).

```
   memory during ingest                 disk during ingest
   ┌───────────────────────────┐        ┌────────────────────────────┐
   │ StoreMeta (schema/ordinals)│       │ temp/  (tempfile, growing) │
   │ per-array pending index    │  ───► │ psal/  (tempfile, growing) │
   │ ≤ one 4 MiB block / array  │       │ (nothing committed yet)    │
   └───────────────────────────┘        └────────────────────────────┘
```

## flush() — the durability boundary

```
   atlas.flush():
     1. drain buffered attribute writes into their .af files
     2. force-init every array referenced in meta
     3. for each touched ArrayFile:  commit pending delta → new immutable
        SIDECAR LAYER; compute/refresh its StatsFile
     4. bump meta_epoch
     5. write atlas.json
```

After a flush, an array file looks like:

```
   temp/data.af
   ┌───────────────────────────────────────────────┐
   │  base layer         (datasets flushed earlier) │
   │  sidecar layer #1   (this flush's datasets)    │  ← newest wins on read
   │  StatsFile          (per-dataset min/max/…)    │
   │  footer                                        │
   └───────────────────────────────────────────────┘
```

Nothing is durable until step 5 completes. If the process dies mid-ingest before
a flush, the on-disk store is exactly as it was after the previous flush.

## What a failed insert leaves behind

Because writes buffer in memory, a mid-insert error persists nothing by itself.
But a partially-created dataset can linger in the *in-memory* store; the Python
`add_xarray_dataset` makes this atomic by rolling back with `delete_dataset` on
failure (see [python-xarray.md](python-xarray.md)).

## compact() — reclaim and renumber

Over time, deletes leave tombstones and array files accumulate sidecar layers.
`compact`:

```
   atlas.compact():
     1. commit + compact every ArrayFile (merge delta layers → one base,
        physically drop deleted datasets' entries)
     2. drop_tombstones() in StoreMeta   ← ordinals RENUMBER here
     3. invalidate the ordinal-map cache
     4. bump meta_epoch, write atlas.json
```

`compact` is the **only** operation that changes ordinals. Any ordinal you held
from before a compact is invalid afterward.

## Why this shape

| Goal | Mechanism |
|------|-----------|
| Fast bulk ingest | buffer in memory, one flush at the end |
| Bounded write memory | array-format spills blocks to tempfiles |
| Crash safety | single durability boundary; nothing partial persists pre-flush |
| Cheap deletes | tombstone (logical); reclaim later at compact |
| Stable cross-dataset row numbers | ordinals only renumber at compact |

## A note on flush cost at scale

`flush` does per-entry work in each array file (committing the delta, computing
stats). For very large collections (≈1M dataset-entries) this is currently the
dominant wall-clock cost of ingestion — far more than the metadata handling.
It's a write-path concern independent of the (on-demand) read path; see the
benchmark in [pruning-index.md](pruning-index.md).
