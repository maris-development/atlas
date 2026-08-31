# Removing datasets

```python
atlas.remove("/data/collection", ["2024-02", "2024-03"])
```

```bash
atlas rm /data/collection 2024-02 2024-03
```

One call, however many datasets. Names may be given as dataset names or as the
NetCDF paths they came from, so the list that built a collection can also tear
part of it down:

```bash
atlas rm /data/collection /data/nc/2024-02.nc
```

## What it actually does

It writes a small `deleted.mask` file beside the container, holding the
ordinals of the removed datasets. **The container is never touched.**

```text
my_collection/
├── data.atlas      unchanged, still holds every byte
└── deleted.mask    20 bytes: which datasets to hide
```

Three consequences:

**No space is reclaimed.** A removed dataset's bytes stay exactly where they
are. `atlas info` says so:

```text
  datasets          2
  removed           1 (of 3 written; space not reclaimed)
```

To reclaim them, rebuild the collection from its sources.

**Ordinals do not move.** A dataset's position is fixed for the life of the
container, so an ordinal you recorded stays valid and no concurrent reader sees
a renumbering.

**It is fast, and it is the only mutation there is.** One GET and one PUT of a
file measured in bytes, whatever the size of the collection.

## Removing something absent

Raises by default:

```python
atlas.remove(collection, ["nope"])
# AtlasError: not in the collection (or already removed): nope
```

`missing_ok` reports instead:

```python
result = atlas.remove(collection, ["2024-01", "nope"], missing_ok=True)
result["removed"]   # ['2024-01']
result["missing"]   # ['nope']
```

```bash
atlas rm /data/collection nope --missing-ok
```

A dataset that was already removed counts as missing — the operation is not
idempotent in its reporting, though the end state is the same.

## Concurrency

Removing re-reads the mask before writing it, so two processes removing
*different* datasets both survive.

Two removals that interleave between that read and the write still lose one:
object stores offer no compare-and-swap here. Serialize removals against a
collection if that matters.

## Why there is no `add`

A collection is written once. Adding a dataset would mean rewriting the
container, at which point you may as well rebuild it:

```bash
atlas create /data/nc /data/collection.new
mv /data/collection.new /data/collection
```

Rebuilding is a single forward pass, which is exactly why the format can get
away without an append path. See [Creating a collection](creating.md).
