# Removing datasets

```python
atlas.remove("/data/collection", ["2024-02", "2024-03"])
```

```bash
atlas rm /data/collection 2024-02 2024-03
```

One call, whatever the number of datasets. A name is a dataset name, or the
NetCDF path the dataset came from. The list that built a collection can
therefore tear part of it down:

```bash
atlas rm /data/collection /data/nc/2024-02.nc
```

## What it actually does

It writes a small `deleted.mask` file beside the container. That file holds the
ordinals of the removed datasets. **The container never changes.**

```text
my_collection/
├── data.atlas      unchanged, still holds every byte
└── deleted.mask    20 bytes: which datasets to hide
```

Three results follow:

**This reclaims no space.** The bytes of a removed dataset stay where they are.
`atlas info` says so:

```text
  datasets          2
  removed           1 (of 3 written; space not reclaimed)
```

To reclaim them, rebuild the collection from its sources.

**No ordinal moves.** A dataset's position holds for the life of the container.
An ordinal you recorded therefore stays valid, and no reader sees a
renumbering.

**It is fast, and it is the one change a collection allows.** One GET and one
PUT of a file measured in bytes, whatever the size of the collection.

## Cost, at scale

One `remove` call writes the mask once. Ten thousand names therefore cost what
one name costs:

```python
dead = [n for n in atlas.list_datasets(collection) if n < "2020"]
atlas.remove(collection, dead)      # one mask write, however long the list
```

The request count does not grow with the list. A head and a tail read open the
collection. `remove` then reads the mask and writes it once. A loop over
`remove` pays all of that per name, so pass the list instead.

A repeated name counts once. The order of the list does not matter, because the
mask holds a sorted set of ordinals.

The same holds on the command line, up to the argument limit of your shell:

```bash
atlas rm /data/collection 2024-01 2024-02 2024-03
```

For a list too long for one command line, call `atlas.remove` from Python.

## Removing something absent

This raises an error by default:

```python
atlas.remove(collection, ["nope"])
# AtlasError: not in the collection (or already removed): nope
```

`missing_ok` reports it instead:

```python
result = atlas.remove(collection, ["2024-01", "nope"], missing_ok=True)
result["removed"]   # ['2024-01']
result["missing"]   # ['nope']
```

```bash
atlas rm /data/collection nope --missing-ok
```

A dataset somebody already removed counts as missing. The report therefore
differs on a second call, and the end state does not.

## Concurrency

A remove reads the mask again before it writes. Two processes that remove
*different* datasets therefore both survive.

Two removes that interleave between that read and the write still lose one. An
object store offers no compare-and-swap here. Serialize the removes against one
collection if that matters.

## Why there is no `add`

One write builds a collection. To add a dataset means a rewrite of the
container. At that point, rebuild it:

```bash
atlas create /data/nc /data/collection.new
mv /data/collection.new /data/collection
```

A rebuild is one forward pass. That is why the format needs no append path.
See [Creating a collection](creating.md).
