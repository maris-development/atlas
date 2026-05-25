# `Atlas`

The store handle. Holds an in-memory `StoreMeta` and the per-array file
caches. All mutations are buffered until [`flush()`](#pyatlas.Atlas.flush);
see [Durability and flushing](../guides/durability.md) for the full
contract.

::: pyatlas.Atlas
    options:
        heading_level: 2
