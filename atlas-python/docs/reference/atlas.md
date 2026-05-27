# `Atlas`

The store handle. Holds an in-memory `StoreMeta` and the per-array file
caches. All mutations are buffered until [`flush()`](#atlas.Atlas.flush);
see [Durability and flushing](../guides/durability.md) for the full
contract.

::: atlas.Atlas
    options:
        heading_level: 2
