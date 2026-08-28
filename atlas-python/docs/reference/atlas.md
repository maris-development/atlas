# `Atlas`

An open collection, read as metadata. Opening reads the container footer and
the deletion mask; every method below is then answered from memory.

There is no array read here. Array data is read through the Rust API — see
[Reading data](../guides/reading-data.md).

::: atlas.Atlas
    options:
        heading_level: 2
