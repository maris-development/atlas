# `DatasetView`

A typed handle into a single dataset within an [`Atlas`](atlas.md).
Mutations go through `define_array` / `write_array` / `set_attribute` /
`delete_array` and are buffered into the parent atlas's in-memory state
until [`Atlas.flush()`](atlas.md#pyatlas.Atlas.flush).

::: pyatlas.DatasetView
    options:
        heading_level: 2
