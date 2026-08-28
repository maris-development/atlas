# `DatasetWriter`

Builds one dataset inside a collection. Declare arrays with `define_array`,
fill them with `write_array` in any order and any number of slabs, then
`finish()`.

Nothing reaches the collection until then: a writer that is dropped or aborted
leaves no trace of its dataset.

::: atlas.DatasetWriter
    options:
        heading_level: 2
