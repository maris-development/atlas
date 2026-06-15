# xarray accessor and helpers

Importing `atlas` registers an accessor at `xr.Dataset.atlas` (the side
effect happens on the very first `import atlas`, so there's no extra
setup). The accessor is a thin convenience wrapper around the
[`Atlas.add_xarray_dataset`](atlas.md#atlas.Atlas.add_xarray_dataset) method.

```python
import atlas, xarray as xr        # ds.atlas is registered here

ds = xr.Dataset(...)
with atlas.Atlas.create("/tmp/store") as atlas:
    ds.atlas.write(atlas, "jan_2024")           # accessor form
    atlas.add_xarray_dataset(ds, "feb_2024")        # equivalent Atlas method
```

See [xarray integration](../guides/xarray.md) for the full storage
conventions and dtype mapping.

## `ds.atlas.write`

::: atlas.xarray._AtlasAccessor.write
    options:
        heading_level: 3
        show_root_heading: false

## `atlas.init_tracing`

Top-level helper for enabling the Rust core's structured logging. Useful
when debugging a slow read path or verifying which chunks a lazy `compute`
actually touched.

```python
atlas.init_tracing("debug")           # everything at debug+
atlas.init_tracing("atlas_python=info")    # just atlas crate at info+
atlas.init_tracing()                  # re-read ATLAS_LOG / RUST_LOG env vars
```

::: atlas.init_tracing
    options:
        heading_level: 3
        show_root_heading: false
