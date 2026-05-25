# xarray accessor and helpers

Importing `pyatlas` registers an accessor at `xr.Dataset.atlas` (the side
effect happens on the very first `import pyatlas`, so there's no extra
setup). The accessor is a thin convenience wrapper around the
[`Atlas.add_xr_dataset`](atlas.md#pyatlas.Atlas.add_xr_dataset) method.

```python
import pyatlas, xarray as xr        # ds.atlas is registered here

ds = xr.Dataset(...)
with pyatlas.Atlas.create("/tmp/store") as atlas:
    ds.atlas.write(atlas, "jan_2024")           # accessor form
    atlas.add_xr_dataset(ds, "feb_2024")        # equivalent Atlas method
```

See [xarray integration](../guides/xarray.md) for the full storage
conventions and dtype mapping.

## `ds.atlas.write`

::: pyatlas.xarray._AtlasAccessor.write
    options:
        heading_level: 3
        show_root_heading: false

## `pyatlas.init_tracing`

Top-level helper for enabling the Rust core's structured logging. Useful
when debugging a slow read path or verifying which chunks a lazy `compute`
actually touched.

```python
pyatlas.init_tracing("debug")           # everything at debug+
pyatlas.init_tracing("pyatlas=info")    # just pyatlas crate at info+
pyatlas.init_tracing()                  # re-read ATLAS_LOG / RUST_LOG env vars
```

::: pyatlas.init_tracing
    options:
        heading_level: 3
        show_root_heading: false
