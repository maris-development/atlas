# xarray accessor and helpers

Importing `atlas` registers an accessor at `xr.Dataset.atlas`. It is a thin
wrapper around
[`AtlasWriter.add_xarray_dataset`](atlas-writer.md#atlas.AtlasWriter.add_xarray_dataset).

```python
import atlas, xarray as xr        # ds.atlas is registered here

ds = xr.Dataset(...)
with atlas.AtlasWriter.create("/tmp/collection") as w:
    ds.atlas.write(w, "jan_2024")            # accessor form
    w.add_xarray_dataset(ds, "feb_2024")     # equivalent method
```

The accessor writes only. See [xarray integration](../guides/xarray.md) for the
storage conventions and the dtype mapping.

## `ds.atlas.write`

::: atlas.xarray._AtlasAccessor.write
    options:
        heading_level: 3
        show_root_heading: false

## `atlas.init_tracing`

Enables the Rust core's structured logging — useful when checking which chunks
a write actually produced, or why an ingest is slow.

```python
atlas.init_tracing("debug")            # everything at debug+
atlas.init_tracing("atlas=info")       # just the atlas crate
atlas.init_tracing()                   # re-read ATLAS_LOG / RUST_LOG
```

::: atlas.init_tracing
    options:
        heading_level: 3
        show_root_heading: false
