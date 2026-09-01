# API reference

Five operations. Each takes a `source`. That is a local path, a URL (`s3://`,
`gs://`, `az://`, `https://`), or an obstore handle. For a remote source, the
extra keyword arguments reach obstore: `region`, `endpoint`, and
`skip_signature`.

```python
import atlas
```

::: atlas.create
    options:
        heading_level: 2

::: atlas.remove
    options:
        heading_level: 2

::: atlas.list_datasets
    options:
        heading_level: 2

::: atlas.describe
    options:
        heading_level: 2

::: atlas.info
    options:
        heading_level: 2

## Helpers

::: atlas.find_netcdf_files
    options:
        heading_level: 3

::: atlas.init_tracing
    options:
        heading_level: 3

## Errors

::: atlas.AtlasError
    options:
        heading_level: 3

::: atlas.SourceError
    options:
        heading_level: 3
