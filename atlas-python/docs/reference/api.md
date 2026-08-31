# API reference

Five operations. Each takes a `source`: a local path, a URL (`s3://`, `gs://`,
`az://`, `https://`), or an obstore handle. Extra keyword arguments are passed
to obstore for remote sources — `region`, `endpoint`, `skip_signature`.

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
