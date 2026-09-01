"""atlas. Thousands of NetCDF datasets in one immutable file.

Five operations, and no more:

    create          build a collection from a directory of NetCDF files
    remove          remove datasets from one, in a single call
    list_datasets   what it holds
    describe        one dataset in detail, like ncdump
    info            the collection as a whole

Every one takes a local path or a URL (``s3://``, ``gs://``, ``az://``,
``https://``). The same call therefore works against a bucket.

The same operations are on the command line as ``atlas create``, ``atlas rm``,
``atlas ls``, ``atlas show``, and ``atlas info``.

The Rust API reads array *data*. This package does not. See the Reading data
guide.
"""

from ._atlas import __version__, init_tracing
from ._ops import (
    AtlasError,
    create,
    describe_dataset as describe,
    find_netcdf_files,
    info,
    list_datasets,
    remove,
)
from ._source import SourceError

__all__ = [
    "create",
    "remove",
    "list_datasets",
    "describe",
    "info",
    "find_netcdf_files",
    "AtlasError",
    "SourceError",
    "init_tracing",
    "__version__",
]
