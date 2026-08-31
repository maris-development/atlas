"""atlas — thousands of NetCDF datasets in one immutable file.

Five operations, no more:

    create          build a collection from a directory of NetCDF files
    remove          remove datasets from one, in a single call
    list_datasets   what it holds
    describe        one dataset in detail, like ncdump
    info            the collection as a whole

Every one of them takes a local path or a URL (``s3://``, ``gs://``, ``az://``,
``https://``), so the same call works against a bucket.

The same operations are on the command line as ``atlas create``, ``atlas rm``,
``atlas ls``, ``atlas show``, and ``atlas info``.

Array *data* is read through the Rust API, not from here. See the Reading data
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
