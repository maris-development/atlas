from ._atlas import DatasetView, DatasetWriter, __version__, init_tracing
from . import xarray as _xarray  # noqa: F401  — registers the `ds.atlas` accessor
from .store import Atlas, AtlasWriter

__all__ = [
    "Atlas",
    "AtlasWriter",
    "DatasetView",
    "DatasetWriter",
    "__version__",
    "init_tracing",
]
