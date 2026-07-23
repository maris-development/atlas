from ._atlas import DatasetView, __version__, init_tracing
from . import xarray as _xarray  # noqa: F401  — registers the `ds.atlas` accessor
from .store import Atlas

__all__ = ["Atlas", "DatasetView", "__version__", "init_tracing"]
