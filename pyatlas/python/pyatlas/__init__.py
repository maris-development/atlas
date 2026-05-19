from ._pyatlas import Atlas, DatasetView, __version__
from . import xarray as _xarray  # noqa: F401  — registers the `ds.atlas` accessor

__all__ = ["Atlas", "DatasetView", "__version__"]
