"""Shared fixtures. A directory of NetCDF files, and a collection from it."""

import numpy as np
import pytest
import xarray as xr

import atlas


def make_dataset(month: int) -> xr.Dataset:
    """One month of a small gridded product, in the dtypes that matter."""
    return xr.Dataset(
        data_vars={
            "temperature": xr.DataArray(
                np.arange(24, dtype=np.float32).reshape(4, 6) + month,
                dims=["lat", "lon"],
                attrs={"units": "celsius", "long_name": "surface temperature"},
            ),
            "counts": xr.DataArray(
                np.array([10, 20, 30, 40], dtype=np.int64), dims=["lat"]
            ),
            "station": xr.DataArray(
                np.array(["a", "b", "c", "d"], dtype=object), dims=["lat"]
            ),
        },
        coords={
            "lat": ("lat", np.arange(4, dtype=np.float64)),
            "lon": ("lon", np.arange(6, dtype=np.float64)),
            "time": np.datetime64(f"2024-{month:02d}-01", "ns"),
        },
        attrs={"month": month, "source": "test", "bounds": [1.0, 2.0]},
    )


@pytest.fixture
def netcdf_dir(tmp_path):
    """Three NetCDF files. Their names give a predictable stem order."""
    d = tmp_path / "nc"
    d.mkdir()
    for month in (1, 2, 3):
        make_dataset(month).to_netcdf(d / f"2024-{month:02d}.nc")
    return d


@pytest.fixture
def collection(netcdf_dir, tmp_path):
    """A collection from `netcdf_dir`."""
    dest = tmp_path / "collection"
    atlas.create(netcdf_dir, str(dest))
    return dest
