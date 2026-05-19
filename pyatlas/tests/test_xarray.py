"""xarray integration tests."""
import os
import tempfile

import dask.array as dask_array
import numpy as np
import pytest
import xarray as xr

import pyatlas  # noqa: F401  — side-effect import registers the ds.atlas accessor


_DATA_DIR = os.path.dirname(__file__)


def _make_dataset() -> xr.Dataset:
    """Build a sample xarray Dataset with coords, data_vars, and per-var attrs."""
    temp = xr.DataArray(
        np.arange(8 * 16, dtype=np.float32).reshape(8, 16),
        dims=["lat", "lon"],
        attrs={"units": "celsius", "long_name": "surface temperature"},
    )
    pressure = xr.DataArray(
        np.full((8, 16), 1013.25, dtype=np.float64),
        dims=["lat", "lon"],
        attrs={"units": "hPa"},
    )
    return xr.Dataset(
        data_vars={"temperature": temp, "pressure": pressure},
        coords={
            "lat": ("lat", np.arange(8, dtype=np.float32)),
            "lon": ("lon", np.arange(16, dtype=np.float32)),
        },
        attrs={"month": 1, "station": "KNMI"},
    )


def test_basic_roundtrip():
    ds = _make_dataset()
    with tempfile.TemporaryDirectory() as d:
        with pyatlas.Atlas.create(d) as atlas:
            atlas.add_xr_dataset(ds, "ds_jan")

        atlas2 = pyatlas.Atlas.open(d)
        ds_back = atlas2.to_xarray("ds_jan")

    xr.testing.assert_identical(ds, ds_back)


def test_per_var_attrs_roundtrip():
    ds = _make_dataset()
    with tempfile.TemporaryDirectory() as d:
        atlas = pyatlas.Atlas.create(d)
        atlas.add_xr_dataset(ds, "ds_jan")
        atlas.flush()

        view = atlas.open_dataset("ds_jan")
        all_attrs = view.attributes()
        assert all_attrs["temperature.units"] == "celsius"
        assert all_attrs["temperature.long_name"] == "surface temperature"
        assert all_attrs["pressure.units"] == "hPa"
        assert all_attrs["month"] == 1
        assert all_attrs["station"] == "KNMI"

        atlas2 = pyatlas.Atlas.open(d)
        ds_back = atlas2.to_xarray("ds_jan")
        assert ds_back["temperature"].attrs == {
            "units": "celsius",
            "long_name": "surface temperature",
        }
        assert ds_back["pressure"].attrs == {"units": "hPa"}


def test_non_scalar_attr_value():
    """List-valued attrs roundtrip via the `json:` prefix marker."""
    da = xr.DataArray(
        np.zeros((4,), dtype=np.int32),
        dims=["x"],
        attrs={"valid_range": [0, 100], "tags": ["draft", "v1"]},
    )
    ds = xr.Dataset(data_vars={"v": da}, attrs={"version_history": [1, 2, 3]})

    with tempfile.TemporaryDirectory() as d:
        atlas = pyatlas.Atlas.create(d)
        atlas.add_xr_dataset(ds, "ds")
        atlas.flush()

        view = atlas.open_dataset("ds")
        on_disk = view.attributes()
        assert on_disk["v.valid_range"].startswith("json:")
        assert on_disk["version_history"].startswith("json:")

        atlas2 = pyatlas.Atlas.open(d)
        ds_back = atlas2.to_xarray("ds")
        assert ds_back["v"].attrs["valid_range"] == [0, 100]
        assert ds_back["v"].attrs["tags"] == ["draft", "v1"]
        assert ds_back.attrs["version_history"] == [1, 2, 3]


def test_no_coords_marker_fallback():
    """If a dataset was written without the _pyatlas_coords marker, a 1-D array
    whose dim name matches its name is auto-promoted to a coordinate."""
    with tempfile.TemporaryDirectory() as d:
        atlas = pyatlas.Atlas.create(d)
        view = atlas.create_dataset("ds")
        view.define_array("lat", dtype="float32", dims=["lat"], shape=[4])
        view.write_array("lat", start=[0], data=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
        view.define_array("temp", dtype="float32", dims=["lat", "lon"], shape=[4, 2])
        view.write_array("temp", start=[0, 0], data=np.full((4, 2), 5.0, dtype=np.float32))
        atlas.flush()

        atlas2 = pyatlas.Atlas.open(d)
        ds_back = atlas2.to_xarray("ds")

    assert "lat" in ds_back.coords
    assert "temp" in ds_back.data_vars
    assert ds_back["temp"].dims == ("lat", "lon")


def test_netcdf_file_roundtrip():
    """Open a real NetCDF file (datetime64[ns] + object-string + scalar vars), roundtrip."""
    ds = xr.open_dataset(os.path.join(_DATA_DIR, "GL_PR_BO_JLKU.nc"))

    assert ds["TIME"].dtype == np.dtype("datetime64[ns]")
    assert ds["DC_REFERENCE"].dtype.kind == "O"
    assert ds["DIRECTION"].dtype.kind == "O"
    assert ds["TRAJECTORY"].ndim == 0  # |S5 scalar

    with tempfile.TemporaryDirectory() as d:
        atlas = pyatlas.Atlas.create(d)
        atlas.add_xr_dataset(ds, "obs")
        atlas.flush()

        view = atlas.open_dataset("obs")
        assert view.array_meta("TIME")["dtype"] == "timestamp_nanoseconds"
        assert view.array_meta("DC_REFERENCE")["dtype"] == "string"
        assert view.array_meta("DIRECTION")["dtype"] == "string"
        assert view.array_meta("TRAJECTORY")["dtype"] == "string"
        assert view.array_meta("TRAJECTORY")["shape"] == []

        atlas2 = pyatlas.Atlas.open(d)
        ds_back = atlas2.to_xarray("obs")

    assert ds_back["TIME"].dtype == np.dtype("datetime64[ns]")
    np.testing.assert_array_equal(ds_back["TIME"].values, ds["TIME"].values)
    np.testing.assert_array_equal(ds_back["LATITUDE"].values, ds["LATITUDE"].values)
    np.testing.assert_array_equal(ds_back["FLU2"].values, ds["FLU2"].values)

    def _decode_one(v):
        return v.decode() if isinstance(v, bytes) else v

    def _decode(arr):
        return np.array([_decode_one(v) for v in arr], dtype=object)

    np.testing.assert_array_equal(ds_back["DC_REFERENCE"].values, _decode(ds["DC_REFERENCE"].values))
    np.testing.assert_array_equal(ds_back["DIRECTION"].values, _decode(ds["DIRECTION"].values))
    assert _decode_one(ds_back["TRAJECTORY"].values.item()) == _decode_one(ds["TRAJECTORY"].values.item())


def test_atlas_xr_batched_roundtrip():
    """Many add_xr_dataset calls accumulate; one atlas.flush persists them all."""
    ds_a = _make_dataset()
    ds_b = _make_dataset()
    with tempfile.TemporaryDirectory() as d:
        with pyatlas.Atlas.create(d) as atlas:
            atlas.add_xr_dataset(ds_a, "jan")
            atlas.add_xr_dataset(ds_b, "feb")

        atlas2 = pyatlas.Atlas.open(d)
        assert sorted(atlas2.list_datasets()) == ["feb", "jan"]
        xr.testing.assert_identical(ds_a, atlas2.to_xarray("jan"))
        xr.testing.assert_identical(ds_b, atlas2.to_xarray("feb"))


def test_atlas_xr_no_implicit_flush():
    """add_xr_dataset doesn't auto-persist — fresh Atlas sees nothing without atlas.flush."""
    ds = _make_dataset()
    with tempfile.TemporaryDirectory() as d:
        atlas = pyatlas.Atlas.create(d)
        atlas.add_xr_dataset(ds, "jan")
        # No flush.

        atlas_peek = pyatlas.Atlas.open(d)
        assert atlas_peek.list_datasets() == []


def test_unsupported_dtype_raises():
    da = xr.DataArray(np.array([True, False, True], dtype=np.bool_), dims=["x"])
    ds = xr.Dataset(data_vars={"flag": da})
    with tempfile.TemporaryDirectory() as d:
        atlas = pyatlas.Atlas.create(d)
        with pytest.raises(NotImplementedError):
            atlas.add_xr_dataset(ds, "ds")


def test_accessor_write():
    """`ds.atlas.write(atlas, name)` performs the same roundtrip as the method form."""
    ds = _make_dataset()
    with tempfile.TemporaryDirectory() as d:
        with pyatlas.Atlas.create(d) as atlas:
            ds.atlas.write(atlas, "ds_jan")  # accessor path

        atlas2 = pyatlas.Atlas.open(d)
        ds_back = atlas2.to_xarray("ds_jan")

    xr.testing.assert_identical(ds, ds_back)


def test_accessor_and_method_equivalent():
    """The accessor and `atlas.add_xr_dataset` produce identical Datasets on roundtrip."""
    ds = _make_dataset()
    with tempfile.TemporaryDirectory() as d_a, tempfile.TemporaryDirectory() as d_b:
        with pyatlas.Atlas.create(d_a) as atlas_a:
            atlas_a.add_xr_dataset(ds, "ds_jan")
        with pyatlas.Atlas.create(d_b) as atlas_b:
            ds.atlas.write(atlas_b, "ds_jan")

        ds_a = pyatlas.Atlas.open(d_a).to_xarray("ds_jan")
        ds_b = pyatlas.Atlas.open(d_b).to_xarray("ds_jan")

    xr.testing.assert_identical(ds_a, ds_b)


# ----- dask-backed (streaming) tests --------------------------------------------------


def test_dask_chunked_roundtrip():
    """A dask-chunked variable's chunks are preserved as the atlas chunk_shape."""
    data = dask_array.arange(8 * 16, dtype=np.float32, chunks=8 * 16).reshape(8, 16)  # type: ignore[arg-type]
    da = xr.DataArray(data, dims=["y", "x"]).chunk({"y": 4, "x": 8})
    ds = xr.Dataset(data_vars={"temp": da})

    with tempfile.TemporaryDirectory() as d:
        atlas = pyatlas.Atlas.create(d)
        atlas.add_xr_dataset(ds, "ds")
        atlas.flush()

        view = atlas.open_dataset("ds")
        meta = view.array_meta("temp")
        assert meta["chunk_shape"] == [4, 8]

        atlas2 = pyatlas.Atlas.open(d)
        ds_back = atlas2.to_xarray("ds")

    xr.testing.assert_identical(ds.compute(), ds_back)


def test_streaming_write_call_count(monkeypatch):
    """`write_array` is called once per dask block (not once for the whole array)."""
    from pyatlas._pyatlas import DatasetView

    arr = dask_array.arange(16, dtype=np.int32, chunks=4)  # type: ignore[arg-type]
    da = xr.DataArray(arr, dims=["x"])
    ds = xr.Dataset(data_vars={"v": da})

    calls: list[tuple[str, list[int], tuple[int, ...]]] = []
    real_write_array = DatasetView.write_array

    def counting_write_array(self, name, start, data):  # type: ignore[no-redef]
        calls.append((name, list(start), tuple(data.shape)))
        return real_write_array(self, name, start, data)

    monkeypatch.setattr(DatasetView, "write_array", counting_write_array)

    with tempfile.TemporaryDirectory() as d:
        atlas = pyatlas.Atlas.create(d)
        atlas.add_xr_dataset(ds, "ds")
        atlas.flush()

    v_calls = [c for c in calls if c[0] == "v"]
    assert len(v_calls) == 4
    assert [c[1] for c in v_calls] == [[0], [4], [8], [12]]
    assert all(c[2] == (4,) for c in v_calls)
