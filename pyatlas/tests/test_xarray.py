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


def test_surrogate_attr_value_is_sanitized():
    """NetCDF backends sometimes hand back attr strs containing lone surrogates
    (bytes-mis-decoded-as-Latin-1 via surrogateescape). The write path recovers
    the original UTF-8 instead of crashing on the pyo3 boundary."""
    # '\udcc2\udcb5mol kg-1' is what xarray produces for the UTF-8 bytes
    # b'\xc2\xb5mol kg-1' = 'µmol kg-1' when the backend mis-decodes them.
    surr = "\udcc2\udcb5mol kg-1"
    da = xr.DataArray(
        np.zeros((2,), dtype=np.float32), dims=["x"], attrs={"units": surr}
    )
    ds = xr.Dataset(data_vars={"v": da})

    with tempfile.TemporaryDirectory() as d:
        with pyatlas.Atlas.create(d) as atlas:
            atlas.add_xr_dataset(ds, "ds")

        atlas2 = pyatlas.Atlas.open(d)
        ds_back = atlas2.to_xarray("ds")

        assert ds_back["v"].attrs["units"] == "µmol kg-1"  # µmol kg-1


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
    """A dask-chunked variable's chunks are preserved as the atlas chunk_shape
    and the read-back variable is dask-backed with matching chunks."""
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

        assert isinstance(ds_back["temp"].data, dask_array.Array)
        assert ds_back["temp"].data.chunks == ((4, 4), (8, 8))
        # `assert_identical` computes both sides for dask-backed comparison.
        xr.testing.assert_identical(ds, ds_back)


def test_batched_iter_preserves_order_and_values():
    """7 chunks of length 4 (chunk count not divisible by the default batch
    size of 8) — exercises the cross-batch-boundary path of the prefetched
    iterator. Values and chunk grid must round-trip byte-identical."""
    expected = np.arange(7 * 4, dtype=np.int32)
    da = xr.DataArray(
        dask_array.from_array(expected, chunks=4),  # type: ignore[arg-type]
        dims=["x"],
    )
    ds = xr.Dataset(data_vars={"v": da})

    with tempfile.TemporaryDirectory() as d:
        with pyatlas.Atlas.create(d) as atlas:
            atlas.add_xr_dataset(ds, "ds")

        atlas2 = pyatlas.Atlas.open(d)
        ds_back = atlas2.to_xarray("ds")
        # Stored chunk grid preserved
        assert ds_back["v"].data.chunks == ((4, 4, 4, 4, 4, 4, 4),)
        np.testing.assert_array_equal(ds_back["v"].compute().data, expected)


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


# ----- dask-backed (lazy) reads -------------------------------------------------------


def test_lazy_read_does_not_compute_until_requested(monkeypatch):
    """Building the dask graph is free; only `.compute()` (or slicing) reads chunks."""
    from pyatlas._pyatlas import DatasetView

    arr = dask_array.arange(16, dtype=np.int32, chunks=4)  # type: ignore[arg-type]
    da = xr.DataArray(arr, dims=["x"])
    ds = xr.Dataset(data_vars={"v": da})

    with tempfile.TemporaryDirectory() as d:
        with pyatlas.Atlas.create(d) as atlas:
            atlas.add_xr_dataset(ds, "ds")

        atlas2 = pyatlas.Atlas.open(d)

        read_calls: list[tuple[str, list[int], list[int]]] = []
        real_read_array = DatasetView.read_array

        def counting_read_array(self, name, start=None, shape=None):  # type: ignore[no-redef]
            read_calls.append((name, list(start or []), list(shape or [])))
            return real_read_array(self, name, start, shape)

        monkeypatch.setattr(DatasetView, "read_array", counting_read_array)

        ds_back = atlas2.to_xarray("ds")
        assert isinstance(ds_back["v"].data, dask_array.Array)
        # Building the graph reads zero chunks.
        assert [c for c in read_calls if c[0] == "v"] == []

        # Slicing one chunk's worth of data triggers exactly one read.
        _ = ds_back["v"].data[0:4].compute()
        v_reads = [c for c in read_calls if c[0] == "v"]
        assert len(v_reads) == 1
        assert v_reads[0] == ("v", [0], [4])

        # Full compute reads all four chunks.
        read_calls.clear()
        result = ds_back["v"].compute()
        v_reads = [c for c in read_calls if c[0] == "v"]
        assert len(v_reads) == 4
        np.testing.assert_array_equal(result.data, np.arange(16, dtype=np.int32))


def test_mixed_eager_and_dask_in_one_dataset():
    """Within one dataset, full-shape arrays are eager and chunked arrays are dask."""
    full = dask_array.arange(8, dtype=np.float32, chunks=8)  # type: ignore[arg-type]
    chunked = dask_array.arange(8, dtype=np.float32, chunks=8).reshape(8)  # type: ignore[arg-type]
    ds = xr.Dataset(
        data_vars={
            "eager_var": xr.DataArray(full, dims=["x"]),
            "lazy_var": xr.DataArray(chunked, dims=["x"]).chunk({"x": 4}),
        }
    )

    with tempfile.TemporaryDirectory() as d:
        with pyatlas.Atlas.create(d) as atlas:
            atlas.add_xr_dataset(ds, "ds")

        atlas2 = pyatlas.Atlas.open(d)
        ds_back = atlas2.to_xarray("ds")

    assert isinstance(ds_back["eager_var"].data, np.ndarray)
    assert isinstance(ds_back["lazy_var"].data, dask_array.Array)
    assert ds_back["lazy_var"].data.chunks == ((4, 4),)


def test_uneven_trailing_chunk():
    """Non-divisible shape produces an uneven trailing chunk; values round-trip."""
    with tempfile.TemporaryDirectory() as d:
        with pyatlas.Atlas.create(d) as atlas:
            view = atlas.create_dataset("ds")
            view.define_array("v", dtype="int32", dims=["x"], shape=[10], chunk_shape=[4])
            view.write_array("v", start=[0], data=np.arange(10, dtype=np.int32))

        atlas2 = pyatlas.Atlas.open(d)
        ds_back = atlas2.to_xarray("ds")

        assert isinstance(ds_back["v"].data, dask_array.Array)
        assert ds_back["v"].data.chunks == ((4, 4, 2),)
        np.testing.assert_array_equal(
            ds_back["v"].compute().data, np.arange(10, dtype=np.int32)
        )


def test_fill_value_attribute_picked_up():
    """`_FillValue` on a variable is consumed by define_array, not flattened.

    The fill value's effect on unwritten cells is exercised in
    `test_smoke.test_fill_value_unwritten_cells_int32`; this test just
    confirms that the xarray accessor strips `_FillValue` from the flattened
    attribute set so it isn't stored twice. Other per-var attrs survive.
    """
    arr = xr.DataArray(
        np.array([10, 20, 30, 40], dtype=np.int32),
        dims=["x"],
        attrs={"_FillValue": np.int32(-1), "units": "K"},
    )
    ds = xr.Dataset({"v": arr})

    with tempfile.TemporaryDirectory() as d:
        with pyatlas.Atlas.create(d) as atlas:
            atlas.add_xr_dataset(ds, "ds")

        atlas2 = pyatlas.Atlas.open(d)
        view = atlas2.open_dataset("ds")
        attrs = view.attributes()
        assert "v._FillValue" not in attrs, attrs
        assert attrs.get("v.units") == "K"

        # The user's source Dataset must not be mutated by the write.
        assert ds["v"].attrs["_FillValue"] == np.int32(-1)


def test_fill_value_attribute_dtype_mismatch_raises():
    """A `_FillValue` whose Python type can't represent the dtype is rejected."""
    arr = xr.DataArray(
        np.zeros(4, dtype=np.int32),
        dims=["x"],
        attrs={"_FillValue": 1.5},  # float for int32 → TypeError
    )
    ds = xr.Dataset({"v": arr})
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(TypeError):
            with pyatlas.Atlas.create(d) as atlas:
                atlas.add_xr_dataset(ds, "ds")
