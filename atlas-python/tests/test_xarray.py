"""xarray integration tests: the write path and the metadata it produces.

Python writes collections and reads their metadata; it does not read array data
back. So these tests assert on what a written collection *says* about itself —
dtypes, shapes, chunk shapes, fill values, attributes — and on the write path's
own behaviour: streaming, rollback, and the warnings it emits.

Array values are verified where they belong, in the Rust suite:
``tests/cross_fixture.rs`` reads a collection written by ``make_fixture.py`` in
this directory and checks the bytes.
"""

import json
import os
import tempfile
import warnings

import dask.array as dask_array
import numpy as np
import pytest
import xarray as xr

import atlas  # noqa: F401  — side-effect import registers the ds.atlas accessor


_DATA_DIR = os.path.dirname(__file__)


def _make_dataset() -> xr.Dataset:
    """A sample Dataset with coords, data_vars, and per-variable attrs."""
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


def _write(ds, name="ds", tmp=None, **kwargs):
    """Write `ds` into a fresh collection and return the open collection."""
    d = tmp or tempfile.mkdtemp()
    with atlas.AtlasWriter.create(str(d)) as w:
        w.add_xarray_dataset(ds, name, **kwargs)
    return atlas.Atlas.open(str(d))


# ── the shape of what gets written ───────────────────────────────────


def test_a_dataset_writes_every_variable_and_coordinate():
    ds = _make_dataset()
    a = _write(ds)

    assert a.list_datasets() == ["ds"]
    view = a.dataset("ds")
    # Coordinates first, then data variables.
    assert view.list_arrays() == ["lat", "lon", "temperature", "pressure"]
    # And which of them were coordinates is recorded.
    assert a.coords("ds") == ["lat", "lon"]

    temp = view.array_meta("temperature")
    assert temp["dtype"] == "float32"
    assert temp["shape"] == [8, 16]
    assert temp["dimension_names"] == ["lat", "lon"]


def test_dataset_and_variable_attributes_land_at_the_right_level():
    ds = _make_dataset()
    a = _write(ds)

    # Dataset attrs, with the coordinate marker filtered out.
    assert a.attributes("ds") == {"month": 1, "station": "KNMI"}

    assert a.array_attributes("ds", "temperature") == {
        "units": "celsius",
        "long_name": "surface temperature",
    }
    assert a.array_attributes("ds", "pressure") == {"units": "hPa"}
    assert a.array_attributes("ds", "lat") == {}


def test_non_scalar_attributes_survive_as_json():
    ds = xr.Dataset(
        {"v": xr.DataArray(np.arange(4, dtype=np.float32), dims=["x"])},
        attrs={
            "bounds": [1.0, 2.0, 3.0],
            "nested": {"a": 1, "b": [2, 3]},
            "plain": "text",
        },
    )
    got = _write(ds).attributes("ds")
    assert got["bounds"] == [1.0, 2.0, 3.0]
    assert got["nested"] == {"a": 1, "b": [2, 3]}
    assert got["plain"] == "text"


def test_a_dataset_with_no_coordinates_reports_none():
    ds = xr.Dataset({"v": xr.DataArray(np.arange(4, dtype=np.float32), dims=["x"])})
    a = _write(ds)
    assert a.coords("ds") == []
    assert a.dataset("ds").list_arrays() == ["v"]


def test_many_datasets_go_into_one_collection():
    with tempfile.TemporaryDirectory() as d:
        with atlas.AtlasWriter.create(d) as w:
            w.add_xarray_dataset(_make_dataset(), "jan")
            w.add_xarray_dataset(_make_dataset(), "feb")
        a = atlas.Atlas.open(d)
        assert a.list_datasets() == ["jan", "feb"]
        # Identical schemas, so the footer interns one copy. Observable here as
        # identical metadata for both.
        assert a.dataset("jan").list_arrays() == a.dataset("feb").list_arrays()
        # One file, no per-array directories.
        assert sorted(p for p in os.listdir(d)) == ["data.atlas"]


# ── dtype mapping ────────────────────────────────────────────────────


def test_a_real_netcdf_file_maps_its_dtypes():
    ds = xr.open_dataset(os.path.join(_DATA_DIR, "GL_PR_BO_JLKU.nc"))

    assert ds["TIME"].dtype == np.dtype("datetime64[ns]")
    assert ds["DC_REFERENCE"].dtype.kind == "O"
    assert ds["TRAJECTORY"].ndim == 0  # |S5 scalar

    view = _write(ds, "obs").dataset("obs")
    assert view.array_meta("TIME")["dtype"] == "timestamp_nanoseconds"
    assert view.array_meta("DC_REFERENCE")["dtype"] == "string"
    assert view.array_meta("DIRECTION")["dtype"] == "string"
    assert view.array_meta("TRAJECTORY")["dtype"] == "string"
    assert view.array_meta("TRAJECTORY")["shape"] == []
    assert view.array_meta("LATITUDE")["dtype"] in ("float32", "float64")


@pytest.mark.parametrize(
    "np_dtype,expected",
    [
        ("int8", "int8"),
        ("int16", "int16"),
        ("int32", "int32"),
        ("int64", "int64"),
        ("uint8", "uint8"),
        ("uint32", "uint32"),
        ("float32", "float32"),
        ("float64", "float64"),
        ("datetime64[ns]", "timestamp_nanoseconds"),
    ],
)
def test_numpy_dtypes_map_one_to_one(np_dtype, expected):
    data = np.zeros(4, dtype=np_dtype)
    ds = xr.Dataset({"v": xr.DataArray(data, dims=["x"])})
    assert _write(ds).dataset("ds").array_meta("v")["dtype"] == expected


def test_an_unsupported_dtype_is_refused():
    ds = xr.Dataset(
        {"v": xr.DataArray(np.array([True, False, True, False]), dims=["x"])}
    )
    with tempfile.TemporaryDirectory() as d:
        with atlas.AtlasWriter.create(d) as w:
            with pytest.raises(NotImplementedError):
                w.add_xarray_dataset(ds, "ds")


def test_timedelta_is_tagged_so_it_can_be_restored():
    arr = np.array([1, 2, 3, 4], dtype="timedelta64[s]")
    ds = xr.Dataset({"dt": xr.DataArray(arr, dims=["x"])})
    view = _write(ds).dataset("ds")
    # Stored as int64 nanoseconds, with a marker naming the unit.
    assert view.array_meta("dt")["dtype"] == "int64"
    assert view.get_array_attribute("dt", "_pyatlas_timedelta") == "ns"


def test_surrogate_escaped_attribute_text_is_sanitized():
    # NetCDF backends surface byte attrs decoded with surrogateescape; those
    # pseudo-codepoints are not representable as UTF-8.
    ds = xr.Dataset(
        {"v": xr.DataArray(np.arange(2, dtype=np.float32), dims=["x"])},
        attrs={"note": "caf\udce9"},
    )
    got = _write(ds).attributes("ds")["note"]
    assert isinstance(got, str)
    got.encode("utf-8")  # must not raise


# ── chunking ─────────────────────────────────────────────────────────


def test_a_numpy_variable_becomes_a_single_chunk():
    ds = xr.Dataset({"v": xr.DataArray(np.arange(16, dtype=np.int32), dims=["x"])})
    meta = _write(ds).dataset("ds").array_meta("v")
    assert meta["shape"] == [16]
    assert meta["chunk_shape"] == [16]


def test_a_dask_variable_keeps_its_chunking():
    arr = dask_array.arange(16, dtype=np.int32, chunks=4)  # type: ignore[arg-type]
    ds = xr.Dataset({"v": xr.DataArray(arr, dims=["x"])})
    assert _write(ds).dataset("ds").array_meta("v")["chunk_shape"] == [4]


def test_an_explicit_chunks_argument_wins():
    ds = xr.Dataset(
        {"v": xr.DataArray(np.arange(16, dtype=np.int32).reshape(4, 4), dims=["x", "y"])}
    )
    a = _write(ds, chunks={"v": [2, 2]})
    assert a.dataset("ds").array_meta("v")["chunk_shape"] == [2, 2]


def test_an_uneven_trailing_chunk_is_written():
    # 10 elements in chunks of 4: 4, 4, 2.
    arr = dask_array.arange(10, dtype=np.int32, chunks=4)  # type: ignore[arg-type]
    ds = xr.Dataset({"v": xr.DataArray(arr, dims=["x"])})
    meta = _write(ds).dataset("ds").array_meta("v")
    assert meta["shape"] == [10]
    assert meta["chunk_shape"] == [4]


def test_dask_blocks_stream_one_write_per_block(monkeypatch):
    """A dask-backed variable is written block by block, not materialized whole."""
    from atlas._atlas import DatasetWriter

    arr = dask_array.arange(16, dtype=np.int32, chunks=4)  # type: ignore[arg-type]
    ds = xr.Dataset({"v": xr.DataArray(arr, dims=["x"])})

    calls: list[tuple[str, list[int], tuple[int, ...]]] = []
    real_write_array = DatasetWriter.write_array

    def counting_write_array(self, name, start, data):
        calls.append((name, list(start), tuple(data.shape)))
        return real_write_array(self, name, start, data)

    monkeypatch.setattr(DatasetWriter, "write_array", counting_write_array)

    with tempfile.TemporaryDirectory() as d:
        with atlas.AtlasWriter.create(d) as w:
            w.add_xarray_dataset(ds, "ds")

    v_calls = [c for c in calls if c[0] == "v"]
    assert len(v_calls) == 4
    assert [c[1] for c in v_calls] == [[0], [4], [8], [12]]
    assert all(c[2] == (4,) for c in v_calls)


def test_eager_and_dask_variables_mix_in_one_dataset():
    ds = xr.Dataset(
        {
            "eager": xr.DataArray(np.arange(8, dtype=np.int32), dims=["x"]),
            "lazy": xr.DataArray(
                dask_array.arange(8, dtype=np.int32, chunks=2),  # type: ignore[arg-type]
                dims=["x"],
            ),
        }
    )
    view = _write(ds).dataset("ds")
    assert view.array_meta("eager")["chunk_shape"] == [8]
    assert view.array_meta("lazy")["chunk_shape"] == [2]


# ── fill values ──────────────────────────────────────────────────────


def test_a_fill_value_attribute_becomes_the_arrays_fill():
    arr = xr.DataArray(
        np.arange(4, dtype=np.int32), dims=["x"], attrs={"_FillValue": np.int32(-999)}
    )
    ds = xr.Dataset({"v": arr})
    a = _write(ds)
    assert a.dataset("ds").array_fill_value("v") == -999
    # And it is not left behind as an ordinary attribute.
    assert "_FillValue" not in a.array_attributes("ds", "v")


def test_float_arrays_default_to_a_nan_fill():
    ds = xr.Dataset({"v": xr.DataArray(np.arange(4, dtype=np.float32), dims=["x"])})
    assert np.isnan(_write(ds).dataset("ds").array_fill_value("v"))


def test_datetime_arrays_default_to_a_nat_fill():
    arr = np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[ns]")
    ds = xr.Dataset({"t": xr.DataArray(arr, dims=["x"])})
    nat = int(np.datetime64("NaT", "ns").view("int64"))
    assert _write(ds).dataset("ds").array_fill_value("t") == nat


def test_integer_arrays_get_no_fill_by_default():
    ds = xr.Dataset({"v": xr.DataArray(np.arange(4, dtype=np.int32), dims=["x"])})
    assert _write(ds).dataset("ds").array_fill_value("v") is None


def test_a_fill_value_dict_targets_named_variables():
    ds = xr.Dataset(
        {
            "a": xr.DataArray(np.arange(4, dtype=np.float32), dims=["x"]),
            "b": xr.DataArray(np.arange(4, dtype=np.float32), dims=["x"]),
        }
    )
    view = _write(ds, fill_value={"a": -1.0}).dataset("ds")
    assert view.array_fill_value("a") == -1.0
    assert np.isnan(view.array_fill_value("b"))  # untouched default


def test_a_scalar_fill_value_applies_to_every_numeric_array():
    ds = xr.Dataset(
        {
            "a": xr.DataArray(np.arange(4, dtype=np.float32), dims=["x"]),
            "b": xr.DataArray(np.arange(4, dtype=np.float64), dims=["x"]),
        }
    )
    view = _write(ds, fill_value=-1.0).dataset("ds")
    assert view.array_fill_value("a") == -1.0
    assert view.array_fill_value("b") == -1.0


def test_a_none_fill_value_disables_the_default():
    ds = xr.Dataset({"v": xr.DataArray(np.arange(4, dtype=np.float32), dims=["x"])})
    assert _write(ds, fill_value={"v": None}).dataset("ds").array_fill_value("v") is None


def test_a_fill_value_of_the_wrong_type_is_refused():
    arr = xr.DataArray(
        np.arange(4, dtype=np.int32), dims=["x"], attrs={"_FillValue": "not a number"}
    )
    ds = xr.Dataset({"v": arr})
    with tempfile.TemporaryDirectory() as d:
        with atlas.AtlasWriter.create(d) as w:
            with pytest.raises((TypeError, ValueError)):
                w.add_xarray_dataset(ds, "ds")


# ── strings ──────────────────────────────────────────────────────────


def test_missing_string_cells_are_filled_with_a_warning():
    arr = xr.DataArray(np.array(["a", None, "c", np.nan], dtype=object), dims=["x"])
    ds = xr.Dataset({"s": arr})
    with tempfile.TemporaryDirectory() as d:
        with pytest.warns(UserWarning, match="2 missing string"):
            with atlas.AtlasWriter.create(d) as w:
                w.add_xarray_dataset(ds, "ds")
        assert atlas.Atlas.open(d).dataset("ds").array_fill_value("s") == ""


def test_a_provided_string_fill_is_used():
    arr = xr.DataArray(np.array(["a", None], dtype=object), dims=["x"])
    ds = xr.Dataset({"s": arr})
    with tempfile.TemporaryDirectory() as d:
        with pytest.warns(UserWarning):
            with atlas.AtlasWriter.create(d) as w:
                w.add_xarray_dataset(ds, "ds", fill_value={"s": "n/a"})
        assert atlas.Atlas.open(d).dataset("ds").array_fill_value("s") == "n/a"


def test_complete_string_arrays_warn_about_nothing():
    arr = xr.DataArray(np.array(["a", "b"], dtype=object), dims=["x"])
    ds = xr.Dataset({"s": arr})
    with tempfile.TemporaryDirectory() as d:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with atlas.AtlasWriter.create(d) as w:
                w.add_xarray_dataset(ds, "ds")


# ── failure handling ─────────────────────────────────────────────────


def test_a_failed_write_leaves_no_trace_of_the_dataset():
    """An unsupported dtype after a valid variable must not half-write."""
    ds = xr.Dataset(
        {
            "ok": xr.DataArray(np.arange(4, dtype=np.float32), dims=["x"]),
            "bad": xr.DataArray(np.array([True, False, True, False]), dims=["x"]),
        }
    )
    with tempfile.TemporaryDirectory() as d:
        with atlas.AtlasWriter.create(d) as w:
            with pytest.raises(NotImplementedError):
                w.add_xarray_dataset(ds, "broken")
            # The collection carries on; the failed dataset simply never lands.
            w.add_xarray_dataset(_make_dataset(), "good")

        assert atlas.Atlas.open(d).list_datasets() == ["good"]


# ── the accessor ─────────────────────────────────────────────────────


def test_the_accessor_writes_like_the_method():
    ds = _make_dataset()
    with tempfile.TemporaryDirectory() as d:
        with atlas.AtlasWriter.create(d) as w:
            ds.atlas.write(w, "via_accessor")
            w.add_xarray_dataset(ds, "via_method")

        a = atlas.Atlas.open(d)
        one = a.dataset("via_accessor")
        two = a.dataset("via_method")
        assert one.list_arrays() == two.list_arrays()
        assert a.attributes("via_accessor") == a.attributes("via_method")
        for name in one.list_arrays():
            a_meta, b_meta = one.array_meta(name), two.array_meta(name)
            # NaN never equals itself, so compare the fill on its own terms.
            a_fill, b_fill = a_meta.pop("fill_value"), b_meta.pop("fill_value")
            assert a_meta == b_meta, name
            assert a_fill == b_fill or (
                isinstance(a_fill, float)
                and isinstance(b_fill, float)
                and np.isnan(a_fill)
                and np.isnan(b_fill)
            ), name


# ── the read path is gone ────────────────────────────────────────────


def test_reading_arrays_back_is_not_offered():
    ds = _make_dataset()
    a = _write(ds)
    for method in ("open_as_xarray_dataset", "open_as_many_xarray_dataset", "read_array"):
        assert not hasattr(a, method)
    assert not hasattr(a.dataset("ds"), "read_array")
    # The accessor writes only.
    assert not hasattr(ds.atlas, "read")


def test_the_coords_marker_is_not_surfaced_as_a_user_attribute():
    ds = _make_dataset()
    a = _write(ds)
    assert "_pyatlas_coords" not in a.attributes("ds")
    # It is still there underneath, which is how coords() answers.
    raw = a.dataset("ds").get_attribute("_pyatlas_coords")
    assert json.loads(raw) == ["lat", "lon"]
