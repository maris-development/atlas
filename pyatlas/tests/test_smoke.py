"""End-to-end smoke test for pyatlas.

Run after `maturin develop`:
    pytest pyatlas/tests/test_smoke.py -v
or
    python pyatlas/tests/test_smoke.py
"""
import tempfile

import numpy as np
import pytest

import pyatlas


def test_numeric_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        s = pyatlas.Atlas.create(d, codec="zstd")
        ds = s.create_dataset("ds_jan")
        ds.define_array(
            "temp",
            dtype="float32",
            dims=["lat", "lon"],
            shape=[4, 4],
            chunk_shape=[2, 2],
        )
        data = np.full((4, 4), 20.0, dtype=np.float32)
        ds.write_array("temp", start=[0, 0], data=data)
        ds.set_attribute("month", 1)
        ds.set_attribute("station", "KNMI")
        s.flush()

        s2 = pyatlas.Atlas.open(d)
        assert s2.dataset_exists("ds_jan")
        assert s2.list_datasets() == ["ds_jan"]

        ds2 = s2.open_dataset("ds_jan")
        arr = ds2.read_array("temp")
        assert arr is not None
        assert arr.shape == (4, 4)
        assert arr.dtype == np.float32
        assert (arr == 20.0).all()

        # Partial read
        chunk = ds2.read_array("temp", start=[0, 0], shape=[2, 2])
        assert chunk is not None
        assert chunk.shape == (2, 2)

        # Attributes
        assert ds2.get_attribute("month") == 1
        assert ds2.get_attribute("station") == "KNMI"
        assert ds2.attributes() == {"month": 1, "station": "KNMI"}

        # Stats (populated on flush)
        stats = ds2.array_stats("temp")
        assert stats is not None
        assert stats["row_count"] == 16
        assert stats["min"] == 20.0
        assert stats["max"] == 20.0


def test_all_numeric_dtypes():
    dtypes_and_values = [
        ("int8", np.int8, [-1, 0, 1, 2]),
        ("int16", np.int16, [-1, 0, 1, 2]),
        ("int32", np.int32, [-1, 0, 1, 2]),
        ("int64", np.int64, [-1, 0, 1, 2]),
        ("uint8", np.uint8, [0, 1, 2, 3]),
        ("uint16", np.uint16, [0, 1, 2, 3]),
        ("uint32", np.uint32, [0, 1, 2, 3]),
        ("uint64", np.uint64, [0, 1, 2, 3]),
        ("float32", np.float32, [0.0, 1.5, 2.5, 3.5]),
        ("float64", np.float64, [0.0, 1.5, 2.5, 3.5]),
    ]
    with tempfile.TemporaryDirectory() as d:
        s = pyatlas.Atlas.create(d)
        ds = s.create_dataset("ds")
        for name, np_dtype, values in dtypes_and_values:
            ds.define_array(name, dtype=name, dims=["x"], shape=[4])
            data = np.array(values, dtype=np_dtype)
            ds.write_array(name, start=[0], data=data)
        s.flush()

        s2 = pyatlas.Atlas.open(d)
        ds2 = s2.open_dataset("ds")
        for name, np_dtype, values in dtypes_and_values:
            arr = ds2.read_array(name)
            assert arr is not None, name
            assert arr.dtype == np_dtype, name
            assert list(arr) == values, name


def test_lz4_codec_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        s = pyatlas.Atlas.create(d, codec="lz4")
        ds = s.create_dataset("ds")
        ds.define_array("arr", dtype="float32", dims=["x"], shape=[4])
        ds.write_array("arr", start=[0], data=np.array([1, 2, 3, 4], dtype=np.float32))
        s.flush()

        s2 = pyatlas.Atlas.open(d)
        ds2 = s2.open_dataset("ds")
        arr = ds2.read_array("arr")
        assert arr is not None
        assert (arr == [1, 2, 3, 4]).all()


def test_attribute_dtypes():
    with tempfile.TemporaryDirectory() as d:
        s = pyatlas.Atlas.create(d)
        ds = s.create_dataset("ds")
        ds.define_array("arr", dtype="f32", dims=["x"], shape=[2])
        ds.set_attribute("flag", True)
        ds.set_attribute("count", 42)
        ds.set_attribute("ratio", 0.5)
        ds.set_attribute("name", "alpha")
        ds.set_attribute("small_int", 7, dtype="int8")
        ds.set_attribute("ratio32", 1.25, dtype="float32")
        ds.set_attribute("created_at", 1700000000000000000, dtype="timestamp_nanoseconds")
        ds.set_attribute("updated_at", 1700000000000000001, dtype="datetime64[ns]")
        s.flush()

        s2 = pyatlas.Atlas.open(d)
        ds2 = s2.open_dataset("ds")
        assert ds2.get_attribute("flag") is True
        assert ds2.get_attribute("count") == 42
        assert ds2.get_attribute("ratio") == 0.5
        assert ds2.get_attribute("name") == "alpha"
        assert ds2.get_attribute("small_int") == 7
        assert abs(ds2.get_attribute("ratio32") - 1.25) < 1e-6
        assert ds2.get_attribute("created_at") == 1700000000000000000
        assert ds2.get_attribute("updated_at") == 1700000000000000001


def test_timestamp_ns_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        s = pyatlas.Atlas.create(d)
        ds = s.create_dataset("ts")
        ds.define_array("event_time", dtype="timestamp_nanoseconds", dims=["t"], shape=[3])

        data = np.array(
            [1700000000000000000, 1700000000000000001, 1700000000000000002],
            dtype=np.int64,
        )
        ds.write_array("event_time", start=[0], data=data)
        s.flush()

        s2 = pyatlas.Atlas.open(d)
        ds2 = s2.open_dataset("ts")
        assert ds2.array_meta("event_time")["dtype"] == "timestamp_nanoseconds"

        arr = ds2.read_array("event_time")
        assert arr is not None
        assert arr.dtype == np.dtype("datetime64[ns]")
        assert list(arr.view(np.int64)) == [
            1700000000000000000,
            1700000000000000001,
            1700000000000000002,
        ]


def test_string_array_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        s = pyatlas.Atlas.create(d)
        ds = s.create_dataset("strs")

        ds.define_array("names", dtype="string", dims=["i"], shape=[3])
        ds.write_array(
            "names",
            start=[0],
            data=np.array(["alpha", "beta", "gamma"], dtype=object),
        )

        # Fixed-size byte array |S5 -> stored as vlen string.
        ds.define_array("codes", dtype="string", dims=["i"], shape=[3])
        ds.write_array(
            "codes",
            start=[0],
            data=np.array([b"AAA", b"BB", b"C"], dtype="|S5"),
        )

        # Fixed-size unicode array |U4 -> stored as vlen string.
        ds.define_array("tags", dtype="string", dims=["i"], shape=[2])
        ds.write_array(
            "tags",
            start=[0],
            data=np.array(["foo", "barr"], dtype="|U4"),
        )

        ds.define_array("grid", dtype="string", dims=["r", "c"], shape=[2, 2])
        ds.write_array(
            "grid",
            start=[0, 0],
            data=np.array([["a", "b"], ["c", "d"]], dtype=object),
        )

        s.flush()

        s2 = pyatlas.Atlas.open(d)
        ds2 = s2.open_dataset("strs")

        assert ds2.array_meta("names")["dtype"] == "string"
        assert list(ds2.read_array("names")) == ["alpha", "beta", "gamma"]
        assert list(ds2.read_array("codes")) == ["AAA", "BB", "C"]
        assert list(ds2.read_array("tags")) == ["foo", "barr"]

        grid = ds2.read_array("grid")
        assert grid.shape == (2, 2)
        assert grid.tolist() == [["a", "b"], ["c", "d"]]


def test_zero_dim_scalar_roundtrip():
    """0-D (shape=()) arrays for all dtype families round-trip end-to-end."""
    with tempfile.TemporaryDirectory() as d:
        s = pyatlas.Atlas.create(d)
        ds = s.create_dataset("scalars")

        ds.define_array("count", dtype="int32", dims=[], shape=[])
        ds.write_array("count", start=[], data=np.array(7, dtype=np.int32))

        ds.define_array("ratio", dtype="float64", dims=[], shape=[])
        ds.write_array("ratio", start=[], data=np.array(0.25, dtype=np.float64))

        ds.define_array("name", dtype="string", dims=[], shape=[])
        ds.write_array("name", start=[], data=np.array("alpha", dtype=object))

        ds.define_array("created_at", dtype="timestamp_nanoseconds", dims=[], shape=[])
        ds.write_array("created_at", start=[], data=np.array(1700000000000000000, dtype=np.int64))

        s.flush()

        s2 = pyatlas.Atlas.open(d)
        ds2 = s2.open_dataset("scalars")

        assert ds2.array_meta("count")["shape"] == []
        assert ds2.read_array("count").item() == 7

        assert ds2.read_array("ratio").item() == 0.25

        assert ds2.read_array("name").item() == "alpha"

        ct = ds2.read_array("created_at")
        assert ct.dtype == np.dtype("datetime64[ns]")
        assert ct.view(np.int64).item() == 1700000000000000000


def test_atlas_batched_roundtrip():
    """Multiple datasets accumulated in memory; single atlas.flush persists everything."""
    with tempfile.TemporaryDirectory() as d:
        s = pyatlas.Atlas.create(d)
        a = s.create_dataset("a")
        a.define_array("temp", dtype="float32", dims=["x"], shape=[4])
        a.write_array("temp", start=[0], data=np.array([1, 2, 3, 4], dtype=np.float32))

        b = s.create_dataset("b")
        b.define_array("temp", dtype="float32", dims=["x"], shape=[4])
        b.write_array("temp", start=[0], data=np.array([5, 6, 7, 8], dtype=np.float32))

        s.flush()

        s2 = pyatlas.Atlas.open(d)
        assert sorted(s2.list_datasets()) == ["a", "b"]
        assert list(s2.open_dataset("a").read_array("temp")) == [1.0, 2.0, 3.0, 4.0]
        assert list(s2.open_dataset("b").read_array("temp")) == [5.0, 6.0, 7.0, 8.0]


def test_atlas_no_implicit_flush():
    """add_xr_dataset (or any mutation) doesn't auto-persist — a fresh Atlas
    sees nothing until atlas.flush() is called."""
    with tempfile.TemporaryDirectory() as d:
        s = pyatlas.Atlas.create(d)
        a = s.create_dataset("a")
        a.define_array("temp", dtype="float32", dims=["x"], shape=[2])
        a.write_array("temp", start=[0], data=np.array([1, 2], dtype=np.float32))
        # No flush.

        s2 = pyatlas.Atlas.open(d)
        assert s2.list_datasets() == []


def test_atlas_context_manager():
    """with atlas: ... auto-flushes on exit."""
    with tempfile.TemporaryDirectory() as d:
        with pyatlas.Atlas.create(d) as s:
            ds = s.create_dataset("x")
            ds.define_array("v", dtype="int32", dims=["i"], shape=[3])
            ds.write_array("v", start=[0], data=np.array([10, 20, 30], dtype=np.int32))

        s2 = pyatlas.Atlas.open(d)
        assert s2.list_datasets() == ["x"]
        assert list(s2.open_dataset("x").read_array("v")) == [10, 20, 30]


def test_delete_dataset_and_array():
    with tempfile.TemporaryDirectory() as d:
        s = pyatlas.Atlas.create(d)
        ds = s.create_dataset("keep")
        ds.define_array("a", dtype="i32", dims=["x"], shape=[2])
        ds.define_array("b", dtype="i32", dims=["x"], shape=[2])
        ds.delete_array("a")
        assert "a" not in ds.list_arrays()
        assert "b" in ds.list_arrays()
        s.flush()

        s.create_dataset("ghost")
        s.delete_dataset("ghost")
        assert not s.dataset_exists("ghost")


def test_array_meta():
    with tempfile.TemporaryDirectory() as d:
        s = pyatlas.Atlas.create(d)
        ds = s.create_dataset("ds")
        ds.define_array(
            "arr",
            dtype="float64",
            dims=["t", "x"],
            shape=[10, 20],
            chunk_shape=[5, 10],
        )
        meta = ds.array_meta("arr")
        assert meta["dtype"] == "float64"
        assert meta["shape"] == [10, 20]
        assert meta["chunk_shape"] == [5, 10]
        assert meta["dimension_names"] == ["t", "x"]


def test_fill_value_unwritten_cells_int32():
    """Cells we never write to come back as the declared fill value."""
    with tempfile.TemporaryDirectory() as d:
        s = pyatlas.Atlas.create(d)
        ds = s.create_dataset("ds")
        ds.define_array("arr", dtype="int32", dims=["x"], shape=[4], fill_value=-1)
        # Only write the first half — the trailing two cells stay unwritten.
        ds.write_array("arr", start=[0], data=np.array([10, 20], dtype=np.int32))
        s.flush()

        s2 = pyatlas.Atlas.open(d)
        arr = s2.open_dataset("ds").read_array("arr")
        assert arr is not None
        assert arr.dtype == np.int32
        assert arr.tolist() == [10, 20, -1, -1]


def test_fill_value_float64_nan():
    with tempfile.TemporaryDirectory() as d:
        s = pyatlas.Atlas.create(d)
        ds = s.create_dataset("ds")
        ds.define_array("arr", dtype="float64", dims=["x"], shape=[4], fill_value=float("nan"))
        ds.write_array("arr", start=[0], data=np.array([1.0, 2.0], dtype=np.float64))
        s.flush()

        s2 = pyatlas.Atlas.open(d)
        arr = s2.open_dataset("ds").read_array("arr")
        assert arr is not None
        assert arr[:2].tolist() == [1.0, 2.0]
        assert np.isnan(arr[2:]).all()


def test_fill_value_float_accepts_int():
    """Floats accept Python ints (coerced); no error."""
    with tempfile.TemporaryDirectory() as d:
        s = pyatlas.Atlas.create(d)
        ds = s.create_dataset("ds")
        ds.define_array("arr", dtype="float32", dims=["x"], shape=[2], fill_value=7)
        s.flush()  # define succeeded; no exception


def test_fill_value_uint_accepts_large():
    """uint64 accepts values larger than i64::MAX."""
    big = 2**63 + 5  # > i64::MAX, well within u64
    with tempfile.TemporaryDirectory() as d:
        s = pyatlas.Atlas.create(d)
        ds = s.create_dataset("ds")
        ds.define_array("arr", dtype="uint64", dims=["x"], shape=[4], fill_value=big)
        ds.write_array("arr", start=[0], data=np.array([1, 2], dtype=np.uint64))
        s.flush()

        s2 = pyatlas.Atlas.open(d)
        arr = s2.open_dataset("ds").read_array("arr")
        assert arr is not None
        assert arr.tolist() == [1, 2, big, big]


def test_fill_value_string_accepted():
    """String fill_value is stored without error (the underlying crate still
    returns "" for unwritten cells, but the binding shouldn't reject it)."""
    with tempfile.TemporaryDirectory() as d:
        s = pyatlas.Atlas.create(d)
        ds = s.create_dataset("ds")
        ds.define_array("arr", dtype="string", dims=["i"], shape=[2], fill_value="N/A")
        ds.write_array("arr", start=[0], data=np.array(["x", "y"], dtype=object))
        s.flush()

        s2 = pyatlas.Atlas.open(d)
        arr = s2.open_dataset("ds").read_array("arr")
        assert arr is not None
        assert list(arr) == ["x", "y"]


@pytest.mark.parametrize(
    "dtype, fill, exc",
    [
        ("int32", 1.5, TypeError),       # float for int
        ("uint32", -1, OverflowError),    # negative for uint
        ("uint8", 256, OverflowError),    # out-of-range uint
        ("int8", 200, OverflowError),     # out-of-range int
        ("float32", "x", TypeError),     # str for float
        ("float32", True, TypeError),    # bool for float
        ("int32", "x", TypeError),       # str for int
        ("string", 1, TypeError),        # int for string
        ("bool", 1, TypeError),          # int (not bool) for bool
    ],
)
def test_fill_value_type_mismatch_raises(dtype, fill, exc):
    with tempfile.TemporaryDirectory() as d:
        s = pyatlas.Atlas.create(d)
        ds = s.create_dataset("ds")
        with pytest.raises(exc):
            ds.define_array("arr", dtype=dtype, dims=["x"], shape=[2], fill_value=fill)


if __name__ == "__main__":
    test_numeric_roundtrip()
    test_all_numeric_dtypes()
    test_lz4_codec_roundtrip()
    test_attribute_dtypes()
    test_timestamp_ns_roundtrip()
    test_string_array_roundtrip()
    test_zero_dim_scalar_roundtrip()
    test_atlas_batched_roundtrip()
    test_atlas_no_implicit_flush()
    test_atlas_context_manager()
    test_delete_dataset_and_array()
    test_array_meta()
    print("All smoke tests passed.")
