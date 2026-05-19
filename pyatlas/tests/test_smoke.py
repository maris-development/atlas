"""End-to-end smoke test for pyatlas.

Run after `maturin develop`:
    pytest pyatlas/tests/test_smoke.py -v
or
    python pyatlas/tests/test_smoke.py
"""
import tempfile

import numpy as np
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
        ds.flush()

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
        ds.flush()

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
        ds.flush()

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
        ds.flush()

        s2 = pyatlas.Atlas.open(d)
        ds2 = s2.open_dataset("ds")
        assert ds2.get_attribute("flag") is True
        assert ds2.get_attribute("count") == 42
        assert ds2.get_attribute("ratio") == 0.5
        assert ds2.get_attribute("name") == "alpha"
        assert ds2.get_attribute("small_int") == 7
        assert abs(ds2.get_attribute("ratio32") - 1.25) < 1e-6


def test_delete_dataset_and_array():
    with tempfile.TemporaryDirectory() as d:
        s = pyatlas.Atlas.create(d)
        ds = s.create_dataset("keep")
        ds.define_array("a", dtype="i32", dims=["x"], shape=[2])
        ds.define_array("b", dtype="i32", dims=["x"], shape=[2])
        ds.delete_array("a")
        assert "a" not in ds.list_arrays()
        assert "b" in ds.list_arrays()
        ds.flush()

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


if __name__ == "__main__":
    test_numeric_roundtrip()
    test_all_numeric_dtypes()
    test_lz4_codec_roundtrip()
    test_attribute_dtypes()
    test_delete_dataset_and_array()
    test_array_meta()
    print("All smoke tests passed.")
