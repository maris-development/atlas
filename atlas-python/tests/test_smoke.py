"""The low-level Python surface: writing a collection, then reading its metadata.

Python cannot read array data back — that is the Rust API's job — so these tests
assert on schemas, attributes, and the writer's error behaviour. Data
correctness is covered by the Rust suite, which reads a collection that these
tests write; see ``tests/golden.rs`` in the repository root.
"""

import numpy as np
import pytest

import atlas


# ── writing and reopening ────────────────────────────────────────────


def build(path, codec="zstd"):
    """A small collection covering the dtypes and shapes that matter."""
    with atlas.AtlasWriter.create(str(path), codec=codec) as w:
        ds = w.add_dataset("grid")
        ds.define_array(
            "temperature",
            dtype="float32",
            dims=["lat", "lon"],
            shape=[4, 6],
            chunk_shape=[2, 3],
            fill_value=float("nan"),
        )
        ds.write_array(
            "temperature",
            start=[0, 0],
            data=np.arange(24, dtype=np.float32).reshape(4, 6),
        )
        ds.define_array("counts", dtype="int64", dims=["lat"], shape=[4])
        ds.write_array("counts", start=[0], data=np.array([1, 2, 3, 4], dtype=np.int64))
        ds.set_attribute("month", 1)
        ds.set_array_attribute("temperature", "units", "celsius")
        ds.finish()

        other = w.add_dataset("labels")
        other.define_array("name", dtype="string", dims=["i"], shape=[3])
        other.write_array("name", start=[0], data=np.array(["a", "b", "c"], dtype=object))
        other.finish()
    return path


def test_a_written_collection_reopens_with_its_metadata(tmp_path):
    build(tmp_path)
    a = atlas.Atlas.open(str(tmp_path))

    assert a.list_datasets() == ["grid", "labels"]
    assert a.list_arrays() == ["counts", "name", "temperature"]
    assert a.dataset_count() == 2
    assert a.dataset_exists("grid")
    assert not a.dataset_exists("nope")

    grid = a.dataset("grid")
    assert grid.name == "grid"
    assert grid.ordinal == 0
    assert grid.list_arrays() == ["temperature", "counts"]

    meta = grid.array_meta("temperature")
    assert meta["dtype"] == "float32"
    assert meta["shape"] == [4, 6]
    assert meta["chunk_shape"] == [2, 3]
    assert meta["dimension_names"] == ["lat", "lon"]
    assert np.isnan(meta["fill_value"])

    assert grid.get_attribute("month") == 1
    assert grid.get_array_attribute("temperature", "units") == "celsius"
    assert dict(grid.array_attributes("temperature")) == {"units": "celsius"}
    # counts was never annotated.
    assert dict(grid.array_attributes("counts")) == {}


def test_the_collection_is_one_file_plus_an_optional_mask(tmp_path):
    build(tmp_path)
    assert (tmp_path / "data.atlas").exists()
    # No deletions yet, so no mask.
    assert not (tmp_path / "deleted.mask").exists()
    # And nothing else: no per-array directories.
    assert sorted(p.name for p in tmp_path.iterdir()) == ["data.atlas"]


def test_dunders_reflect_the_collection(tmp_path):
    build(tmp_path)
    a = atlas.Atlas.open(str(tmp_path))
    assert len(a) == 2
    assert "grid" in a
    assert "nope" not in a
    assert list(a) == ["grid", "labels"]

    grid = a.dataset("grid")
    assert len(grid) == 2
    assert "temperature" in grid
    assert "nope" not in grid


def test_segment_range_covers_a_standalone_array_format_file(tmp_path):
    build(tmp_path)
    a = atlas.Atlas.open(str(tmp_path))
    start, end = a.dataset("grid").segment_range
    assert start == 8  # right after the container header
    blob = (tmp_path / "data.atlas").read_bytes()[start:end]
    # array-format is footer-addressed, so a complete file ends in its magic.
    assert blob[-4:] == b"ARRF"


@pytest.mark.parametrize("codec", ["zstd", "lz4", "none"])
def test_every_codec_produces_a_readable_collection(tmp_path, codec):
    build(tmp_path, codec=codec)
    a = atlas.Atlas.open(str(tmp_path))
    assert a.list_datasets() == ["grid", "labels"]


def test_an_unknown_codec_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown codec"):
        atlas.AtlasWriter.create(str(tmp_path), codec="brotli")


# ── deletion ─────────────────────────────────────────────────────────


def test_deleting_hides_a_dataset_without_touching_the_container(tmp_path):
    build(tmp_path)
    before = (tmp_path / "data.atlas").stat().st_size

    a = atlas.Atlas.open(str(tmp_path))
    a.delete_dataset("labels")
    assert a.list_datasets() == ["grid"]
    assert atlas.Atlas.open(str(tmp_path)).list_datasets() == ["grid"]

    assert (tmp_path / "data.atlas").stat().st_size == before
    assert (tmp_path / "deleted.mask").exists()


def test_deleting_a_missing_dataset_raises_key_error(tmp_path):
    build(tmp_path)
    a = atlas.Atlas.open(str(tmp_path))
    a.delete_dataset("labels")
    with pytest.raises(KeyError):
        a.delete_dataset("labels")
    with pytest.raises(KeyError):
        a.dataset("labels")


def test_ordinals_survive_a_deletion(tmp_path):
    build(tmp_path)
    a = atlas.Atlas.open(str(tmp_path))
    assert a.dataset("labels").ordinal == 1
    a.delete_dataset("grid")
    reopened = atlas.Atlas.open(str(tmp_path))
    assert reopened.dataset("labels").ordinal == 1


# ── writer behaviour ─────────────────────────────────────────────────


def test_a_collection_with_no_datasets_is_valid(tmp_path):
    with atlas.AtlasWriter.create(str(tmp_path)):
        pass
    a = atlas.Atlas.open(str(tmp_path))
    assert a.list_datasets() == []
    assert len(a) == 0


def test_an_exception_inside_the_context_abandons_the_collection(tmp_path):
    class Boom(Exception):
        pass

    with pytest.raises(Boom):
        with atlas.AtlasWriter.create(str(tmp_path)) as w:
            ds = w.add_dataset("d")
            ds.define_array("x", dtype="float32", dims=["i"], shape=[2])
            ds.finish()
            raise Boom()

    # No trailer was written, so nothing opens.
    with pytest.raises((ValueError, RuntimeError, OSError)):
        atlas.Atlas.open(str(tmp_path))


def test_an_aborted_dataset_never_enters_the_collection(tmp_path):
    with atlas.AtlasWriter.create(str(tmp_path)) as w:
        gone = w.add_dataset("gone")
        gone.define_array("x", dtype="float32", dims=["i"], shape=[2])
        gone.abort()

        kept = w.add_dataset("kept")
        kept.define_array("x", dtype="float32", dims=["i"], shape=[2])
        kept.finish()

    assert atlas.Atlas.open(str(tmp_path)).list_datasets() == ["kept"]


def test_a_dataset_writer_works_as_a_context_manager(tmp_path):
    with atlas.AtlasWriter.create(str(tmp_path)) as w:
        with w.add_dataset("committed") as ds:
            ds.define_array("x", dtype="int32", dims=["i"], shape=[2])

        class Boom(Exception):
            pass

        with pytest.raises(Boom):
            with w.add_dataset("discarded") as ds:
                ds.define_array("x", dtype="int32", dims=["i"], shape=[2])
                raise Boom()

    assert atlas.Atlas.open(str(tmp_path)).list_datasets() == ["committed"]


def test_finishing_twice_is_an_error(tmp_path):
    w = atlas.AtlasWriter.create(str(tmp_path))
    w.finish()
    assert w.closed
    with pytest.raises(ValueError):
        w.finish()
    with pytest.raises(ValueError):
        w.add_dataset("d")


def test_duplicate_dataset_names_are_rejected(tmp_path):
    with atlas.AtlasWriter.create(str(tmp_path)) as w:
        w.add_dataset("d").finish()
        with pytest.raises(FileExistsError):
            w.add_dataset("d")


def test_duplicate_array_names_are_rejected(tmp_path):
    with atlas.AtlasWriter.create(str(tmp_path)) as w:
        ds = w.add_dataset("d")
        ds.define_array("x", dtype="int32", dims=["i"], shape=[2])
        with pytest.raises(FileExistsError):
            ds.define_array("x", dtype="int32", dims=["i"], shape=[2])
        ds.finish()


@pytest.mark.parametrize("name", ["", "_hidden", "a/b", "..", "."])
def test_invalid_names_are_rejected(tmp_path, name):
    with atlas.AtlasWriter.create(str(tmp_path)) as w:
        with pytest.raises(ValueError):
            w.add_dataset(name)
        ds = w.add_dataset("ok")
        with pytest.raises(ValueError):
            ds.define_array(name, dtype="int32", dims=["i"], shape=[2])
        ds.finish()


def test_writing_an_undefined_array_raises_key_error(tmp_path):
    with atlas.AtlasWriter.create(str(tmp_path)) as w:
        ds = w.add_dataset("d")
        with pytest.raises(KeyError):
            ds.write_array("nope", start=[0], data=np.zeros(2, dtype=np.float32))
        with pytest.raises(KeyError):
            ds.set_array_attribute("nope", "k", 1)
        ds.finish()


def test_a_dtype_mismatch_on_write_raises_type_error(tmp_path):
    with atlas.AtlasWriter.create(str(tmp_path)) as w:
        ds = w.add_dataset("d")
        ds.define_array("x", dtype="float32", dims=["i"], shape=[4])
        with pytest.raises(TypeError):
            ds.write_array("x", start=[0], data=np.zeros(4, dtype=np.float64))
        ds.finish()


def test_a_non_contiguous_input_is_rejected(tmp_path):
    with atlas.AtlasWriter.create(str(tmp_path)) as w:
        ds = w.add_dataset("d")
        ds.define_array("x", dtype="float32", dims=["i"], shape=[4])
        strided = np.arange(8, dtype=np.float32)[::2]
        assert not strided.flags["C_CONTIGUOUS"]
        with pytest.raises(ValueError, match="C-contiguous"):
            ds.write_array("x", start=[0], data=strided)
        ds.finish()


def test_an_unknown_dtype_string_is_rejected(tmp_path):
    with atlas.AtlasWriter.create(str(tmp_path)) as w:
        ds = w.add_dataset("d")
        with pytest.raises(ValueError, match="unknown dtype"):
            ds.define_array("x", dtype="complex128", dims=["i"], shape=[2])
        ds.finish()


# ── fill values ──────────────────────────────────────────────────────


def test_fill_values_are_type_checked(tmp_path):
    with atlas.AtlasWriter.create(str(tmp_path)) as w:
        ds = w.add_dataset("d")
        with pytest.raises(TypeError):
            ds.define_array("a", dtype="int32", dims=["i"], shape=[2], fill_value=1.5)
        with pytest.raises(TypeError):
            ds.define_array("b", dtype="string", dims=["i"], shape=[2], fill_value=7)
        with pytest.raises(OverflowError):
            ds.define_array("c", dtype="int8", dims=["i"], shape=[2], fill_value=1000)
        with pytest.raises(OverflowError):
            ds.define_array("e", dtype="uint8", dims=["i"], shape=[2], fill_value=-1)
        ds.finish()


def test_fill_values_round_trip_per_dtype(tmp_path):
    cases = [
        ("i", "int32", -7),
        ("u", "uint16", 9),
        ("f", "float64", 1.5),
        ("s", "string", "n/a"),
        ("t", "timestamp_nanoseconds", -(2**63)),
    ]
    with atlas.AtlasWriter.create(str(tmp_path)) as w:
        ds = w.add_dataset("d")
        for name, dtype, fill in cases:
            ds.define_array(name, dtype=dtype, dims=["i"], shape=[2], fill_value=fill)
        ds.finish()

    view = atlas.Atlas.open(str(tmp_path)).dataset("d")
    for name, _dtype, fill in cases:
        assert view.array_fill_value(name) == fill, name


# ── attributes ───────────────────────────────────────────────────────


def test_attribute_types_round_trip(tmp_path):
    values = {
        "an_int": 42,
        "a_float": 2.5,
        "a_str": "hello",
        "a_bool": True,
        "a_list": [1, 2, 3],
        "a_str_list": ["a", "b"],
    }
    with atlas.AtlasWriter.create(str(tmp_path)) as w:
        ds = w.add_dataset("d")
        for key, value in values.items():
            ds.set_attribute(key, value)
        ds.finish()

    view = atlas.Atlas.open(str(tmp_path)).dataset("d")
    got = dict(view.attributes())
    for key, value in values.items():
        assert got[key] == value, key


def test_a_repeated_attribute_key_keeps_the_last_value(tmp_path):
    with atlas.AtlasWriter.create(str(tmp_path)) as w:
        ds = w.add_dataset("d")
        ds.set_attribute("k", 1)
        ds.set_attribute("k", 2)
        ds.finish()
    view = atlas.Atlas.open(str(tmp_path)).dataset("d")
    assert view.get_attribute("k") == 2
    assert len(view.attributes()) == 1


def test_a_dtype_hint_narrows_an_attribute(tmp_path):
    with atlas.AtlasWriter.create(str(tmp_path)) as w:
        ds = w.add_dataset("d")
        ds.set_attribute("small", 7, dtype="int32")
        ds.finish()
    assert atlas.Atlas.open(str(tmp_path)).dataset("d").get_attribute("small") == 7


# ── rejecting things that are not collections ────────────────────────


def test_opening_an_empty_directory_fails_clearly(tmp_path):
    with pytest.raises(ValueError, match="not an atlas collection"):
        atlas.Atlas.open(str(tmp_path))


def test_opening_an_atlas_014_store_says_so(tmp_path):
    (tmp_path / "atlas.json").write_text('{"version": 3}')
    with pytest.raises(ValueError, match="0.14"):
        atlas.Atlas.open(str(tmp_path))


def test_a_damaged_mask_is_reported(tmp_path):
    build(tmp_path)
    (tmp_path / "deleted.mask").write_bytes(b"not a mask at all")
    with pytest.raises(RuntimeError, match="mask"):
        atlas.Atlas.open(str(tmp_path))


# ── the read API is deliberately absent ──────────────────────────────


@pytest.mark.parametrize(
    "method",
    [
        "read_array",
        "read_arrays",
        "read_array_across",
        "read_array_across_stacked",
        "open_as_xarray_dataset",
        "open_as_many_xarray_dataset",
        "pruning_index",
        "column_summaries",
        "merged_schema",
        "create_dataset",
        "flush",
        "compact",
    ],
)
def test_mutation_and_read_apis_are_gone(tmp_path, method):
    build(tmp_path)
    a = atlas.Atlas.open(str(tmp_path))
    assert not hasattr(a, method), f"Atlas should not expose {method}"
    assert not hasattr(a.dataset("grid"), method), f"DatasetView should not expose {method}"


# ── object store backend ─────────────────────────────────────────────


def test_a_collection_round_trips_through_an_obstore_handle(tmp_path):
    # LocalStore rather than MemoryStore: when the installed obstore was built
    # against a different pyo3-object_store than these bindings, a handle
    # crosses the boundary by being reconstructed from its configuration.
    # A LocalStore has a path to rebuild from; a MemoryStore has nothing.
    obstore = pytest.importorskip("obstore")
    store = obstore.store.LocalStore(str(tmp_path))

    with atlas.AtlasWriter.create(store) as w:
        ds = w.add_dataset("d")
        ds.define_array("x", dtype="float32", dims=["i"], shape=[3])
        ds.write_array("x", start=[0], data=np.array([1, 2, 3], dtype=np.float32))
        ds.set_attribute("k", "v")
        ds.finish()

    a = atlas.Atlas.open(store)
    assert a.list_datasets() == ["d"]
    assert a.dataset("d").get_attribute("k") == "v"
    a.delete_dataset("d")
    assert atlas.Atlas.open(store).list_datasets() == []


def test_version_is_exposed():
    assert atlas.__version__.count(".") == 2
