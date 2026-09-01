"""The five operations, as a library.

These tests check no array *value*, because Python cannot read one. The Rust
suite checks the bytes these tests write. See ``tests/cross_fixture.rs``.
"""

import logging

import numpy as np
import pytest
import xarray as xr

import atlas

from conftest import make_dataset


# ── create ───────────────────────────────────────────────────────────


def test_create_makes_one_dataset_per_netcdf_file(netcdf_dir, tmp_path):
    dest = tmp_path / "c"
    result = atlas.create(netcdf_dir, str(dest))

    assert result["dataset_count"] == 3
    assert result["written"] == ["2024-01.nc", "2024-02.nc", "2024-03.nc"]
    assert result["skipped"] == []
    # One file. A mask appears only after a remove.
    assert sorted(p.name for p in dest.iterdir()) == ["data.atlas"]


def test_datasets_are_named_after_the_whole_file_name(netcdf_dir, tmp_path):
    atlas.create(netcdf_dir, str(tmp_path / "c"))
    assert atlas.list_datasets(str(tmp_path / "c")) == [
        "2024-01.nc",
        "2024-02.nc",
        "2024-03.nc",
    ]


def test_create_is_ordered_by_filename(netcdf_dir, tmp_path):
    """An ordinal comes from the write order, so that order must be stable."""
    atlas.create(netcdf_dir, str(tmp_path / "a"))
    atlas.create(netcdf_dir, str(tmp_path / "b"))
    for name in atlas.list_datasets(str(tmp_path / "a")):
        assert atlas.describe(str(tmp_path / "a"), name)["ordinal"] == (
            atlas.describe(str(tmp_path / "b"), name)["ordinal"]
        )


def test_create_finds_files_recursively_when_asked(tmp_path):
    nested = tmp_path / "nc" / "2024" / "q1"
    nested.mkdir(parents=True)
    make_dataset(1).to_netcdf(nested / "jan.nc")

    flat = atlas.find_netcdf_files(tmp_path / "nc")
    assert flat == []

    found = atlas.find_netcdf_files(tmp_path / "nc", recursive=True)
    assert [p.name for p in found] == ["jan.nc"]

    atlas.create(tmp_path / "nc", str(tmp_path / "c"), recursive=True)
    assert atlas.list_datasets(str(tmp_path / "c")) == ["jan.nc"]


def test_an_empty_directory_is_an_error(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(atlas.AtlasError, match="no NetCDF files"):
        atlas.create(tmp_path / "empty", str(tmp_path / "c"))


def test_a_missing_directory_is_an_error(tmp_path):
    with pytest.raises(atlas.AtlasError, match="not a directory"):
        atlas.create(tmp_path / "nope", str(tmp_path / "c"))


def test_two_files_with_the_same_name_collide(tmp_path):
    src = tmp_path / "nc"
    (src / "a").mkdir(parents=True)
    (src / "b").mkdir()
    make_dataset(1).to_netcdf(src / "a" / "jan.nc")
    make_dataset(2).to_netcdf(src / "b" / "jan.nc")

    with pytest.raises(atlas.AtlasError, match="duplicate dataset name"):
        atlas.create(src, str(tmp_path / "c"), recursive=True)


def test_a_bad_file_abandons_the_collection_by_default(tmp_path):
    src = tmp_path / "nc"
    src.mkdir()
    make_dataset(1).to_netcdf(src / "good.nc")
    # bool is no supported array dtype.
    xr.Dataset({"flag": xr.DataArray(np.array([True, False]), dims=["x"])}).to_netcdf(
        src / "bad.nc"
    )

    with pytest.raises(atlas.AtlasError):
        atlas.create(src, str(tmp_path / "c"))
    # No trailer landed, so nothing opens.
    with pytest.raises((ValueError, atlas.AtlasError)):
        atlas.list_datasets(str(tmp_path / "c"))


def test_skip_errors_keeps_the_good_files(tmp_path):
    src = tmp_path / "nc"
    src.mkdir()
    make_dataset(1).to_netcdf(src / "good.nc")
    xr.Dataset({"flag": xr.DataArray(np.array([True, False]), dims=["x"])}).to_netcdf(
        src / "bad.nc"
    )

    result = atlas.create(src, str(tmp_path / "c"), on_error="skip")
    assert result["written"] == ["good.nc"]
    assert len(result["skipped"]) == 1
    assert result["skipped"][0]["file"].endswith("bad.nc")
    assert atlas.list_datasets(str(tmp_path / "c")) == ["good.nc"]


def test_progress_is_called_per_dataset(netcdf_dir, tmp_path):
    seen = []
    atlas.create(netcdf_dir, str(tmp_path / "c"), progress=seen.append)
    assert seen == ["2024-01.nc", "2024-02.nc", "2024-03.nc"]


@pytest.mark.parametrize("codec", ["zstd", "lz4", "none"])
def test_every_codec_round_trips(netcdf_dir, tmp_path, codec):
    dest = tmp_path / codec
    atlas.create(netcdf_dir, str(dest), codec=codec)
    assert atlas.info(str(dest))["codec"] == codec
    assert len(atlas.list_datasets(str(dest))) == 3


# ── unsupported dtypes ───────────────────────────────────────────────


@pytest.fixture
def netcdf_dir_with_a_bool(tmp_path):
    """One file holding a bool variable, which atlas cannot store."""
    d = tmp_path / "nc"
    d.mkdir()
    xr.Dataset(
        data_vars={
            "temperature": xr.DataArray(np.arange(6, dtype=np.float32), dims=["x"]),
            "flag": xr.DataArray(np.array([True, False] * 3), dims=["x"]),
        },
        coords={"x": ("x", np.arange(6, dtype=np.float64))},
        attrs={"source": "test"},
    ).to_netcdf(d / "a.nc")
    return d


def test_an_unsupported_dtype_fails_the_file_by_default(
    netcdf_dir_with_a_bool, tmp_path
):
    with pytest.raises(atlas.AtlasError, match="bool"):
        atlas.create(netcdf_dir_with_a_bool, str(tmp_path / "c"))


def test_skipping_an_unsupported_array_keeps_the_rest_of_the_dataset(
    netcdf_dir_with_a_bool, tmp_path
):
    dest = tmp_path / "c"
    result = atlas.create(netcdf_dir_with_a_bool, str(dest), on_unsupported="skip")

    assert result["written"] == ["a.nc"]
    assert result["skipped"] == []
    assert result["skipped_arrays"] == [
        {
            "array": "flag",
            "dtype": "bool",
            "error": result["skipped_arrays"][0]["error"],
            "dataset": "a.nc",
        }
    ]
    assert "not supported" in result["skipped_arrays"][0]["error"]

    # The supported arrays and the attributes all landed.
    detail = atlas.describe(str(dest), "a.nc")
    assert [a["name"] for a in detail["arrays"]] == ["x", "temperature"]
    assert detail["attributes"] == {"source": "test"}


def test_a_skipped_array_leaves_no_partial_array(netcdf_dir_with_a_bool, tmp_path):
    dest = tmp_path / "c"
    atlas.create(netcdf_dir_with_a_bool, str(dest), on_unsupported="skip")
    # The name is absent from the schema, not present and empty.
    assert "flag" not in atlas.info(str(dest))["distinct_arrays"]


def test_an_unknown_on_unsupported_mode_is_refused(netcdf_dir, tmp_path):
    with pytest.raises(atlas.AtlasError, match="on_unsupported"):
        atlas.create(netcdf_dir, str(tmp_path / "c"), on_unsupported="sometimes")


def test_a_clean_ingest_reports_no_skipped_arrays(netcdf_dir, tmp_path):
    result = atlas.create(netcdf_dir, str(tmp_path / "c"), on_unsupported="skip")
    assert result["skipped_arrays"] == []


# ── the log file ─────────────────────────────────────────────────────


def test_the_log_file_records_a_skipped_array(netcdf_dir_with_a_bool, tmp_path):
    log = tmp_path / "ingest.log"
    handler = atlas.log_to_file(log)
    try:
        atlas.create(netcdf_dir_with_a_bool, str(tmp_path / "c"), on_unsupported="skip")
    finally:
        logging.getLogger("atlas").removeHandler(handler)
        handler.close()

    text = log.read_text()
    assert "WARNING" in text
    # The line names the file, the array, the dtype, and the reason.
    assert "a.nc" in text
    assert "skipped array 'flag' of dtype bool" in text
    assert "is not supported by atlas" in text
    assert "wrote 1 dataset(s)" in text


def test_the_log_file_records_a_skipped_file(netcdf_dir, tmp_path):
    (netcdf_dir / "broken.nc").write_bytes(b"not a netcdf file")
    log = tmp_path / "ingest.log"
    handler = atlas.log_to_file(log)
    try:
        result = atlas.create(netcdf_dir, str(tmp_path / "c"), on_error="skip")
    finally:
        logging.getLogger("atlas").removeHandler(handler)
        handler.close()

    assert len(result["skipped"]) == 1
    text = log.read_text()
    assert "skipping" in text and "broken.nc" in text


def test_the_log_file_appends_across_runs(netcdf_dir, tmp_path):
    log = tmp_path / "ingest.log"
    for run in (1, 2):
        handler = atlas.log_to_file(log)
        try:
            atlas.create(netcdf_dir, str(tmp_path / f"c{run}"))
        finally:
            logging.getLogger("atlas").removeHandler(handler)
            handler.close()
    assert log.read_text().count("ingesting 3 file(s)") == 2


def test_atlas_logs_nowhere_by_default(netcdf_dir, tmp_path, caplog):
    # The library adds no handler of its own, so a host application decides.
    assert logging.getLogger("atlas").handlers == [
        h for h in logging.getLogger("atlas").handlers if isinstance(h, logging.NullHandler)
    ]
    with caplog.at_level(logging.INFO, logger="atlas"):
        atlas.create(netcdf_dir, str(tmp_path / "c"))
    # Records still exist for anyone who attaches a handler.
    assert any("ingesting" in r.message for r in caplog.records)


# ── chunked ingest ───────────────────────────────────────────────────


def test_a_large_variable_streams_in_blocks(tmp_path, monkeypatch):
    """A file above the block budget lands block by block."""
    from atlas._atlas import DatasetWriter

    src = tmp_path / "nc"
    src.mkdir()
    # 8 MiB of float64, under a 1 MiB block budget.
    rows, cols = 1024, 1024
    xr.Dataset(
        {"big": (("y", "x"), np.zeros((rows, cols), dtype=np.float64))}
    ).to_netcdf(src / "big.nc")

    calls = []
    real = DatasetWriter.write_array

    def counting(self, name, start, data):
        calls.append((name, tuple(data.shape)))
        return real(self, name, start, data)

    monkeypatch.setattr(DatasetWriter, "write_array", counting)
    atlas.create(src, str(tmp_path / "c"), chunk_size="1MiB")

    blocks = [c for c in calls if c[0] == "big"]
    assert len(blocks) > 1, "a large variable should not be written in one call"
    # No single block is the whole array.
    assert all(shape != (rows, cols) for _, shape in blocks)

    stored = atlas.describe(str(tmp_path / "c"), "big.nc")["arrays"][0]
    assert stored["shape"] == [rows, cols]
    assert stored["chunk_shape"] != [rows, cols]


def test_chunk_size_controls_the_stored_chunk_shape(tmp_path):
    src = tmp_path / "nc"
    src.mkdir()
    xr.Dataset(
        {"big": (("y", "x"), np.zeros((1024, 1024), dtype=np.float64))}
    ).to_netcdf(src / "big.nc")

    def chunk_shape(dest, **kwargs):
        atlas.create(src, str(dest), **kwargs)
        return atlas.describe(str(dest), "big.nc")["arrays"][0]["chunk_shape"]

    small = chunk_shape(tmp_path / "small", chunk_size="1MiB")
    large = chunk_shape(tmp_path / "large", chunk_size="64MiB")

    # A larger budget gives larger blocks, and fewer of them.
    assert np.prod(small) < np.prod(large)
    # The larger budget covers the whole 8 MiB array in one block.
    assert large == [1024, 1024]


def test_small_files_still_land_as_a_single_chunk(netcdf_dir, tmp_path):
    """Auto chunking must not split an array that fits with room to spare."""
    atlas.create(netcdf_dir, str(tmp_path / "c"))
    for array in atlas.describe(str(tmp_path / "c"), "2024-01.nc")["arrays"]:
        assert array["chunk_shape"] == array["shape"], array["name"]


def test_open_chunks_none_reads_each_variable_whole(tmp_path, monkeypatch):
    from atlas._atlas import DatasetWriter

    src = tmp_path / "nc"
    src.mkdir()
    xr.Dataset(
        {"big": (("y", "x"), np.zeros((1024, 1024), dtype=np.float64))}
    ).to_netcdf(src / "big.nc")

    calls = []
    real = DatasetWriter.write_array
    monkeypatch.setattr(
        DatasetWriter,
        "write_array",
        lambda self, name, start, data: (
            calls.append((name, tuple(data.shape))), real(self, name, start, data)
        )[1],
    )
    atlas.create(src, str(tmp_path / "c"), open_chunks=None, chunk_size="1MiB")

    blocks = [c for c in calls if c[0] == "big"]
    assert blocks == [("big", (1024, 1024))], "open_chunks=None should not chunk"


def test_open_chunks_native_uses_the_files_own_chunking(tmp_path):
    src = tmp_path / "nc"
    src.mkdir()
    # Ask netCDF4 for one exact on-disk chunking.
    xr.Dataset(
        {"big": (("y", "x"), np.zeros((512, 512), dtype=np.float64))}
    ).to_netcdf(
        src / "big.nc",
        engine="netcdf4",
        encoding={"big": {"chunksizes": (128, 256), "zlib": True}},
    )

    atlas.create(src, str(tmp_path / "c"), open_chunks="native")
    stored = atlas.describe(str(tmp_path / "c"), "big.nc")["arrays"][0]
    assert stored["chunk_shape"] == [128, 256]


def test_open_chunks_accepts_an_explicit_dict(tmp_path):
    src = tmp_path / "nc"
    src.mkdir()
    xr.Dataset(
        {"big": (("y", "x"), np.zeros((512, 512), dtype=np.float64))}
    ).to_netcdf(src / "big.nc")

    atlas.create(src, str(tmp_path / "c"), open_chunks={"y": 128, "x": 256})
    stored = atlas.describe(str(tmp_path / "c"), "big.nc")["arrays"][0]
    assert stored["chunk_shape"] == [128, 256]


@pytest.mark.parametrize("bad", ["sometimes", 42, 3.5])
def test_an_unknown_open_chunks_mode_is_rejected(netcdf_dir, tmp_path, bad):
    with pytest.raises(atlas.AtlasError, match="open_chunks"):
        atlas.create(netcdf_dir, str(tmp_path / "c"), open_chunks=bad)


def test_explicit_chunks_reach_the_stored_array(netcdf_dir, tmp_path):
    dest = tmp_path / "c"
    atlas.create(netcdf_dir, str(dest), chunks={"temperature": [2, 3]})
    arrays = {a["name"]: a for a in atlas.describe(str(dest), "2024-01.nc")["arrays"]}
    assert arrays["temperature"]["chunk_shape"] == [2, 3]


def test_on_error_must_be_a_known_mode(netcdf_dir, tmp_path):
    with pytest.raises(atlas.AtlasError, match="on_error"):
        atlas.create(netcdf_dir, str(tmp_path / "c"), on_error="explode")


# ── remove ───────────────────────────────────────────────────────────


def test_remove_takes_several_datasets_in_one_call(collection):
    result = atlas.remove(str(collection), ["2024-01.nc", "2024-03.nc"])
    assert result["removed"] == ["2024-01.nc", "2024-03.nc"]
    assert result["remaining"] == 1
    assert atlas.list_datasets(str(collection)) == ["2024-02.nc"]


def test_remove_accepts_the_netcdf_path_it_came_from(collection, netcdf_dir):
    atlas.remove(str(collection), [netcdf_dir / "2024-02.nc"])
    assert atlas.list_datasets(str(collection)) == ["2024-01.nc", "2024-03.nc"]


def test_remove_writes_a_mask_and_leaves_the_container_alone(collection):
    before = (collection / "data.atlas").stat().st_size
    atlas.remove(str(collection), ["2024-01.nc"])
    assert (collection / "data.atlas").stat().st_size == before
    assert (collection / "deleted.mask").exists()


def test_removals_accumulate_across_calls(collection):
    atlas.remove(str(collection), ["2024-01.nc"])
    atlas.remove(str(collection), ["2024-03.nc"])
    assert atlas.list_datasets(str(collection)) == ["2024-02.nc"]


def test_removing_many_datasets_is_one_call(collection, monkeypatch):
    from atlas import _atlas

    # One mask write covers the batch, so a big removal costs what one costs.
    calls = []
    original = _atlas.Atlas.delete_datasets

    def spy(self, names):
        calls.append(list(names))
        return original(self, names)

    monkeypatch.setattr(_atlas.Atlas, "delete_datasets", spy)
    result = atlas.remove(str(collection), ["2024-01.nc", "2024-03.nc"])

    assert calls == [["2024-01.nc", "2024-03.nc"]]
    assert result["removed"] == ["2024-01.nc", "2024-03.nc"]
    assert result["remaining"] == 1
    assert atlas.list_datasets(str(collection)) == ["2024-02.nc"]


def test_a_repeated_target_counts_once(collection):
    result = atlas.remove(str(collection), ["2024-01.nc", "2024-01.nc"])
    assert result["removed"] == ["2024-01.nc"]
    assert result["remaining"] == 2


def test_removing_something_absent_is_an_error(collection):
    with pytest.raises(atlas.AtlasError, match="not in the collection"):
        atlas.remove(str(collection), ["nope"])


def test_missing_ok_reports_instead_of_raising(collection):
    result = atlas.remove(str(collection), ["2024-01.nc", "nope"], missing_ok=True)
    assert result["removed"] == ["2024-01.nc"]
    assert result["missing"] == ["nope"]


def test_removing_nothing_is_an_error(collection):
    with pytest.raises(atlas.AtlasError, match="no datasets given"):
        atlas.remove(str(collection), [])


def test_ordinals_do_not_shift_when_a_dataset_is_removed(collection):
    before = atlas.describe(str(collection), "2024-03.nc")["ordinal"]
    atlas.remove(str(collection), ["2024-01.nc"])
    assert atlas.describe(str(collection), "2024-03.nc")["ordinal"] == before


# ── list ─────────────────────────────────────────────────────────────


def test_list_applies_the_mask(collection):
    assert len(atlas.list_datasets(str(collection))) == 3
    atlas.remove(str(collection), ["2024-02.nc"])
    assert atlas.list_datasets(str(collection)) == ["2024-01.nc", "2024-03.nc"]


def test_listing_a_non_collection_fails_clearly(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="not an atlas collection"):
        atlas.list_datasets(str(tmp_path / "empty"))


# ── describe ─────────────────────────────────────────────────────────


def test_describe_reports_the_whole_dataset(collection):
    d = atlas.describe(str(collection), "2024-01.nc")

    assert d["name"] == "2024-01.nc"
    assert d["ordinal"] == 0
    assert d["dimensions"] == {"lat": 4, "lon": 6}
    assert sorted(d["coordinates"]) == ["lat", "lon", "time"]
    assert d["attributes"] == {"month": 1, "source": "test", "bounds": [1.0, 2.0]}
    # The coordinate marker is internal to atlas, and no user attribute.
    assert "_pyatlas_coords" not in d["attributes"]

    start, end = d["segment_range"]
    assert 0 < start < end


def test_describe_reports_each_array(collection):
    arrays = {a["name"]: a for a in atlas.describe(str(collection), "2024-01.nc")["arrays"]}
    assert set(arrays) == {"lat", "lon", "time", "temperature", "counts", "station"}

    temp = arrays["temperature"]
    assert temp["dtype"] == "float32"
    assert temp["shape"] == [4, 6]
    assert temp["dimensions"] == ["lat", "lon"]
    assert temp["attributes"] == {
        "units": "celsius",
        "long_name": "surface temperature",
    }
    assert not temp["is_coordinate"]
    assert arrays["lat"]["is_coordinate"]

    assert arrays["time"]["dtype"] == "timestamp_nanoseconds"
    assert arrays["station"]["dtype"] == "string"
    assert arrays["counts"]["dtype"] == "int64"


def test_describe_reports_the_statistics_recorded_at_write_time(collection):
    arrays = {a["name"]: a for a in atlas.describe(str(collection), "2024-01.nc")["arrays"]}

    # temperature is arange(24) + 1. That gives 1..24, with nothing missing.
    temp = arrays["temperature"]["stats"]
    assert temp["row_count"] == 24
    assert temp["null_count"] == 0
    assert temp["min"] == 1.0
    assert temp["max"] == 24.0

    assert arrays["counts"]["stats"]["min"] == 10
    assert arrays["counts"]["stats"]["max"] == 40

    # A string compares lexicographically, and comes back as bytes.
    station = arrays["station"]["stats"]
    assert station["min"] == b"a"
    assert station["max"] == b"d"
    assert station["row_count"] == 4


def test_describe_accepts_a_netcdf_path(collection, netcdf_dir):
    by_name = atlas.describe(str(collection), "2024-01.nc")
    by_path = atlas.describe(str(collection), netcdf_dir / "2024-01.nc")
    # Two NaN fill values never compare equal. Compare everything else.
    for key in ("name", "ordinal", "segment_range", "dimensions", "coordinates",
                "attributes"):
        assert by_name[key] == by_path[key], key
    assert [a["name"] for a in by_name["arrays"]] == [
        a["name"] for a in by_path["arrays"]
    ]


def test_describing_a_missing_dataset_is_an_error(collection):
    with pytest.raises(atlas.AtlasError, match="no dataset"):
        atlas.describe(str(collection), "nope")


def test_describing_a_removed_dataset_is_an_error(collection):
    atlas.remove(str(collection), ["2024-01.nc"])
    with pytest.raises(atlas.AtlasError, match="removed"):
        atlas.describe(str(collection), "2024-01.nc")


# ── info ─────────────────────────────────────────────────────────────


def test_info_summarises_the_collection(collection):
    i = atlas.info(str(collection))

    assert i["format_version"] == 1
    assert i["codec"] == "zstd"
    assert i["dataset_count"] == 3
    assert i["total_datasets"] == 3
    assert i["deleted_count"] == 0
    assert i["container_bytes"] == (collection / "data.atlas").stat().st_size
    assert i["created_unix_ms"] > 0
    assert i["distinct_arrays"] == [
        "counts",
        "lat",
        "lon",
        "station",
        "temperature",
        "time",
    ]
    # All three months declare the same arrays, so one schema is stored.
    assert i["interned_schemas"] == 1


def test_info_folds_array_stats_over_the_collection(collection):
    i = atlas.info(str(collection))
    stats = i["array_stats"]

    # Every distinct array gets an entry.
    assert set(stats) == set(i["distinct_arrays"])

    # Three months of a 4x6 grid. Each month adds its own number. The result
    # runs from month 1 (min 1.0) to month 3 (max 26.0), over 72 elements.
    assert stats["temperature"] == {
        "min": 1.0,
        "max": 26.0,
        "null_count": 0,
        "row_count": 72,
    }
    # One dataset holds a third of that.
    one = atlas.describe(str(collection), "2024-01.nc")
    temperature = next(a for a in one["arrays"] if a["name"] == "temperature")
    assert temperature["stats"]["row_count"] == 24
    assert temperature["stats"]["max"] == 24.0

    # Strings compare as raw bytes.
    assert stats["station"]["min"] == b"a"
    assert stats["station"]["max"] == b"d"
    assert stats["station"]["row_count"] == 12


def test_info_stats_leave_out_a_removed_dataset(collection):
    atlas.remove(str(collection), ["2024-01.nc"])
    stats = atlas.info(str(collection))["array_stats"]
    # January held the lowest temperature, so the minimum rises.
    assert stats["temperature"]["min"] == 2.0
    assert stats["temperature"]["row_count"] == 48


def test_info_counts_removals_separately(collection):
    atlas.remove(str(collection), ["2024-01.nc"])
    i = atlas.info(str(collection))
    assert i["dataset_count"] == 2
    assert i["deleted_count"] == 1
    assert i["total_datasets"] == 3


# ── the rest of the API is gone ──────────────────────────────────────


@pytest.mark.parametrize(
    "name",
    [
        "Atlas",
        "AtlasWriter",
        "DatasetWriter",
        "DatasetView",
        "read_array",
        "open_as_xarray_dataset",
        "pruning_index",
    ],
)
def test_the_old_surface_is_gone(name):
    assert not hasattr(atlas, name), f"atlas should not expose {name}"


def test_the_public_surface_is_exactly_the_five_operations():
    operations = {"create", "remove", "list_datasets", "describe", "info"}
    assert operations <= set(atlas.__all__)
    # Everything else exported is a helper or an error type, not an operation.
    assert set(atlas.__all__) - operations == {
        "find_netcdf_files",
        "log_to_file",
        "AtlasError",
        "SourceError",
        "init_tracing",
        "__version__",
    }


def test_version_is_exposed():
    assert atlas.__version__.count(".") == 2
