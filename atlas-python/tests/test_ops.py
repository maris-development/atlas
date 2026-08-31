"""The five operations, as a library.

Array *values* are not checked here — Python cannot read them. The Rust suite
verifies the bytes these tests write; see ``tests/cross_fixture.rs``.
"""

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
    assert result["written"] == ["2024-01", "2024-02", "2024-03"]
    assert result["skipped"] == []
    # One file, plus a mask only once something is removed.
    assert sorted(p.name for p in dest.iterdir()) == ["data.atlas"]


def test_datasets_are_named_after_the_file_stem(netcdf_dir, tmp_path):
    atlas.create(netcdf_dir, str(tmp_path / "c"))
    assert atlas.list_datasets(str(tmp_path / "c")) == [
        "2024-01",
        "2024-02",
        "2024-03",
    ]


def test_create_is_ordered_by_filename(netcdf_dir, tmp_path):
    """Ordinals are handed out in write order, so the order must be stable."""
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
    assert atlas.list_datasets(str(tmp_path / "c")) == ["jan"]


def test_an_empty_directory_is_an_error(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(atlas.AtlasError, match="no NetCDF files"):
        atlas.create(tmp_path / "empty", str(tmp_path / "c"))


def test_a_missing_directory_is_an_error(tmp_path):
    with pytest.raises(atlas.AtlasError, match="not a directory"):
        atlas.create(tmp_path / "nope", str(tmp_path / "c"))


def test_two_files_with_the_same_stem_collide(tmp_path):
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
    # bool is not a supported array dtype.
    xr.Dataset({"flag": xr.DataArray(np.array([True, False]), dims=["x"])}).to_netcdf(
        src / "bad.nc"
    )

    with pytest.raises(atlas.AtlasError):
        atlas.create(src, str(tmp_path / "c"))
    # No trailer was written, so nothing opens.
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
    assert result["written"] == ["good"]
    assert len(result["skipped"]) == 1
    assert result["skipped"][0]["file"].endswith("bad.nc")
    assert atlas.list_datasets(str(tmp_path / "c")) == ["good"]


def test_progress_is_called_per_dataset(netcdf_dir, tmp_path):
    seen = []
    atlas.create(netcdf_dir, str(tmp_path / "c"), progress=seen.append)
    assert seen == ["2024-01", "2024-02", "2024-03"]


@pytest.mark.parametrize("codec", ["zstd", "lz4", "none"])
def test_every_codec_round_trips(netcdf_dir, tmp_path, codec):
    dest = tmp_path / codec
    atlas.create(netcdf_dir, str(dest), codec=codec)
    assert atlas.info(str(dest))["codec"] == codec
    assert len(atlas.list_datasets(str(dest))) == 3


# ── chunked ingest ───────────────────────────────────────────────────


def test_a_large_variable_streams_in_blocks(tmp_path, monkeypatch):
    """A file bigger than the block budget is written block by block."""
    from atlas._atlas import DatasetWriter

    src = tmp_path / "nc"
    src.mkdir()
    # 8 MiB of float64, ingested with a 1 MiB block budget.
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

    stored = atlas.describe(str(tmp_path / "c"), "big")["arrays"][0]
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
        return atlas.describe(str(dest), "big")["arrays"][0]["chunk_shape"]

    small = chunk_shape(tmp_path / "small", chunk_size="1MiB")
    large = chunk_shape(tmp_path / "large", chunk_size="64MiB")

    # A bigger budget means bigger blocks, so fewer of them.
    assert np.prod(small) < np.prod(large)
    # The larger budget covers the whole 8 MiB array in one block.
    assert large == [1024, 1024]


def test_small_files_still_land_as_a_single_chunk(netcdf_dir, tmp_path):
    """Auto chunking must not fragment arrays that comfortably fit."""
    atlas.create(netcdf_dir, str(tmp_path / "c"))
    for array in atlas.describe(str(tmp_path / "c"), "2024-01")["arrays"]:
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
    # Ask netCDF4 for a specific on-disk chunking.
    xr.Dataset(
        {"big": (("y", "x"), np.zeros((512, 512), dtype=np.float64))}
    ).to_netcdf(
        src / "big.nc",
        engine="netcdf4",
        encoding={"big": {"chunksizes": (128, 256), "zlib": True}},
    )

    atlas.create(src, str(tmp_path / "c"), open_chunks="native")
    stored = atlas.describe(str(tmp_path / "c"), "big")["arrays"][0]
    assert stored["chunk_shape"] == [128, 256]


def test_open_chunks_accepts_an_explicit_dict(tmp_path):
    src = tmp_path / "nc"
    src.mkdir()
    xr.Dataset(
        {"big": (("y", "x"), np.zeros((512, 512), dtype=np.float64))}
    ).to_netcdf(src / "big.nc")

    atlas.create(src, str(tmp_path / "c"), open_chunks={"y": 128, "x": 256})
    stored = atlas.describe(str(tmp_path / "c"), "big")["arrays"][0]
    assert stored["chunk_shape"] == [128, 256]


@pytest.mark.parametrize("bad", ["sometimes", 42, 3.5])
def test_an_unknown_open_chunks_mode_is_rejected(netcdf_dir, tmp_path, bad):
    with pytest.raises(atlas.AtlasError, match="open_chunks"):
        atlas.create(netcdf_dir, str(tmp_path / "c"), open_chunks=bad)


def test_explicit_chunks_reach_the_stored_array(netcdf_dir, tmp_path):
    dest = tmp_path / "c"
    atlas.create(netcdf_dir, str(dest), chunks={"temperature": [2, 3]})
    arrays = {a["name"]: a for a in atlas.describe(str(dest), "2024-01")["arrays"]}
    assert arrays["temperature"]["chunk_shape"] == [2, 3]


def test_on_error_must_be_a_known_mode(netcdf_dir, tmp_path):
    with pytest.raises(atlas.AtlasError, match="on_error"):
        atlas.create(netcdf_dir, str(tmp_path / "c"), on_error="explode")


# ── remove ───────────────────────────────────────────────────────────


def test_remove_takes_several_datasets_in_one_call(collection):
    result = atlas.remove(str(collection), ["2024-01", "2024-03"])
    assert result["removed"] == ["2024-01", "2024-03"]
    assert result["remaining"] == 1
    assert atlas.list_datasets(str(collection)) == ["2024-02"]


def test_remove_accepts_the_netcdf_path_it_came_from(collection, netcdf_dir):
    atlas.remove(str(collection), [netcdf_dir / "2024-02.nc"])
    assert atlas.list_datasets(str(collection)) == ["2024-01", "2024-03"]


def test_remove_writes_a_mask_and_leaves_the_container_alone(collection):
    before = (collection / "data.atlas").stat().st_size
    atlas.remove(str(collection), ["2024-01"])
    assert (collection / "data.atlas").stat().st_size == before
    assert (collection / "deleted.mask").exists()


def test_removals_accumulate_across_calls(collection):
    atlas.remove(str(collection), ["2024-01"])
    atlas.remove(str(collection), ["2024-03"])
    assert atlas.list_datasets(str(collection)) == ["2024-02"]


def test_removing_something_absent_is_an_error(collection):
    with pytest.raises(atlas.AtlasError, match="not in the collection"):
        atlas.remove(str(collection), ["nope"])


def test_missing_ok_reports_instead_of_raising(collection):
    result = atlas.remove(str(collection), ["2024-01", "nope"], missing_ok=True)
    assert result["removed"] == ["2024-01"]
    assert result["missing"] == ["nope"]


def test_removing_nothing_is_an_error(collection):
    with pytest.raises(atlas.AtlasError, match="no datasets given"):
        atlas.remove(str(collection), [])


def test_ordinals_do_not_shift_when_a_dataset_is_removed(collection):
    before = atlas.describe(str(collection), "2024-03")["ordinal"]
    atlas.remove(str(collection), ["2024-01"])
    assert atlas.describe(str(collection), "2024-03")["ordinal"] == before


# ── list ─────────────────────────────────────────────────────────────


def test_list_applies_the_mask(collection):
    assert len(atlas.list_datasets(str(collection))) == 3
    atlas.remove(str(collection), ["2024-02"])
    assert atlas.list_datasets(str(collection)) == ["2024-01", "2024-03"]


def test_listing_a_non_collection_fails_clearly(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="not an atlas collection"):
        atlas.list_datasets(str(tmp_path / "empty"))


# ── describe ─────────────────────────────────────────────────────────


def test_describe_reports_the_whole_dataset(collection):
    d = atlas.describe(str(collection), "2024-01")

    assert d["name"] == "2024-01"
    assert d["ordinal"] == 0
    assert d["dimensions"] == {"lat": 4, "lon": 6}
    assert sorted(d["coordinates"]) == ["lat", "lon", "time"]
    assert d["attributes"] == {"month": 1, "source": "test", "bounds": [1.0, 2.0]}
    # The coordinate marker is atlas bookkeeping, not a user attribute.
    assert "_pyatlas_coords" not in d["attributes"]

    start, end = d["segment_range"]
    assert 0 < start < end


def test_describe_reports_each_array(collection):
    arrays = {a["name"]: a for a in atlas.describe(str(collection), "2024-01")["arrays"]}
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
    arrays = {a["name"]: a for a in atlas.describe(str(collection), "2024-01")["arrays"]}

    # temperature is arange(24) + 1, so 1..24 with nothing missing.
    temp = arrays["temperature"]["stats"]
    assert temp["row_count"] == 24
    assert temp["null_count"] == 0
    assert temp["min"] == 1.0
    assert temp["max"] == 24.0

    assert arrays["counts"]["stats"]["min"] == 10
    assert arrays["counts"]["stats"]["max"] == 40

    # Strings compare lexicographically, and come back as bytes.
    station = arrays["station"]["stats"]
    assert station["min"] == b"a"
    assert station["max"] == b"d"
    assert station["row_count"] == 4


def test_describe_accepts_a_netcdf_path(collection, netcdf_dir):
    by_name = atlas.describe(str(collection), "2024-01")
    by_path = atlas.describe(str(collection), netcdf_dir / "2024-01.nc")
    # NaN fill values never compare equal, so compare everything else.
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
    atlas.remove(str(collection), ["2024-01"])
    with pytest.raises(atlas.AtlasError, match="removed"):
        atlas.describe(str(collection), "2024-01")


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


def test_info_counts_removals_separately(collection):
    atlas.remove(str(collection), ["2024-01"])
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
        "AtlasError",
        "SourceError",
        "init_tracing",
        "__version__",
    }


def test_version_is_exposed():
    assert atlas.__version__.count(".") == 2
