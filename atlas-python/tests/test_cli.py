"""The `atlas` command.

Each subcommand is driven through `main()` with an argv list, so the tests
exercise argument parsing and output formatting rather than just the functions
underneath.
"""

import json

import pytest

from atlas import _cli


def run(capsys, *argv):
    """Run the CLI and return `(exit_code, stdout, stderr)`."""
    code = _cli.main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


# ── create ───────────────────────────────────────────────────────────


def test_create_reports_what_it_wrote(capsys, netcdf_dir, tmp_path):
    dest = tmp_path / "c"
    code, out, err = run(capsys, "create", str(netcdf_dir), str(dest))

    assert code == 0
    assert "3 dataset(s) written" in out
    # Progress goes to stderr, so stdout stays pipeable.
    assert "2024-01" in err
    assert (dest / "data.atlas").exists()


def test_create_is_quiet_when_asked(capsys, netcdf_dir, tmp_path):
    code, out, err = run(capsys, "create", str(netcdf_dir), str(tmp_path / "c"), "-q")
    assert code == 0
    assert "2024-01" not in err


def test_create_accepts_a_codec(capsys, netcdf_dir, tmp_path):
    dest = tmp_path / "c"
    run(capsys, "create", str(netcdf_dir), str(dest), "--codec", "lz4", "-q")
    code, out, _ = run(capsys, "info", str(dest), "--json")
    assert json.loads(out)["codec"] == "lz4"


def test_create_accepts_chunks_as_json(capsys, netcdf_dir, tmp_path):
    dest = tmp_path / "c"
    run(
        capsys, "create", str(netcdf_dir), str(dest),
        "--chunks", '{"temperature": [2, 3]}', "-q",
    )
    code, out, _ = run(capsys, "show", str(dest), "2024-01", "--json")
    arrays = {a["name"]: a for a in json.loads(out)["arrays"]}
    assert arrays["temperature"]["chunk_shape"] == [2, 3]


def test_create_chunk_size_controls_the_stored_chunk_shape(capsys, tmp_path):
    import numpy as np
    import xarray as xr

    src = tmp_path / "nc"
    src.mkdir()
    xr.Dataset(
        {"big": (("y", "x"), np.zeros((1024, 1024), dtype=np.float64))}
    ).to_netcdf(src / "big.nc")

    run(capsys, "create", str(src), str(tmp_path / "small"),
        "--chunk-size", "1MiB", "-q")
    code, out, _ = run(capsys, "show", str(tmp_path / "small"), "big", "--json")
    chunked = json.loads(out)["arrays"][0]["chunk_shape"]
    assert chunked != [1024, 1024]

    run(capsys, "create", str(src), str(tmp_path / "large"),
        "--chunk-size", "64MiB", "-q")
    code, out, _ = run(capsys, "show", str(tmp_path / "large"), "big", "--json")
    assert json.loads(out)["arrays"][0]["chunk_shape"] == [1024, 1024]


@pytest.mark.parametrize("mode", ["auto", "native", "none", '{"lat": 2}'])
def test_open_chunks_modes_are_accepted(capsys, netcdf_dir, tmp_path, mode):
    code, out, err = run(
        capsys, "create", str(netcdf_dir), str(tmp_path / mode[:4]),
        "--open-chunks", mode, "-q",
    )
    assert code == 0, err


def test_an_unknown_open_chunks_mode_fails_clearly(capsys, netcdf_dir, tmp_path):
    code, out, err = run(
        capsys, "create", str(netcdf_dir), str(tmp_path / "c"),
        "--open-chunks", "sometimes",
    )
    assert code == 1
    assert "--open-chunks" in err


def test_open_chunks_rejects_non_object_json(capsys, netcdf_dir, tmp_path):
    code, out, err = run(
        capsys, "create", str(netcdf_dir), str(tmp_path / "c"),
        "--open-chunks", "[1, 2]",
    )
    assert code == 1
    assert "JSON must be an object" in err


def test_create_on_an_empty_directory_fails(capsys, tmp_path):
    (tmp_path / "empty").mkdir()
    code, out, err = run(capsys, "create", str(tmp_path / "empty"), str(tmp_path / "c"))
    assert code == 1
    assert "no NetCDF files" in err


# ── ls ───────────────────────────────────────────────────────────────


def test_ls_prints_one_name_per_line(capsys, collection):
    code, out, _ = run(capsys, "ls", str(collection))
    assert code == 0
    assert out.split() == ["2024-01", "2024-02", "2024-03"]


def test_ls_json_is_a_list(capsys, collection):
    code, out, _ = run(capsys, "ls", str(collection), "--json")
    assert json.loads(out) == ["2024-01", "2024-02", "2024-03"]


def test_ls_on_a_non_collection_fails_clearly(capsys, tmp_path):
    (tmp_path / "empty").mkdir()
    code, out, err = run(capsys, "ls", str(tmp_path / "empty"))
    assert code == 1
    assert "not an atlas collection" in err


# ── rm ───────────────────────────────────────────────────────────────


def test_rm_removes_several_in_one_call(capsys, collection):
    code, out, _ = run(capsys, "rm", str(collection), "2024-01", "2024-03")
    assert code == 0
    assert "removed 2" in out
    assert "1 dataset(s) remain" in out

    _, listing, _ = run(capsys, "ls", str(collection))
    assert listing.split() == ["2024-02"]


def test_rm_accepts_netcdf_paths(capsys, collection, netcdf_dir):
    code, _, _ = run(capsys, "rm", str(collection), str(netcdf_dir / "2024-02.nc"))
    assert code == 0
    _, listing, _ = run(capsys, "ls", str(collection))
    assert listing.split() == ["2024-01", "2024-03"]


def test_rm_of_something_absent_fails(capsys, collection):
    code, out, err = run(capsys, "rm", str(collection), "nope")
    assert code == 1
    assert "not in the collection" in err


def test_rm_missing_ok_succeeds(capsys, collection):
    code, out, err = run(capsys, "rm", str(collection), "nope", "--missing-ok")
    assert code == 0
    assert "not present: nope" in err


# ── show ─────────────────────────────────────────────────────────────


def test_show_renders_ncdump_style(capsys, collection):
    code, out, _ = run(capsys, "show", str(collection), "2024-01")
    assert code == 0

    assert out.startswith("dataset 2024-01 {")
    assert out.rstrip().endswith("}")
    assert "dimensions:" in out
    assert "lat = 4 ;" in out
    assert "variables:" in out
    assert "float32 temperature(lat, lon) ;" in out
    assert "// coordinate" in out
    assert 'temperature:units = "celsius" ;' in out
    assert "// global attributes:" in out
    assert ":month = 1 ;" in out


def test_show_includes_array_statistics(capsys, collection):
    code, out, _ = run(capsys, "show", str(collection), "2024-01")
    assert "// stats:" in out
    # temperature is arange(24) + 1.
    assert "count=24" in out
    assert "min=1.0" in out
    assert "max=24.0" in out


def test_show_notes_a_non_default_chunk_shape(capsys, netcdf_dir, tmp_path):
    dest = tmp_path / "c"
    run(capsys, "create", str(netcdf_dir), str(dest),
        "--chunks", '{"temperature": [2, 3]}', "-q")
    code, out, _ = run(capsys, "show", str(dest), "2024-01")
    assert "_ChunkShape = [2, 3]" in out


def test_show_json_carries_the_full_structure(capsys, collection):
    code, out, _ = run(capsys, "show", str(collection), "2024-01", "--json")
    d = json.loads(out)
    assert d["name"] == "2024-01"
    assert d["dimensions"] == {"lat": 4, "lon": 6}
    arrays = {a["name"]: a for a in d["arrays"]}
    assert arrays["temperature"]["stats"]["row_count"] == 24
    # bytes are decoded for JSON, so the output is valid.
    assert arrays["station"]["stats"]["min"] == "a"


def test_show_of_a_missing_dataset_fails(capsys, collection):
    code, out, err = run(capsys, "show", str(collection), "nope")
    assert code == 1
    assert "no dataset" in err


# ── info ─────────────────────────────────────────────────────────────


def test_info_summarises(capsys, collection):
    code, out, _ = run(capsys, "info", str(collection))
    assert code == 0
    assert "format version    1" in out
    assert "codec             zstd" in out
    assert "datasets          3" in out
    assert "interned schemas  1" in out
    assert "temperature" in out


def test_info_reports_removals(capsys, collection):
    run(capsys, "rm", str(collection), "2024-01")
    code, out, _ = run(capsys, "info", str(collection))
    assert "datasets          2" in out
    assert "removed           1" in out
    assert "space not reclaimed" in out


def test_info_json(capsys, collection):
    code, out, _ = run(capsys, "info", str(collection), "--json")
    i = json.loads(out)
    assert i["dataset_count"] == 3
    assert i["format_version"] == 1


# ── parsing ──────────────────────────────────────────────────────────


def test_no_subcommand_is_an_error(capsys):
    with pytest.raises(SystemExit) as exc:
        _cli.main([])
    assert exc.value.code != 0


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        _cli.main(["--version"])
    assert exc.value.code == 0
    assert "atlas" in capsys.readouterr().out


@pytest.mark.parametrize("command", ["create", "rm", "ls", "show", "info"])
def test_every_command_offers_json_and_store_flags(command):
    """The remote flags have to be on every subcommand, not just the read ones."""
    parser = _cli.build_parser()
    sub = parser._subparsers._group_actions[0].choices[command]
    flags = {action.option_strings[0] for action in sub._actions if action.option_strings}
    assert {"--json", "--region", "--endpoint", "--anonymous"} <= flags


def test_an_unsupported_url_scheme_fails_clearly(capsys):
    code, out, err = run(capsys, "ls", "ftp://host/path")
    assert code == 1
    assert "unsupported scheme" in err
