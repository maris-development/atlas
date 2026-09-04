"""The block helpers of the xarray layer, on their own.

`test_ops` drives these through a whole ingest. These reach one function at a
time, which is where the edge cases live.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import atlas
from atlas import xarray as ax


# ── missing strings ──────────────────────────────────────────────────
#
# Atlas cannot store a missing string as null, because the format has no
# string null sentinel. Every missing cell therefore takes a real string.


@pytest.mark.parametrize(
    "block, expected_filled",
    [
        (np.array(["a", None, "b"], dtype=object), 1),
        (np.array(["a", float("nan")], dtype=object), 1),
        (np.array([["a", None], [float("nan"), "b"]], dtype=object), 2),
        # pandas markers. A `None`-or-`NaN` test misses these, and the
        # bindings would then reject the cell as a non-string.
        (np.array(["a", pd.NA], dtype=object), 1),
        (np.array(["a", pd.NaT], dtype=object), 1),
        # Nothing missing, and nothing to do.
        (np.array(["a", "b"], dtype=object), 0),
        (np.array([], dtype=object), 0),
        (np.zeros((0, 3), dtype=object), 0),
    ],
)
def test_every_missing_cell_takes_the_fill(block, expected_filled):
    out, n = ax._fill_missing_strings(block, "")
    assert n == expected_filled
    assert out.shape == block.shape
    assert not any(_is_missing(v) for v in np.asarray(out).reshape(-1))


def _is_missing(v):
    return v is None or v is pd.NA or v is pd.NaT or (isinstance(v, float) and v != v)


def test_a_fixed_width_string_block_is_left_alone():
    # `|S` and `|U` hold no missing cell, so the scan does not run.
    block = np.array(["a", "b"], dtype="U1")
    out, n = ax._fill_missing_strings(block, "")
    assert n == 0
    assert out is block


def test_the_fill_is_the_string_it_is_given():
    block = np.array(["a", None], dtype=object)
    out, n = ax._fill_missing_strings(block, "unknown")
    assert n == 1
    assert list(out) == ["a", "unknown"]


def test_an_element_that_is_a_sequence_is_not_missing():
    # `pandas.isna` on a nested sequence must give one bool for the cell, and
    # never a per-item array.
    block = np.array([[1, 2], None, "x"], dtype=object)
    out, n = ax._fill_missing_strings(block, "")
    assert n == 1
    assert list(out) == [[1, 2], "", "x"]


# ── through a whole ingest ───────────────────────────────────────────


def test_a_masked_string_variable_lands_with_the_fill(tmp_path):
    """A char variable with unwritten cells, as an Argo QC array has.

    `to_netcdf` of a `None` writes an empty string, so it round-trips with
    nothing missing. Only a real fill value leaves a masked cell, which
    xarray decodes to a float `nan` inside an object array. That is the case
    the scan exists for, so the fixture is built with netCDF4 directly.
    """
    import netCDF4

    d = tmp_path / "nc"
    d.mkdir()
    with netCDF4.Dataset(d / "a.nc", "w") as nc:
        nc.createDimension("x", 4)
        nc.createDimension("s", 1)
        v = nc.createVariable("qc", "S1", ("x", "s"), fill_value=b" ")
        v[0] = b"1"
        v[2] = b"2"  # 1 and 3 stay unwritten, and therefore masked

    dest = tmp_path / "c"
    with pytest.warns(UserWarning, match="missing string"):
        atlas.create(d, str(dest))

    qc = {a["name"]: a for a in atlas.describe(str(dest), "a.nc")["arrays"]}["qc"]
    assert qc["dtype"] == "string"
    assert qc["shape"] == [4]
    # The two masked cells took the resolved string fill, which is "".
    assert qc["fill_value"] == ""
    # `null_count` counts the elements equal to the fill, so the filled cells
    # land there and stay out of the bounds. Nothing is lost, and nothing
    # reaches the bindings as a non-string.
    assert qc["stats"] == {
        "min": b"1",
        "max": b"2",
        "null_count": 2,
        "row_count": 4,
    }
