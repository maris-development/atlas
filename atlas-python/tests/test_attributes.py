"""What an attribute value can hold, and what comes back.

An attribute carries its own type to disk and back, so a read consults no
schema. Every type therefore returns itself. The `dtype=` argument picks a
narrower type than the inferred one, and nothing else.
"""

import pytest

from atlas import _atlas


def write(tmp_path, values, array_values=None):
    """One dataset carrying `values`, reopened. Each value is `(key, v, dtype)`."""
    dest = tmp_path / "c"
    with _atlas.AtlasWriter.create(str(dest)) as writer:
        ds = writer.add_dataset("d")
        ds.define_array("x", "int64", ["i"], [2], None, None)
        for key, value, dtype in values:
            ds.set_attribute(key, value, dtype)
        for key, value, dtype in array_values or ():
            ds.set_array_attribute("x", key, value, dtype)
        ds.finish()
    return _atlas.Atlas.open(str(dest)).dataset("d")


def test_an_inferred_value_reads_back_as_itself(tmp_path):
    view = write(
        tmp_path,
        [
            ("flag", True, None),
            ("count", 7, None),
            ("scale", 0.5, None),
            ("name", "buoy", None),
            ("raw", b"\xde\xad", None),
            ("tags", ["a", "b"], None),
            ("bounds", [1.0, 2.0], None),
        ],
    )
    assert view.attributes() == {
        "flag": True,
        "count": 7,
        "scale": 0.5,
        "name": "buoy",
        "raw": b"\xde\xad",
        "tags": ["a", "b"],
        "bounds": [1.0, 2.0],
    }


def test_a_dtype_picks_the_stored_width(tmp_path):
    view = write(tmp_path, [("small", 7, "int32"), ("wide", 7, "int64")])
    # Python sees an int either way. The width is what the segment stores, and
    # the schema records it.
    assert view.get_attribute("small") == 7
    assert view.get_attribute("wide") == 7


def test_an_out_of_range_value_for_its_dtype_is_an_error(tmp_path):
    with pytest.raises(OverflowError):
        write(tmp_path, [("small", 70_000, "int8")])


def test_an_attribute_cannot_hold_a_timestamp(tmp_path):
    # The storage layer has no timestamp attribute, so the value could not
    # read back as one. The error says what to do instead.
    for dtype in ("timestamp_ns", "timestamp_nanoseconds", "datetime64[ns]"):
        with pytest.raises(ValueError, match="cannot hold a timestamp"):
            write(tmp_path, [("when", 1_700_000_000_000_000_000, dtype)])


def test_the_int64_replacement_round_trips(tmp_path):
    # What the error tells a caller to write instead.
    view = write(
        tmp_path,
        [
            ("when", 1_700_000_000_000_000_000, "int64"),
            ("when_units", "ns since epoch", None),
        ],
    )
    assert view.get_attribute("when") == 1_700_000_000_000_000_000
    assert view.get_attribute("when_units") == "ns since epoch"


def test_an_unknown_dtype_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="unknown attribute dtype"):
        write(tmp_path, [("x", 1, "complex128")])


def test_an_array_attribute_takes_the_same_types(tmp_path):
    view = write(
        tmp_path,
        [],
        array_values=[("units", "celsius", None), ("scale", 2, "int32")],
    )
    assert view.array_attributes("x") == {"units": "celsius", "scale": 2}
    with pytest.raises(ValueError, match="cannot hold a timestamp"):
        write(tmp_path, [], array_values=[("t", 1, "datetime64[ns]")])
