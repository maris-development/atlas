"""The collection-level reader, one column at a time.

`atlas.info` and `atlas.describe` answer about the whole collection and about
one dataset. Between the two sits a third shape: one array, or one attribute
key, over every dataset at once. That is the row group index of a Parquet
file, and these tests cover it.

Every column keys on the dataset name, so the columns join by name and never
by position. A column holds only the datasets that carry the key, so it is
shorter than the dataset list whenever one of them does not.
"""

import pytest

import atlas
from atlas import _atlas

from conftest import make_dataset


@pytest.fixture
def opened(collection):
    """The collection of `conftest`, opened through the low-level reader."""
    return _atlas.Atlas.open(str(collection))


@pytest.fixture
def gappy(tmp_path):
    """Three datasets, where only the outer two carry `season`.

    The shared fixture gives every dataset the same keys. This one does not,
    which is the case a column has to answer for.
    """
    source = tmp_path / "nc"
    source.mkdir()
    for month, season in ((1, "winter"), (2, None), (3, "spring")):
        ds = make_dataset(month)
        if season is not None:
            ds.attrs["season"] = season
        ds.to_netcdf(source / f"2024-{month:02d}.nc")
    dest = tmp_path / "gappy"
    atlas.create(source, str(dest))
    return _atlas.Atlas.open(str(dest))


# ── attributes over the collection ───────────────────────────────────


def test_a_dataset_attribute_comes_back_keyed_by_dataset(opened):
    months = opened.attributes_by_dataset("month")
    assert months == {"2024-01.nc": 1, "2024-02.nc": 2, "2024-03.nc": 3}

    # Name for name, it equals what each dataset reports on its own.
    for name in opened.list_datasets():
        assert months[name] == opened.dataset(name).get_attribute("month")


def test_a_column_leaves_out_a_dataset_that_does_not_carry_the_key(gappy):
    # Only the outer two carry `season`, so the column is shorter than the
    # dataset list. It does not line up with `list_datasets` position for
    # position. Walk the names and look each one up.
    assert gappy.list_datasets() == ["2024-01.nc", "2024-02.nc", "2024-03.nc"]

    seasons = gappy.attributes_by_dataset("season")
    assert seasons == {"2024-01.nc": "winter", "2024-03.nc": "spring"}
    assert "2024-02.nc" not in seasons

    # The keys keep write order. They are the datasets that carry the key, in
    # the order `list_datasets` gives them, and no placeholder for the rest.
    assert list(seasons) == ["2024-01.nc", "2024-03.nc"]

    column = [seasons.get(name) for name in gappy.list_datasets()]
    assert column == ["winter", None, "spring"]


def test_an_array_attribute_takes_the_array_name(opened):
    units = opened.attributes_by_dataset("units", "temperature")
    assert units == {name: "celsius" for name in opened.list_datasets()}


def test_a_key_nobody_carries_gives_an_empty_dict(opened):
    assert opened.attributes_by_dataset("nope") == {}
    assert opened.attributes_by_dataset("nope", "temperature") == {}


def test_an_array_no_dataset_declares_gives_an_empty_dict(opened):
    assert opened.attributes_by_dataset("units", "missing") == {}


def test_a_removed_dataset_leaves_the_attribute_column(opened):
    opened.delete_dataset("2024-01.nc")
    months = opened.attributes_by_dataset("month")
    assert months == {"2024-02.nc": 2, "2024-03.nc": 3}
    assert list(months) == ["2024-02.nc", "2024-03.nc"]


# ── statistics over the collection ───────────────────────────────────


def test_per_dataset_stats_key_on_the_dataset_and_keep_write_order(opened):
    per = opened.array_stats_by_dataset("temperature")
    assert list(per) == ["2024-01.nc", "2024-02.nc", "2024-03.nc"]

    # Each month adds its own number to the same 4x6 grid.
    assert per["2024-01.nc"] == {
        "min": 1.0,
        "max": 24.0,
        "null_count": 0,
        "row_count": 24,
    }
    assert per["2024-03.nc"]["max"] == 26.0

    # The rows sum to what `array_stats` reports for the whole collection.
    merged = opened.array_stats("temperature")
    assert merged["row_count"] == sum(s["row_count"] for s in per.values())
    assert merged["min"] == min(s["min"] for s in per.values())
    assert merged["max"] == max(s["max"] for s in per.values())


def test_an_array_no_dataset_declares_gives_no_rows(opened):
    assert opened.array_stats_by_dataset("missing") == {}
    assert opened.array_stats("missing") is None


def test_a_removed_dataset_leaves_the_stats_column(opened):
    opened.delete_dataset("2024-01.nc")
    per = opened.array_stats_by_dataset("temperature")
    assert list(per) == ["2024-02.nc", "2024-03.nc"]


# ── the join ─────────────────────────────────────────────────────────


def test_the_columns_join_by_name_into_one_table(gappy):
    # `list_datasets` gives the rows. Every column is a lookup on the name,
    # because a column leaves out the datasets that do not carry its key.
    seasons = gappy.attributes_by_dataset("season")
    months = gappy.attributes_by_dataset("month")
    bounds = gappy.array_stats_by_dataset("temperature")

    table = [
        {
            "dataset": name,
            "month": months.get(name),
            "season": seasons.get(name),
            "max": (bounds.get(name) or {}).get("max"),
        }
        for name in gappy.list_datasets()
    ]

    assert table == [
        {"dataset": "2024-01.nc", "month": 1, "season": "winter", "max": 24.0},
        {"dataset": "2024-02.nc", "month": 2, "season": None, "max": 25.0},
        {"dataset": "2024-03.nc", "month": 3, "season": "spring", "max": 26.0},
    ]
