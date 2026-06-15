"""Missing data and fill values when ingesting an `xr.Dataset`.

When a NetCDF dataset is read with `mask_and_scale=True` (xarray's default),
missing cells become `NaN` (floats) / `NaT` (datetimes) and the CF `_FillValue`
moves into `var.encoding`. Atlas records those masked cells as **null** by
defaulting each array to a sentinel fill on write:

    float32/float64        -> NaN
    datetime64[ns]         -> NaT
    string (object/|S/|U)  -> ""   (missing cells substituted, with a warning)
    integer                -> none (no missing sentinel; set one explicitly)

Demonstrates:
    1. The default fills and the resulting `null_count` in `array_stats`.
    2. Overriding per variable with `fill_value={var: scalar}` (or a bare scalar).
    3. The warning emitted when missing string cells are substituted.

Run:
    python atlas-python/examples/09_missing_data.py
"""
import tempfile
import warnings

import numpy as np
import xarray as xr

import atlas  # noqa: F401  — registers the ds.atlas accessor


def build_dataset() -> xr.Dataset:
    """A small Dataset with masked-style missing cells in three dtypes."""
    # Float field with two NaN holes (what mask_and_scale leaves behind).
    temperature = xr.DataArray(
        np.array([[1.0, np.nan], [3.0, np.nan]], dtype=np.float64),
        dims=["x", "y"],
        attrs={"units": "celsius"},
    )
    # Datetime field with a NaT hole.
    observed = xr.DataArray(
        np.array(["2024-01-01", "NaT", "2024-01-03", "2024-01-04"], dtype="datetime64[ns]"),
        dims=["t"],
    )
    # Object-string field with a missing cell.
    station = xr.DataArray(
        np.array(["KNMI", None, "DWD"], dtype=object),
        dims=["s"],
    )
    # Integer field with a user-chosen sentinel via the CF `_FillValue` attr.
    count = xr.DataArray(
        np.array([5, -1, 7], dtype=np.int32),
        dims=["s"],
        attrs={"_FillValue": np.int32(-1)},
    )
    return xr.Dataset(
        {"temperature": temperature, "observed": observed, "station": station, "count": count}
    )


def main() -> None:
    ds = build_dataset()

    with tempfile.TemporaryDirectory() as store_dir:
        with atlas.Atlas.create(store_dir) as store:
            # Default fills for temperature/observed/station; `count` uses its
            # _FillValue=-1. Missing string cells emit a warning as they're
            # substituted with "" — surface it so the example shows it.
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                store.add_xarray_dataset(ds, "defaults")
            for w in caught:
                print(f"warning: {w.message}")

            # Override the string fill so missing 'station' cells become "N/A"
            # (still tracked as null) instead of the default "".
            store.add_xarray_dataset(ds, "explicit", fill_value={"station": "N/A"})

        store = atlas.Atlas.open(store_dir)
        view = store.open_dataset("defaults")

        print("\nDefault fills and null counts:")
        for name in ["temperature", "observed", "station", "count"]:
            fv = view.array_fill_value(name)
            null_count = view.array_stats(name)["null_count"]
            print(f"  {name:12s} fill={fv!r:24s} null_count={null_count}")

        explicit = store.open_dataset("explicit")
        print("\nExplicit override for 'station':")
        print(f"  fill={explicit.array_fill_value('station')!r}"
              f"  values={list(explicit.read_array('station'))}"
              f"  null_count={explicit.array_stats('station')['null_count']}")

        # Round-trip preserves NaN/NaT; the "" / NaN / NaT sentinels are not
        # re-emitted as redundant _FillValue attrs.
        ds_back = store.open_as_xarray_dataset("defaults")
        assert np.isnan(ds_back["temperature"].values).sum() == 2
        assert "_FillValue" not in ds_back["temperature"].attrs
        print("\nRound-tripped: NaN/NaT preserved, no redundant _FillValue attrs.")


if __name__ == "__main__":
    main()
