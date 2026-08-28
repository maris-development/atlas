"""Missing data and fill values when writing an `xr.Dataset`.

Read a NetCDF file with `mask_and_scale=True` (xarray's default) and missing
cells arrive as `NaN` for floats and `NaT` for datetimes, with the CF
`_FillValue` moved into `var.encoding`. Atlas records those cells as unwritten
by giving each array a sentinel fill:

    float32 / float64      -> NaN
    datetime64[ns]         -> NaT
    string                 -> ""    (missing cells substituted, with a warning)
    integer                -> none  (no sentinel; set one explicitly)

Shown here:
    1. The defaults, as reported by `array_fill_value`.
    2. Overriding per variable with `fill_value={var: scalar}`, or all at once
       with a bare scalar.
    3. The warning when missing string cells are substituted.

Run:
    python atlas-python/examples/09_missing_data.py
"""
import tempfile
import warnings

import numpy as np
import xarray as xr

import atlas  # noqa: F401  — registers the ds.atlas accessor


def build_dataset() -> xr.Dataset:
    """A small Dataset with masked-style holes in three dtypes."""
    temperature = xr.DataArray(
        np.array([[1.0, np.nan], [3.0, np.nan], [5.0, 6.0], [np.nan, 8.0]], dtype=np.float64),
        dims=["x", "y"],
        attrs={"units": "celsius"},
    )
    observed = xr.DataArray(
        np.array(["2024-01-01", "NaT", "2024-01-03", "2024-01-04"], dtype="datetime64[ns]"),
        dims=["x"],
    )
    label = xr.DataArray(
        np.array(["north", None, "south", np.nan], dtype=object),
        dims=["x"],
    )
    counts = xr.DataArray(np.array([1, 2, 3, 4], dtype=np.int32), dims=["x"])
    return xr.Dataset(
        {
            "temperature": temperature,
            "observed": observed,
            "label": label,
            "counts": counts,
        }
    )


def report(path: str, name: str) -> None:
    view = atlas.Atlas.open(path).dataset(name)
    print(f"  {'array':<14}{'dtype':<26}fill value")
    for array in view.list_arrays():
        meta = view.array_meta(array)
        print(f"  {array:<14}{meta['dtype']:<26}{meta['fill_value']!r}")


def main() -> None:
    ds = build_dataset()

    print("1. Defaults\n")
    with tempfile.TemporaryDirectory() as path:
        # Substituting missing strings is worth knowing about, so it warns.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with atlas.AtlasWriter.create(path) as writer:
                writer.add_xarray_dataset(ds, "defaults")
        report(path, "defaults")
        for w in caught:
            print(f"\n  warning: {w.message}")

    print("\n2. Explicit fills\n")
    with tempfile.TemporaryDirectory() as path:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with atlas.AtlasWriter.create(path) as writer:
                writer.add_xarray_dataset(
                    ds,
                    "explicit",
                    fill_value={
                        "counts": -999,      # integers have no default
                        "label": "unknown",  # instead of ""
                        "temperature": None,  # opt out of the NaN default
                    },
                )
        report(path, "explicit")

    print("\n3. One scalar for every numeric array\n")
    with tempfile.TemporaryDirectory() as path:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with atlas.AtlasWriter.create(path) as writer:
                writer.add_xarray_dataset(ds, "scalar", fill_value=-1)
        report(path, "scalar")


if __name__ == "__main__":
    main()
