"""xarray integration: write an `xr.Dataset` into atlas two equivalent ways, then read it back.

Demonstrates:
    1. `store.add_xarray_dataset(ds, name)` — the atlas-side instance method.
    2. `ds.atlas.write(atlas, name)`    — the xarray accessor (registered automatically
                                          when `atlas` is imported).
    3. `store.open_as_xarray_dataset(name)`          — read back as a fresh `xr.Dataset`.

Per-variable attributes (`units`, `long_name`, …) round-trip through the
flattened `{var}.{attr}` convention.

Run:
    python atlas-python/examples/02_xarray.py
"""
import tempfile

import numpy as np
import xarray as xr

import atlas  # noqa: F401  — registers the ds.atlas accessor


def build_dataset() -> xr.Dataset:
    """A small weather Dataset with named coords, two data variables, and attrs."""
    temperature = xr.DataArray(
        np.arange(8 * 16, dtype=np.float32).reshape(8, 16),
        dims=["lat", "lon"],
        attrs={"units": "celsius", "long_name": "surface temperature"},
    )
    pressure = xr.DataArray(
        np.full((8, 16), 1013.25, dtype=np.float64),
        dims=["lat", "lon"],
        attrs={"units": "hPa"},
    )
    return xr.Dataset(
        data_vars={"temperature": temperature, "pressure": pressure},
        coords={
            "lat": ("lat", np.arange(8, dtype=np.float32)),
            "lon": ("lon", np.arange(16, dtype=np.float32)),
        },
        attrs={"month": 1, "station": "KNMI"},
    )


def main() -> None:
    ds = build_dataset()

    with tempfile.TemporaryDirectory() as store_dir:
        with atlas.Atlas.create(store_dir) as store:
            # Two equivalent ways to write
            store.add_xarray_dataset(ds, "jan_2024")
            ds.atlas.write(store, "feb_2024")

            print(f"Datasets in store: {store.list_datasets()}")
        # `with` block calls store.close() (== flush) on exit; without it
        # the in-memory writes would never reach disk.

        # Reopen and read back
        store = atlas.Atlas.open(store_dir)
        ds_jan = store.open_as_xarray_dataset("jan_2024")
        ds_feb = store.open_as_xarray_dataset("feb_2024")

        print("\njan_2024 (store.open_as_xarray_dataset):")
        print(ds_jan)
        print("\njan_2024.temperature.attrs:", ds_jan["temperature"].attrs)
        print("feb_2024 attrs:               ", ds_feb.attrs)

        # The roundtrip is bit-identical
        xr.testing.assert_identical(ds, ds_jan)
        xr.testing.assert_identical(ds, ds_feb)
        print("\nBoth Datasets round-tripped identically.")


if __name__ == "__main__":
    main()
