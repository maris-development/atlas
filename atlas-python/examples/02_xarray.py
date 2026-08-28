"""Write xarray Datasets into a collection, one file for the lot.

`add_xarray_dataset` maps an `xarray.Dataset` onto atlas: coordinates and data
variables become arrays, variable attrs become per-array attributes, dataset
attrs become dataset-level attributes, and which variables were coordinates is
recorded so `Atlas.coords` can tell you afterwards.

Reading the data back is the Rust API's job. From Python you get the structure.

Run:
    python atlas-python/examples/02_xarray.py
"""
import tempfile

import numpy as np
import xarray as xr

import atlas


def monthly_grid(month: int) -> xr.Dataset:
    return xr.Dataset(
        data_vars={
            "temperature": xr.DataArray(
                np.full((4, 6), 10.0 + month, dtype=np.float32),
                dims=["lat", "lon"],
                attrs={"units": "celsius", "long_name": "surface temperature"},
            ),
            "station": xr.DataArray(
                np.array(["a", "b", "c", "d"], dtype=object),
                dims=["lat"],
            ),
        },
        coords={
            "lat": ("lat", np.arange(4, dtype=np.float64)),
            "lon": ("lon", np.arange(6, dtype=np.float64)),
            "time": np.datetime64(f"2024-{month:02d}-01", "ns"),
        },
        attrs={"month": month, "source": "example"},
    )


def main() -> None:
    with tempfile.TemporaryDirectory() as path:
        # A whole directory of NetCDF files would go in the same loop.
        with atlas.AtlasWriter.create(path) as writer:
            for month in (1, 2, 3):
                writer.add_xarray_dataset(monthly_grid(month), name=f"2024-{month:02d}")
                print(f"  wrote 2024-{month:02d}")

        # The accessor is the same call spelled the other way round:
        #     ds.atlas.write(writer, name)

        collection = atlas.Atlas.open(path)
        print(f"\ndatasets: {collection.list_datasets()}")
        print(f"arrays:   {collection.list_arrays()}")

        name = "2024-01"
        print(f"\n{name}:")
        print(f"  coordinates: {collection.coords(name)}")
        print(f"  attributes:  {collection.attributes(name)}")

        view = collection.dataset(name)
        for array in view.list_arrays():
            meta = view.array_meta(array)
            print(f"  {array:12s} {meta['dtype']:24s} {meta['shape']}")

        print(f"\n  temperature attrs: {collection.array_attributes(name, 'temperature')}")

        # dtype mapping worth knowing:
        #   datetime64[ns] -> timestamp_nanoseconds
        #   timedelta64    -> int64 nanoseconds, plus a marker attribute
        #   object/bytes/unicode -> variable-length string
        print(f"\n  time is stored as {view.array_meta('time')['dtype']}")
        print(f"  station is stored as {view.array_meta('station')['dtype']}")


if __name__ == "__main__":
    main()
