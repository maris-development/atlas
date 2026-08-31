"""The five operations, as a library.

Builds a small collection from NetCDF files, then inspects and prunes it.
Everything here also works against object storage — see 02_object_store.py.

Run:
    python atlas-python/examples/01_library.py
"""
import tempfile
from pathlib import Path

import numpy as np
import xarray as xr

import atlas


def write_source_files(directory: Path) -> None:
    """Three monthly grids, the kind of thing a model run produces."""
    for month in (1, 2, 3):
        xr.Dataset(
            data_vars={
                "temperature": xr.DataArray(
                    np.full((4, 6), 10.0 + month, dtype=np.float32),
                    dims=["lat", "lon"],
                    attrs={"units": "celsius", "long_name": "surface temperature"},
                ),
                "station": xr.DataArray(
                    np.array(["a", "b", "c", "d"], dtype=object), dims=["lat"]
                ),
            },
            coords={
                "lat": ("lat", np.arange(4, dtype=np.float64)),
                "lon": ("lon", np.arange(6, dtype=np.float64)),
            },
            attrs={"month": month, "source": "example"},
        ).to_netcdf(directory / f"2024-{month:02d}.nc")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "netcdf"
        source.mkdir()
        write_source_files(source)
        collection = str(Path(tmp) / "collection")

        # ── 1. create ────────────────────────────────────────────────
        # One dataset per file, named after the file stem. Nothing is
        # readable until every file is written and the footer lands.
        print("Files to ingest:", [p.name for p in atlas.find_netcdf_files(source)])
        result = atlas.create(source, collection)
        print(f"created {result['dataset_count']} datasets\n")

        # ── 2. list ──────────────────────────────────────────────────
        # One range read of the container tail, whatever the size.
        print("datasets:", atlas.list_datasets(collection), "\n")

        # ── 3. info ──────────────────────────────────────────────────
        summary = atlas.info(collection)
        print("collection:")
        print(f"  {summary['dataset_count']} datasets, "
              f"{summary['container_bytes']} bytes, codec {summary['codec']}")
        print(f"  arrays: {summary['distinct_arrays']}")
        # All three months declare the same arrays, so one schema is stored.
        print(f"  interned schemas: {summary['interned_schemas']}\n")

        # ── 4. describe ──────────────────────────────────────────────
        detail = atlas.describe(collection, "2024-01")
        print(f"dataset {detail['name']} (ordinal {detail['ordinal']}):")
        print(f"  dimensions: {detail['dimensions']}")
        print(f"  coordinates: {detail['coordinates']}")
        print(f"  attributes: {detail['attributes']}")
        for array in detail["arrays"]:
            dims = ", ".join(array["dimensions"])
            print(f"  {array['dtype']:>22s} {array['name']}({dims})")
            # Statistics were computed when the array was written and live in
            # the footer, so reading them costs nothing.
            stats = array["stats"]
            print(f"  {'':>22s}   count={stats['row_count']} "
                  f"nulls={stats['null_count']} "
                  f"min={stats['min']!r} max={stats['max']!r}")
        print()

        # ── 5. remove ────────────────────────────────────────────────
        # Several at once, by name or by the file they came from.
        removed = atlas.remove(collection, ["2024-02", source / "2024-03.nc"])
        print(f"removed {removed['removed']}, {removed['remaining']} remain")
        print("datasets now:", atlas.list_datasets(collection))

        after = atlas.info(collection)
        print(f"  container is still {after['container_bytes']} bytes: removing "
              f"writes a mask, it does not rewrite the file")
        print("  rebuild the collection to reclaim the space")


if __name__ == "__main__":
    main()
