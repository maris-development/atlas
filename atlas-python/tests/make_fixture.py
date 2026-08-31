"""Write the cross-language fixture that the Rust suite reads back.

Python writes collections but cannot read their arrays, so a pytest run alone
cannot prove that the bytes it wrote are the bytes it meant. This script builds
a collection from NetCDF files the way `atlas create` does, and
`tests/cross_fixture.rs` in the repository root opens it with the Rust reader
and checks every value.

Regenerate after an intentional change to the write path or the format:

    python atlas-python/tests/make_fixture.py

The output goes to `tests/fixtures/from_python/` and is committed.
"""

from __future__ import annotations

import pathlib
import shutil
import tempfile

import numpy as np
import xarray as xr

import atlas

# Repository root, from atlas-python/tests/make_fixture.py.
FIXTURE_DIR = pathlib.Path(__file__).resolve().parents[2] / "tests/fixtures/from_python"


def build_dataset() -> xr.Dataset:
    """Values the Rust side asserts on. Keep in sync with cross_fixture.rs."""
    return xr.Dataset(
        data_vars={
            "temperature": xr.DataArray(
                np.arange(4 * 6, dtype=np.float32).reshape(4, 6),
                dims=["lat", "lon"],
                attrs={"units": "celsius", "long_name": "surface temperature"},
            ),
            "counts": xr.DataArray(
                np.array([10, 20, 30, 40], dtype=np.int64), dims=["lat"]
            ),
            "label": xr.DataArray(
                np.array(["alpha", "beta", "gamma", "delta"], dtype=object),
                dims=["lat"],
            ),
            "observed": xr.DataArray(
                np.array(
                    [f"2024-01-0{d}T00:00:00" for d in (1, 2, 3, 4)],
                    dtype="datetime64[ns]",
                ),
                dims=["lat"],
            ),
        },
        coords={
            "lat": ("lat", np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64)),
            "lon": ("lon", np.arange(6, dtype=np.float64)),
        },
        attrs={"month": 1, "station": "KNMI", "bounds": [1.0, 2.0]},
    )


def main() -> None:
    if FIXTURE_DIR.exists():
        shutil.rmtree(FIXTURE_DIR)
    FIXTURE_DIR.parent.mkdir(parents=True, exist_ok=True)

    # `atlas.create` ingests a directory, so lay the sources out as one.
    with tempfile.TemporaryDirectory() as staging:
        source = pathlib.Path(staging)
        dataset = build_dataset()
        # Two files with identical contents. `create` applies one chunking to
        # every file, so the two datasets end up with the same schema — which
        # is what the interning assertion in cross_fixture.rs turns on.
        dataset.to_netcdf(source / "grid.nc")
        dataset.to_netcdf(source / "grid_copy.nc")

        atlas.create(
            source,
            str(FIXTURE_DIR),
            chunks={"temperature": [2, 3]},
            progress=lambda name: print(f"  wrote {name}"),
        )

    print(f"wrote {FIXTURE_DIR}")
    info = atlas.info(str(FIXTURE_DIR))
    print(f"  datasets: {atlas.list_datasets(str(FIXTURE_DIR))}")
    print(f"  arrays:   {info['distinct_arrays']}")


if __name__ == "__main__":
    main()
