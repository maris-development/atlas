"""Write the cross-language fixture that the Rust suite reads back.

Python writes collections but cannot read their arrays, so a pytest run alone
cannot prove that the bytes it wrote are the bytes it meant. This script writes
a collection from xarray, and ``tests/cross_fixture.rs`` in the repository root
opens it with the Rust reader and checks every value.

Regenerate after an intentional change to the write path or the format:

    python atlas-python/tests/make_fixture.py

The output goes to ``tests/fixtures/from_python/`` and is committed.
"""

from __future__ import annotations

import pathlib
import shutil

import numpy as np
import xarray as xr

import atlas

# Repository root, from atlas-python/tests/make_fixture.py.
FIXTURE_DIR = pathlib.Path(__file__).resolve().parents[2] / "tests/fixtures/from_python"


def build_dataset() -> xr.Dataset:
    """Values the Rust side asserts on. Keep in sync with cross_fixture.rs."""
    temperature = xr.DataArray(
        np.arange(4 * 6, dtype=np.float32).reshape(4, 6),
        dims=["lat", "lon"],
        attrs={"units": "celsius", "long_name": "surface temperature"},
    )
    counts = xr.DataArray(
        np.array([10, 20, 30, 40], dtype=np.int64),
        dims=["lat"],
    )
    label = xr.DataArray(
        np.array(["alpha", "beta", "gamma", "delta"], dtype=object),
        dims=["lat"],
    )
    observed = xr.DataArray(
        np.array(
            [
                "2024-01-01T00:00:00",
                "2024-01-02T00:00:00",
                "2024-01-03T00:00:00",
                "2024-01-04T00:00:00",
            ],
            dtype="datetime64[ns]",
        ),
        dims=["lat"],
    )
    return xr.Dataset(
        data_vars={
            "temperature": temperature,
            "counts": counts,
            "label": label,
            "observed": observed,
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

    ds = build_dataset()
    with atlas.AtlasWriter.create(str(FIXTURE_DIR)) as w:
        # One chunked, to exercise the chunk grid on the read side.
        w.add_xarray_dataset(ds, "grid", chunks={"temperature": [2, 3]})
        w.add_xarray_dataset(ds, "grid_copy")

    print(f"wrote {FIXTURE_DIR}")
    a = atlas.Atlas.open(str(FIXTURE_DIR))
    print("  datasets:", a.list_datasets())
    print("  arrays:  ", a.list_arrays())


if __name__ == "__main__":
    main()
