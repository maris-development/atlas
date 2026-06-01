"""Multiple datasets, one physical array file per array name.

Atlas's defining trick: when many datasets define an array of the same name,
those datasets all write into the *same* physical `<name>/data.af` file,
keyed by dataset name inside the file. So adding the 1000th dataset that
defines `temperature` doesn't create a 1000th file — it just appends to the
existing `temperature/data.af`.

This is what makes atlas suitable for "thousands of small datasets with the
same schema" workloads (weather stations, sensors, document collections) —
each dataset is logically independent but they share a small number of
physical files.

Run:
    python atlas-python/examples/07_shared_arrays.py
"""
import tempfile
from pathlib import Path

import numpy as np

import atlas

N_STATIONS = 50


def main() -> None:
    with tempfile.TemporaryDirectory() as store_dir:
        # 50 stations × 2 variables = 100 logical arrays.
        with atlas.Atlas.create(store_dir) as store:
            for i in range(N_STATIONS):
                ds = store.create_dataset(f"station_{i:03d}")
                ds.define_array(
                    "temperature",
                    dtype="float32",
                    dims=["hour"],
                    shape=[24],
                    chunk_shape=[24],
                )
                ds.define_array(
                    "pressure",
                    dtype="float32",
                    dims=["hour"],
                    shape=[24],
                    chunk_shape=[24],
                )
                ds.write_array(
                    "temperature",
                    start=[0],
                    data=np.full(24, 20.0 + i * 0.1, dtype=np.float32),
                )
                ds.write_array(
                    "pressure",
                    start=[0],
                    data=np.full(24, 1013.0 + i * 0.01, dtype=np.float32),
                )
                ds.set_attribute("station_id", i)

            n_datasets = len(store.list_datasets())
            arrays = store.list_arrays()
            print(f"Logical datasets       : {n_datasets}")
            print(f"Physical arrays        : {arrays}")
            print(f"  → {n_datasets} datasets share {len(arrays)} files\n")

        # Inspect the directory layout: 2 array dirs total, regardless of N_STATIONS.
        store_path = Path(store_dir)
        array_dirs = sorted(p.name for p in store_path.iterdir() if p.is_dir())
        print(f"On-disk array directories: {array_dirs}")

        # Read the same array name back from different datasets.
        store = atlas.Atlas.open(store_dir)
        t0 = store.open_dataset("station_000").read_array("temperature")
        t49 = store.open_dataset("station_049").read_array("temperature")
        assert t0 is not None and t49 is not None
        print(f"\nstation_000.temperature[0] = {t0[0]:.2f}")
        print(f"station_049.temperature[0] = {t49[0]:.2f}")
        print("Two datasets, same `temperature` name → same physical file, "
              "distinct contents.")


if __name__ == "__main__":
    main()
