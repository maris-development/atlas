"""Create an atlas store, add a few datasets with arrays + attributes, reopen, and read back.

Run:
    python atlas-python/examples/01_basics.py
"""
import tempfile

import numpy as np

import atlas


def main() -> None:
    with tempfile.TemporaryDirectory() as store_dir:
        print(f"Creating atlas store at {store_dir}")
        store = atlas.Atlas.create(store_dir, codec="zstd")

        # ── Dataset 1: a 4×4 temperature grid for January ──────────────────
        jan = store.create_dataset("jan_2024")
        jan.define_array(
            "temperature",
            dtype="float32",
            dims=["lat", "lon"],
            shape=[4, 4],
            chunk_shape=[2, 2],
        )
        jan.write_array("temperature", start=[0, 0],
                        data=np.full((4, 4), 5.0, dtype=np.float32))
        jan.set_attribute("month", 1)
        jan.set_attribute("station", "KNMI")

        # ── Dataset 2: same shape, different values, shares the "temperature"
        #              physical array file with jan_2024 inside store. ─────
        feb = store.create_dataset("feb_2024")
        feb.define_array(
            "temperature",
            dtype="float32",
            dims=["lat", "lon"],
            shape=[4, 4],
            chunk_shape=[2, 2],
        )
        feb.write_array("temperature", start=[0, 0],
                        data=np.full((4, 4), 7.5, dtype=np.float32))
        feb.set_attribute("month", 2)

        # Persist atlas.json + every cached array file in one shot.
        store.flush()

        print("\nDatasets:", store.list_datasets())
        print("Physical arrays:", store.list_arrays())

        # ── Reopen and read back ───────────────────────────────────────────
        atlas2 = atlas.Atlas.open(store_dir)
        jan2 = atlas2.open_dataset("jan_2024")
        temp = jan2.read_array("temperature")
        assert temp is not None
        print(f"\njan_2024 temperature[0,0] = {temp[0, 0]}  (expected 5.0)")
        print(f"jan_2024 attrs           = {jan2.attributes()}")

        stats = jan2.array_stats("temperature")
        assert stats is not None
        print(f"jan_2024 temperature stats: rows={stats['row_count']} "
              f"min={stats['min']} max={stats['max']}")


if __name__ == "__main__":
    main()
