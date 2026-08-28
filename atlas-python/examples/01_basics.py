"""Write a collection, then reopen it and inspect what it holds.

A collection is one immutable file. You build it once and it never changes,
except that a dataset can be hidden with `delete_dataset`.

Opening a collection from Python gives you its structure — datasets, arrays,
dtypes, shapes, attributes — but not its array data. Reading arrays is the Rust
API's job.

Run:
    python atlas-python/examples/01_basics.py
"""
import tempfile

import numpy as np

import atlas


def main() -> None:
    with tempfile.TemporaryDirectory() as path:
        print(f"Writing a collection at {path}\n")

        # Nothing is readable until the `with` block exits and the footer is
        # written. An exception here would leave no collection behind at all.
        with atlas.AtlasWriter.create(path, codec="zstd") as writer:
            # ── A 4x4 temperature grid for January ──────────────────────
            jan = writer.add_dataset("jan_2024")
            jan.define_array(
                "temperature",
                dtype="float32",
                dims=["lat", "lon"],
                shape=[4, 4],
                chunk_shape=[2, 2],  # four chunks, so partial reads stay partial
                fill_value=float("nan"),
            )
            jan.write_array(
                "temperature",
                start=[0, 0],
                data=np.full((4, 4), 5.0, dtype=np.float32),
            )
            jan.set_attribute("month", 1)
            jan.set_attribute("station", "KNMI")
            jan.set_array_attribute("temperature", "units", "celsius")
            jan.finish()  # only now does the dataset enter the file
            print("  wrote jan_2024")

            # ── February, same shape ────────────────────────────────────
            feb = writer.add_dataset("feb_2024")
            feb.define_array(
                "temperature",
                dtype="float32",
                dims=["lat", "lon"],
                shape=[4, 4],
                chunk_shape=[2, 2],
                fill_value=float("nan"),
            )
            feb.write_array(
                "temperature",
                start=[0, 0],
                data=np.full((4, 4), 7.5, dtype=np.float32),
            )
            feb.set_attribute("month", 2)
            feb.finish()
            print("  wrote feb_2024")

        # ── Reopen ──────────────────────────────────────────────────────
        print("\nReopening (reads the footer, nothing else)")
        collection = atlas.Atlas.open(path)
        print(f"  datasets: {collection.list_datasets()}")
        print(f"  arrays:   {collection.list_arrays()}")

        jan_view = collection.dataset("jan_2024")
        print(f"\n  jan_2024 holds {jan_view.list_arrays()}")
        print(f"  temperature: {jan_view.array_meta('temperature')}")
        print(f"  attributes:  {collection.attributes('jan_2024')}")
        print(f"  units:       {jan_view.get_array_attribute('temperature', 'units')}")

        # Both datasets declare the same arrays, so the footer stores that
        # schema once and each dataset points at it.
        print(f"\n  feb_2024 has the same arrays: "
              f"{jan_view.list_arrays() == collection.dataset('feb_2024').list_arrays()}")

        # ── Delete ──────────────────────────────────────────────────────
        print("\nDeleting feb_2024")
        collection.delete_dataset("feb_2024")
        print(f"  datasets now: {atlas.Atlas.open(path).list_datasets()}")
        print("  the container is untouched; only a small mask file was written")
        print("  rewrite the collection to reclaim the space")


if __name__ == "__main__":
    main()
