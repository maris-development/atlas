"""Write and open a collection through an obstore object-store handle.

`AtlasWriter.create` and `Atlas.open` accept either a local filesystem path
(`str` / `os.PathLike`) or any [obstore][obstore] store: `S3Store`, `GCSStore`,
`AzureStore`, `LocalStore`, `MemoryStore`, `HttpStore`. Everything else works
the same against either.

The single-file format suits object storage: writing streams one multipart
upload, and opening costs one range read of the tail however large the
collection is.

This example uses `LocalStore` so it runs anywhere with no credentials. The
cloud variants are commented out in `main()` — uncomment the one you need and
leave the rest alone.

Install (once):

    pip install "atlas-python[cloud]"

Run:

    python atlas-python/examples/08_object_store.py

[obstore]: https://github.com/developmentseed/obstore
"""
import tempfile

import numpy as np

import atlas

try:
    import obstore
except ImportError:  # pragma: no cover - guidance, not logic
    raise SystemExit(
        'obstore is not installed. Run: pip install "atlas-python[cloud]"'
    )


def main() -> None:
    with tempfile.TemporaryDirectory() as path:
        # Swap in whichever you need; nothing below changes.
        #   store = obstore.store.S3Store("my-bucket", prefix="collections/2024")
        #   store = obstore.store.GCSStore("my-bucket", prefix="collections/2024")
        #   store = obstore.store.AzureStore("my-container", prefix="collections/2024")
        #   store = obstore.store.MemoryStore()
        store = obstore.store.LocalStore(path)

        print(f"Writing through {type(store).__name__}")
        with atlas.AtlasWriter.create(store) as writer:
            for month in (1, 2):
                ds = writer.add_dataset(f"2024-{month:02d}")
                ds.define_array(
                    "temperature",
                    dtype="float32",
                    dims=["lat", "lon"],
                    shape=[8, 8],
                    chunk_shape=[4, 4],
                )
                ds.write_array(
                    "temperature",
                    start=[0, 0],
                    data=np.full((8, 8), 10.0 + month, dtype=np.float32),
                )
                ds.set_attribute("month", month)
                ds.finish()
                print(f"  wrote 2024-{month:02d}")

        # Opening issues one range read for the tail of data.atlas.
        collection = atlas.Atlas.open(store)
        print(f"\ndatasets: {collection.list_datasets()}")
        view = collection.dataset("2024-01")
        print(f"  temperature: {view.array_meta('temperature')}")
        print(f"  month:       {view.get_attribute('month')}")

        # Deleting writes one small object beside the container.
        collection.delete_dataset("2024-02")
        print(f"\nafter delete: {atlas.Atlas.open(store).list_datasets()}")


if __name__ == "__main__":
    main()
