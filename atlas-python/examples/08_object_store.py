"""Open and create atlas stores through an obstore object_store handle.

`Atlas.create` and `Atlas.open` accept either a local filesystem path
(string / `os.PathLike`) or any [obstore][obstore]-constructed store —
`obstore.store.S3Store`, `obstore.store.GCSStore`,
`obstore.store.AzureStore`, `obstore.store.LocalStore`,
`obstore.store.HttpStore`. The rest of the API (`define_array`,
`write_array`, `read_array`, `set_attribute`, `flush`, `add_xarray_dataset`,
`open_as_xarray_dataset`, all the bulk reads) works identically against either.

This example uses `obstore.store.LocalStore` so it runs anywhere with no
credentials. The S3 / GCS / Azure variants are shown commented out at
the top of `main()` — uncomment the one you need, leave the rest of the
script alone.

Install (once):

    pip install "atlas-python[cloud]"   # pulls obstore alongside atlas

Run:

    python atlas-python/examples/08_object_store.py

[obstore]: https://github.com/developmentseed/obstore
"""
import tempfile
from pathlib import Path

import numpy as np
import obstore as obs
import xarray as xr

import atlas


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        # ── Choose ONE backend ────────────────────────────────────────
        # Local — the only one that runs without external credentials.
        store = obs.store.LocalStore(tmp)

        # S3 — set AWS_* env vars or pass creds explicitly via kwargs.
        # store = obs.store.S3Store(
        #     "my-bucket",
        #     prefix="atlas-stores/demo",
        #     region="us-east-1",
        #     # access_key_id="AKIA...",         # optional; falls back to env
        #     # secret_access_key="...",         # optional; falls back to env
        #     # endpoint="https://minio.local",  # for MinIO / non-AWS S3
        # )

        # Google Cloud Storage:
        # store = obs.store.GCSStore("my-bucket", prefix="atlas-stores/demo")

        # Azure Blob Storage:
        # store = obs.store.AzureStore(
        #     container_name="my-container",
        #     prefix="atlas-stores/demo",
        # )

        print(f"Backend: {type(store).__module__}.{type(store).__name__}")

        # ── 1. Create — pass the obstore handle exactly where you would
        #               have passed a path string. ──────────────────────
        with atlas.Atlas.create(store, codec="zstd") as atlas_store:
            jan = atlas_store.create_dataset("jan_2024")
            jan.define_array(
                "temperature",
                dtype="float32",
                dims=["lat", "lon"],
                shape=[8, 16],
                chunk_shape=[4, 8],
                fill_value=float("nan"),
            )
            jan.write_array(
                "temperature",
                start=[0, 0],
                data=np.full((8, 16), 20.0, dtype=np.float32),
            )
            jan.set_attribute("month", 1)
            jan.set_attribute("station", "KNMI")

            # xarray ingestion works the same way — one helper call
            # streams an xr.Dataset (dask-chunked or eager) into atlas
            # through the obstore backend.
            ds = xr.Dataset(
                data_vars={
                    "pressure": (
                        ["lat", "lon"],
                        np.full((8, 16), 1013.25, dtype=np.float64),
                        {"units": "hPa"},
                    ),
                },
                coords={
                    "lat": np.arange(8, dtype=np.float32),
                    "lon": np.arange(16, dtype=np.float32),
                },
                attrs={"month": 2, "station": "KNMI"},
            )
            atlas_store.add_xarray_dataset(ds, "feb_2024")
        # `with` exit calls atlas_store.close() == single flush. On a cloud
        # backend this is the moment one `PutObject` per touched array
        # file + one for the metadata happens — call flush() at coarse
        # grain (per-batch, not per-dataset) to amortise.

        # ── 2. Reopen — pass the same handle in. Codec, meta format
        #               and meta compression are auto-detected from the
        #               on-disk filename in both directions. ───────────
        atlas2 = atlas.Atlas.open(store)
        print(f"\nDatasets: {atlas2.list_datasets()}")
        print(f"Physical arrays: {atlas2.list_arrays()}")

        jan_back = atlas2.open_dataset("jan_2024")
        temp = jan_back.read_array("temperature")
        assert temp is not None
        print(f"\njan_2024.temperature[0,0] = {temp[0, 0]}")
        print(f"jan_2024 attrs            = {jan_back.attributes()}")

        feb_back = atlas2.open_as_xarray_dataset("feb_2024")
        xr.testing.assert_identical(ds, feb_back)
        print(f"\nfeb_2024 round-tripped identically through xarray ✓")

        # Persisted stats are populated after flush; works on cloud too.
        stats = jan_back.array_stats("temperature")
        assert stats is not None
        print(
            f"\njan_2024.temperature stats: "
            f"rows={stats['row_count']} min={stats['min']} max={stats['max']}"
        )

        # ── 3. (Optional) inspect what landed on disk. Local-only — on
        #      S3 you'd use `aws s3 ls s3://bucket/prefix/`. ─────────────
        if isinstance(store, obs.store.LocalStore):
            root = Path(tmp)
            entries = sorted(p.relative_to(root) for p in root.rglob("*") if p.is_file())
            print(f"\nOn-disk layout under {tmp}:")
            for entry in entries:
                print(f"  {entry}")


if __name__ == "__main__":
    main()
