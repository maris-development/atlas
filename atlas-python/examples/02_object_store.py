"""The same five operations, against object storage.

Every operation takes a URL as readily as a path:

    s3://bucket/prefix      gs://bucket/prefix
    az://container/prefix   https://host/path

It also takes an obstore handle you built yourself, to set credentials or other
options. Atlas never sees a credential. obstore does.

The single-file format suits object storage. A write is one multipart upload.
An open is one range read of the tail, whatever the number of datasets.

Install (once):

    pip install "atlas-python[cloud]"

Run:

    python atlas-python/examples/02_object_store.py
"""
import tempfile
from pathlib import Path

import numpy as np
import xarray as xr

import atlas

try:
    import obstore
except ImportError:  # pragma: no cover - guidance, not logic
    raise SystemExit('obstore is not installed. Run: pip install "atlas-python[cloud]"')


def write_source_files(directory: Path) -> None:
    for month in (1, 2):
        xr.Dataset(
            {"temperature": xr.DataArray(
                np.full((4, 6), 10.0 + month, dtype=np.float32),
                dims=["lat", "lon"], attrs={"units": "celsius"})},
            coords={"lat": ("lat", np.arange(4.0)), "lon": ("lon", np.arange(6.0))},
            attrs={"month": month},
        ).to_netcdf(directory / f"2024-{month:02d}.nc")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "netcdf"
        source.mkdir()
        write_source_files(source)

        # A real deployment swaps this line for one of:
        #   store = "s3://my-bucket/collections/2024"
        #   store = obstore.store.S3Store("my-bucket", prefix="collections/2024",
        #                                 region="eu-west-1")
        #   store = obstore.store.GCSStore("my-bucket", prefix="collections/2024")
        #   store = obstore.store.AzureStore("my-container", account="...")
        # Nothing below changes.
        store = obstore.store.LocalStore(str(Path(tmp) / "bucket"), mkdir=True)
        print(f"Using {type(store).__name__}\n")

        result = atlas.create(source, store)
        print(f"create   {result['dataset_count']} datasets -> {result['written']}")

        print(f"ls       {atlas.list_datasets(store)}")

        summary = atlas.info(store)
        print(f"info     {summary['dataset_count']} datasets, "
              f"{summary['container_bytes']} bytes")

        detail = atlas.describe(store, "2024-01.nc")
        temperature = next(a for a in detail["arrays"] if a["name"] == "temperature")
        print(f"show     temperature {temperature['dtype']}{temperature['shape']} "
              f"stats={temperature['stats']}")

        removed = atlas.remove(store, ["2024-02.nc"])
        print(f"rm       {removed['removed']}, {removed['remaining']} remain")

        # Credentials and endpoints pass straight through to obstore:
        #   atlas.list_datasets("s3://bucket/prefix", region="eu-west-1")
        #   atlas.list_datasets("s3://public-bucket/x", skip_signature=True)
        #
        # The CLI takes the same settings as flags:
        #   atlas ls s3://bucket/prefix --region eu-west-1
        #   atlas info s3://public-bucket/x --anonymous


if __name__ == "__main__":
    main()
