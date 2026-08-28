"""Stream a dask-backed xarray Dataset into a collection, one chunk at a time.

When a variable's `.data` is a `dask.array.Array`, `add_xarray_dataset` walks
the dask chunk grid and writes one block at a time. Peak memory is bounded by a
single block per variable, not the whole array, so a dataset far larger than RAM
can be written.

The dask chunk shape also becomes the on-disk chunk shape, which is what makes a
later partial read cheap: a reader fetches only the chunks a region overlaps.

Run:
    python atlas-python/examples/03_dask_streaming.py
"""
import tempfile

import dask.array as dask_array
import numpy as np
import xarray as xr

import atlas  # noqa: F401


def main() -> None:
    # A 32x64 dask array split into 8x16 blocks: 4 x 4 = 16 blocks.
    raw = dask_array.arange(  # type: ignore[arg-type]
        32 * 64, dtype=np.float32, chunks=32 * 64
    ).reshape(32, 64)
    temperature = xr.DataArray(raw, dims=["y", "x"]).chunk({"y": 8, "x": 16})
    ds = xr.Dataset(data_vars={"temperature": temperature})

    print("Input:")
    print(f"  shape  = {temperature.shape}")
    print(f"  chunks = {temperature.chunks}")
    print(f"  blocks = {int(np.prod([len(c) for c in temperature.chunks]))}")

    with tempfile.TemporaryDirectory() as path:
        with atlas.AtlasWriter.create(path) as writer:
            # One write_array call per dask block. Nothing is materialized whole.
            writer.add_xarray_dataset(ds, "ds")

        view = atlas.Atlas.open(path).dataset("ds")
        meta = view.array_meta("temperature")
        print("\nStored:")
        print(f"  shape       = {meta['shape']}")
        print(f"  chunk_shape = {meta['chunk_shape']}  (from the dask chunking)")

        # To choose a different on-disk chunking, pass it explicitly:
        #     writer.add_xarray_dataset(ds, "ds", chunks={"temperature": [16, 32]})

    # Reading the data back is the Rust API's job. What the chunk shape buys you
    # there: reading temperature[10:12, 20:22] fetches one 8x16 chunk, not the
    # 32x64 array.


if __name__ == "__main__":
    main()
