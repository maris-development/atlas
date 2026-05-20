"""Stream a dask-backed xarray Dataset into atlas, one chunk at a time.

When a variable's `.data` is a `dask.array.Array`, `atlas.add_xr_dataset` (and
the `ds.atlas.write` accessor) iterates the dask chunk grid and calls
`view.write_array(start=..., data=chunk)` once per chunk — so peak memory is
bounded by a single chunk per variable rather than the full array.

The dask chunk shape is also used as the atlas on-disk `chunk_shape`, which
keeps cross-dataset scans fast.

Run:
    python pyatlas/examples/03_dask_streaming.py
"""
import tempfile

import dask.array as dask_array
import numpy as np
import xarray as xr

import pyatlas  # noqa: F401


def main() -> None:
    # A 32×64 dask array split into 8×16 blocks → 4×4 = 16 blocks total.
    raw = dask_array.arange(  # type: ignore[arg-type]
        32 * 64, dtype=np.float32, chunks=32 * 64
    ).reshape(32, 64)
    temperature = xr.DataArray(raw, dims=["y", "x"]).chunk({"y": 8, "x": 16})
    ds = xr.Dataset(data_vars={"temperature": temperature})

    print("Input Dataset:")
    print(f"  temperature.shape   = {temperature.shape}")
    print(f"  temperature.chunks  = {temperature.chunks}")
    print(f"  total blocks        = {int(np.prod([len(c) for c in temperature.chunks]))}")

    with tempfile.TemporaryDirectory() as store_dir:
        with pyatlas.Atlas.create(store_dir) as atlas:
            atlas.add_xr_dataset(ds, "ds")  # streams chunk-by-chunk

            # Verify the dask chunk shape became the atlas on-disk chunk shape
            view = atlas.open_dataset("ds")
            meta = view.array_meta("temperature")
            print(f"\nAtlas chunk_shape for 'temperature': {meta['chunk_shape']}")
        # `with` block calls atlas.close() (== flush) on exit.

        # Reopen — chunked variables come back dask-backed (one task per on-disk chunk)
        atlas = pyatlas.Atlas.open(store_dir)
        ds_back = atlas.to_xarray("ds")
        print(f"Read-back shape:   {ds_back['temperature'].shape}")
        print(f"Read-back chunks:  {ds_back['temperature'].data.chunks}")
        print(f"Read-back .data:   {type(ds_back['temperature'].data).__module__}.{type(ds_back['temperature'].data).__name__}")
        xr.testing.assert_identical(ds, ds_back)  # both dask, same chunks
        print("Roundtrip preserves lazy dask structure.")


if __name__ == "__main__":
    main()
