"""Compare the three array compression codecs (zstd, lz4, none) on a real workload.

The `codec` kwarg on `Atlas.create` sets the codec used when *new* blocks are
written. The codec is recorded per-array, so reading is automatic regardless
of which codec the store was opened with. Existing blocks always decompress
with whichever codec they were originally written with.

Rule of thumb:
    * `zstd` — default. Best ratio at moderate CPU. Pick this unless you have
      a reason not to.
    * `lz4`  — faster to decompress; ~2× larger files. Worth it for read-heavy
      scan loops where the compressed-bytes-per-second matters more than space.
    * `none` — fastest write path, no size reduction. Useful only for tiny
      stores or when you'll compress the entire directory externally.

Run:
    python pyatlas/examples/05_codecs.py
"""
import tempfile
from pathlib import Path

import numpy as np

import pyatlas

# A workload that compresses well: sinusoidal data with lots of structure.
N_DATASETS = 8
SHAPE = (256, 256)


def populate(atlas: pyatlas.Atlas) -> None:
    """Write the same data into every dataset, so codec is the only variable."""
    # A smooth field with no noise — realistic for geophysical / model output.
    # Random/noisy data resists compression and would show all three codecs
    # tied; smooth data is where codec choice actually matters.
    xs = np.linspace(0, 4 * np.pi, SHAPE[0])
    ys = np.linspace(0, 4 * np.pi, SHAPE[1])
    data = (np.sin(xs)[:, None] + np.cos(ys)[None, :]).astype(np.float32)

    for i in range(N_DATASETS):
        ds = atlas.create_dataset(f"sensor_{i:02d}")
        ds.define_array(
            "readings",
            dtype="float32",
            dims=["lat", "lon"],
            shape=list(SHAPE),
            chunk_shape=[64, 64],
        )
        ds.write_array("readings", start=[0, 0], data=data)
    atlas.flush()


def array_bytes(store_dir: Path) -> int:
    """Total bytes used by array files (everything except the top-level atlas.* metadata)."""
    total = 0
    for path in store_dir.rglob("*"):
        if path.is_file() and not path.name.startswith("atlas."):
            total += path.stat().st_size
    return total


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        results: list[tuple[str, int]] = []

        for codec in ["zstd", "lz4", "none"]:
            store_dir = tmp_path / codec
            with pyatlas.Atlas.create(str(store_dir), codec=codec) as atlas:
                populate(atlas)
            results.append((codec, array_bytes(store_dir)))

        baseline = next(size for codec, size in results if codec == "none")
        print(f"Workload: {N_DATASETS} datasets × {SHAPE[0]}×{SHAPE[1]} float32 "
              f"= {N_DATASETS * SHAPE[0] * SHAPE[1] * 4 / 1024:.0f} KiB raw per codec\n")
        print(f"{'codec':<8} {'array bytes':>14}   ratio vs raw")
        print("─" * 44)
        for codec, size in results:
            print(f"{codec:<8} {size:>14,}   {size / baseline:.2f}×")

        # Codec is auto-detected per array on open — no kwarg required.
        print("\nReopening the lz4 store with no kwargs; read-back is automatic:")
        atlas = pyatlas.Atlas.open(str(tmp_path / "lz4"))
        ds = atlas.open_dataset("sensor_00")
        readings = ds.read_array("readings")
        assert readings is not None
        print(f"  sensor_00.readings.shape = {readings.shape}, dtype = {readings.dtype} ✓")


if __name__ == "__main__":
    main()
