"""Compare the three block codecs on one workload.

The `codec` argument to `AtlasWriter.create` sets how array blocks are
compressed. Each block records the codec that produced it, so a reader decodes
whatever it finds without being told — the choice affects only writing.

Rule of thumb:
    zstd  default. Best ratio at moderate CPU. Pick this unless you have a
          reason not to.
    lz4   faster to decompress, larger files. Worth it for scan-heavy reading
          where compressed bytes per second matters more than space.
    none  fastest write path, no size reduction. Useful for tiny collections,
          or when you compress the whole file externally.

Run:
    python atlas-python/examples/05_codecs.py
"""
import tempfile
from pathlib import Path

import numpy as np

import atlas

N_DATASETS = 8
SHAPE = (256, 256)


def field() -> np.ndarray:
    """A smooth field, the kind of data where codec choice actually matters.

    Random noise resists compression and would leave all three codecs tied.
    """
    xs = np.linspace(0, 4 * np.pi, SHAPE[0])
    ys = np.linspace(0, 4 * np.pi, SHAPE[1])
    return (np.sin(xs)[:, None] + np.cos(ys)[None, :]).astype(np.float32)


def write(path: str, codec: str, data: np.ndarray) -> int:
    """Write the same data under `codec` and return the file size in bytes."""
    with atlas.AtlasWriter.create(path, codec=codec) as writer:
        for i in range(N_DATASETS):
            ds = writer.add_dataset(f"grid_{i:02d}")
            ds.define_array(
                "field",
                dtype="float32",
                dims=["y", "x"],
                shape=list(SHAPE),
                chunk_shape=[64, 64],
            )
            ds.write_array("field", start=[0, 0], data=data)
            ds.finish()
    return (Path(path) / "data.atlas").stat().st_size


def main() -> None:
    data = field()
    raw = data.nbytes * N_DATASETS
    print(f"{N_DATASETS} datasets x {SHAPE[0]}x{SHAPE[1]} float32 = {raw / 1e6:.2f} MB raw\n")
    print(f"{'codec':<8} {'file size':>12} {'ratio':>8}")
    print("-" * 30)

    for codec in ("zstd", "lz4", "none"):
        with tempfile.TemporaryDirectory() as path:
            size = write(path, codec, data)
            print(f"{codec:<8} {size / 1e6:>9.2f} MB {raw / size:>7.1f}x")

            # Whatever the codec, the reader is told nothing about it.
            collection = atlas.Atlas.open(path)
            assert collection.dataset_count() == N_DATASETS


if __name__ == "__main__":
    main()
