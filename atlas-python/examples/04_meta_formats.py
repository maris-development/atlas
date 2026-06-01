"""Compare the six on-disk metadata format / compression combinations.

The metadata file (`atlas.json`, `atlas.msgpack`, optionally with `.zst` /
`.lz4` suffix) is read on every `Atlas.open` and written on every `flush`.
JSON is the default — human-readable and backwards compatible — but for stores
with thousands of datasets/arrays you can shrink it significantly by switching
to MessagePack and/or applying compression. The on-disk format is encoded in
the filename, so `Atlas.open` auto-detects it without any extra argument.

Run:
    python atlas-python/examples/04_meta_formats.py
"""
import tempfile
from pathlib import Path

import numpy as np

import atlas


def populate(store: "atlas.Atlas") -> None:
    """Build a moderately-sized store: 30 datasets × 4 arrays each."""
    for i in range(30):
        ds = store.create_dataset(f"dataset_{i:02d}")
        for j in range(4):
            ds.define_array(
                f"arr_{j}",
                dtype="float32",
                dims=["x", "y", "z"],
                shape=[64, 64, 64],
                chunk_shape=[8, 8, 8],
            )
        ds.set_attribute("dataset_id", i)
        ds.set_attribute("station", "KNMI")


COMBOS = [
    ("json", "none", "atlas.json"),
    ("json", "zstd", "atlas.json.zst"),
    ("json", "lz4", "atlas.json.lz4"),
    ("msgpack", "none", "atlas.msgpack"),
    ("msgpack", "zstd", "atlas.msgpack.zst"),
    ("msgpack", "lz4", "atlas.msgpack.lz4"),
]


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        results: list[tuple[str, str, str, int]] = []

        for meta_format, meta_compression, expected_file in COMBOS:
            store_dir = tmp_path / f"{meta_format}_{meta_compression}"
            with atlas.Atlas.create(
                str(store_dir),
                meta_format=meta_format,
                meta_compression=meta_compression,
            ) as store:
                populate(store)
            # `with` block flushes on exit.

            size = (store_dir / expected_file).stat().st_size
            results.append((meta_format, meta_compression, expected_file, size))

        baseline = results[0][3]  # uncompressed JSON
        print(f"{'format':<8} {'compression':<12} {'filename':<22} {'bytes':>8}   ratio")
        print("─" * 60)
        for meta_format, meta_compression, name, size in results:
            ratio = size / baseline
            print(f"{meta_format:<8} {meta_compression:<12} {name:<22} {size:>8}   {ratio:.2f}×")

        # Auto-detection: reopening any of the stores with no kwargs works.
        smallest = min(results, key=lambda r: r[3])
        smallest_dir = tmp_path / f"{smallest[0]}_{smallest[1]}"
        print(f"\nSmallest variant: {smallest[2]} ({smallest[3]} bytes) — reopening "
              f"with no kwargs to confirm auto-detection.")
        store = atlas.Atlas.open(str(smallest_dir))
        assert len(store.list_datasets()) == 30
        print(f"  list_datasets() returned {len(store.list_datasets())} datasets ✓")


if __name__ == "__main__":
    main()
