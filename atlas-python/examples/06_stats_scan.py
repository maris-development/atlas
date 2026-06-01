"""Scan stats across many datasets without loading any raw data.

Atlas maintains per-array statistics (`min`, `max`, `row_count`) alongside the
data, persisted on flush. `array_stats(name)` returns these without
decompressing or reading the data itself — useful for cross-dataset summary
queries over thousands of datasets.

This example writes a fleet of sensor datasets, then finds the one with the
highest peak reading by scanning stats alone.

Run:
    python atlas-python/examples/06_stats_scan.py
"""
import tempfile

import numpy as np

import atlas

N_SENSORS = 32
N_READINGS = 24 * 7  # one week, hourly


def main() -> None:
    with tempfile.TemporaryDirectory() as store_dir:
        rng = np.random.default_rng(seed=0)

        # ── Write one dataset per sensor ──────────────────────────────
        # Use lz4 — readings are random so they compress poorly, and lz4
        # decompresses faster on the stats-scan read path.
        with atlas.Atlas.create(store_dir, codec="lz4") as store:
            for i in range(N_SENSORS):
                ds = store.create_dataset(f"sensor_{i:03d}")
                ds.define_array(
                    "readings",
                    dtype="float32",
                    dims=["hour"],
                    shape=[N_READINGS],
                    chunk_shape=[24],
                )
                # Give each sensor a distinct baseline so stats differ.
                baseline = 18.0 + (i % 8) * 1.5
                noise = rng.normal(scale=2.0, size=N_READINGS).astype(np.float32)
                ds.write_array("readings", start=[0], data=baseline + noise)
                ds.set_attribute("sensor_id", i)
        # `with` block flushes on exit; stats are written here too.

        # ── Reopen and scan stats only ────────────────────────────────
        store = atlas.Atlas.open(store_dir)
        print(f"Scanning stats for {len(store.list_datasets())} sensors "
              f"(no raw data read)…\n")

        peak_sensor = ""
        peak_max = -np.inf
        for name in sorted(store.list_datasets()):
            ds = store.open_dataset(name)
            stats = ds.array_stats("readings")
            assert stats is not None
            if stats["max"] > peak_max:
                peak_max = stats["max"]
                peak_sensor = name

        print(f"{'sensor':<12} {'min':>7} {'max':>7} {'rows':>6}")
        print("─" * 36)
        # Show a handful around the peak so the output is readable.
        for name in sorted(store.list_datasets())[:5]:
            ds = store.open_dataset(name)
            stats = ds.array_stats("readings")
            assert stats is not None
            print(f"{name:<12} {stats['min']:>7.2f} {stats['max']:>7.2f} "
                  f"{stats['row_count']:>6}")
        print("…")
        print(f"\nPeak: {peak_sensor} max={peak_max:.2f}")


if __name__ == "__main__":
    main()
