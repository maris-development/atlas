"""Regenerate the benchmark sweep SVGs embedded in docs/benchmarks.md.

Each chart groups bars by dataset count (3 groups: 100 / 500 / 1000) and
colors them by backend. Output is theme-neutral SVG with a transparent
background, so it sits cleanly on Material's light or dark palette.

Re-run after `bench_collection.py` produces new numbers:

    python atlas-python/benchmarks/generate_bench_charts.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Repo layout: atlas-python/benchmarks/generate_bench_charts.py → docs/assets/
OUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "assets"

# Backend → color. Ordered slowest → fastest within each dataset-count
# group. The chart focuses on the parallel paths every backend actually
# uses in production xarray code: `xr.open_mfdataset(parallel=True)` for
# netcdf and zarr (dask under the hood), `dask.delayed` for atlas, and
# atlas's bulk PyO3 path.
#
# The serial-loop variants are skipped here — pass `--netcdf-no-dask` /
# `--zarr-no-dask` and run `bench_collection.py` without `--use-dask` to
# capture them. atlas (default, serial) tracks atlas+dask × ~3 across
# the sweep; reach for the parallel path once you cross a few hundred
# datasets.
BACKENDS: list[tuple[str, str]] = [
    ("netcdf+dask",  "#ffa726"),  # orange 400  — open_mfdataset(parallel=True)
    ("zarr+dask",    "#ef5350"),  # red 400     — open_mfdataset(parallel=True)
    ("atlas+dask",   "#7c4dff"),  # purple A200 — dask.delayed fan-out
    ("atlas-bulk",   "#00bfa5"),  # teal A700   — Atlas.read_array_across_stacked
]

DATASET_COUNTS = [100, 500, 1000]

# Numbers from the sweep recorded in atlas-python/docs/benchmarks.md.
# read slice (seconds), indexed [backend_idx][dataset_count_idx];
# the order of these rows must match BACKENDS above.
RESULTS: dict[str, list[list[float]]] = {
    "profile": [
        # netcdf+dask: 100ds, 500ds, 1000ds
        [0.564, 3.901, 7.409],
        # zarr+dask
        [0.543, 2.734, 5.879],
        # atlas+dask (--use-dask)
        [0.023, 0.081, 0.148],
        # atlas-bulk
        [0.013, 0.064, 0.122],
    ],
    "gridded": [
        # netcdf+dask
        [0.873, 6.025, 14.037],
        # zarr+dask
        [0.506, 2.711,  5.968],
        # atlas+dask
        [0.233, 1.410,  3.763],
        # atlas-bulk
        [0.220, 1.114,  2.031],
    ],
}

CASE_META: dict[str, dict[str, str | int]] = {
    "profile": dict(
        title="Profile — read slice (seconds, lower is better)",
        subtitle="(50, 168) per variable × 4 variables, slice 25%",
        ymax=8,
    ),
    "gridded": dict(
        title="Gridded — read slice (seconds, lower is better)",
        subtitle="(100, 100, 48) per variable × 3 variables, chunks (50, 50, 24), slice 25%",
        ymax=16,
    ),
}


def render(case: str) -> Path:
    meta = CASE_META[case]
    data = RESULTS[case]
    n_backends = len(BACKENDS)
    n_groups = len(DATASET_COUNTS)

    fig, ax = plt.subplots(figsize=(9.6, 4.4), dpi=110)
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    group_x = np.arange(n_groups)
    bar_w = 0.78 / n_backends

    for i, (label, color) in enumerate(BACKENDS):
        offsets = group_x - 0.39 + bar_w * (i + 0.5)
        bars = ax.bar(offsets, data[i], bar_w, label=label, color=color,
                      edgecolor="none")
        # Value labels — small, just above each bar
        for x, y in zip(offsets, data[i]):
            ax.annotate(f"{y:.2f}", (x, y), ha="center", va="bottom",
                        fontsize=7.5, color="#555")

    ax.set_xticks(group_x)
    ax.set_xticklabels([f"{n} datasets" for n in DATASET_COUNTS])
    ax.set_ylabel("Read slice (s)")
    ax.set_ylim(0, meta["ymax"])
    ax.set_title(meta["title"], fontsize=11, pad=10)
    # Subtitle under the main title
    ax.text(0.5, 1.01, meta["subtitle"], transform=ax.transAxes,
            ha="center", va="bottom", fontsize=8.5, color="#666")

    # Subtle grid; spines off except bottom
    ax.yaxis.grid(True, linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#999")
    ax.spines["bottom"].set_color("#999")
    ax.tick_params(colors="#555")

    ax.legend(loc="upper left", frameon=False, fontsize=9, ncol=len(BACKENDS),
              bbox_to_anchor=(0.0, -0.10))

    plt.tight_layout()
    out_path = OUT_DIR / f"bench_{case}.svg"
    fig.savefig(out_path, format="svg", bbox_inches="tight",
                transparent=True)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    repo_root = OUT_DIR.parent.parent.parent  # …/array-store
    for case in ("profile", "gridded"):
        path = render(case)
        try:
            shown = path.relative_to(repo_root)
        except ValueError:
            shown = path
        print(f"wrote {shown}")
