"""Shared helpers for the atlas / netCDF / zarr benchmark."""
from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr


# Workload shape — fixed across backends so the comparison is apples-to-apples.
N_TIME = 24                  # 24 hourly readings per dataset
SLICE_HOURS = 6              # read the first 6 hours from each dataset
VARIABLES = ("temperature", "pressure", "humidity")


def generate_dataset(idx: int, use_dask: bool = False) -> xr.Dataset:
    """Deterministic per-station Dataset; same content for the same idx across
    backends so storage / read comparisons reflect format choices, not data.

    When `use_dask=True`, the three data variables are dask-backed with two
    chunks along the time axis (`chunks=N_TIME // 2`). The values are
    bit-identical to the numpy version — the dask wrapping just stresses
    each backend's chunk-streaming write path.
    """
    rng = np.random.default_rng(seed=idx)
    time_coord = np.arange(
        np.datetime64("2024-01-01T00:00:00", "ns"),
        np.datetime64("2024-01-01T00:00:00", "ns") + np.timedelta64(N_TIME, "h"),
        np.timedelta64(1, "h"),
    )
    baseline = 18.0 + (idx % 8) * 1.5
    temp = (baseline + rng.normal(scale=2.0, size=N_TIME)).astype(np.float32)
    pres = (1013.25 + rng.normal(scale=3.0, size=N_TIME)).astype(np.float64)
    humid = np.clip(rng.normal(loc=60.0, scale=10.0, size=N_TIME), 0, 100).astype(
        np.float32
    )

    if use_dask:
        import dask.array as da

        chunk = max(1, N_TIME // 2)
        temp_arr = da.from_array(temp, chunks=chunk)
        pres_arr = da.from_array(pres, chunks=chunk)
        humid_arr = da.from_array(humid, chunks=chunk)
    else:
        temp_arr = temp
        pres_arr = pres
        humid_arr = humid

    return xr.Dataset(
        data_vars={
            "temperature": xr.DataArray(temp_arr, dims=["time"], attrs={"units": "celsius"}),
            "pressure": xr.DataArray(pres_arr, dims=["time"], attrs={"units": "hPa"}),
            "humidity": xr.DataArray(humid_arr, dims=["time"], attrs={"units": "percent"}),
        },
        coords={"time": ("time", time_coord)},
        attrs={"station_id": idx},
    )


def dir_size_bytes(path: Path) -> int:
    """Recursive sum of file sizes under `path`. Ignores fs block overhead."""
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())


@dataclass
class TimedBlock:
    elapsed: float = 0.0


@contextlib.contextmanager
def time_block():
    """`with time_block() as t: ...; t.elapsed` — wall time via perf_counter."""
    tb = TimedBlock()
    start = time.perf_counter()
    try:
        yield tb
    finally:
        tb.elapsed = time.perf_counter() - start


def progress_tick(i: int, total: int, label: str, every: int = 0) -> None:
    """Print "label: 200/1000 (1.4s)" at ~10 evenly-spaced points (or every
    `every` items if positive). Always prints the final tick."""
    step = every if every > 0 else max(1, total // 10)
    if (i + 1) % step == 0 or (i + 1) == total:
        print(f"  {label}: {i + 1}/{total}", flush=True)


def log_phase(backend: str, phase: str, msg: str = "") -> None:
    suffix = f" — {msg}" if msg else ""
    print(f"[{backend}] {phase}{suffix}", flush=True)


def parallel_compute(tasks: list, num_workers: int | None) -> list:
    """Run a list of `dask.delayed` tasks on the threaded scheduler.

    Threads not processes — pyatlas / netCDF / zarr handles are not picklable.
    Threaded scheduler is fine for I/O-bound work since the GIL is released
    inside Rust / C extensions during the heavy lifting.
    """
    import dask

    with dask.config.set(scheduler="threads", num_workers=num_workers):
        return list(dask.compute(*tasks))


@dataclass
class BackendResult:
    name: str
    write_s: float
    read_s: float
    size_bytes: int


def print_results_table(results: list[BackendResult], n_datasets: int) -> None:
    print()
    print(
        f"Workload: {n_datasets} datasets × {len(VARIABLES)} variables × "
        f"{N_TIME} time elements, read slice [0:{SLICE_HOURS}]"
    )
    print("─" * 68)
    print(f"{'backend':<10} {'write (s)':>12} {'read slice (s)':>16} {'storage (MiB)':>16}")
    print("─" * 68)
    for r in results:
        mib = r.size_bytes / (1024 * 1024)
        print(f"{r.name:<10} {r.write_s:>12.3f} {r.read_s:>16.3f} {mib:>16.3f}")
