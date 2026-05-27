"""Shared helpers for the atlas / netCDF / zarr benchmark."""
from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr


@dataclass(frozen=True)
class VarSpec:
    """One variable in a workload case."""
    name: str
    dtype: str  # numpy dtype string, e.g. "float32"
    units: str
    # baseline + scale used by the generator's RNG: value = baseline + N(0, scale).
    baseline: float
    scale: float


@dataclass(frozen=True)
class DimSpec:
    """One dimension in a workload case."""
    name: str
    size: int
    # How to fill the coord array. "time" -> hourly datetime64[ns];
    # "linspace" -> evenly-spaced float32 from start..stop.
    coord: str = "linspace"
    start: float = 0.0
    stop: float = 1.0


@dataclass(frozen=True)
class Case:
    """A named workload — variables, dims, attrs, optional chunk shape."""
    name: str
    vars: tuple[VarSpec, ...]
    dims: tuple[DimSpec, ...]
    attr_key: str  # name of the per-dataset attr that records idx
    # Optional per-variable on-disk chunk shape (one value per dim, in dim order).
    # `None` = single chunk per variable (chunk_shape == shape) — the
    # pessimal-for-everyone default. For workloads where the read slice is
    # smaller than the full shape, set a chunk smaller than the slice so
    # zarr/netCDF/atlas can each push the slice down to chunk-level reads.
    chunk_shape: tuple[int, ...] | None = None

    @property
    def var_names(self) -> tuple[str, ...]:
        return tuple(v.name for v in self.vars)

    @property
    def dim_names(self) -> tuple[str, ...]:
        return tuple(d.name for d in self.dims)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(d.size for d in self.dims)

    def bytes_per_dataset(self) -> int:
        n = int(np.prod(self.shape))
        return sum(np.dtype(v.dtype).itemsize * n for v in self.vars)


# ── Case registry ────────────────────────────────────────────────────────

CASES: dict[str, Case] = {
    "sensors": Case(
        name="sensors",
        vars=(
            VarSpec("temperature", "float32", "celsius", baseline=18.0, scale=2.0),
            VarSpec("pressure", "float64", "hPa", baseline=1013.25, scale=3.0),
            VarSpec("humidity", "float32", "percent", baseline=60.0, scale=10.0),
        ),
        dims=(DimSpec("time", 24, coord="time"),),
        attr_key="station_id",
    ),
    "gridded": Case(
        name="gridded",
        vars=(
            VarSpec("temperature", "float32", "celsius", baseline=15.0, scale=5.0),
            VarSpec("pressure", "float64", "hPa", baseline=1013.0, scale=8.0),
            VarSpec("humidity", "float32", "percent", baseline=65.0, scale=15.0),
        ),
        dims=(
            DimSpec("lon", 100, coord="linspace", start=-180.0, stop=180.0),
            DimSpec("lat", 100, coord="linspace", start=-90.0, stop=90.0),
            DimSpec("time", 48, coord="time"),
        ),
        # 2x2x2 = 8 chunks per variable, ~120 KiB raw per chunk — a typical
        # climate-data chunk size. The default 0.25 slice (lon=0:25, lat=0:25,
        # time=0:12) fits inside one chunk per dim, so all three backends can
        # push the isel down and decompress 1/8 of the data per dataset
        # instead of the full chunk. Without this — single-chunk arrays —
        # all three are forced to decompress the entire (100,100,48) volume.
        chunk_shape=(50, 50, 24),
        attr_key="grid_id",
    ),
    "profile": Case(
        name="profile",
        vars=(
            VarSpec("temperature", "float32", "celsius", baseline=10.0, scale=4.0),
            VarSpec("salinity", "float32", "psu", baseline=35.0, scale=1.5),
        ),
        dims=(
            DimSpec("depth", 50, coord="linspace", start=0.0, stop=500.0),
            DimSpec("time", 168, coord="time"),
        ),
        # Profile data is small; one chunk per dataset is fine.
        attr_key="cast_id",
    ),
}


# ── Coordinate generators ────────────────────────────────────────────────


def _build_coord(d: DimSpec) -> np.ndarray:
    if d.coord == "time":
        return np.arange(
            np.datetime64("2024-01-01T00:00:00", "ns"),
            np.datetime64("2024-01-01T00:00:00", "ns") + np.timedelta64(d.size, "h"),
            np.timedelta64(1, "h"),
        )
    # linspace fallback — float32 coord
    return np.linspace(d.start, d.stop, d.size, dtype=np.float32)


def _dask_chunks(case: Case) -> tuple[int, ...]:
    """Split the largest dim into ~2 chunks; leave others whole. Cheap default
    that exercises chunk-streaming without aggressive fragmentation."""
    biggest = max(range(len(case.dims)), key=lambda i: case.dims[i].size)
    return tuple(
        max(1, d.size // 2) if i == biggest else d.size
        for i, d in enumerate(case.dims)
    )


# ── Generator ────────────────────────────────────────────────────────────


def generate_dataset(idx: int, case: Case, use_dask: bool = False) -> xr.Dataset:
    """Deterministic per-dataset xr.Dataset for `case`. Same content for the
    same `(idx, case)` across backends so storage / read comparisons reflect
    format choices, not data."""
    rng = np.random.default_rng(seed=idx)
    shape = case.shape

    data_vars: dict[str, xr.DataArray] = {}
    if use_dask:
        import dask.array as da
        chunks = _dask_chunks(case)
    for var in case.vars:
        arr = (var.baseline + rng.normal(scale=var.scale, size=shape)).astype(var.dtype)
        if var.name == "humidity":
            arr = np.clip(arr, 0, 100).astype(var.dtype)
        if use_dask:
            arr = da.from_array(arr, chunks=chunks)
        data_vars[var.name] = xr.DataArray(arr, dims=list(case.dim_names), attrs={"units": var.units})

    coords = {d.name: (d.name, _build_coord(d)) for d in case.dims}
    return xr.Dataset(data_vars=data_vars, coords=coords, attrs={case.attr_key: idx})


# ── Slicing ──────────────────────────────────────────────────────────────


def slice_indexers(case: Case, fraction: float) -> dict[str, slice]:
    """`{dim_name: slice(0, int(fraction * dim_size))}` for every dim in the
    case. Always at least 1 element per dim so slices remain non-empty."""
    out: dict[str, slice] = {}
    for d in case.dims:
        n = max(1, int(fraction * d.size))
        out[d.name] = slice(0, n)
    return out


def sliced_shape(case: Case, fraction: float) -> tuple[int, ...]:
    return tuple(s.stop - s.start for s in slice_indexers(case, fraction).values())


# ── Misc helpers ─────────────────────────────────────────────────────────


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
    """Print "label: 200/1000" at ~10 evenly-spaced points (or every `every`
    items if positive). Always prints the final tick."""
    step = every if every > 0 else max(1, total // 10)
    if (i + 1) % step == 0 or (i + 1) == total:
        print(f"  {label}: {i + 1}/{total}", flush=True)


def log_phase(backend: str, phase: str, msg: str = "") -> None:
    suffix = f" — {msg}" if msg else ""
    print(f"[{backend}] {phase}{suffix}", flush=True)


def parallel_compute(tasks: list, num_workers: int | None) -> list:
    """Run a list of `dask.delayed` tasks on the threaded scheduler.

    Threads not processes — atlas / netCDF / zarr handles are not picklable.
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


def print_results_table(
    results: list[BackendResult], n_datasets: int, case: Case, slice_fraction: float
) -> None:
    shape_str = ", ".join(f"{d.name}={d.size}" for d in case.dims)
    slice_str = ", ".join(f"{d.name}=[0:{max(1, int(slice_fraction * d.size))}]" for d in case.dims)
    print()
    print(
        f"Workload: {n_datasets} datasets × case={case.name!r} × {len(case.vars)} vars\n"
        f"  shape per var : ({shape_str})\n"
        f"  read slice    : {slice_str}  (fraction={slice_fraction})"
    )
    print("─" * 68)
    print(f"{'backend':<14} {'write (s)':>12} {'read slice (s)':>16} {'storage (MiB)':>16}")
    print("─" * 68)
    for r in results:
        mib = r.size_bytes / (1024 * 1024)
        print(f"{r.name:<14} {r.write_s:>12.3f} {r.read_s:>16.3f} {mib:>16.3f}")
