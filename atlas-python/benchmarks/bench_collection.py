"""Compare atlas / netCDF / zarr on a 1000-dataset collection workload.

Each backend uses its canonical "many datasets" layout and read pattern:
    atlas  — 1 store, 1000 datasets;       iterate `open_as_xarray_dataset(name)`
    netCDF — 1000 .nc files;               `xr.open_mfdataset(files, ...)`
    zarr   — 1 store, 1000 groups;         iterate `xr.open_zarr(..., group=name)`

Measures write time, read-slice time, and on-disk size. Compression is matched
where each ecosystem supports it (zstd for atlas + zarr, zlib for netCDF).

Run:
    pip install -e "atlas-python[bench]"     # one-time
    python atlas-python/benchmarks/bench_collection.py --datasets 1000
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Callable

# Make `_common` importable when run directly (`python bench_collection.py`)
# or as a module (`python -m atlas.benchmarks.bench_collection`).
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import xarray as xr

import atlas
from _common import (
    CASES,
    BackendResult,
    Case,
    dir_size_bytes,
    generate_dataset,
    log_phase,
    parallel_compute,
    print_results_table,
    progress_tick,
    slice_indexers,
    time_block,
)


# ── atlas ─────────────────────────────────────────────────────────────────


def bench_atlas(
    n_datasets: int,
    root: Path,
    case: Case,
    slice_fraction: float,
    use_dask: bool = False,
    dask_workers: int | None = None,
) -> BackendResult:
    if root.exists():
        shutil.rmtree(root)

    var_names = list(case.var_names)
    indexers = slice_indexers(case, slice_fraction)

    chunks_per_var = (
        {v: list(case.chunk_shape) for v in var_names} if case.chunk_shape else None
    )

    log_phase("atlas", f"writing {n_datasets} datasets (case={case.name})")
    with time_block() as wt:
        with atlas.Atlas.create(
            str(root),
            codec="zstd",
            meta_format="msgpack",
            meta_compression="zstd",
        ) as store:
            for i in range(n_datasets):
                store.add_xarray_dataset(
                    generate_dataset(i, case, use_dask=use_dask),
                    f"ds_{i:04d}",
                    chunks=chunks_per_var,
                )
                progress_tick(i, n_datasets, "write")
    log_phase("atlas", "write", f"{wt.elapsed:.3f}s")

    size = dir_size_bytes(root)
    log_phase("atlas", "size", f"{size / (1024 * 1024):.3f} MiB")

    log_phase(
        "atlas",
        f"reading slice from {n_datasets} datasets"
        + (f" (dask, workers={dask_workers or 'default'})" if use_dask else ""),
    )
    store = atlas.Atlas.open(str(root))
    if use_dask:
        from dask.delayed import delayed

        # Fast path: per-dataset slice reads via `view.read_arrays(...)`
        # bypass open_as_xarray_dataset's xr.Dataset + dask-graph build overhead. The
        # dask scheduler still parallelises across datasets; each task does
        # just one Rust call per variable with the slice already applied.
        start = [s.start for s in indexers.values()]
        shape = [s.stop - s.start for s in indexers.values()]

        @delayed
        def load_one(idx: int):
            view = store.open_dataset(f"ds_{idx:04d}")
            return view.read_arrays(var_names, start=start, shape=shape)

        tasks = [load_one(i) for i in range(n_datasets)]
        with time_block() as rt:
            parallel_compute(tasks, dask_workers)
    else:
        with time_block() as rt:
            for i in range(n_datasets):
                ds = store.open_as_xarray_dataset(f"ds_{i:04d}")
                _ = ds[var_names].isel(indexers).load()
                progress_tick(i, n_datasets, "read")
    log_phase("atlas", "read", f"{rt.elapsed:.3f}s")

    return BackendResult(name="atlas", write_s=wt.elapsed, read_s=rt.elapsed, size_bytes=size)


# ── atlas (bulk, single-call open_as_many_xarray_dataset) ────────────────────────────


def bench_atlas_bulk(
    n_datasets: int,
    root: Path,
    case: Case,
    slice_fraction: float,
    use_dask: bool = False,
    dask_workers: int | None = None,
) -> BackendResult:
    """Same write phase as `bench_atlas`; read phase is one
    `Atlas.open_as_many_xarray_dataset(...)` call — the atlas-native equivalent of
    `xr.open_mfdataset`. Stacked Dataset is sliced and `.load()`-ed in one
    dask compute."""
    import dask.config as dask_config

    if root.exists():
        shutil.rmtree(root)

    var_names = list(case.var_names)
    indexers = slice_indexers(case, slice_fraction)
    names = [f"ds_{i:04d}" for i in range(n_datasets)]
    chunks_per_var = (
        {v: list(case.chunk_shape) for v in var_names} if case.chunk_shape else None
    )

    log_phase("atlas-bulk", f"writing {n_datasets} datasets (case={case.name})")
    with time_block() as wt:
        with atlas.Atlas.create(
            str(root),
            codec="zstd",
            meta_format="msgpack",
            meta_compression="zstd",
        ) as store:
            for i, name in enumerate(names):
                store.add_xarray_dataset(
                    generate_dataset(i, case, use_dask=use_dask),
                    name,
                    chunks=chunks_per_var,
                )
                progress_tick(i, n_datasets, "write")
    log_phase("atlas-bulk", "write", f"{wt.elapsed:.3f}s")

    size = dir_size_bytes(root)
    log_phase("atlas-bulk", "size", f"{size / (1024 * 1024):.3f} MiB")

    log_phase(
        "atlas-bulk",
        f"read_array_across_stacked × {len(var_names)} vars + slice push-down"
        + (f" (dask, workers={dask_workers or 'default'})" if use_dask else ""),
    )
    store = atlas.Atlas.open(str(root))

    # Push the slice down through the low-level API so atlas only decompresses
    # the chunks overlapping the slice (matches what zarr/netcdf's
    # `open_mfdataset(...).isel(...).load()` does via dask graph optimization).
    # `open_as_many_xarray_dataset(...).isel(...)` would also work but slices in numpy
    # *after* decompressing the full per-dataset chunks, wasting work when
    # `chunk_shape != shape`.
    start = [s.start for s in indexers.values()]
    shape = [s.stop - s.start for s in indexers.values()]
    scheduler_ctx = (
        dask_config.set(scheduler="threads", num_workers=dask_workers)
        if use_dask
        else contextlib.nullcontext()
    )
    with scheduler_ctx, time_block() as rt:
        # One Rust call per variable; each returns a stacked
        # (N, *slice_shape) numpy array directly.
        for var in var_names:
            _ = store.read_array_across_stacked(var, names, start=start, shape=shape)
    log_phase("atlas-bulk", "read", f"{rt.elapsed:.3f}s")

    return BackendResult(name="atlas-bulk", write_s=wt.elapsed, read_s=rt.elapsed, size_bytes=size)


# ── netCDF ────────────────────────────────────────────────────────────────


def bench_netcdf(
    n_datasets: int,
    root: Path,
    case: Case,
    slice_fraction: float,
    use_groups: bool = False,
    use_dask: bool = False,
    dask_workers: int | None = None,
    serial: bool = False,
) -> BackendResult:
    """Default layout is 1000 separate .nc files (the standard CMIP /
    observational pattern). Pass `use_groups=True` to write a single .nc file
    containing 1000 netCDF4 groups. Pass `serial=True` to disable
    `open_mfdataset`'s dask parallelism on read — iterate files in a plain
    Python loop instead; the apples-to-apples comparison against atlas
    (default, serial).
    """
    label = "netcdf-groups" if use_groups else (
        "netcdf-no-dask" if serial else "netcdf"
    )
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    var_names = list(case.var_names)
    indexers = slice_indexers(case, slice_fraction)

    netcdf_encoding = {
        var: {
            "zlib": True,
            "complevel": 4,
            **({"chunksizes": list(case.chunk_shape)} if case.chunk_shape else {}),
        }
        for var in var_names
    }

    if not use_groups:
        # Default: 1000 separate .nc files. open_mfdataset is the canonical
        # xarray idiom for this layout.
        log_phase(label, f"writing {n_datasets} .nc files (case={case.name})")
        with time_block() as wt:
            for i in range(n_datasets):
                ds = generate_dataset(i, case, use_dask=use_dask)
                ds.to_netcdf(
                    root / f"ds_{i:04d}.nc",
                    engine="netcdf4",
                    encoding=netcdf_encoding,
                )
                progress_tick(i, n_datasets, "write")
        log_phase(label, "write", f"{wt.elapsed:.3f}s")

        size = dir_size_bytes(root)
        log_phase(label, "size", f"{size / (1024 * 1024):.3f} MiB")

        files = sorted(root.glob("ds_*.nc"))
        if serial:
            # No open_mfdataset, no dask: iterate per file, slice, load.
            # Apples-to-apples with `atlas` (default, serial).
            log_phase(label, f"serial open_dataset across {len(files)} files + load slice")
            with time_block() as rt:
                for i, path in enumerate(files):
                    ds = xr.open_dataset(path, engine="netcdf4")
                    _ = ds[var_names].isel(indexers).load()
                    ds.close()
                    progress_tick(i, n_datasets, "read")
            log_phase(label, "read", f"{rt.elapsed:.3f}s")
        else:
            # open_mfdataset already uses dask under the hood; --use-dask just
            # tightens the worker count when set.
            log_phase(label, f"open_mfdataset across {len(files)} files + load slice")
            import dask.config as dask_config

            scheduler_ctx = (
                dask_config.set(scheduler="threads", num_workers=dask_workers)
                if use_dask
                else contextlib.nullcontext()
            )
            with scheduler_ctx, time_block() as rt:
                ds = xr.open_mfdataset(
                    files,
                    combine="nested",
                    concat_dim="station",
                    parallel=True,
                    engine="netcdf4",
                    combine_attrs="drop_conflicts",
                )
                _ = ds[var_names].isel(indexers).load()
                ds.close()
            log_phase(label, "read", f"{rt.elapsed:.3f}s")
    else:
        # Single .nc file with 1000 netCDF4 groups. Less common in the wild
        # but the apples-to-apples analog to zarr/atlas one-container layouts.
        path = root / "store.nc"
        log_phase(label, f"writing {n_datasets} groups into one .nc file (case={case.name})")
        with time_block() as wt:
            for i in range(n_datasets):
                ds = generate_dataset(i, case, use_dask=use_dask)
                # First write creates the file; subsequent writes append a group.
                ds.to_netcdf(
                    path,
                    group=f"ds_{i:04d}",
                    mode="w" if i == 0 else "a",
                    engine="netcdf4",
                    encoding=netcdf_encoding,
                )
                progress_tick(i, n_datasets, "write")
        log_phase(label, "write", f"{wt.elapsed:.3f}s")

        size = dir_size_bytes(root)
        log_phase(label, "size", f"{size / (1024 * 1024):.3f} MiB")

        # No open_mfdataset for groups-in-one-file — iterate, optionally via dask.delayed.
        log_phase(
            label,
            f"reading slice from {n_datasets} groups"
            + (f" (dask, workers={dask_workers or 'default'})" if use_dask else ""),
        )
        if use_dask:
            from dask.delayed import delayed

            @delayed
            def load_one(idx: int):
                ds = xr.open_dataset(path, group=f"ds_{idx:04d}", engine="netcdf4")
                out = ds[var_names].isel(indexers).load()
                ds.close()
                return out

            tasks = [load_one(i) for i in range(n_datasets)]
            with time_block() as rt:
                parallel_compute(tasks, dask_workers)
        else:
            with time_block() as rt:
                for i in range(n_datasets):
                    ds = xr.open_dataset(path, group=f"ds_{i:04d}", engine="netcdf4")
                    _ = ds[var_names].isel(indexers).load()
                    ds.close()
                    progress_tick(i, n_datasets, "read")
        log_phase(label, "read", f"{rt.elapsed:.3f}s")

    return BackendResult(name=label, write_s=wt.elapsed, read_s=rt.elapsed, size_bytes=size)


# ── zarr ──────────────────────────────────────────────────────────────────


def bench_zarr(
    n_datasets: int,
    root: Path,
    case: Case,
    slice_fraction: float,
    use_groups: bool = False,
    use_dask: bool = False,
    dask_workers: int | None = None,
    serial: bool = False,
) -> BackendResult:
    """Default layout is 1000 separate zarr stores (mirrors the netcdf
    1000-files default). Pass `use_groups=True` for one store with 1000
    groups inside (zarr's other canonical multi-dataset pattern). Pass
    `serial=True` to disable `open_mfdataset`'s dask parallelism on read —
    iterate stores one at a time instead.
    """
    import warnings

    import dask.config as dask_config
    from zarr.codecs import ZstdCodec

    label = "zarr-groups" if use_groups else (
        "zarr-no-dask" if serial else "zarr"
    )
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    var_names = list(case.var_names)
    indexers = slice_indexers(case, slice_fraction)

    # xarray writes consolidated metadata by default, which zarr 3 warns about
    # on every group write — drowns the benchmark output. The warning is
    # informational (consolidated metadata isn't in the v3 spec yet, but works);
    # silence it to keep the timing reads readable.
    warnings.filterwarnings(
        "ignore",
        message=".*Consolidated metadata.*",
        category=UserWarning,
    )

    # zarr 3 encoding: `compressors` is a list of v3 codec instances per
    # variable. (Pass a `zarr.codecs.ZstdCodec` — numcodecs codecs are not
    # accepted directly by zarr 3's v3 array path.)
    zarr_encoding = {
        var: {
            "compressors": (ZstdCodec(level=3),),
            **({"chunks": tuple(case.chunk_shape)} if case.chunk_shape else {}),
        }
        for var in var_names
    }

    if not use_groups:
        # Default: 1000 separate zarr stores. open_mfdataset works for
        # multiple zarr stores via engine="zarr".
        log_phase(label, f"writing {n_datasets} .zarr stores (case={case.name})")
        with time_block() as wt:
            for i in range(n_datasets):
                ds = generate_dataset(i, case, use_dask=use_dask)
                ds.to_zarr(
                    str(root / f"ds_{i:04d}.zarr"),
                    mode="w",
                    encoding=zarr_encoding,
                    zarr_format=3,
                )
                progress_tick(i, n_datasets, "write")
        log_phase(label, "write", f"{wt.elapsed:.3f}s")

        size = dir_size_bytes(root)
        log_phase(label, "size", f"{size / (1024 * 1024):.3f} MiB")

        stores = sorted(str(p) for p in root.glob("ds_*.zarr"))
        if serial:
            log_phase(label, f"serial open_zarr across {len(stores)} stores + load slice")
            with time_block() as rt:
                for i, store in enumerate(stores):
                    ds = xr.open_zarr(store)
                    _ = ds[var_names].isel(indexers).load()
                    ds.close()
                    progress_tick(i, n_datasets, "read")
            log_phase(label, "read", f"{rt.elapsed:.3f}s")
        else:
            # open_mfdataset already uses dask under the hood; --use-dask just
            # tightens the worker count when set.
            log_phase(label, f"open_mfdataset across {len(stores)} stores + load slice")
            scheduler_ctx = (
                dask_config.set(scheduler="threads", num_workers=dask_workers)
                if use_dask
                else contextlib.nullcontext()
            )
            with scheduler_ctx, time_block() as rt:
                ds = xr.open_mfdataset(
                    stores,
                    combine="nested",
                    concat_dim="station",
                    parallel=True,
                    engine="zarr",
                    combine_attrs="drop_conflicts",
                )
                _ = ds[var_names].isel(indexers).load()
                ds.close()
            log_phase(label, "read", f"{rt.elapsed:.3f}s")
    else:
        # One store with 1000 groups. open_mfdataset doesn't apply (it's for
        # multiple stores, not groups in one); per-group iteration is the
        # only option.
        log_phase(label, f"writing {n_datasets} groups into one store (case={case.name})")
        with time_block() as wt:
            for i in range(n_datasets):
                ds = generate_dataset(i, case, use_dask=use_dask)
                ds.to_zarr(
                    str(root),
                    group=f"ds_{i:04d}",
                    mode="a",
                    encoding=zarr_encoding,
                    zarr_format=3,
                )
                progress_tick(i, n_datasets, "write")
        log_phase(label, "write", f"{wt.elapsed:.3f}s")

        size = dir_size_bytes(root)
        log_phase(label, "size", f"{size / (1024 * 1024):.3f} MiB")

        log_phase(
            label,
            f"reading slice from {n_datasets} groups"
            + (f" (dask, workers={dask_workers or 'default'})" if use_dask else ""),
        )
        if use_dask:
            from dask.delayed import delayed

            @delayed
            def load_one(idx: int):
                ds = xr.open_zarr(str(root), group=f"ds_{idx:04d}")
                return ds[var_names].isel(indexers).load()

            tasks = [load_one(i) for i in range(n_datasets)]
            with time_block() as rt:
                parallel_compute(tasks, dask_workers)
        else:
            with time_block() as rt:
                for i in range(n_datasets):
                    ds = xr.open_zarr(str(root), group=f"ds_{i:04d}")
                    _ = ds[var_names].isel(indexers).load()
                    progress_tick(i, n_datasets, "read")
        log_phase(label, "read", f"{rt.elapsed:.3f}s")

    return BackendResult(name=label, write_s=wt.elapsed, read_s=rt.elapsed, size_bytes=size)


# ── orchestration ────────────────────────────────────────────────────────


BACKENDS = {
    "atlas": bench_atlas,
    "netcdf": bench_netcdf,
    "zarr": bench_zarr,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--datasets", type=int, default=1000)
    p.add_argument(
        "--backends",
        default="atlas,netcdf,zarr",
        help="Comma-separated subset of: atlas,netcdf,zarr",
    )
    p.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Re-run each backend N times and report the mean (size is single-run).",
    )
    p.add_argument(
        "--case",
        default="sensors",
        choices=sorted(CASES.keys()),
        help=(
            "Workload case (default 'sensors' — preserves the original "
            "sensor-fleet behavior). 'gridded' is (lon, lat, time); "
            "'profile' is (depth, time). Scale --datasets down for bigger cases."
        ),
    )
    p.add_argument(
        "--slice-fraction",
        type=float,
        default=0.25,
        help=(
            "Read slice covers the first int(F * dim_len) elements of every "
            "dim. Default 0.25; for 'sensors' time=24 this gives time=[0:6]."
        ),
    )
    p.add_argument(
        "--atlas-bulk",
        action="store_true",
        help=(
            "Add an `atlas-bulk` row that reads via Atlas.open_as_many_xarray_dataset "
            "(atlas-native equivalent of xr.open_mfdataset) instead of "
            "iterating per-dataset. Additive — the default `atlas` row still runs."
        ),
    )
    p.add_argument(
        "--netcdf-groups",
        action="store_true",
        help="Use one .nc file with N groups instead of N separate .nc files (default).",
    )
    p.add_argument(
        "--netcdf-no-dask",
        action="store_true",
        help=(
            "Add a `netcdf-no-dask` row that reads N .nc files in a serial "
            "Python loop (no `open_mfdataset`, no dask). Apples-to-apples "
            "against atlas (default, serial). Additive — the default "
            "`netcdf` row still runs."
        ),
    )
    p.add_argument(
        "--zarr-groups",
        action="store_true",
        help="Use one zarr store with N groups instead of N separate stores (default).",
    )
    p.add_argument(
        "--zarr-no-dask",
        action="store_true",
        help=(
            "Add a `zarr-no-dask` row that reads N zarr stores in a serial "
            "Python loop (no `open_mfdataset`, no dask). Apples-to-apples "
            "against atlas (default, serial). Additive — the default "
            "`zarr` row still runs."
        ),
    )
    p.add_argument(
        "--use-dask",
        action="store_true",
        help=(
            "Use dask to parallelize iteration-based reads (atlas, "
            "netcdf-groups, zarr-groups) via dask.delayed, and produce "
            "dask-backed source xr.Datasets so the write path exercises "
            "each backend's chunk-streaming code. The open_mfdataset variants "
            "(default netcdf, default zarr) already use dask under the hood; "
            "this flag just constrains the worker count when set."
        ),
    )
    p.add_argument(
        "--dask-workers",
        type=int,
        default=None,
        help=(
            "Number of dask threads when --use-dask is set. Defaults to "
            "dask's default (typically CPU count)."
        ),
    )
    p.add_argument(
        "--keep-output",
        action="store_true",
        help="Don't delete the tempdir on exit (useful for inspecting files).",
    )
    p.add_argument(
        "--n-vars",
        type=int,
        default=None,
        help=(
            "Override the number of variables in the case by cycling the "
            "case's default var list and suffixing copies (`_2`, `_3`, ...). "
            "Default = case's native var count."
        ),
    )
    return p.parse_args()


def _build_tasks(
    backends: list[str], n_datasets: int, args: argparse.Namespace,
    case: Case | None = None,
) -> list[tuple[str, Callable[[Path], BackendResult]]]:
    """Expand the backend list + optional --*-groups flags into a flat list of
    (label, runner) tasks. Each `--*-groups` flag adds an *additional* row
    alongside the default layout for that backend, not in place of it."""
    use_dask = args.use_dask
    dask_workers = args.dask_workers
    if case is None:
        case = CASES[args.case]
    slice_fraction = args.slice_fraction
    tasks: list[tuple[str, Callable[[Path], BackendResult]]] = []
    for name in backends:
        if name == "atlas":
            tasks.append((
                "atlas",
                lambda root: bench_atlas(
                    n_datasets, root, case, slice_fraction,
                    use_dask=use_dask, dask_workers=dask_workers,
                ),
            ))
            if args.atlas_bulk:
                tasks.append((
                    "atlas-bulk",
                    lambda root: bench_atlas_bulk(
                        n_datasets, root, case, slice_fraction,
                        use_dask=use_dask, dask_workers=dask_workers,
                    ),
                ))
        elif name == "netcdf":
            tasks.append((
                "netcdf",
                lambda root: bench_netcdf(
                    n_datasets, root, case, slice_fraction,
                    use_groups=False, use_dask=use_dask, dask_workers=dask_workers,
                ),
            ))
            if args.netcdf_groups:
                tasks.append((
                    "netcdf-groups",
                    lambda root: bench_netcdf(
                        n_datasets, root, case, slice_fraction,
                        use_groups=True, use_dask=use_dask, dask_workers=dask_workers,
                    ),
                ))
            if args.netcdf_no_dask:
                tasks.append((
                    "netcdf-no-dask",
                    lambda root: bench_netcdf(
                        n_datasets, root, case, slice_fraction,
                        serial=True,
                    ),
                ))
        elif name == "zarr":
            tasks.append((
                "zarr",
                lambda root: bench_zarr(
                    n_datasets, root, case, slice_fraction,
                    use_groups=False, use_dask=use_dask, dask_workers=dask_workers,
                ),
            ))
            if args.zarr_groups:
                tasks.append((
                    "zarr-groups",
                    lambda root: bench_zarr(
                        n_datasets, root, case, slice_fraction,
                        use_groups=True, use_dask=use_dask, dask_workers=dask_workers,
                    ),
                ))
            if args.zarr_no_dask:
                tasks.append((
                    "zarr-no-dask",
                    lambda root: bench_zarr(
                        n_datasets, root, case, slice_fraction,
                        serial=True,
                    ),
                ))
    return tasks


def main() -> None:
    args = parse_args()
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    for b in backends:
        if b not in BACKENDS:
            sys.exit(f"unknown backend {b!r}; choose from {list(BACKENDS)}")

    case = CASES[args.case]
    if args.n_vars is not None and args.n_vars > 0:
        original = case.vars
        if args.n_vars <= len(original):
            new_vars = original[: args.n_vars]
        else:
            new_vars = list(original)
            for i in range(len(original), args.n_vars):
                src = original[i % len(original)]
                suffix = f"_{(i // len(original)) + 1}"
                new_vars.append(dataclasses.replace(src, name=src.name + suffix))
            new_vars = tuple(new_vars)
        case = dataclasses.replace(case, vars=new_vars)
    tasks = _build_tasks(backends, args.datasets, args, case=case)

    tmp = Path(tempfile.mkdtemp(prefix="atlas-bench-"))
    print(f"Working dir: {tmp}")
    print(f"Case:        {case.name} — shape={case.shape} dims={case.dim_names}")
    print(f"Tasks:       {[t[0] for t in tasks]}")
    if args.use_dask:
        print(f"Dask:        enabled, workers={args.dask_workers or 'default'}")
    try:
        results: list[BackendResult] = []
        for label, runner in tasks:
            print(f"\n── {label} ──")
            runs: list[BackendResult] = []
            for r in range(args.repeats):
                print(f"  repeat {r + 1}/{args.repeats}…", flush=True)
                runs.append(runner(tmp / label))
            mean_write = float(np.mean([x.write_s for x in runs]))
            mean_read = float(np.mean([x.read_s for x in runs]))
            results.append(
                BackendResult(
                    name=label,
                    write_s=mean_write,
                    read_s=mean_read,
                    size_bytes=runs[-1].size_bytes,
                )
            )
        print_results_table(results, args.datasets, case, args.slice_fraction)
    finally:
        if args.keep_output:
            print(f"\nOutput kept at: {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
