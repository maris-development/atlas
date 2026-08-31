"""The six operations atlas supports from Python.

Everything here works the same against a local directory or an object store;
see `_source.resolve`.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any, Iterable, Optional, Sequence

from . import _atlas
from . import _source
from . import xarray as _xarray

# Files that look like NetCDF. `create` scans a directory for these.
NETCDF_SUFFIXES = (".nc", ".nc4", ".cdf", ".netcdf")


class AtlasError(RuntimeError):
    """An operation could not complete. The message says what went wrong."""


def dataset_name(path: "pathlib.Path | str") -> str:
    """The dataset name a NetCDF file is stored under: its stem.

    `/data/2024/jan.nc` becomes `jan`. Accepts a bare name too, so callers can
    pass either a path or a name and get the same answer.
    """
    stem = pathlib.PurePath(str(path)).stem
    return stem or str(path)


def find_netcdf_files(
    directory: "pathlib.Path | str", recursive: bool = False
) -> list[pathlib.Path]:
    """NetCDF files in `directory`, sorted, so a collection's order is stable."""
    root = pathlib.Path(directory)
    if not root.is_dir():
        raise AtlasError(f"not a directory: {root}")
    walk = root.rglob("*") if recursive else root.glob("*")
    return sorted(p for p in walk if p.is_file() and p.suffix.lower() in NETCDF_SUFFIXES)


# ── create ───────────────────────────────────────────────────────────


def create(
    directory: "pathlib.Path | str",
    destination: Any,
    *,
    recursive: bool = False,
    codec: str = "zstd",
    chunks: Optional[dict[str, Sequence[int]]] = None,
    on_error: str = "stop",
    progress: Optional[Any] = None,
    **store_options: Any,
) -> dict[str, Any]:
    """Build a collection at `destination` from the NetCDF files in `directory`.

    Each file becomes one dataset, named after the file stem. Nothing is
    readable at `destination` until every file has been written and the footer
    lands, so a failure leaves no half-built collection behind.

    `on_error` is `"stop"` (the default, abandoning the whole collection) or
    `"skip"` (recording the failure and carrying on). `progress` is called with
    each file's name as it is written.

    Returns a summary: how many datasets were written, and what was skipped.
    """
    if on_error not in ("stop", "skip"):
        raise AtlasError(f"on_error must be 'stop' or 'skip', got {on_error!r}")

    import xarray as xr

    files = find_netcdf_files(directory, recursive=recursive)
    if not files:
        raise AtlasError(f"no NetCDF files under {directory}")

    names: set[str] = set()
    written: list[str] = []
    skipped: list[dict[str, str]] = []

    target = _source.resolve(destination, **store_options)
    with _atlas.AtlasWriter.create(target, codec) as writer:
        for path in files:
            name = dataset_name(path)
            if name in names:
                message = f"duplicate dataset name {name!r} from {path}"
                if on_error == "stop":
                    raise AtlasError(message)
                skipped.append({"file": str(path), "error": message})
                continue

            try:
                with xr.open_dataset(path) as ds:
                    _xarray._write_xarray_dataset(writer, ds, name, chunks, None)
            except Exception as exc:
                if on_error == "stop":
                    raise AtlasError(f"{path}: {exc}") from exc
                skipped.append({"file": str(path), "error": str(exc)})
                continue

            names.add(name)
            written.append(name)
            if progress is not None:
                progress(name)

    return {
        "destination": _source.describe(destination),
        "written": written,
        "skipped": skipped,
        "dataset_count": len(written),
    }


# ── remove ───────────────────────────────────────────────────────────


def remove(
    source: Any,
    targets: Iterable[Any],
    *,
    missing_ok: bool = False,
    **store_options: Any,
) -> dict[str, Any]:
    """Remove datasets from a collection, in one call.

    `targets` are dataset names or NetCDF file paths — a path is reduced to its
    stem, so the same list that built the collection can tear part of it down.

    This updates the deletion mask beside the container; the container itself
    is untouched, so no space is reclaimed and no ordinal moves. Rewrite the
    collection to reclaim the bytes.

    With `missing_ok`, a name that is absent or already removed is reported
    rather than raised.
    """
    wanted = [dataset_name(t) for t in targets]
    if not wanted:
        raise AtlasError("no datasets given")

    collection = _atlas.Atlas.open(_source.resolve(source, **store_options))
    present = set(collection.list_datasets())

    missing = [n for n in wanted if n not in present]
    if missing and not missing_ok:
        raise AtlasError(
            f"not in the collection (or already removed): {', '.join(sorted(missing))}"
        )

    removed = []
    for name in wanted:
        if name in present:
            collection.delete_dataset(name)
            removed.append(name)

    return {
        "removed": removed,
        "missing": missing,
        "remaining": collection.dataset_count(),
    }


# ── list ─────────────────────────────────────────────────────────────


def list_datasets(source: Any, **store_options: Any) -> list[str]:
    """Names of the datasets in the collection, with the deletion mask applied."""
    collection = _atlas.Atlas.open(_source.resolve(source, **store_options))
    return collection.list_datasets()


# ── show ─────────────────────────────────────────────────────────────


def describe_dataset(source: Any, name: Any, **store_options: Any) -> dict[str, Any]:
    """Everything the collection records about one dataset.

    `name` may be a dataset name or the NetCDF path it came from.

    Returns dimensions, every array's type, shape, chunking, fill value,
    attributes, and the statistics recorded when it was written.
    """
    wanted = dataset_name(name)
    collection = _atlas.Atlas.open(_source.resolve(source, **store_options))
    try:
        view = collection.dataset(wanted)
    except KeyError as exc:
        raise AtlasError(
            f"no dataset {wanted!r} in the collection "
            f"(it may have been removed)"
        ) from exc

    coords = set(_coords_of(view))
    dimensions: dict[str, int] = {}
    arrays = []
    for array in view.list_arrays():
        meta = view.array_meta(array)
        for dim, size in zip(meta["dimension_names"], meta["shape"]):
            dimensions.setdefault(dim, size)
        arrays.append(
            {
                "name": array,
                "dtype": meta["dtype"],
                "shape": meta["shape"],
                "chunk_shape": meta["chunk_shape"],
                "dimensions": meta["dimension_names"],
                "fill_value": meta["fill_value"],
                "is_coordinate": array in coords,
                "attributes": {
                    k: _xarray._decode_attr_value(v)
                    for k, v in view.array_attributes(array).items()
                },
                "stats": view.array_stats(array),
            }
        )

    return {
        "name": view.name,
        "ordinal": view.ordinal,
        "segment_range": list(view.segment_range),
        "dimensions": dimensions,
        "coordinates": sorted(coords),
        "arrays": arrays,
        "attributes": _global_attrs(view),
    }


# ── info ─────────────────────────────────────────────────────────────


def info(source: Any, **store_options: Any) -> dict[str, Any]:
    """A summary of the whole collection."""
    collection = _atlas.Atlas.open(_source.resolve(source, **store_options))
    summary = collection.summary()
    live = collection.list_datasets()
    return {
        "source": _source.describe(source),
        "format_version": summary["format_version"],
        "created_unix_ms": summary["created_unix_ms"],
        "codec": summary["codec"],
        "container_bytes": summary["container_bytes"],
        "dataset_count": len(live),
        "deleted_count": summary["total_datasets"] - len(live),
        "total_datasets": summary["total_datasets"],
        "distinct_arrays": collection.list_arrays(),
        "interned_schemas": summary["interned_schemas"],
    }


# ── shared helpers ───────────────────────────────────────────────────


def _coords_of(view: Any) -> list[str]:
    """Which of a dataset's arrays were xarray coordinates, if atlas wrote it."""
    raw = view.get_attribute(_xarray._COORDS_ATTR)
    if raw is None:
        return []
    try:
        return [str(n) for n in json.loads(raw)]
    except (TypeError, ValueError):
        return []


def _global_attrs(view: Any) -> dict[str, Any]:
    """Dataset attributes, decoded, without atlas's own bookkeeping keys."""
    return {
        key: _xarray._decode_attr_value(value)
        for key, value in view.attributes().items()
        if key != _xarray._COORDS_ATTR
    }
