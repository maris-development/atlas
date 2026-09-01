"""The five operations atlas offers from Python.

Every one behaves the same against a local directory and an object store. See
`_source.resolve`.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any, Iterable, Optional, Sequence

from . import _atlas
from . import _source
from . import xarray as _xarray
from ._log import describe_exception, get_logger

_LOG = get_logger("ops")

# Suffixes `create` treats as NetCDF. It scans a directory for these.
NETCDF_SUFFIXES = (".nc", ".nc4", ".cdf", ".netcdf")

# How `create` opens a NetCDF file. This sets two things at once. How much of a
# file stays in memory, and the stored chunk shape. `chunks=` overrides the
# second.
#
#   "auto"    dask sizes blocks to `chunk_size`. The default. A file far larger
#             than memory streams. A small one still lands as one chunk.
#   "native"  use the chunk encoding of the file. Ingest reads no extra bytes.
#             A netCDF4 file with tiny chunks gives tiny atlas chunks, and a
#             netCDF3 file has no chunking to use.
#   None      read each variable whole. Only for a file you know is small.
#   a dict    explicit, per dimension: {"time": 100, "lat": -1}
OPEN_CHUNK_MODES = ("auto", "native")

# Block size dask aims at under "auto". It also caps how much of one variable
# stays in memory during a write.
DEFAULT_CHUNK_SIZE = "128MiB"


class AtlasError(RuntimeError):
    """An operation did not complete. The message says what went wrong."""


def dataset_name(path: "pathlib.Path | str") -> str:
    """The dataset name of a NetCDF file. That is its file name, with suffix.

    `/data/2024/jan.nc` becomes `jan.nc`. This also takes a bare name, so a
    path and a name give the same answer.

    The suffix stays because it tells two files apart. `jan.nc` and `jan.nc4`
    are two datasets, not one duplicate.
    """
    name = pathlib.PurePath(str(path)).name
    return name or str(path)


def find_netcdf_files(
    directory: "pathlib.Path | str", recursive: bool = True
) -> list[pathlib.Path]:
    """NetCDF files in `directory`, sorted. The sort fixes a collection's order.

    The walk descends into every subdirectory. Pass `recursive=False` for the
    top directory alone.
    """
    root = pathlib.Path(directory)
    if not root.is_dir():
        raise AtlasError(f"not a directory: {root}")
    walk = root.rglob("*") if recursive else root.glob("*")
    return sorted(p for p in walk if p.is_file() and p.suffix.lower() in NETCDF_SUFFIXES)


def _open_kwargs(open_chunks: Any) -> dict[str, Any]:
    """Translates `open_chunks` into `xr.open_dataset` keyword arguments."""
    if open_chunks is None:
        # No `chunks=` at all. xarray then reads each variable whole.
        return {}
    if isinstance(open_chunks, str):
        if open_chunks == "native":
            # xarray spells "the chunking of the file" as an empty dict.
            return {"chunks": {}}
        if open_chunks == "auto":
            return {"chunks": "auto"}
        raise AtlasError(
            f"open_chunks must be one of {OPEN_CHUNK_MODES}, a dict, or None; "
            f"got {open_chunks!r}"
        )
    if isinstance(open_chunks, dict):
        return {"chunks": open_chunks}
    raise AtlasError(
        f"open_chunks must be one of {OPEN_CHUNK_MODES}, a dict, or None; "
        f"got {type(open_chunks).__name__}"
    )


# ── create ───────────────────────────────────────────────────────────


def create(
    directory: "pathlib.Path | str",
    destination: Any,
    *,
    recursive: bool = True,
    codec: str = "zstd",
    chunks: Optional[dict[str, Sequence[int]]] = None,
    open_chunks: Any = "auto",
    chunk_size: str = DEFAULT_CHUNK_SIZE,
    decode_times: bool = True,
    convert_calendar: bool = False,
    on_error: str = "stop",
    on_unsupported: str = "stop",
    progress: Optional[Any] = None,
    **store_options: Any,
) -> dict[str, Any]:
    """Builds a collection at `destination` from the NetCDF files in `directory`.

    The scan descends into every subdirectory. Pass `recursive=False` for the
    top directory alone.

    Each file becomes one dataset, named after the file. `2024-01.nc` becomes
    `2024-01.nc`, suffix and all. A name carries no directory, so two files of
    one name in two subdirectories collide. `on_error="skip"` keeps the first
    and reports the second.

    Nothing at `destination` is readable until every file lands, with the
    footer. A failure therefore leaves no half-built collection.

    Each file opens with dask chunking, under `open_chunks="auto"` by default.
    A file far larger than memory then streams block by block. Those blocks
    also become the stored chunk shape of the arrays, unless `chunks` names
    one.

    Two settings decide what a failure costs. Both default to `"stop"`.

    `on_error` covers a whole file. `"stop"` abandons the collection. `"skip"`
    records the failure and continues with the next file.

    `on_unsupported` covers one array. `"stop"` fails the file, which
    `on_error` then handles. `"skip"` leaves that array out, and the rest of
    the dataset still lands.

    `decode_times` controls how xarray reads a time axis. Under the default,
    a calendar it cannot map to `datetime64[ns]`, such as a Julian one,
    decodes to cftime objects, which atlas cannot store. Set it false to keep
    the raw numbers and their `units` and `calendar` attributes instead.

    `convert_calendar` turns those cftime objects into exact Gregorian
    timestamps. Each one keeps its instant, so a Julian 1973-02-25 becomes the
    Gregorian 1973-03-10 that names the same moment. A calendar with no real
    instant, such as `360_day`, and a date outside the nanosecond range both
    raise instead.

    `progress` takes each file name as that file lands.

    Returns a summary. How many datasets landed, which files the run skipped,
    and which arrays it left out.
    """
    if on_error not in ("stop", "skip"):
        raise AtlasError(f"on_error must be 'stop' or 'skip', got {on_error!r}")
    if on_unsupported not in ("stop", "skip"):
        raise AtlasError(
            f"on_unsupported must be 'stop' or 'skip', got {on_unsupported!r}"
        )

    open_kwargs = _open_kwargs(open_chunks)
    if not decode_times:
        open_kwargs["decode_times"] = False

    import dask
    import xarray as xr

    files = find_netcdf_files(directory, recursive=recursive)
    if not files:
        raise AtlasError(f"no NetCDF files under {directory}")

    names: set[str] = set()
    written: list[str] = []
    skipped: list[dict[str, str]] = []
    skipped_arrays: list[dict[str, str]] = []
    _LOG.info("ingesting %d file(s) into %s", len(files), _source.describe(destination))

    target = _source.resolve(destination, **store_options)
    # "auto" sizes its blocks against `array.chunk-size`. That value therefore
    # also caps how much of one variable stays in memory during a write.
    with dask.config.set({"array.chunk-size": chunk_size}):
        with _atlas.AtlasWriter.create(target, codec) as writer:
            for path in files:
                name = dataset_name(path)
                if name in names:
                    message = f"duplicate dataset name {name!r} from {path}"
                    if on_error == "stop":
                        _LOG.error("%s", message)
                        raise AtlasError(message)
                    _LOG.warning("skipping %s: %s", path, message)
                    skipped.append({"file": str(path), "error": message})
                    continue

                try:
                    # The write runs inside the `with`, so every block lands
                    # before the file closes.
                    with xr.open_dataset(path, **open_kwargs) as ds:
                        left_out = _xarray._write_xarray_dataset(
                            writer,
                            ds,
                            name,
                            chunks,
                            None,
                            on_unsupported,
                            convert_calendar,
                        )
                    for item in left_out:
                        _LOG.warning(
                            "%s: skipped array %r of dtype %s: %s",
                            path,
                            item["array"],
                            item["dtype"],
                            item["error"],
                        )
                    skipped_arrays.extend(left_out)
                except Exception as exc:
                    if on_error == "stop":
                        _LOG.error("%s: %s", path, describe_exception(exc))
                        raise AtlasError(f"{path}: {exc}") from exc
                    _LOG.warning(
                        "skipping %s: %s", path, describe_exception(exc)
                    )
                    skipped.append({"file": str(path), "error": str(exc)})
                    continue

                names.add(name)
                written.append(name)
                if progress is not None:
                    progress(name)

    _LOG.info(
        "wrote %d dataset(s); skipped %d file(s) and %d array(s)",
        len(written),
        len(skipped),
        len(skipped_arrays),
    )
    return {
        "destination": _source.describe(destination),
        "written": written,
        "skipped": skipped,
        "skipped_arrays": skipped_arrays,
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
    """Removes datasets from a collection, in one call.

    Each entry of `targets` is a dataset name or a NetCDF file path. A path
    reduces to its file name. The list that built the collection can therefore
    tear part of it down.

    This updates the deletion mask beside the container. The container does not
    change, so this reclaims no space and moves no ordinal. Rewrite the
    collection to reclaim the bytes.

    One mask write covers the whole call. Ten thousand names therefore cost
    what one name costs. A repeated name counts once.

    Under `missing_ok`, a name that is absent or already removed appears in the
    result instead of an error.
    """
    # dict.fromkeys drops a repeat and keeps the order the caller gave.
    wanted = list(dict.fromkeys(dataset_name(t) for t in targets))
    if not wanted:
        raise AtlasError("no datasets given")

    collection = _atlas.Atlas.open(_source.resolve(source, **store_options))
    present = set(collection.list_datasets())

    missing = [n for n in wanted if n not in present]
    if missing and not missing_ok:
        raise AtlasError(
            f"not in the collection (or already removed): {', '.join(sorted(missing))}"
        )

    removed = [n for n in wanted if n in present]
    if removed:
        collection.delete_datasets(removed)

    return {
        "removed": removed,
        "missing": missing,
        "remaining": collection.dataset_count(),
    }


# ── list ─────────────────────────────────────────────────────────────


def list_datasets(source: Any, **store_options: Any) -> list[str]:
    """Names of the datasets in the collection. The deletion mask applies."""
    collection = _atlas.Atlas.open(_source.resolve(source, **store_options))
    return collection.list_datasets()


# ── show ─────────────────────────────────────────────────────────────


def describe_dataset(source: Any, name: Any, **store_options: Any) -> dict[str, Any]:
    """Everything the collection records about one dataset.

    `name` is a dataset name, or the NetCDF path the dataset came from.

    Returns the dimensions. For every array it returns the type, the shape, the
    chunking, the fill value, the attributes, and the statistics of the write.
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
    """A summary of the whole collection.

    `array_stats` combines each array over every live dataset that holds it.
    Use `describe_dataset` for the statistics of one dataset on its own.
    """
    collection = _atlas.Atlas.open(_source.resolve(source, **store_options))
    summary = collection.summary()
    live = collection.list_datasets()
    arrays = collection.list_arrays()
    return {
        "source": _source.describe(source),
        "format_version": summary["format_version"],
        "created_unix_ms": summary["created_unix_ms"],
        "codec": summary["codec"],
        "container_bytes": summary["container_bytes"],
        "dataset_count": len(live),
        "deleted_count": summary["total_datasets"] - len(live),
        "total_datasets": summary["total_datasets"],
        "distinct_arrays": arrays,
        "array_stats": {name: collection.array_stats(name) for name in arrays},
        "interned_schemas": summary["interned_schemas"],
    }


# ── shared helpers ───────────────────────────────────────────────────


def _coords_of(view: Any) -> list[str]:
    """Which arrays of a dataset were xarray coordinates. Atlas records this."""
    raw = view.get_attribute(_xarray._COORDS_ATTR)
    if raw is None:
        return []
    try:
        return [str(n) for n in json.loads(raw)]
    except (TypeError, ValueError):
        return []


def _global_attrs(view: Any) -> dict[str, Any]:
    """Dataset attributes, decoded, without the internal keys of atlas."""
    return {
        key: _xarray._decode_attr_value(value)
        for key, value in view.attributes().items()
        if key != _xarray._COORDS_ATTR
    }
