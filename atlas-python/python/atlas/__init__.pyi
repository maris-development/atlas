"""Typed surface of the ``atlas`` package.

Five operations, and nothing else. A collection is one immutable file: you
build it from a directory of NetCDF files, and afterwards you can inspect it or
remove datasets from it. Changing a dataset means rebuilding the collection.

Array *data* is read through the Rust API. From Python you get the structure —
which datasets exist, what arrays they hold, their types, shapes, attributes,
and the statistics recorded when they were written.

The same operations are on the command line: ``atlas create``, ``atlas rm``,
``atlas ls``, ``atlas show``, ``atlas info``.
"""

import os
import pathlib
from typing import Any, Callable, Iterable, Optional, Sequence, Union

__version__: str

__all__ = [
    "create",
    "remove",
    "list_datasets",
    "describe",
    "info",
    "find_netcdf_files",
    "AtlasError",
    "SourceError",
    "init_tracing",
    "__version__",
]

# A local path, a URL (``s3://``, ``gs://``, ``az://``, ``http(s)://``), or an
# obstore store handle. URLs and handles need ``pip install
# "atlas-python[cloud]"``.
Source = Union[str, os.PathLike[str], Any]

# File suffixes `create` treats as NetCDF.
NETCDF_SUFFIXES: tuple[str, ...]

# Accepted string values for `create(open_chunks=...)`.
OPEN_CHUNK_MODES: tuple[str, ...]

# Default block size `open_chunks="auto"` aims for.
DEFAULT_CHUNK_SIZE: str

class AtlasError(RuntimeError):
    """An operation could not complete. The message says what went wrong."""

class SourceError(ValueError):
    """A source could not be resolved: bad URL, or obstore is not installed."""

def create(
    directory: Union[str, os.PathLike[str]],
    destination: Source,
    *,
    recursive: bool = False,
    codec: str = "zstd",
    chunks: Optional[dict[str, Sequence[int]]] = None,
    open_chunks: Union[str, dict[str, int], None] = "auto",
    chunk_size: str = "128MiB",
    on_error: str = "stop",
    progress: Optional[Callable[[str], None]] = None,
    **store_options: Any,
) -> dict[str, Any]:
    """Build a collection at ``destination`` from the NetCDF files in ``directory``.

    Each file becomes one dataset named after its stem, so ``jan_2024.nc``
    becomes ``jan_2024``. Files are written in sorted order, which fixes the
    ordinals a collection hands out.

    Nothing is readable at ``destination`` until every file has been written
    and the footer lands. A failure part-way leaves no collection at all,
    rather than a partial one.

    Source files are opened with dask chunking, so a file far larger than
    memory streams block by block instead of being read whole. Those blocks
    also become each array's stored chunk shape, unless ``chunks`` names one.

    Args:
        directory: Where the NetCDF files are.
        destination: Where to write the collection. A local path, or a URL for
            object storage.
        recursive: Descend into subdirectories. Off by default.
        codec: Block compression: ``"zstd"`` (default), ``"lz4"``, or
            ``"none"``. Blocks record their own codec, so readers need no
            argument.
        chunks: Per-variable **stored** chunk shape, ``{var: [d0, d1, ...]}``.
            Overrides whatever ``open_chunks`` produced. Note that source
            blocks then no longer align with stored chunks, so writes become
            read-modify-write — correct, but slower.
        open_chunks: How source files are read, which also sets the stored
            chunk shape unless ``chunks`` overrides it.

            - ``"auto"`` (default) — dask picks blocks sized to
              ``chunk_size``. A large file streams; a small one still lands as
              a single chunk.
            - ``"native"`` — use the file's own chunk encoding. No read
              amplification during ingest, but a netCDF4 file with tiny chunks
              gives tiny atlas chunks, and a netCDF3 file has no chunking to
              use.
            - ``None`` — read each variable whole. Only for files you know
              are small.
            - a dict — explicit, per dimension: ``{"time": 100, "lat": -1}``.
        chunk_size: Block size ``"auto"`` aims for, as a dask size string.
            Roughly the memory ceiling per variable during ingest. Defaults to
            ``"128MiB"``.
        on_error: ``"stop"`` (default) abandons the whole collection on the
            first bad file. ``"skip"`` records it and carries on.
        progress: Called with each dataset name as it is written.
        **store_options: Passed to the obstore constructor for remote
            destinations — ``region``, ``endpoint``, ``skip_signature``.

    Returns:
        ``{"destination", "written", "skipped", "dataset_count"}``. ``skipped``
        holds ``{"file", "error"}`` for each file that failed under
        ``on_error="skip"``.

    Raises:
        AtlasError: The directory holds no NetCDF files, two files share a
            stem, ``open_chunks`` is not a recognised mode, or a file failed
            under ``on_error="stop"``.
        SourceError: ``destination`` could not be resolved.
    """
    ...

def remove(
    source: Source,
    targets: Iterable[Union[str, os.PathLike[str]]],
    *,
    missing_ok: bool = False,
    **store_options: Any,
) -> dict[str, Any]:
    """Remove datasets from a collection, in one call.

    ``targets`` are dataset names or NetCDF paths — a path is reduced to its
    stem, so the list that built a collection can also tear part of it down.

    This writes the deletion mask beside the container. The container is not
    touched: no space is reclaimed, and no ordinal moves. Rebuild the
    collection to reclaim the bytes.

    Args:
        source: The collection.
        targets: Dataset names, or the NetCDF files they came from.
        missing_ok: Report names that are absent or already removed instead of
            raising.
        **store_options: As for :func:`create`.

    Returns:
        ``{"removed", "missing", "remaining"}``.

    Raises:
        AtlasError: ``targets`` was empty, or named something absent while
            ``missing_ok`` is false.
    """
    ...

def list_datasets(source: Source, **store_options: Any) -> list[str]:
    """Dataset names in the collection, in write order.

    Removed datasets are not listed. Costs one range read of the container
    tail, whatever the collection size.
    """
    ...

def describe(source: Source, name: Union[str, os.PathLike[str]], **store_options: Any) -> dict[str, Any]:
    """Everything the collection records about one dataset.

    ``name`` may be a dataset name or the NetCDF path it came from.

    Returns:
        ``{"name", "ordinal", "segment_range", "dimensions", "coordinates",
        "arrays", "attributes"}``. Each entry of ``arrays`` is ``{"name",
        "dtype", "shape", "chunk_shape", "dimensions", "fill_value",
        "is_coordinate", "attributes", "stats"}``, where ``stats`` is
        ``{"min", "max", "null_count", "row_count"}`` as recorded when the
        array was written — ``null_count`` counts elements equal to the fill
        value, and ``min``/``max`` are ``None`` for a dtype with no ordering
        and ``bytes`` for strings.

    Raises:
        AtlasError: No such dataset, or it has been removed.
    """
    ...

def info(source: Source, **store_options: Any) -> dict[str, Any]:
    """Summarise the whole collection.

    Returns:
        ``{"source", "format_version", "created_unix_ms", "codec",
        "container_bytes", "dataset_count", "deleted_count",
        "total_datasets", "distinct_arrays", "interned_schemas"}``.

        ``deleted_count`` is how many datasets the mask hides; their bytes are
        still in the container. ``interned_schemas`` is how many distinct
        schemas the datasets share between them — lower than
        ``total_datasets`` whenever datasets have the same arrays.
    """
    ...

def find_netcdf_files(
    directory: Union[str, os.PathLike[str]], recursive: bool = False
) -> list[pathlib.Path]:
    """NetCDF files in ``directory``, sorted — what :func:`create` would ingest.

    Useful for checking what a ``create`` call will pick up before running it.

    Raises:
        AtlasError: ``directory`` is not a directory.
    """
    ...

def init_tracing(filter: Optional[str] = ...) -> None:
    """Install a Rust ``tracing`` subscriber, writing to stderr.

    Set ``ATLAS_LOG`` or ``RUST_LOG`` to have this happen on import.
    """
    ...
