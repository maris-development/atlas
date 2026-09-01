"""Typed surface of the ``atlas`` package.

Five operations, and nothing else. A collection is one immutable file. Build it
from a directory of NetCDF files. Then inspect it, or remove datasets from it.
To change a dataset, rebuild the collection.

The Rust API reads array *data*. Python gives the structure. Which datasets
exist, what arrays they hold, their types, their shapes, their attributes, and
the statistics of the write.

The same operations are on the command line: ``atlas create``, ``atlas rm``,
``atlas ls``, ``atlas show``, ``atlas info``.
"""

import logging
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
    "log_to_file",
    "AtlasError",
    "SourceError",
    "init_tracing",
    "__version__",
]

# A local path, a URL (``s3://``, ``gs://``, ``az://``, ``http(s)://``), or an
# obstore store handle. A URL and a handle need ``pip install
# "atlas-python[cloud]"``.
Source = Union[str, os.PathLike[str], Any]

# File suffixes `create` treats as NetCDF.
NETCDF_SUFFIXES: tuple[str, ...]

# Accepted string values for `create(open_chunks=...)`.
OPEN_CHUNK_MODES: tuple[str, ...]

# Default block size `open_chunks="auto"` aims at.
DEFAULT_CHUNK_SIZE: str

class AtlasError(RuntimeError):
    """An operation did not complete. The message says what went wrong."""

class SourceError(ValueError):
    """A source did not resolve. The URL is bad, or obstore is absent."""

def create(
    directory: Union[str, os.PathLike[str]],
    destination: Source,
    *,
    recursive: bool = True,
    codec: str = "zstd",
    chunks: Optional[dict[str, Sequence[int]]] = None,
    open_chunks: Union[str, dict[str, int], None] = "auto",
    chunk_size: str = "128MiB",
    decode_times: bool = True,
    convert_calendar: bool = False,
    on_error: str = "stop",
    on_unsupported: str = "stop",
    progress: Optional[Callable[[str], None]] = None,
    **store_options: Any,
) -> dict[str, Any]:
    """Builds a collection at ``destination`` from the NetCDF files in ``directory``.

    The scan descends into every subdirectory. Pass ``recursive=False`` for the
    top directory alone.

    Each file becomes one dataset, named after the file. ``jan_2024.nc``
    becomes ``jan_2024.nc``, suffix and all. The suffix tells two files apart,
    so ``jan.nc`` and ``jan.nc4`` are two datasets. A name carries no
    directory, so two files of one name in two subdirectories collide. The
    files land in sorted order, which fixes the ordinals of the collection.

    Nothing at ``destination`` is readable until every file lands, with the
    footer. A failure part-way leaves no collection, not a partial one.

    Each source file opens with dask chunking. A file far larger than memory
    therefore streams block by block. Those blocks also become the stored chunk
    shape of each array, unless ``chunks`` names one.

    Args:
        directory: Where the NetCDF files are.
        destination: Where to write the collection. A local path, or a URL for
            object storage.
        recursive: Descend into the subdirectories. On by default. Set it
            false for the top directory alone.
        codec: Block compression. ``"zstd"`` is the default. ``"lz4"`` and
            ``"none"`` are the others. Each block records its own codec, so a
            reader needs no argument.
        chunks: The **stored** chunk shape per variable,
            ``{var: [d0, d1, ...]}``. It overrides what ``open_chunks``
            produced. The source blocks then no longer align with the stored
            chunks. Each write becomes a read-modify-write, which is correct
            but slower.
        open_chunks: How to read the source files. This also sets the stored
            chunk shape, unless ``chunks`` overrides it.

            - ``"auto"``, the default. dask sizes the blocks to
              ``chunk_size``. A large file streams. A small one still lands as
              one chunk.
            - ``"native"``. Use the chunk encoding of the file. Ingest reads
              no extra bytes. A netCDF4 file with tiny chunks gives tiny atlas
              chunks, and a netCDF3 file has no chunking to use.
            - ``None``. Read each variable whole. Only for a file you know is
              small.
            - A dict, explicit and per dimension: ``{"time": 100, "lat": -1}``.
        chunk_size: The block size ``"auto"`` aims at, as a dask size string.
            It is about the memory ceiling per variable during ingest. It
            defaults to ``"128MiB"``.
        decode_times: How xarray reads a time axis. Under the default, a
            calendar it cannot map to ``datetime64[ns]``, such as a Julian
            one, decodes to cftime objects, which atlas cannot store. Set it
            false to keep the raw numbers, with their ``units`` and
            ``calendar`` attributes.
        convert_calendar: Turn a cftime axis into exact Gregorian timestamps.
            Each one keeps its instant, so a Julian ``1973-02-25`` becomes the
            Gregorian ``1973-03-10`` that names the same moment. A calendar
            with no real instant, such as ``360_day``, and a date outside the
            nanosecond range both raise instead.
        on_error: ``"stop"`` is the default. It abandons the whole collection
            on the first bad file. ``"skip"`` records that file and continues.
        on_unsupported: What one array of an unsupported dtype costs.
            ``"stop"`` is the default, and fails the file, which ``on_error``
            then handles. ``"skip"`` leaves that array out, and the rest of
            the dataset still lands.
        progress: Takes each dataset name as that dataset lands.
        **store_options: These reach the obstore constructor for a remote
            destination. ``region``, ``endpoint``, and ``skip_signature``.

    Returns:
        ``{"destination", "written", "skipped", "skipped_arrays",
        "dataset_count"}``. ``skipped`` holds ``{"file", "error"}`` for each
        file that failed under ``on_error="skip"``. ``skipped_arrays`` holds
        ``{"dataset", "array", "dtype", "error"}`` for each array left out
        under ``on_unsupported="skip"``.

    Raises:
        AtlasError: The directory holds no NetCDF file. Or two files share a
            name. Or ``open_chunks`` names no known mode. Or a file failed
            under ``on_error="stop"``.
        SourceError: ``destination`` did not resolve.
    """
    ...

def remove(
    source: Source,
    targets: Iterable[Union[str, os.PathLike[str]]],
    *,
    missing_ok: bool = False,
    **store_options: Any,
) -> dict[str, Any]:
    """Removes datasets from a collection, in one call.

    Each entry of ``targets`` is a dataset name or a NetCDF path. A path
    reduces to its file name. The list that built a collection can therefore
    tear part of it down.

    This writes the deletion mask beside the container. The container does not
    change. This reclaims no space, and moves no ordinal. Rebuild the
    collection to reclaim the bytes.

    One mask write covers the whole call. Ten thousand names therefore cost
    what one name costs. A repeated name counts once.

    Args:
        source: The collection.
        targets: Dataset names, or the NetCDF files they came from.
        missing_ok: Report a name that is absent or already removed, instead of
            an error.
        **store_options: As for :func:`create`.

    Returns:
        ``{"removed", "missing", "remaining"}``. ``removed`` holds the names
        the mask gained, in the order ``targets`` gave them.

    Raises:
        AtlasError: ``targets`` was empty. Or it named something absent while
            ``missing_ok`` is false.
    """
    ...

def list_datasets(source: Source, **store_options: Any) -> list[str]:
    """Dataset names in the collection, in write order.

    A removed dataset does not appear. This costs one range read of the
    container tail, whatever the size of the collection.
    """
    ...

def describe(source: Source, name: Union[str, os.PathLike[str]], **store_options: Any) -> dict[str, Any]:
    """Everything the collection records about one dataset.

    ``name`` is a dataset name, or the NetCDF path the dataset came from.

    Returns:
        ``{"name", "ordinal", "segment_range", "dimensions", "coordinates",
        "arrays", "attributes"}``. Each entry of ``arrays`` is ``{"name",
        "dtype", "shape", "chunk_shape", "dimensions", "fill_value",
        "is_coordinate", "attributes", "stats"}``.

        ``stats`` is ``{"min", "max", "null_count", "row_count"}``, as the
        write recorded it. ``null_count`` counts the elements equal to the fill
        value. ``min`` and ``max`` are ``None`` for a dtype with no order, and
        ``bytes`` for a string.

    Raises:
        AtlasError: There is no such dataset, or somebody removed it.
    """
    ...

def info(source: Source, **store_options: Any) -> dict[str, Any]:
    """Summarizes the whole collection.

    Returns:
        ``{"source", "format_version", "created_unix_ms", "codec",
        "container_bytes", "dataset_count", "deleted_count",
        "total_datasets", "distinct_arrays", "array_stats",
        "interned_schemas"}``.

        ``deleted_count`` is how many datasets the mask hides. Their bytes
        stay in the container. ``interned_schemas`` is how many distinct
        schemas the datasets share between them. It falls below
        ``total_datasets`` when two datasets hold the same arrays.

        ``array_stats`` maps each name in ``distinct_arrays`` to
        ``{"min", "max", "null_count", "row_count"}`` for the whole
        collection. The counts add up over every live dataset that holds the
        array. The minimum is the smallest of the minimums. The maximum is the
        largest of the maximums. The value is ``None`` when no live dataset
        records statistics for that array. Use :func:`describe` for the
        statistics of one dataset on its own.
    """
    ...

def find_netcdf_files(
    directory: Union[str, os.PathLike[str]], recursive: bool = True
) -> list[pathlib.Path]:
    """NetCDF files in ``directory``, sorted. :func:`create` ingests these.

    The walk descends into every subdirectory, as :func:`create` does.

    Call it to see what a ``create`` call picks up, before you run that call.

    Raises:
        AtlasError: ``directory`` is no directory.
    """
    ...

def init_tracing(filter: Optional[str] = ...) -> None:
    """Installs a Rust ``tracing`` subscriber that writes to stderr.

    Set ``ATLAS_LOG`` or ``RUST_LOG``, and this runs on import.
    """
    ...

def log_to_file(
    path: Union[str, os.PathLike[str]],
    *,
    level: int = ...,
    capture_warnings: bool = True,
) -> logging.Handler:
    """Writes the atlas log to ``path``. Returns the handler it attached.

    The file gets every record at ``level`` and above. That covers each file an
    ingest skips, each array it skips, and every error an operation reports.
    It opens in append mode, so a second run adds to it.

    With ``capture_warnings``, a Python warning goes to the same file. That
    also moves it off stderr, because ``logging.captureWarnings`` is
    process-wide.

    Atlas logs to the ``atlas`` logger and adds no handler of its own, so
    attach your own handler instead when you want a different destination.
    The Rust core logs through ``tracing``, which :func:`init_tracing` sends to
    stderr.

    The ``atlas`` command takes ``--log-file PATH`` for the same thing.
    """
    ...
