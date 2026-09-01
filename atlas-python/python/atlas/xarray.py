"""How an `xarray.Dataset` becomes an atlas dataset.

This module is internal. `atlas.create` is the public entry point. Here sit the
mappings for dtypes, fill values, and attribute encoding. Here too sits the
block-by-block stream that keeps memory flat for a dask-backed variable.

There is no read path here, and that is deliberate. A collection opened from
Python gives metadata only. The Rust API reads array data.
"""
from __future__ import annotations

import itertools
import json
import math
import time
import warnings
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Iterator, Optional, Sequence

import numpy as np

from ._atlas import log_chunk_event as _log_chunk_event
from ._log import get_logger as _get_logger

_LOG = _get_logger("xarray")

if TYPE_CHECKING:
    import xarray as xr

    from ._atlas import AtlasWriter, DatasetWriter


_COORDS_ATTR = "_pyatlas_coords"
_JSON_PREFIX = "json:"

# Marker attribute for a variable that arrived as `timedelta64`. Atlas has no
# duration type, so a timedelta stores as int64 nanoseconds. That is the same
# int64 view datetime64 uses. The value of this attribute is the unit to
# restore on read. The read path removes it, so no user ever sees it.
_TIMEDELTA_ATTR = "_pyatlas_timedelta"

_NUMPY_TO_ATLAS = {
    np.dtype("int8"): "int8",
    np.dtype("int16"): "int16",
    np.dtype("int32"): "int32",
    np.dtype("int64"): "int64",
    np.dtype("uint8"): "uint8",
    np.dtype("uint16"): "uint16",
    np.dtype("uint32"): "uint32",
    np.dtype("uint64"): "uint64",
    np.dtype("float32"): "float32",
    np.dtype("float64"): "float64",
    np.dtype("datetime64[ns]"): "timestamp_nanoseconds",
}


def _np_to_atlas_dtype(np_dtype: np.dtype) -> str:
    if np_dtype in _NUMPY_TO_ATLAS:
        return _NUMPY_TO_ATLAS[np_dtype]
    # timedelta64 of any unit has no atlas type. It stores as int64
    # nanoseconds, with a `_TIMEDELTA_ATTR` tag. The read path restores the
    # duration dtype from that tag, as it does for datetime64.
    if np_dtype.kind == "m":
        return "int64"
    # Object (Python str or bytes) and fixed-size byte or unicode strings all
    # become variable-length atlas strings.
    if np_dtype.kind in ("O", "S", "U"):
        return "string"
    supported = ", ".join(sorted(set(_NUMPY_TO_ATLAS.values())))
    raise NotImplementedError(
        f"numpy dtype {np_dtype!r} is not supported by atlas "
        f"(supported: {supported}, plus object/bytes/unicode → string)"
    )


def _sanitize_str(s: str) -> str:
    """Removes a lone Unicode surrogate from a Python str.

    A NetCDF backend often gives a byte attribute as a Python str that it
    decoded with ``errors='surrogateescape'``. That str then holds a
    pseudo-codepoint in U+DC80..U+DCFF, which a Rust UTF-8 str cannot hold.

    This first recovers the original bytes through surrogateescape, and decodes
    them again as UTF-8. That covers the common case, where the bytes were
    valid UTF-8 and something upstream read them as Latin-1. A failure falls
    back to lossy replacement.
    """
    try:
        s.encode("utf-8")
        return s
    except UnicodeEncodeError:
        pass
    raw = s.encode("utf-8", errors="surrogateescape")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("utf-8", errors="replace")


def _encode_attr_value(value: Any) -> Any:
    """Converts an xarray attribute value into a value atlas can store.

    A bool, an int, a float, and a str pass through as they are. Everything
    else encodes as JSON behind a ``json:`` prefix, so a read decodes it back
    without loss. A numpy scalar unwraps first. A numpy array becomes a list. A
    value JSON cannot serialize raises ``TypeError``.
    """
    # Unwrap a numpy scalar.
    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, str):
        return _sanitize_str(value)
    if isinstance(value, (bool, int, float)):
        return value

    # Convert a numpy array to nested lists.
    if isinstance(value, np.ndarray):
        value = value.tolist()

    try:
        return _JSON_PREFIX + json.dumps(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"attribute value of type {type(value).__name__} is not JSON-serialisable "
            f"and cannot be stored: {value!r}"
        ) from exc


def _decode_attr_value(value: Any) -> Any:
    """The reverse of :func:`_encode_attr_value`."""
    if isinstance(value, str) and value.startswith(_JSON_PREFIX):
        return json.loads(value[len(_JSON_PREFIX):])
    return value


def _normalize_fill_value(value: Any, np_dtype: np.dtype) -> Any:
    """Converts an xarray `_FillValue` into a Python scalar of the array dtype.

    The `define_array(fill_value=...)` binding expects a plain Python scalar
    that matches the array dtype. xarray often holds `_FillValue` as a 0-D
    numpy array or a numpy scalar. This unwraps that. For a datetime64 array it
    reads the value as nanoseconds from the epoch. For a string array (object,
    bytes, or unicode) it decodes `bytes` to `str`, because NetCDF stores a
    fixed-width string fill value as bytes.
    """
    if value is None:
        return None
    if isinstance(value, np.ndarray) and value.ndim == 0:
        if np_dtype.kind == "M":
            value = value.view(np.int64).item()
        elif np_dtype.kind == "m":  # timedelta64 -> int64 nanoseconds
            value = value.astype("timedelta64[ns]").view(np.int64).item()
        else:
            value = value.item()
    elif isinstance(value, np.generic):
        if isinstance(value, np.datetime64):
            return value.astype("datetime64[ns]").view(np.int64).item()
        if isinstance(value, np.timedelta64):
            return value.astype("timedelta64[ns]").view(np.int64).item()
        value = value.item()
    if np_dtype.kind in ("O", "S", "U") and isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


# A sentinel that separates two cases. The fill_value dict maps this variable
# to None, which turns the default off. Or the dict omits the variable, which
# leaves the default in place.
_UNSET = object()

# The int64 bit pattern of `datetime64[ns]` NaT, which equals i64::MIN. It is
# the default fill value for a timestamp array. NaT is the masked-datetime
# sentinel of xarray, so this records it as null, as NaN does for a float.
_NAT_INT64 = int(np.datetime64("NaT", "ns").view("int64"))


def _resolve_fill_value(
    var_name: str,
    np_dtype: np.dtype,
    attr_fill: Any,
    fill_value_arg: Any,
) -> Any:
    """Decides the atlas fill value for one variable.

    The order runs from highest to lowest:

    1. The explicit ``fill_value`` argument. A ``{var: scalar}`` dict names the
       variables. An entry of ``None`` turns the default off for that
       variable. A bare scalar applies to a numeric array only, that is int,
       uint, or float.
    2. The CF ``_FillValue`` attribute of the variable.
    3. The default. ``NaN`` for a float array, ``NaT`` for a datetime array,
       and ``""`` for a string array. A masked cell then records as null.
       Every other dtype gets ``None``.
    """
    override = _UNSET
    if isinstance(fill_value_arg, dict):
        override = fill_value_arg.get(var_name, _UNSET)
    elif fill_value_arg is not None and np_dtype.kind in ("i", "u", "f"):
        override = fill_value_arg
    if override is not _UNSET:
        return None if override is None else _normalize_fill_value(override, np_dtype)

    if attr_fill is not None:
        return _normalize_fill_value(attr_fill, np_dtype)

    if np_dtype.kind == "f":
        return float("nan")
    if np_dtype.kind in ("M", "m"):  # datetime64 and timedelta64 use the NaT sentinel
        return _NAT_INT64
    if np_dtype.kind in ("O", "S", "U"):  # a string uses "" for a missing cell
        return ""
    return None


def _is_missing_str(x: Any) -> bool:
    """True when an object-array cell holds a missing string, `None` or `NaN`.

    A masked string variable gives a missing cell as `None`. After numpy
    conversion it can instead give a float `NaN` inside an object array.
    """
    return x is None or (isinstance(x, float) and math.isnan(x))


def _fill_missing_strings(block: np.ndarray, fill: str) -> tuple[np.ndarray, int]:
    """Replaces every `None` or `NaN` cell of an object-dtype `block` with `fill`.

    Atlas cannot store a missing string as null, because the `.af` format has
    no string null sentinel. A masked string cell therefore takes a real
    string. Returns ``(block, n_filled)``. A block that is not object dtype
    (`\\|S` or `\\|U`) holds no `None` and no `NaN`, so it returns unchanged
    with ``n_filled == 0``.
    """
    if block.dtype.kind != "O":
        return block, 0
    flat = block.reshape(-1)
    mask = np.fromiter(
        (_is_missing_str(x) for x in flat), dtype=bool, count=flat.size
    ).reshape(block.shape)
    n = int(mask.sum())
    if not n:
        return block, 0
    return np.where(mask, fill, block), n


# How far into an object array to look for a real value. Enough to tell a
# string array from an array of something else, and cheap on a dask block.
_OBJECT_SAMPLE = 64


def _sample_object_element(var: Any) -> Any:
    """The first value of an object array that is no missing marker.

    This scans a bounded prefix, and computes one dask block at most.
    """
    data = var.data
    if _is_dask_array(data):
        import dask

        (head,) = dask.compute(data.reshape(-1)[:_OBJECT_SAMPLE])
    else:
        head = np.asarray(data).reshape(-1)[:_OBJECT_SAMPLE]
    for value in head:
        if not _is_missing_str(value):
            return value
    return None


def _reject_unstorable_object_array(var_name: str, var: Any) -> None:
    """Raises when an object array holds something atlas cannot store.

    Atlas stores an object array as a string array. numpy reports only
    `object` for the dtype, so the element type settles what the array really
    holds. A missing marker and an empty array both pass, because the string
    path handles them.
    """
    sample = _sample_object_element(var)
    if sample is None or isinstance(sample, (str, bytes)):
        return

    kind = type(sample)
    if kind.__module__.split(".")[0] == "cftime":
        raise NotImplementedError(
            f"variable {var_name!r} holds cftime objects ({kind.__name__}), "
            f"which atlas cannot store. xarray decodes a calendar it cannot "
            f"map to datetime64[ns], such as a Julian one, into cftime. Pass "
            f"decode_times=False, or --no-decode-times, to keep the raw "
            f"numbers and their units. Or convert the calendar first with "
            f"ds.convert_calendar('standard')"
        )
    raise NotImplementedError(
        f"variable {var_name!r} is an object array of {kind.__name__}. Atlas "
        f"stores an object array as a string, so every element must be a str "
        f"or bytes"
    )


def _is_dask_array(arr: Any) -> bool:
    """True when `arr` is a `dask.array.Array`. False when dask is absent."""
    try:
        import dask.array as da
    except ImportError:
        return False
    return isinstance(arr, da.Array)


def _dask_chunk_shape(arr: Any) -> list[int]:
    """The first chunk size on each axis of a dask array. This is the atlas
    chunk_shape."""
    return [c[0] for c in arr.chunks]


def _contiguous(a: np.ndarray) -> np.ndarray:
    # np.ascontiguousarray promotes a 0-D array to 1-D, because ndmin defaults
    # to 1. That breaks a scalar-array write. Copy a non-contiguous array with
    # np.asarray and .copy(order='C') instead, which keeps the rank.
    if a.flags["C_CONTIGUOUS"]:
        return a
    return a.copy(order="C")


# A batch serves the many-small-NetCDF-chunks case. It turns N dask scheduler
# calls into N/batch_size. A batch sized by *count* would ruin the large-block
# case. Eight 128 MiB blocks per batch, with two batches in flight, holds 2 GiB
# in memory. The batch is therefore sized by bytes, and the count is a ceiling.
_MAX_BATCH_SIZE = 8
_DEFAULT_PREFETCH_DEPTH = 2

# Bytes a batch aims at. Peak memory is about
# `prefetch_depth * max(_BATCH_BYTE_BUDGET, one block)`. A variable chunked at
# 128 MiB therefore holds two blocks in flight, not sixteen.
_BATCH_BYTE_BUDGET = 64 * 1024 * 1024


def _block_nbytes(arr: Any) -> int:
    """Bytes in the largest block of a dask array.

    An object array of strings reports the pointer size, which is below the
    real payload. Such an array is small in practice, and the floor below keeps
    the batch reasonable.
    """
    largest = 1
    for sizes in arr.chunks:
        largest *= max(sizes) if sizes else 1
    return max(1, largest * max(1, arr.dtype.itemsize))


def _batch_size_for(arr: Any) -> int:
    """How many blocks one scheduler call computes.

    The count covers the dask overhead on a small block. It never holds too
    much of a large variable in memory.
    """
    return max(1, min(_MAX_BATCH_SIZE, _BATCH_BYTE_BUDGET // _block_nbytes(arr)))


def _iter_blocks(
    arr: Any,
    var_name: str = "",
    batch_size: Optional[int] = None,
    prefetch_depth: int = _DEFAULT_PREFETCH_DEPTH,
) -> Iterator[tuple[list[int], np.ndarray]]:
    """Yields ``(start_index, block_np)`` for each chunk of `arr`.

    A numpy-backed input gives one full-shape block. A dask-backed input
    prefetches. One background thread runs ``dask.compute(*K)`` on the next
    batch, while the main thread consumes the blocks already in memory. NetCDF
    reads therefore overlap atlas writes.

    Bytes size the batch. See :data:`_BATCH_BYTE_BUDGET`. Peak memory then
    tracks the block size, not the block count. Pass ``batch_size`` to override
    it.

    Each batch that completes emits one ``event=dask_compute`` debug event
    through the Rust tracing subscriber. See ``log_chunk_event``.
    """
    if not _is_dask_array(arr):
        a = np.asarray(arr)
        yield [0] * a.ndim, _contiguous(a)
        return

    import dask

    if batch_size is None:
        batch_size = _batch_size_for(arr)

    chunks = arr.chunks
    offsets = [[0, *itertools.accumulate(c)][:-1] for c in chunks]
    pairs: list[tuple[list[int], Any]] = []
    for block_idx in itertools.product(*[range(len(c)) for c in chunks]):
        start = [offsets[d][i] for d, i in enumerate(block_idx)]
        pairs.append((start, arr.blocks[block_idx]))

    def _fetch(batch: list[tuple[list[int], Any]]) -> list[tuple[list[int], np.ndarray]]:
        starts = [s for s, _ in batch]
        delayed = [b for _, b in batch]
        t0 = time.perf_counter_ns()
        materialised = dask.compute(*delayed)
        elapsed_us = (time.perf_counter_ns() - t0) // 1000
        _log_chunk_event("dask_compute", var_name, elapsed_us, chunks=len(batch))
        return [(s, _contiguous(b)) for s, b in zip(starts, materialised)]

    with ThreadPoolExecutor(max_workers=1) as executor:
        in_flight: list[Future] = []
        cursor = 0
        while cursor < len(pairs) and len(in_flight) < prefetch_depth:
            in_flight.append(executor.submit(_fetch, pairs[cursor : cursor + batch_size]))
            cursor += batch_size
        while in_flight:
            future = in_flight.pop(0)
            for start, block in future.result():
                yield start, block
            if cursor < len(pairs):
                in_flight.append(
                    executor.submit(_fetch, pairs[cursor : cursor + batch_size])
                )
                cursor += batch_size


def _write_xarray_to_view(
    writer: "DatasetWriter",
    ds: "xr.Dataset",
    chunks: Optional[dict[str, Sequence[int]]] = None,
    fill_value: Any = None,
    on_unsupported: str = "stop",
) -> list[dict[str, str]]:
    """Fills an empty `DatasetWriter` from an xarray Dataset.

    Every coordinate and data variable becomes an atlas array. The coordinate
    names become ``_pyatlas_coords``, a JSON list. Every dataset attribute
    becomes a dataset-level attribute. The attributes of a variable become
    attributes of that variable's array.

    Under `on_unsupported="skip"`, a variable of a dtype atlas cannot store
    leaves the dataset instead of failing it. Returns one record per variable
    it left out.
    """
    if on_unsupported not in ("stop", "skip"):
        raise ValueError(
            f"on_unsupported must be 'stop' or 'skip', got {on_unsupported!r}"
        )

    coord_names = [str(n) for n in ds.coords.keys()]
    order = coord_names + [str(n) for n in ds.data_vars.keys()]

    # Resolve every dtype before the first `define_array`. An array that is
    # already defined cannot leave the schema, so a mid-write skip would store
    # a half-written array. This settles the whole dataset up front.
    dtypes: dict[str, str] = {}
    skipped: list[dict[str, str]] = []
    for var_name in order:
        try:
            np_dtype = np.dtype(ds[var_name].dtype)
            atlas_dtype = _np_to_atlas_dtype(np_dtype)
            # An object array maps to string on the dtype alone. Check what it
            # really holds before any array is defined, so a skip leaves no
            # half-written array behind.
            if np_dtype.kind == "O":
                _reject_unstorable_object_array(var_name, ds[var_name])
            dtypes[var_name] = atlas_dtype
        except NotImplementedError as exc:
            if on_unsupported == "stop":
                raise
            # The caller logs this. It knows the file the dataset came from.
            skipped.append(
                {
                    "array": var_name,
                    "dtype": str(ds[var_name].dtype),
                    "error": str(exc),
                }
            )

    order = [n for n in order if n in dtypes]
    coord_names = [n for n in coord_names if n in dtypes]

    # Write the coords first, then the data_vars. Atlas ignores the order. The
    # order makes the on-disk layout predictable.
    for var_name in order:
        var = ds[var_name]
        np_dtype = np.dtype(var.dtype)
        atlas_dtype = dtypes[var_name]
        dims = [str(d) for d in var.dims]
        shape = [int(s) for s in var.shape]

        # Pick the atlas chunk_shape, in this order:
        #   1. the explicit `chunks=` argument, else
        #   2. the dask chunk shape, when the variable is dask-backed, else
        #   3. None, so atlas stores one full-shape chunk.
        if chunks is not None and var_name in chunks:
            chunk_shape: Optional[list[int]] = [int(s) for s in chunks[var_name]]
        elif _is_dask_array(var.data):
            chunk_shape = _dask_chunk_shape(var.data)
        else:
            chunk_shape = None

        # Resolve the typed fill value. Remove any CF `_FillValue` attribute,
        # so it reaches define_array and does not store as a flat atlas
        # attribute. `_resolve_fill_value` then applies the explicit
        # `fill_value` override and the NaN-for-floats default. Copy the attrs
        # first, so the user's Dataset does not change.
        var_attrs = dict(var.attrs)
        resolved_fill = _resolve_fill_value(
            var_name,
            np_dtype,
            var_attrs.pop("_FillValue", None),
            fill_value,
        )

        writer.define_array(
            var_name,
            dtype=atlas_dtype,
            dims=dims,
            shape=shape,
            chunk_shape=chunk_shape,
            fill_value=resolved_fill,
        )

        # Tag a timedelta64 array, which stores as int64 ns. The read path
        # then restores the duration dtype. The values convert to ns below, so
        # the recorded unit is always "ns".
        if np_dtype.kind == "m":
            writer.set_array_attribute(var_name, _TIMEDELTA_ATTR, "ns")

        # A string array cannot store a missing cell as null. Every `None` and
        # `NaN` therefore takes the resolved string fill, which defaults to "".
        str_fill = resolved_fill if isinstance(resolved_fill, str) else ""
        n_filled_strings = 0

        # Stream the blocks. Dask-backed data arrives in prefetched batches.
        # Numpy-backed data arrives as one full-shape block.
        for start, block in _iter_blocks(var.data, var_name=var_name):
            # A TimestampNs column reaches the bindings as np.int64 only. View
            # the numpy datetime64 as int64, with no copy.
            if block.dtype.kind == "M":
                block = block.view(np.int64)
            # timedelta64 becomes int64 nanoseconds. Convert the unit first,
            # then take a view with no copy. The marker restores the duration
            # on read.
            elif block.dtype.kind == "m":
                block = block.astype("timedelta64[ns]").view(np.int64)
            elif atlas_dtype == "string":
                block, n = _fill_missing_strings(block, str_fill)
                n_filled_strings += n
            t0 = time.perf_counter_ns()
            writer.write_array(var_name, start=start, data=block)
            _log_chunk_event(
                "write",
                var_name,
                (time.perf_counter_ns() - t0) // 1000,
                bytes=int(block.nbytes),
            )

        if n_filled_strings:
            message = (
                f"{var_name!r}: replaced {n_filled_strings} missing string "
                f"cell(s) (None/NaN) with {str_fill!r}. Atlas cannot store "
                f"a missing string as null"
            )
            _LOG.warning("%s", message)
            warnings.warn(message, stacklevel=2)

        # Each variable attribute becomes a real per-array attribute in the
        # collection footer. `_FillValue` is the exception. It became the fill.
        for attr_key, attr_val in var_attrs.items():
            encoded = _encode_attr_value(attr_val)
            writer.set_array_attribute(var_name, _sanitize_str(str(attr_key)), encoded)

    # Dataset-level attrs
    for attr_key, attr_val in ds.attrs.items():
        encoded = _encode_attr_value(attr_val)
        writer.set_attribute(_sanitize_str(str(attr_key)), encoded)

    # A marker. A read uses it to tell a coordinate from a data variable. A
    # skipped coordinate is not in the list, because it is not in the dataset.
    writer.set_attribute(_COORDS_ATTR, json.dumps(coord_names))
    return skipped


def _write_xarray_dataset(
    writer: "AtlasWriter",
    ds: "xr.Dataset",
    name: str,
    chunks: Optional[dict[str, Sequence[int]]] = None,
    fill_value: Any = None,
    on_unsupported: str = "stop",
) -> list[dict[str, str]]:
    """Writes an ``xarray.Dataset`` into an open writer, under ``name``.

    This is atomic. A dataset reaches the container only when it finishes. A
    failure part-way aborts the dataset, and the collection never sees it.

    Under `on_unsupported="skip"`, a variable of a dtype atlas cannot store
    leaves the dataset, and the rest of it still lands. Returns one record per
    variable it left out, each tagged with the dataset name.
    """
    dataset_writer = writer.add_dataset(name)
    try:
        skipped = _write_xarray_to_view(
            dataset_writer,
            ds,
            chunks=chunks,
            fill_value=fill_value,
            on_unsupported=on_unsupported,
        )
    except BaseException:
        dataset_writer.abort()
        raise
    dataset_writer.finish()
    for record in skipped:
        record["dataset"] = name
    return skipped
