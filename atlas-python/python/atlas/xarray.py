"""xarray integration for atlas: the write path.

`xarray` and `dask` are required dependencies.

This module turns `xarray.Dataset` objects into atlas datasets. It has no read
path, and that is deliberate: a collection opened from Python is metadata only.
Use the Rust API to read array data back.

Dask-backed variables stream block by block, so a dataset far larger than memory
can be written. The on-disk chunk grid follows the dask chunking unless
`chunks=` says otherwise.

Bulk ingestion, one collection from many files:

    with atlas.AtlasWriter.create(path) as w:
        for nc_path in nc_paths:
            ds = xr.open_dataset(nc_path)
            w.add_xarray_dataset(ds, name=Path(nc_path).stem)
    # w.finish() runs on __exit__; nothing is readable before it.
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

if TYPE_CHECKING:
    import xarray as xr

    from ._atlas import AtlasWriter, DatasetWriter


_COORDS_ATTR = "_pyatlas_coords"
_JSON_PREFIX = "json:"

# Per-array marker attribute for variables that were `timedelta64` on the way in.
# Atlas has no native duration type, so a timedelta is stored as int64
# nanoseconds (the same int64 view datetime64 uses) and tagged with this
# attribute; its value is the unit to reconstruct on read. The read path pops
# it so it never surfaces as a user-visible attribute.
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
    # timedelta64 (any unit) has no native atlas type; it's stored as int64
    # nanoseconds and tagged with `_TIMEDELTA_ATTR` so the read path restores
    # the duration dtype — the datetime64 parallel.
    if np_dtype.kind == "m":
        return "int64"
    # Object (Python str/bytes) and fixed-size byte/unicode strings all
    # become variable-length atlas strings.
    if np_dtype.kind in ("O", "S", "U"):
        return "string"
    supported = ", ".join(sorted(set(_NUMPY_TO_ATLAS.values())))
    raise NotImplementedError(
        f"numpy dtype {np_dtype!r} is not supported by atlas "
        f"(supported: {supported}, plus object/bytes/unicode → string)"
    )


def _sanitize_str(s: str) -> str:
    """Strip lone Unicode surrogates from a Python str.

    NetCDF backends often surface byte attrs as Python strs that were decoded
    with ``errors='surrogateescape'``; the resulting strs hold pseudo-codepoints
    in U+DC80..U+DCFF which Rust's UTF-8 strs can't represent. We try to recover
    the original bytes via surrogateescape and re-decode as UTF-8 (the common
    case: bytes that *were* valid UTF-8 all along but were treated as Latin-1
    upstream); if that fails we fall back to lossy replacement.
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
    """Coerce an xarray attr value to something atlas can store.

    Primitives (bool/int/float/str) are returned as-is. Everything else is
    JSON-encoded and prefixed with ``json:`` so it can be decoded losslessly
    on read. Numpy scalars are unwrapped first. Numpy arrays are converted to
    lists. Values that can't be JSON-serialised raise ``TypeError``.
    """
    # Unwrap numpy scalars
    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, str):
        return _sanitize_str(value)
    if isinstance(value, (bool, int, float)):
        return value

    # Convert numpy arrays to nested lists
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
    """Inverse of :func:`_encode_attr_value`."""
    if isinstance(value, str) and value.startswith(_JSON_PREFIX):
        return json.loads(value[len(_JSON_PREFIX):])
    return value


def _normalize_fill_value(value: Any, np_dtype: np.dtype) -> Any:
    """Coerce an xarray `_FillValue` attribute into a Python scalar matching the array dtype.

    The binding's `define_array(fill_value=...)` expects a plain Python scalar
    that is type-consistent with the array dtype. xarray often stores `_FillValue`
    as a 0-D numpy array or numpy scalar; this unwraps that and (for datetime64
    arrays) reinterprets the value as nanoseconds since the epoch. For
    string-kind arrays (object/bytes/unicode), `bytes` are decoded to `str`
    since NetCDF stores fixed-width string fill values as bytes.
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


# Sentinel distinguishing "the fill_value dict maps this var to None" (explicitly
# disable the default) from "this var isn't in the dict" (fall through to default).
_UNSET = object()

# The int64 bit pattern of `datetime64[ns]` NaT (== i64::MIN). Used as the default
# fill value for timestamp arrays so that NaT (xarray's masked-datetime sentinel)
# is recorded as null — the datetime parallel to NaN for floats.
_NAT_INT64 = int(np.datetime64("NaT", "ns").view("int64"))


def _resolve_fill_value(
    var_name: str,
    np_dtype: np.dtype,
    attr_fill: Any,
    fill_value_arg: Any,
) -> Any:
    """Decide the atlas fill value for one variable.

    Precedence (highest first):
      1. The explicit ``fill_value`` kwarg. A ``{var: scalar}`` dict targets named
         vars (an entry of ``None`` *disables* the default for that var); a bare
         scalar applies to numeric (int/uint/float) arrays only.
      2. The variable's CF ``_FillValue`` attribute.
      3. Default: ``NaN`` for float arrays, ``NaT`` for datetime arrays, and ``""``
         for string arrays (so mask_and_scale'd / masked missing cells are recorded
         as null), else ``None``.
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
    if np_dtype.kind in ("M", "m"):  # datetime64/timedelta64 -> NaT (i64::MIN sentinel)
        return _NAT_INT64
    if np_dtype.kind in ("O", "S", "U"):  # string -> "" sentinel for missing cells
        return ""
    return None


def _is_missing_str(x: Any) -> bool:
    """True if an object-array cell represents a missing string (None or NaN).

    Masked string variables surface missing cells as `None` or (after numpy
    coercion) a float `NaN` inside an object array.
    """
    return x is None or (isinstance(x, float) and math.isnan(x))


def _fill_missing_strings(block: np.ndarray, fill: str) -> tuple[np.ndarray, int]:
    """Replace None/NaN cells in an object-dtype string `block` with `fill`.

    Atlas can't store a missing string as null (the `.af` format has no string
    null sentinel), so masked string cells are substituted with a real string.
    Returns ``(block, n_filled)``; non-object blocks (`\\|S` / `\\|U`, which can't
    hold None/NaN) are returned unchanged with ``n_filled == 0``.
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


def _is_dask_array(arr: Any) -> bool:
    """Return True if `arr` is a `dask.array.Array`. False if dask isn't installed."""
    try:
        import dask.array as da
    except ImportError:
        return False
    return isinstance(arr, da.Array)


def _dask_chunk_shape(arr: Any) -> list[int]:
    """First chunk size along each dim of a dask array — used as the atlas chunk_shape."""
    return [c[0] for c in arr.chunks]


def _contiguous(a: np.ndarray) -> np.ndarray:
    # np.ascontiguousarray promotes 0-D arrays to 1-D (ndmin=1 default),
    # which breaks scalar-array writes. Force-copy non-contiguous arrays
    # via np.asarray + .copy(order='C') instead, which preserves rank.
    if a.flags["C_CONTIGUOUS"]:
        return a
    return a.copy(order="C")


# Tuned for the "many small NetCDF chunks" case. batch_size controls how many
# dask blocks ride one scheduler invocation (lower dask plumbing overhead);
# prefetch_depth bounds peak memory at batch_size * prefetch_depth chunks.
_DEFAULT_BATCH_SIZE = 8
_DEFAULT_PREFETCH_DEPTH = 2


def _iter_blocks(
    arr: Any,
    var_name: str = "",
    batch_size: int = _DEFAULT_BATCH_SIZE,
    prefetch_depth: int = _DEFAULT_PREFETCH_DEPTH,
) -> Iterator[tuple[list[int], np.ndarray]]:
    """Yield ``(start_index, block_np)`` for each chunk in `arr`.

    Numpy-backed inputs are yielded as one full-shape block. Dask-backed inputs
    are prefetched in batches: a single background thread runs
    ``dask.compute(*K)`` on the next ``batch_size`` blocks while the main thread
    consumes already-materialised blocks, capping in-flight work at
    ``prefetch_depth`` batches. This collapses N per-chunk dask scheduler
    invocations into N/batch_size, and overlaps NetCDF I/O with atlas writes.

    Emits one ``event=dask_compute`` debug event per fulfilled batch via the
    Rust tracing subscriber (see ``log_chunk_event``).
    """
    if not _is_dask_array(arr):
        a = np.asarray(arr)
        yield [0] * a.ndim, _contiguous(a)
        return

    import dask

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
) -> None:
    """Populate an empty `DatasetWriter` from an xarray Dataset.

    Writes every coordinate and data variable as an atlas array, the coordinate
    names as ``_pyatlas_coords`` (a JSON list), all dataset attrs as
    dataset-level attributes, and each variable's attrs as attributes on that
    variable's array.
    """
    coord_names = [str(n) for n in ds.coords.keys()]

    # Write coords first, then data_vars. Order doesn't matter to atlas but
    # makes the on-disk file layout predictable.
    for var_name in coord_names + [str(n) for n in ds.data_vars.keys()]:
        var = ds[var_name]
        np_dtype = np.dtype(var.dtype)
        atlas_dtype = _np_to_atlas_dtype(np_dtype)
        dims = [str(d) for d in var.dims]
        shape = [int(s) for s in var.shape]

        # Pick the atlas chunk_shape:
        #   1. explicit user override via the `chunks=` kwarg, else
        #   2. the dask chunk shape if the variable is dask-backed, else
        #   3. None (atlas defaults to a single full-shape chunk).
        if chunks is not None and var_name in chunks:
            chunk_shape: Optional[list[int]] = [int(s) for s in chunks[var_name]]
        elif _is_dask_array(var.data):
            chunk_shape = _dask_chunk_shape(var.data)
        else:
            chunk_shape = None

        # Resolve the typed fill value. The CF/netCDF `_FillValue` attribute (if
        # any) is popped so it's passed to define_array, not stored as a flattened
        # atlas attribute; `_resolve_fill_value` layers the explicit `fill_value`
        # override and the NaN-for-floats default on top. Copy attrs first to
        # avoid mutating the user's Dataset.
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

        # Tag timedelta64 arrays (stored as int64 ns) so the read path can
        # restore the duration dtype. Values are normalised to ns below, so
        # the recorded unit is always "ns".
        if np_dtype.kind == "m":
            writer.set_array_attribute(var_name, _TIMEDELTA_ATTR, "ns")

        # For string arrays, missing (None/NaN) cells can't be stored as null,
        # so they're substituted with the resolved string fill (default "").
        str_fill = resolved_fill if isinstance(resolved_fill, str) else ""
        n_filled_strings = 0

        # Stream blocks: prefetched batches for dask-backed data, a single
        # full-shape block for numpy-backed data.
        for start, block in _iter_blocks(var.data, var_name=var_name):
            # TimestampNs columns: the bindings accept np.int64 only; cast the
            # numpy datetime64 view to int64 without copying.
            if block.dtype.kind == "M":
                block = block.view(np.int64)
            # timedelta64 -> int64 nanoseconds (normalise the unit first, then
            # a zero-copy view). Restored to a duration on read via the marker.
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
            warnings.warn(
                f"{var_name!r}: replaced {n_filled_strings} missing string "
                f"cell(s) (None/NaN) with {str_fill!r} — atlas cannot store "
                f"missing strings as null",
                stacklevel=2,
            )

        # Per-variable attrs become real per-array attributes in the
        # collection footer (minus `_FillValue`, which became the fill).
        for attr_key, attr_val in var_attrs.items():
            encoded = _encode_attr_value(attr_val)
            writer.set_array_attribute(var_name, _sanitize_str(str(attr_key)), encoded)

    # Dataset-level attrs
    for attr_key, attr_val in ds.attrs.items():
        encoded = _encode_attr_value(attr_val)
        writer.set_attribute(_sanitize_str(str(attr_key)), encoded)

    # Marker so we can faithfully restore coord/var distinction on read.
    writer.set_attribute(_COORDS_ATTR, json.dumps(coord_names))


def _write_xarray_dataset(
    writer: "AtlasWriter",
    ds: "xr.Dataset",
    name: str,
    chunks: Optional[dict[str, Sequence[int]]] = None,
    fill_value: Any = None,
) -> None:
    """Write an ``xarray.Dataset`` into an open :class:`AtlasWriter` as ``name``.

    Both ``AtlasWriter.add_xarray_dataset`` and the ``ds.atlas.write`` accessor
    route through here.

    The write is atomic. A ``DatasetWriter`` reaches the container only when it
    finishes, so a failure partway through — an unsupported dtype, say — is
    discarded by aborting it, and the collection never sees the dataset.
    """
    dataset_writer = writer.add_dataset(name)
    try:
        _write_xarray_to_view(dataset_writer, ds, chunks=chunks, fill_value=fill_value)
    except BaseException:
        dataset_writer.abort()
        raise
    dataset_writer.finish()


# --- xarray accessor ----------------------------------------------------------
# Registered as `ds.atlas` once `atlas` is imported. The whole module is
# side-effect-imported from `atlas/__init__.py`, so importing `atlas` is enough
# to activate this.

import xarray as _xr  # noqa: E402


@_xr.register_dataset_accessor("atlas")
class _AtlasAccessor:
    """Methods exposed on an `xr.Dataset` as `ds.atlas.*`."""

    def __init__(self, ds: "_xr.Dataset") -> None:
        self._ds = ds

    def write(
        self,
        writer: "AtlasWriter",
        name: str,
        chunks: Optional[dict[str, Sequence[int]]] = None,
        fill_value: Any = None,
    ) -> None:
        """Write this Dataset into the open collection under `name`.

        Equivalent to `writer.add_xarray_dataset(self_ds, name, chunks, fill_value)`.

        `fill_value` overrides the per-array fill: a bare scalar applies to
        numeric arrays, a `{var: scalar}` dict targets named vars (`None`
        disables the default for that var). When omitted, float arrays default
        to a `NaN` fill so mask_and_scale'd missing cells are recorded as null.
        """
        writer.add_xarray_dataset(self._ds, name, chunks, fill_value)
