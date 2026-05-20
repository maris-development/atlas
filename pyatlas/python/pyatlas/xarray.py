"""xarray integration for pyatlas.

`xarray` and `dask` are required dependencies.

Reads automatically return dask-backed variables when an array was stored with a
non-trivial chunk shape (`chunk_shape != shape`); full-shape arrays come back
eager as numpy. The dask chunks mirror the on-disk chunk grid one-to-one.

Bulk ingestion — Atlas itself batches; explicit flush on the atlas persists:
    with atlas:
        for nc_path in nc_paths:
            ds = xr.open_dataset(nc_path)
            atlas.add_xr_dataset(ds, name=Path(nc_path).stem)
    # atlas.close() (== flush) runs on __exit__.
"""
from __future__ import annotations

import itertools
import json
from typing import TYPE_CHECKING, Any, Iterator, Optional, Sequence

import numpy as np

if TYPE_CHECKING:
    import xarray as xr

    from ._pyatlas import Atlas, DatasetView


_COORDS_ATTR = "_pyatlas_coords"
_JSON_PREFIX = "json:"

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
    # Object (Python str/bytes) and fixed-size byte/unicode strings all
    # become variable-length atlas strings.
    if np_dtype.kind in ("O", "S", "U"):
        return "string"
    supported = ", ".join(sorted(set(_NUMPY_TO_ATLAS.values())))
    raise NotImplementedError(
        f"numpy dtype {np_dtype!r} is not supported by pyatlas "
        f"(supported: {supported}, plus object/bytes/unicode → string)"
    )


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

    if isinstance(value, (bool, int, float, str)):
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


def _iter_blocks(arr: Any) -> Iterator[tuple[list[int], np.ndarray]]:
    """Yield ``(start_index, block_np)`` for each chunk in `arr`.

    For numpy-backed inputs the whole array is yielded as a single block (no
    behaviour change vs the old eager path). For dask-backed inputs blocks
    are computed one at a time via ``arr.blocks[idx].compute()``, so peak
    memory is bounded by a single chunk per variable rather than the full
    array.
    """
    def _contiguous(a: np.ndarray) -> np.ndarray:
        # np.ascontiguousarray promotes 0-D arrays to 1-D (ndmin=1 default),
        # which breaks scalar-array writes. Force-copy non-contiguous arrays
        # via np.asarray + .copy(order='C') instead, which preserves rank.
        if a.flags["C_CONTIGUOUS"]:
            return a
        return a.copy(order="C")

    if _is_dask_array(arr):
        chunks = arr.chunks
        # Per-dim cumulative starts: e.g. ((4,4,4,4),) -> [[0,4,8,12]]
        offsets = [[0, *itertools.accumulate(c)][:-1] for c in chunks]
        for block_idx in itertools.product(*[range(len(c)) for c in chunks]):
            start = [offsets[d][i] for d, i in enumerate(block_idx)]
            block = _contiguous(arr.blocks[block_idx].compute())
            yield start, block
    else:
        a = np.asarray(arr)
        yield [0] * a.ndim, _contiguous(a)


def _write_xarray_to_view(
    view: "DatasetView",
    ds: "xr.Dataset",
    chunks: Optional[dict[str, Sequence[int]]] = None,
) -> None:
    """Populate an empty `DatasetView` with the contents of an xarray Dataset.

    Writes every coordinate and data variable as an atlas array, the
    coordinate names as ``_pyatlas_coords`` (JSON list), all dataset attrs,
    and all per-variable attrs flattened as ``{var}.{attr}``.
    """
    coord_names = [str(n) for n in ds.coords.keys()]

    # Write coords first, then data_vars. Order doesn't matter to atlas but
    # makes the on-disk file layout predictable.
    for var_name in coord_names + [str(n) for n in ds.data_vars.keys()]:
        var = ds[var_name]
        atlas_dtype = _np_to_atlas_dtype(np.dtype(var.dtype))
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

        view.define_array(
            var_name,
            dtype=atlas_dtype,
            dims=dims,
            shape=shape,
            chunk_shape=chunk_shape,
        )

        # Stream blocks: one chunk at a time for dask-backed data, a single
        # full-shape block for numpy-backed data.
        for start, block in _iter_blocks(var.data):
            # TimestampNs columns: the bindings accept np.int64 only; cast the
            # numpy datetime64 view to int64 without copying.
            if block.dtype.kind == "M":
                block = block.view(np.int64)
            view.write_array(var_name, start=start, data=block)

        # Per-variable attrs → flattened as `{var}.{attr}`
        for attr_key, attr_val in var.attrs.items():
            encoded = _encode_attr_value(attr_val)
            view.set_attribute(f"{var_name}.{attr_key}", encoded)

    # Dataset-level attrs
    for attr_key, attr_val in ds.attrs.items():
        encoded = _encode_attr_value(attr_val)
        view.set_attribute(str(attr_key), encoded)

    # Marker so we can faithfully restore coord/var distinction on read.
    view.set_attribute(_COORDS_ATTR, json.dumps(coord_names))


_ATLAS_TO_NUMPY = {atlas: np_dt for np_dt, atlas in _NUMPY_TO_ATLAS.items()}


def _atlas_to_numpy_dtype(atlas_dtype: str) -> np.dtype:
    """Numpy dtype that `view.read_array` returns for a given atlas dtype string."""
    if atlas_dtype in _ATLAS_TO_NUMPY:
        return _ATLAS_TO_NUMPY[atlas_dtype]
    if atlas_dtype == "string":
        return np.dtype("object")
    raise NotImplementedError(
        f"atlas dtype {atlas_dtype!r} is not supported on the dask read path"
    )


def _dask_chunks_for(shape: Sequence[int], chunk_shape: Sequence[int]) -> tuple:
    """Per-dim chunk-length tuples in the form dask expects."""
    chunks: list[tuple[int, ...]] = []
    for dim_size, dim_chunk in zip(shape, chunk_shape):
        if dim_chunk <= 0 or dim_size == 0:
            chunks.append((dim_size,))
            continue
        full = dim_size // dim_chunk
        rem = dim_size - full * dim_chunk
        c = (dim_chunk,) * full + ((rem,) if rem else ())
        chunks.append(c if c else (0,))
    return tuple(chunks)


def _view_to_dask_array(view: "DatasetView", name: str) -> Any:
    """Build a `dask.array.Array` that lazily reads `name` chunk-by-chunk.

    Each on-disk chunk becomes one dask task; values are fetched via
    `view.read_array(name, start, block_shape)` on demand. Used by
    `_view_to_xarray` when an array's `chunk_shape != shape`.
    """
    import dask
    import dask.array as da

    meta = view.array_meta(name)
    shape: list[int] = list(meta["shape"])
    chunk_shape: list[int] = list(meta["chunk_shape"])
    np_dtype = _atlas_to_numpy_dtype(meta["dtype"])

    per_dim_chunks = _dask_chunks_for(shape, chunk_shape)
    offsets = [
        [0, *itertools.accumulate(dim_chunks)][:-1] for dim_chunks in per_dim_chunks
    ]

    def _read_block(start: list[int], block_shape: list[int]) -> np.ndarray:
        return view.read_array(name, start=start, shape=block_shape)

    def _nested_blocks(axis: int, prefix_start: list[int], prefix_shape: list[int]):
        if axis == len(shape):
            block_shape = list(prefix_shape)
            block_start = list(prefix_start)
            delayed = dask.delayed(_read_block)(block_start, block_shape)
            return da.from_delayed(delayed, shape=tuple(block_shape), dtype=np_dtype)
        return [
            _nested_blocks(
                axis + 1,
                prefix_start + [offsets[axis][i]],
                prefix_shape + [per_dim_chunks[axis][i]],
            )
            for i in range(len(per_dim_chunks[axis]))
        ]

    nested = _nested_blocks(0, [], [])
    return da.block(nested)


def _view_to_xarray(view: "DatasetView") -> "xr.Dataset":
    """Convert an atlas `DatasetView` into an xarray Dataset.

    Variables stored with `chunk_shape != shape` come back dask-backed (one task
    per on-disk chunk); full-shape arrays come back eager as numpy.
    """
    import xarray as xr

    array_names = list(view.list_arrays())
    array_name_set = set(array_names)

    # Coord/var assignment
    coords_marker = view.get_attribute(_COORDS_ATTR)
    if isinstance(coords_marker, str):
        try:
            coord_names = set(json.loads(coords_marker))
        except (TypeError, ValueError):
            coord_names = set()
    else:
        # Fallback heuristic: 1-D array whose single dim matches its name.
        coord_names = set()
        for name in array_names:
            meta = view.array_meta(name)
            if (
                len(meta["dimension_names"]) == 1
                and meta["dimension_names"][0] == name
            ):
                coord_names.add(name)

    # Pull array data. Chunked arrays (chunk_shape != shape) come back as a
    # lazy dask.array; full-shape and 0-D arrays come back eager.
    data_vars: dict[str, tuple] = {}
    coords: dict[str, tuple] = {}
    for name in array_names:
        meta = view.array_meta(name)
        shape = list(meta["shape"])
        chunk_shape = list(meta["chunk_shape"])
        if not shape or chunk_shape == shape:
            arr = view.read_array(name)
            if arr is None:
                continue
        else:
            arr = _view_to_dask_array(view, name)
        dims = list(meta["dimension_names"])
        entry = (dims, arr, {})  # placeholder for per-var attrs; filled below
        if name in coord_names:
            coords[name] = entry
        else:
            data_vars[name] = entry

    # Split dataset attrs vs flattened per-var attrs
    raw_attrs = dict(view.attributes())
    raw_attrs.pop(_COORDS_ATTR, None)

    dataset_attrs: dict[str, Any] = {}
    per_var_attrs: dict[str, dict[str, Any]] = {n: {} for n in array_names}
    for key, value in raw_attrs.items():
        if "." in key:
            var, rest = key.split(".", 1)
            if var in array_name_set:
                per_var_attrs[var][rest] = _decode_attr_value(value)
                continue
        dataset_attrs[key] = _decode_attr_value(value)

    # Inject per-var attrs into the (dims, data, attrs) triples
    def _with_attrs(name: str, triple: tuple) -> tuple:
        dims, arr, _ = triple
        return (dims, arr, per_var_attrs.get(name, {}))

    data_vars = {n: _with_attrs(n, t) for n, t in data_vars.items()}
    coords = {n: _with_attrs(n, t) for n, t in coords.items()}

    return xr.Dataset(data_vars=data_vars, coords=coords, attrs=dataset_attrs)


def _write_xarray_new_dataset(
    atlas: "Atlas",
    ds: "xr.Dataset",
    name: str,
    chunks: Optional[dict[str, Sequence[int]]] = None,
) -> None:
    """Rust-delegated helper: create a fresh atlas dataset and populate it.

    Both `atlas.add_xr_dataset` and the `ds.atlas.write` accessor route through
    this function.
    """
    view = atlas.create_dataset(name)
    _write_xarray_to_view(view, ds, chunks=chunks)


# --- xarray accessor ----------------------------------------------------------
# Registered as `ds.atlas` once `pyatlas` is imported. The whole module is
# side-effect-imported from `pyatlas/__init__.py` so importing `pyatlas` is
# enough to activate this.

import xarray as _xr  # noqa: E402


@_xr.register_dataset_accessor("atlas")
class _AtlasAccessor:
    """Methods exposed on an `xr.Dataset` as `ds.atlas.*`."""

    def __init__(self, ds: "_xr.Dataset") -> None:
        self._ds = ds

    def write(
        self,
        atlas: "Atlas",
        name: str,
        chunks: Optional[dict[str, Sequence[int]]] = None,
    ) -> None:
        """Append this Dataset to the open atlas store under `name`.

        Equivalent to `atlas.add_xr_dataset(self_ds, name, chunks)`.
        """
        atlas.add_xr_dataset(self._ds, name, chunks)
