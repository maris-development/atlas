"""xarray integration for pyatlas.

Requires xarray as an optional dependency:

    pip install xarray

Public API:
    to_atlas(ds, atlas, name)    — write an xr.Dataset into a new atlas dataset
    from_atlas(atlas, name)      — read an atlas dataset back as an xr.Dataset

These functions are also reachable via methods on `Atlas` and `DatasetView`:
    Atlas.from_xarray(ds, path, name, codec="zstd")
    Atlas.to_xarray(name)
    DatasetView.write_xarray(ds)
    DatasetView.to_xarray()
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
}


def _np_to_atlas_dtype(np_dtype: np.dtype) -> str:
    try:
        return _NUMPY_TO_ATLAS[np_dtype]
    except KeyError as exc:
        supported = ", ".join(sorted(set(_NUMPY_TO_ATLAS.values())))
        raise NotImplementedError(
            f"numpy dtype {np_dtype!r} is not supported by pyatlas "
            f"(supported: {supported})"
        ) from exc


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
    if _is_dask_array(arr):
        chunks = arr.chunks
        # Per-dim cumulative starts: e.g. ((4,4,4,4),) -> [[0,4,8,12]]
        offsets = [[0, *itertools.accumulate(c)][:-1] for c in chunks]
        for block_idx in itertools.product(*[range(len(c)) for c in chunks]):
            start = [offsets[d][i] for d, i in enumerate(block_idx)]
            block = np.ascontiguousarray(arr.blocks[block_idx].compute())
            yield start, block
    else:
        yield [0] * np.ndim(arr), np.ascontiguousarray(np.asarray(arr))


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

    view.flush()


def _view_to_xarray(view: "DatasetView") -> "xr.Dataset":
    """Convert an atlas `DatasetView` into an xarray Dataset (eager read)."""
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

    # Pull array data
    data_vars: dict[str, tuple] = {}
    coords: dict[str, tuple] = {}
    for name in array_names:
        meta = view.array_meta(name)
        arr = view.read_array(name)
        if arr is None:
            continue
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
