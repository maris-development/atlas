"""The public :class:`Atlas` and :class:`AtlasWriter` facades.

The Rust extension (``atlas._atlas``) holds the primitives: building a
collection, streaming numpy blocks into it, and reading its metadata back. Those
release the GIL and move data without copying, so they belong in Rust.

Everything pythonic — the xarray integration and the attribute decoding — lives
here. Primitives are not re-declared; ``__getattr__`` forwards them to the core.
The authoritative, typed surface is ``__init__.pyi``.

Note what is missing on purpose: a collection opened from Python exposes its
datasets, schemas, and attributes, but not its array data. Use the Rust API to
read arrays.
"""

import json as _json
from typing import Any, Optional, Sequence, Union

from . import _atlas
from . import xarray as _xarray

# Re-exported in the type stub; kept loose here since the stub is the contract.
_AtlasSource = Any


class AtlasWriter:
    """Builds one collection, then finishes.

    A collection is written once. There is no reopening it to add datasets:
    rewrite it instead. Nothing at the target is readable until
    :meth:`finish` runs, so use this as a context manager and let an exception
    abandon the whole write.

    >>> with AtlasWriter.create("/tmp/my_collection") as w:  # doctest: +SKIP
    ...     w.add_xarray_dataset(ds, name="jan_2024")
    """

    __slots__ = ("_inner",)

    def __init__(self, inner: "_atlas.AtlasWriter") -> None:
        object.__setattr__(self, "_inner", inner)

    @staticmethod
    def create(
        source: "_AtlasSource",
        codec: str = "zstd",
        block_target_size: Optional[int] = None,
    ) -> "AtlasWriter":
        """Start a collection. See the type stub for argument details."""
        return AtlasWriter(_atlas.AtlasWriter.create(source, codec, block_target_size))

    def __getattr__(self, name: str) -> Any:
        # Reached only for names not defined on the facade: add_dataset,
        # finish, dataset_count, and so on forward straight to the core.
        # `_inner` is set via object.__setattr__, so this never recurses.
        return getattr(self._inner, name)

    def add_xarray_dataset(
        self,
        ds: "Any",
        name: str,
        chunks: Optional[dict[str, Sequence[int]]] = None,
        fill_value: Union[Any, dict[str, Any], None] = None,
    ) -> None:
        """Write an ``xarray.Dataset`` into the collection as ``name``.

        Dask-backed variables stream block by block, so the dataset need not
        fit in memory. The write is atomic: a failure partway leaves no trace
        of the dataset in the collection.
        """
        _xarray._write_xarray_dataset(self._inner, ds, name, chunks, fill_value)

    # Dunders bypass __getattr__, so define them explicitly.
    def __enter__(self) -> "AtlasWriter":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        return bool(self._inner.__exit__(exc_type, exc_value, traceback))

    def __repr__(self) -> str:
        return repr(self._inner)


class Atlas:
    """An open collection, read as metadata.

    Lists datasets, reports their schemas and attributes, and deletes datasets.
    It does **not** read array data; :class:`AtlasWriter` writes collections and
    the Rust API reads them.
    """

    __slots__ = ("_inner",)

    def __init__(self, inner: "_atlas.Atlas") -> None:
        object.__setattr__(self, "_inner", inner)

    @staticmethod
    def open(source: "_AtlasSource") -> "Atlas":
        """Open an existing collection. Reads the footer and nothing else."""
        return Atlas(_atlas.Atlas.open(source))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def dataset(self, name: str) -> "_atlas.DatasetView":
        """A metadata view of one dataset. Raises ``KeyError`` if absent."""
        return self._inner.dataset(name)

    def coords(self, name: str) -> list[str]:
        """Names of the variables that were xarray coordinates in ``name``.

        Empty for a dataset that atlas did not write from xarray.
        """
        raw = self._inner.dataset(name).get_attribute(_xarray._COORDS_ATTR)
        if raw is None:
            return []
        # The marker is a bare JSON list, not one of the `json:`-prefixed
        # attribute values, so decode it directly.
        return [str(n) for n in _json.loads(raw)]

    def attributes(self, name: str) -> dict[str, Any]:
        """Dataset-level attributes of ``name``, decoded back to Python values.

        Complex values that were JSON-encoded on the way in are restored here;
        the reserved ``_pyatlas_coords`` marker is omitted (see :meth:`coords`).
        """
        attrs = self._inner.dataset(name).attributes()
        return {
            key: _xarray._decode_attr_value(value)
            for key, value in attrs.items()
            if key != _xarray._COORDS_ATTR
        }

    def array_attributes(self, name: str, array: str) -> dict[str, Any]:
        """Attributes of one array of ``name``, decoded back to Python values."""
        attrs = self._inner.dataset(name).array_attributes(array)
        return {key: _xarray._decode_attr_value(value) for key, value in attrs.items()}

    def __contains__(self, name: str) -> bool:
        return bool(self._inner.dataset_exists(name))

    def __len__(self) -> int:
        return int(self._inner.dataset_count())

    def __iter__(self) -> Any:
        return iter(self._inner.list_datasets())

    def __repr__(self) -> str:
        return repr(self._inner)
