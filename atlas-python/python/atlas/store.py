"""The public :class:`Atlas` — a thin Python facade over the Rust core.

The Rust extension (``atlas._atlas``) exposes the fast primitives: creating
datasets, defining and reading arrays, attributes, the pruning index. Those all
release the GIL and move data zero-copy, so they belong in Rust.

Everything *pythonic* — the xarray integration, and any future high-level
convenience — lives here, in pure Python. This module owns the ergonomics; the
Rust core owns the performance. Adding a new high-level method is a pure-Python
edit with no rebuild, and the hot paths stay direct: the xarray writer loops over
the **core** ``DatasetView`` (returned straight from the core), never through
this wrapper.

Primitive methods aren't re-declared here; ``__getattr__`` forwards them to the
core. The authoritative, typed surface is ``__init__.pyi``.
"""

from typing import Any, Optional, Sequence, Union

from . import _atlas
from . import xarray as _xarray

# Re-exported in the type stub; kept loose here since the stub is the contract.
_AtlasSource = Any


class Atlas:
    """A directory-based store for many named datasets of N-dimensional arrays.

    Construct with :meth:`create` / :meth:`open`; see ``atlas`` package docs for
    the full API. Instances wrap the Rust core and forward primitive calls to
    it, so `isinstance(store, Atlas)` holds and every method behaves as the stub
    documents.
    """

    __slots__ = ("_inner",)

    def __init__(self, inner: "_atlas.Atlas") -> None:
        # Bypass __setattr__/__getattr__ machinery for the one real field.
        object.__setattr__(self, "_inner", inner)

    # ── Construction ────────────────────────────────────────────────────
    @staticmethod
    def create(
        source: "_AtlasSource",
        codec: str = "zstd",
        meta_format: str = "json",
        meta_compression: str = "none",
        on_type_mismatch: str = "warn",
    ) -> "Atlas":
        """Create a new store. See the type stub for argument details."""
        return Atlas(
            _atlas.Atlas.create(source, codec, meta_format, meta_compression, on_type_mismatch)
        )

    @staticmethod
    def open(source: "_AtlasSource", on_type_mismatch: str = "warn") -> "Atlas":
        """Open an existing store."""
        return Atlas(_atlas.Atlas.open(source, on_type_mismatch))

    # ── Primitive delegation ────────────────────────────────────────────
    def __getattr__(self, name: str) -> Any:
        # Reached only for names not defined on the facade — every primitive
        # (create_dataset, read_array_across, pruning_index, flush, …) forwards
        # straight to the Rust core. `_inner` is set in __init__ via
        # object.__setattr__, so it never recurses here.
        return getattr(self._inner, name)

    # ── xarray integration (pure Python, calling the core directly) ─────
    def add_xarray_dataset(
        self,
        ds: "Any",
        name: str,
        chunks: Optional[dict[str, Sequence[int]]] = None,
        fill_value: Union[Any, dict[str, Any], None] = None,
    ) -> None:
        """Append an atlas dataset populated from an ``xarray.Dataset``.

        The per-variable write loop runs against the **core** ``DatasetView``,
        so this convenience adds no per-chunk overhead.
        """
        _xarray._write_xarray_new_dataset(self._inner, ds, name, chunks, fill_value)

    def open_as_xarray_dataset(self, name: str) -> "Any":
        """Open ``name`` and return it as an ``xarray.Dataset`` (eager read)."""
        return _xarray._view_to_xarray(self._inner.open_dataset(name))

    def open_as_many_xarray_dataset(
        self,
        names: Sequence[str],
        concat_dim: str = "dataset",
        parallel: bool = True,
    ) -> "Any":
        """Open many datasets stacked along ``concat_dim`` as one ``xr.Dataset``."""
        return _xarray._atlas_to_xarray_many(self._inner, list(names), concat_dim, parallel)

    # ── Dunders (implicit calls bypass __getattr__, so define explicitly) ─
    def __enter__(self) -> "Atlas":
        return self

    def __exit__(self, *_exc: object) -> bool:
        self._inner.close()
        return False

    def __repr__(self) -> str:
        return repr(self._inner)
