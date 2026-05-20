from typing import TYPE_CHECKING, Any, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import xarray as xr

__version__: str

class Atlas:
    """A directory-based store for many named datasets of N-dimensional arrays."""

    @staticmethod
    def create(path: str, codec: str = "zstd") -> "Atlas":
        """Create a new store at the given local filesystem path.

        Args:
            path: Directory to create the store in. Will be created if missing.
            codec: Compression codec for new array blocks.
                One of `"zstd"` (default), `"lz4"`, `"none"` / `"uncompressed"`.
        """
        ...

    @staticmethod
    def open(path: str) -> "Atlas":
        """Open an existing store at the given local filesystem path.

        The codec is read from `atlas.json` — no codec argument required.
        """
        ...

    def create_dataset(self, name: str) -> "DatasetView":
        """Create a new dataset. Raises if a dataset with this name already exists."""
        ...

    def open_dataset(self, name: str) -> "DatasetView":
        """Open an existing dataset. Raises `KeyError` if not found."""
        ...

    def delete_dataset(self, name: str) -> None:
        """Remove a dataset and tombstone its entries in every shared array file."""
        ...

    def list_datasets(self) -> list[str]:
        """Names of all datasets in this store."""
        ...

    def list_arrays(self) -> list[str]:
        """Distinct array names across all datasets."""
        ...

    def dataset_exists(self, name: str) -> bool: ...

    def __repr__(self) -> str: ...

    def add_xr_dataset(
        self,
        ds: "xr.Dataset",
        name: str,
        chunks: Optional[dict[str, Sequence[int]]] = None,
    ) -> None:
        """Append an atlas dataset populated from an `xarray.Dataset`.

        Per-variable attributes are flattened as `{var}.{attr}` alongside the
        dataset attrs. Dask-backed variables are streamed chunk-by-chunk; their
        chunk shape is used as the atlas chunk shape unless `chunks` overrides
        it per-variable.
        """
        ...

    def to_xarray(self, name: str) -> "xr.Dataset":
        """Open dataset `name` and return it as an `xarray.Dataset`.

        Variables stored with `chunk_shape != shape` come back dask-backed (one
        dask task per on-disk chunk); full-shape and 0-D variables come back
        eager as numpy arrays.
        """
        ...

    def flush(self) -> None:
        """Persist the in-memory atlas.json + every cached array file.

        This is the single durability boundary; until called, no mutation
        reaches disk (dropping the Atlas without flushing abandons every
        pending write).
        """
        ...

    def close(self) -> None:
        """Final flush; alias for `flush()`. Mirrors the context-manager exit."""
        ...

    def compact(self) -> None:
        """Compact every cached array file in place (reclaim tombstoned space)."""
        ...

    def __enter__(self) -> "Atlas": ...
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None: ...


class DatasetView:
    """A handle to a single dataset within an `Atlas` store.

    Holds the per-dataset array schemas and attributes. Mutations are buffered
    in-memory until `flush()` is called.
    """

    @property
    def name(self) -> str: ...

    def list_arrays(self) -> list[str]:
        """Names of all arrays defined in this dataset."""
        ...

    def define_array(
        self,
        name: str,
        dtype: str,
        dims: Sequence[str],
        shape: Sequence[int],
        chunk_shape: Optional[Sequence[int]] = None,
    ) -> None:
        """Declare a new N-dimensional array.

        Args:
            name: Array name (no `/`, no leading `_`, non-empty).
            dtype: e.g. `"float32"`, `"int64"`, `"uint8"`. See the module README for the full list.
            dims: Named dimensions, one per axis.
            shape: Logical shape, one entry per axis.
            chunk_shape: Optional chunk shape; defaults to `shape` (a single chunk).
        """
        ...

    def write_array(
        self,
        name: str,
        start: Sequence[int],
        data: NDArray[Any],
    ) -> None:
        """Write a numpy array at the given starting index.

        The numpy dtype must match the stored dtype and the array must be C-contiguous.
        """
        ...

    def read_array(
        self,
        name: str,
        start: Optional[Sequence[int]] = None,
        shape: Optional[Sequence[int]] = None,
    ) -> Optional[NDArray[Any]]:
        """Read a full or partial array.

        With `start` and `shape` omitted, returns the entire array. Returns `None`
        if the array does not exist in this dataset.
        """
        ...

    def delete_array(self, name: str) -> None:
        """Remove the array from this dataset (tombstone)."""
        ...

    def array_meta(self, name: str) -> Optional[dict[str, Any]]:
        """Schema for `name` (`{"dtype", "shape", "chunk_shape", "dimension_names"}`),
        or `None` if no array with that name exists in this dataset."""
        ...

    def array_stats(self, name: str) -> Optional[dict[str, Any]]:
        """Persisted statistics or `None` if not yet computed.

        After `flush()` returns `{"row_count", "null_count", "min", "max"}`.
        """
        ...

    def attributes(self) -> dict[str, Any]:
        """All attributes as a dict."""
        ...

    def set_attribute(
        self,
        key: str,
        value: Any,
        dtype: Optional[str] = None,
    ) -> None:
        """Set a typed attribute.

        Type is inferred from the Python type by default. Pass `dtype` to force
        a narrower variant (e.g. `dtype="int8"`).
        """
        ...

    def get_attribute(self, key: str) -> Any:
        """Returns the attribute value or `None` if not set."""
        ...

    def __repr__(self) -> str: ...
