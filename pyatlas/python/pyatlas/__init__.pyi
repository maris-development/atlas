import os
from typing import TYPE_CHECKING, Any, Optional, Sequence, Union

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import xarray as xr
    # `obstore` is an optional dependency — only needed if you want to
    # open / create stores against S3, GCS, Azure, or HTTP backends.
    # `pip install pyatlas[cloud]` pulls it in.
    import obstore.store as _obstore_store

# Public alias for the polymorphic source argument accepted by
# `Atlas.create` / `Atlas.open`. Either a local filesystem path or an
# obstore-constructed store handle (`obstore.store.S3Store`,
# `obstore.store.GCSStore`, `obstore.store.AzureStore`,
# `obstore.store.LocalStore`, `obstore.store.HttpStore`).
AtlasSource = Union[str, "os.PathLike[str]", "_obstore_store.ObjectStore"]

__version__: str

class Atlas:
    """A directory-based store for many named datasets of N-dimensional arrays."""

    @staticmethod
    def create(
        source: AtlasSource,
        codec: str = "zstd",
        meta_format: str = "json",
        meta_compression: str = "none",
    ) -> "Atlas":
        """Create a new store.

        Args:
            source: Either a local filesystem path (created with `mkdir -p`
                semantics) or an [obstore](https://github.com/developmentseed/obstore)-
                constructed store handle (`obstore.store.S3Store`,
                `obstore.store.GCSStore`, `obstore.store.AzureStore`,
                `obstore.store.LocalStore`, `obstore.store.HttpStore`).
                Cloud credentials, region, endpoint and retry policy are
                obstore's responsibility — pyatlas writes through the handle.
            codec: Compression codec for new array blocks.
                One of `"zstd"` (default), `"lz4"`, `"none"` / `"uncompressed"`.
            meta_format: On-disk encoding for the metadata file.
                `"json"` (default, written as `atlas.json`) or
                `"msgpack"` / `"mp"` (written as `atlas.msgpack`, ~30-50%
                smaller and faster to parse, but not human-readable).
            meta_compression: Compression applied to the encoded metadata file.
                `"none"` / `"uncompressed"` (default — filename has no extra
                suffix), `"zstd"` (suffix `.zst`), or `"lz4"` (suffix `.lz4`).
                Mostly useful for stores with thousands of datasets on a
                high-latency object store.
        """
        ...

    @staticmethod
    def open(source: AtlasSource) -> "Atlas":
        """Open an existing store.

        Accepts the same shapes as [`Atlas.create`][pyatlas.Atlas.create]:
        a local filesystem path or an obstore-constructed store handle.
        The codec, metadata format, and metadata compression are
        auto-detected from the on-disk filename (`atlas.json`,
        `atlas.msgpack`, `atlas.json.zst`, `atlas.msgpack.lz4`, etc.) —
        no extra arguments required.
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

    def to_xarray_many(
        self,
        names: Sequence[str],
        concat_dim: str = "dataset",
        parallel: bool = True,
    ) -> "xr.Dataset":
        """Open many datasets and stack them along `concat_dim` as one Dataset.

        atlas-native equivalent of `xr.open_mfdataset(...)`. Each variable comes
        back shape `(len(names), *original_shape)` as eager numpy. Wrap with
        `.chunk(...)` downstream if you need dask laziness.

        Implementation calls `Atlas.read_array_across` once per variable —
        N per-dataset reads share one `RwLock::read` guard on the shared
        physical file and dispatch concurrently on the tokio runtime.

        Variable names + dtypes must match across all listed datasets.
        Coordinates and dataset-level attrs are taken from the first dataset.

        The `parallel` parameter is accepted for API compatibility but no
        longer selects an implementation — the bulk path is always taken.
        """
        ...

    def read_array_across(
        self,
        array: str,
        dataset_names: Sequence[str],
        start: Optional[Sequence[int]] = None,
        shape: Optional[Sequence[int]] = None,
    ) -> list[Optional[NDArray[Any]]]:
        """Bulk-read the same slice of `array` across many datasets.

        Returns a list of length `len(dataset_names)` — one numpy array per
        dataset, or `None` for datasets that don't declare `array`. All reads
        share one `RwLock::read` guard on the array's shared physical file and
        dispatch concurrently on the tokio runtime via a `JoinSet` capped at
        `num_cpus` in-flight tasks. Replaces N individual `read_array` calls
        with a single Python ↔ Rust round-trip.

        `start` / `shape` follow the same conventions as
        `DatasetView.read_array`: omit both to read each dataset's full array.

        For the common "stack and use as one array" pattern, prefer
        [`read_array_across_stacked`] which skips the Python-side
        `np.stack` copy.
        """
        ...

    def read_array_across_stacked(
        self,
        array: str,
        dataset_names: Sequence[str],
        start: Optional[Sequence[int]] = None,
        shape: Optional[Sequence[int]] = None,
    ) -> NDArray[Any]:
        """Stacked variant of `read_array_across`: returns one numpy ndarray
        of shape `(len(dataset_names), *per_dataset_shape)` instead of a list.

        The output buffer is pre-allocated in Rust; each parallel read writes
        its row in as the task completes. Skips the ~N × per_dataset_size of
        memory copies that `np.stack(read_array_across(...))` would do.

        Errors if any listed dataset doesn't declare `array` (the stacked
        representation has no positional "missing" sentinel).
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
        fill_value: Optional[Any] = None,
    ) -> None:
        """Declare a new N-dimensional array.

        Args:
            name: Array name (no `/`, no leading `_`, non-empty).
            dtype: e.g. `"float32"`, `"int64"`, `"uint8"`. See the module README for the full list.
            dims: Named dimensions, one per axis.
            shape: Logical shape, one entry per axis.
            chunk_shape: Optional chunk shape; defaults to `shape` (a single chunk).
            fill_value: Optional scalar returned for unwritten cells. Must match
                the array dtype: a Python `int` for int/uint/timestamp arrays
                (range-checked), a `float` (or `int`) for float arrays, a `bool`
                for bool arrays, a `str` for string arrays. Raises `TypeError`
                on a mismatch and `OverflowError` if the value is out of range.
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

    def read_arrays(
        self,
        names: Sequence[str],
        start: Optional[Sequence[int]] = None,
        shape: Optional[Sequence[int]] = None,
    ) -> dict[str, Optional[NDArray[Any]]]:
        """Bulk-read multiple arrays in one PyO3 call.

        Returns `{name: ndarray | None}` — `None` for arrays not in this
        dataset. Same `start` / `shape` apply to every array.

        Fast path for per-dataset slice reads (e.g. inside a dask worker)
        where `to_xarray(name).isel(...).load()` overhead would dominate the
        actual I/O cost. Skips the xr.Dataset construction and per-chunk
        dask graph that `to_xarray` builds. See the benchmarks for the
        ~3-4× speedup over `to_xarray` iteration on chunked storage.
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

    def array_fill_value(self, name: str) -> Optional[Any]:
        """The fill value passed to `define_array`, or `None` if the array
        doesn't exist in this dataset or was defined without one."""
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
