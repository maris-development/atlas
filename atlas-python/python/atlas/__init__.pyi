import os
from typing import TYPE_CHECKING, Any, Optional, Sequence, Union

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import xarray as xr
    # `obstore` is an optional dependency — only needed if you want to
    # open / create stores against S3, GCS, Azure, or HTTP backends.
    # `pip install atlas-python[cloud]` pulls it in.
    import obstore.store as _obstore_store

# Public alias for the polymorphic source argument accepted by
# `Atlas.create` / `Atlas.open`. Either a local filesystem path or an
# obstore-constructed store handle (`obstore.store.S3Store`,
# `obstore.store.GCSStore`, `obstore.store.AzureStore`,
# `obstore.store.LocalStore`, `obstore.store.HttpStore`).
AtlasSource = Union[str, "os.PathLike[str]", "_obstore_store.ObjectStore"]

__version__: str

def init_tracing(filter: Optional[str] = None) -> None:
    """Install a `tracing` subscriber that writes atlas's internal logs to stderr.

    `filter` is an [`env_logger`/`tracing`-style directive](https://docs.rs/tracing-subscriber/latest/tracing_subscriber/filter/struct.EnvFilter.html)
    (e.g. `"atlas=debug"`); when omitted, the `ATLAS_LOG` / `RUST_LOG`
    environment variables are consulted, defaulting to `"info"`. Idempotent —
    safe to call more than once; only the first call installs the subscriber.
    Raises `ValueError` if `filter` is not a valid directive.
    """
    ...

class Atlas:
    """A directory-based store for many named datasets of N-dimensional arrays."""

    @staticmethod
    def create(
        source: AtlasSource,
        codec: str = "zstd",
        meta_format: str = "json",
        meta_compression: str = "none",
        on_type_mismatch: str = "warn",
    ) -> "Atlas":
        """Create a new store.

        Args:
            source: Either a local filesystem path (created with `mkdir -p`
                semantics) or an [obstore](https://github.com/developmentseed/obstore)-
                constructed store handle (`obstore.store.S3Store`,
                `obstore.store.GCSStore`, `obstore.store.AzureStore`,
                `obstore.store.LocalStore`, `obstore.store.HttpStore`).
                Cloud credentials, region, endpoint and retry policy are
                obstore's responsibility — atlas writes through the handle.
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
    def open(source: AtlasSource, on_type_mismatch: str = "warn") -> "Atlas":
        """Open an existing store.

        `on_type_mismatch` (`"warn"` | `"error"`) sets the per-session policy
        for a dataset whose type can't merge with the collection's existing
        type for that array/attribute. It is not read from disk.

        Accepts the same shapes as [`Atlas.create`][atlas.Atlas.create]:
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

    def merged_schema(self) -> dict[str, Any]:
        """Collection-wide merged schema.

        Returns ``{"arrays": {name: {"dtype", "dimension_names", "attributes"}},
        "global_attributes": {key: dtype}}`` — every unique array and attribute
        with its type widened across all datasets. Descriptive only.
        """
        ...

    def pruning_index(
        self,
        arrays: Optional[list[str]] = None,
        global_attrs: Optional[list[str]] = None,
        array_attrs: Optional[list[tuple[str, str]]] = None,
    ) -> dict[str, Any]:
        """Flattened statistics for **only** the requested columns.

        Columns are addressed by ``arrays`` (array names), ``global_attrs``
        (dataset-level attribute keys), and ``array_attrs`` (``(array, key)``
        pairs). In the result, an array column is keyed by its name, a global
        attribute by its key, and a per-array attribute by ``"array:key"``.

        Returns ``{"rows", "datasets", "live", "columns"}``. Each entry of
        ``columns`` holds numpy arrays over the full row space: ``present``,
        ``stats_valid``, ``min``, ``max``, ``row_count``, ``null_count``.
        ``row_count`` is 0 for a dataset that doesn't declare the column.

        Statistics keep the type they were computed with, so ``min``/``max``
        come back as int64, uint64, float64, or a list of ``bytes | None``
        depending on what the column actually holds. For a column's
        collection-wide declared type, use `merged_schema()`. **Attribute
        columns currently carry presence only** — ``present`` marks which
        datasets have the key, but ``stats_valid`` is False and ``min``/``max``
        are unset.

        Row ``i`` is the dataset at ordinal ``i`` (see `dataset_row`);
        ``datasets[i]`` names it, or is ``None`` for a deleted slot. Datasets
        that don't declare a column are explicit gaps (``present`` is False).
        Always ``&`` in ``live`` to exclude deleted datasets.

        Only the named columns are fetched from storage, so the cost is
        independent of how many other columns the collection has. Raises
        `RuntimeError` if the on-disk index is stale relative to the metadata
        (flush to rebuild it) or corrupt.
        """
        ...

    def column_summaries(self) -> dict[str, Any]:
        """Every column's ``{"min", "max", "present_count"}``, read from the
        index footer alone — no column data is fetched.

        Use it to rule a column out before requesting it: if its
        collection-wide range can't satisfy a predicate, no dataset can.
        """
        ...

    def dataset_row(self, name: str) -> Optional[int]:
        """This dataset's fixed row ordinal in the pruning index.

        Stable across deletions — a deleted dataset keeps its slot so no other
        dataset's row moves. Only `compact()` renumbers.
        """
        ...

    def row_slots(self) -> int:
        """Total row slots including tombstoned ones — the pruning index's
        height. Larger than the number of live datasets until `compact()`."""
        ...

    def dataset_exists(self, name: str) -> bool: ...

    def __repr__(self) -> str: ...

    def add_xarray_dataset(
        self,
        ds: "xr.Dataset",
        name: str,
        chunks: Optional[dict[str, Sequence[int]]] = None,
        fill_value: Union[Any, dict[str, Any], None] = None,
    ) -> None:
        """Append an atlas dataset populated from an `xarray.Dataset`.

        Per-variable attributes are flattened as `{var}.{attr}` alongside the
        dataset attrs. Dask-backed variables are streamed chunk-by-chunk; their
        chunk shape is used as the atlas chunk shape unless `chunks` overrides
        it per-variable.

        `fill_value` overrides the per-array fill value: a bare scalar applies to
        numeric arrays, a `{var: scalar}` dict targets named variables (`None`
        disables the default for that variable). When omitted, arrays default to a
        sentinel fill so mask_and_scale'd missing cells are recorded as null: `NaN`
        for floats, `NaT` for `datetime64[ns]`, and `""` for strings (integers
        have none). Missing string cells (None/NaN) are substituted with the string
        fill and a warning is emitted, since a string can't be stored as null
        directly.
        """
        ...

    def open_as_xarray_dataset(self, name: str) -> "xr.Dataset":
        """Open dataset `name` and return it as an `xarray.Dataset`.

        Variables stored with `chunk_shape != shape` come back dask-backed (one
        dask task per on-disk chunk); full-shape and 0-D variables come back
        eager as numpy arrays.
        """
        ...

    def open_as_many_xarray_dataset(
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

    Exposes the dataset's array schemas plus its attributes. Dataset-level
    (global) attributes live in the reserved `_global` file; per-variable
    attributes live on each array's own file. Attribute *values* are stored in
    the `.af` files, not in `atlas.json`. Mutations are buffered in-memory until
    `flush()` is called.
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
        where `open_as_xarray_dataset(name).isel(...).load()` overhead would dominate the
        actual I/O cost. Skips the xr.Dataset construction and per-chunk
        dask graph that `open_as_xarray_dataset` builds. See the benchmarks for the
        ~3-4× speedup over `open_as_xarray_dataset` iteration on chunked storage.
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
        """Persisted statistics, or `None` if the array isn't in this dataset or
        hasn't been flushed yet.

        After `flush()` returns `{"row_count", "null_count", "min", "max"}`.
        `min`/`max` keep the array's own type: `int`/`float` for numeric arrays,
        `bytes` for string arrays (lexicographic order), and `int` nanoseconds
        since the epoch for `timestamp_nanoseconds` arrays. `min`/`max` are
        `None` for dtypes with no natural ordering (e.g. lists).
        """
        ...

    def array_fill_value(self, name: str) -> Optional[Any]:
        """The fill value passed to `define_array`, or `None` if the array
        doesn't exist in this dataset or was defined without one."""
        ...

    def attributes(self) -> dict[str, Any]:
        """All dataset-level (global) attributes as a dict."""
        ...

    def set_attribute(
        self,
        key: str,
        value: Any,
        dtype: Optional[str] = None,
    ) -> None:
        """Set a typed dataset-level (global) attribute.

        Type is inferred from the Python type by default. Pass `dtype` to force
        a narrower variant (e.g. `dtype="int8"`).
        """
        ...

    def get_attribute(self, key: str) -> Any:
        """Returns the dataset-level attribute value or `None` if not set."""
        ...

    def set_array_attribute(
        self,
        array: str,
        key: str,
        value: Any,
        dtype: Optional[str] = None,
    ) -> None:
        """Set a typed per-variable attribute on `array` (e.g. `units`).

        Raises `KeyError` if the array isn't defined in this dataset. Type is
        inferred from the Python type by default; pass `dtype` to force a
        narrower variant.
        """
        ...

    def get_array_attribute(self, array: str, key: str) -> Any:
        """Returns the per-variable attribute value on `array`, or `None`."""
        ...

    def array_attributes(self, array: str) -> dict[str, Any]:
        """All per-variable attributes on `array` as a dict."""
        ...

    def __repr__(self) -> str: ...
