"""Typed surface of the ``atlas`` package.

A collection is one immutable file. You build it with :class:`AtlasWriter`, and
once it is finished it never changes. Opening it with :class:`Atlas` gives you
its datasets, their schemas, and their attributes — but not their array data.
Reading arrays is the Rust API's job.

The one thing you can do to a finished collection is
:meth:`Atlas.delete_dataset`, which hides a dataset by writing a small mask file
beside the container. It reclaims no space and moves no ordinals.
"""

import os
from typing import Any, Iterator, Optional, Sequence, Union

import xarray as xr

__version__: str

__all__ = [
    "Atlas",
    "AtlasWriter",
    "DatasetView",
    "DatasetWriter",
    "__version__",
    "init_tracing",
]

# A local filesystem path, or an obstore store handle
# (``obstore.store.S3Store``, ``GCSStore``, ``AzureStore``, ``MemoryStore``, …).
AtlasSource = Union[str, os.PathLike[str], Any]

def init_tracing(filter: Optional[str] = ...) -> None:
    """Install a Rust `tracing` subscriber.

    Set the ``ATLAS_LOG`` or ``RUST_LOG`` environment variable to have this
    happen automatically on import.
    """
    ...

class AtlasWriter:
    """Builds one collection, then finishes.

    Nothing at the target is readable until :meth:`finish` runs. Use the writer
    as a context manager: a clean exit finishes the collection, and an exception
    abandons it entirely.

    >>> with AtlasWriter.create("/tmp/weather") as w:
    ...     w.add_xarray_dataset(ds, name="jan_2024")
    """

    @staticmethod
    def create(
        source: AtlasSource,
        codec: str = "zstd",
        block_target_size: Optional[int] = None,
    ) -> "AtlasWriter":
        """Start a collection at ``source``.

        Args:
            source: A local directory path, or an obstore store handle. With a
                store handle, the collection is written at the store's root.
            codec: Block compression: ``"zstd"`` (default), ``"lz4"``, or
                ``"none"``. Each block records its own codec, so a reader never
                needs to be told which was used.
            block_target_size: Target compressed block size in bytes. Defaults
                to 8 MiB. Chunks smaller than this share a block.

        Raises:
            OSError: The path could not be created.
        """
        ...

    def add_dataset(self, name: str) -> "DatasetWriter":
        """Begin a dataset. Call :meth:`DatasetWriter.finish` to commit it.

        Several datasets may be open at once; each enters the collection when it
        finishes, in finish order.

        Raises:
            FileExistsError: A dataset of this name was already added.
            ValueError: The name is empty, contains ``/``, is ``.`` or ``..``,
                starts with ``_``, or the writer has already finished.
        """
        ...

    def add_xarray_dataset(
        self,
        ds: xr.Dataset,
        name: str,
        chunks: Optional[dict[str, Sequence[int]]] = None,
        fill_value: Union[Any, dict[str, Any], None] = None,
    ) -> None:
        """Write an ``xarray.Dataset`` into the collection as ``name``.

        Every coordinate and data variable becomes an atlas array. Variable
        attributes become per-array attributes, dataset attributes become
        dataset-level attributes, and which variables were coordinates is
        recorded so :meth:`Atlas.coords` can report it.

        Dask-backed variables stream block by block, so the dataset need not fit
        in memory. The write is atomic: a failure partway leaves no trace of the
        dataset in the collection.

        Args:
            ds: The dataset to write.
            name: Name for it in the collection. Must be unique.
            chunks: Per-variable on-disk chunk shape, ``{var: [d0, d1, ...]}``.
                Defaults to the dask chunking, or one full-shape chunk for
                numpy-backed variables.
            fill_value: Overrides the per-array fill. A bare scalar applies to
                every numeric array; a ``{var: scalar}`` dict targets named
                variables, with ``None`` disabling the default for that
                variable. When omitted, float arrays default to a ``NaN`` fill
                and datetime arrays to ``NaT``.

        Raises:
            FileExistsError: A dataset of this name was already added.
            NotImplementedError: A variable has a dtype atlas cannot store.

        Note:
            ``dtype`` mapping: the signed and unsigned integer widths and
            ``float32``/``float64`` map straight through;
            ``datetime64[ns]`` becomes ``timestamp_nanoseconds``;
            ``timedelta64`` becomes ``int64`` nanoseconds plus a marker
            attribute; object, bytes, and unicode arrays become variable-length
            strings. Missing string cells cannot be stored as null and are
            replaced with the fill value, with a warning.
        """
        ...

    def dataset_count(self) -> int:
        """How many datasets have been committed so far."""
        ...

    @property
    def closed(self) -> bool:
        """Whether :meth:`finish` has run."""
        ...

    def finish(self) -> None:
        """Write the footer and close the collection.

        Nothing at the target is readable before this returns, and the
        collection cannot be changed after it.

        Raises:
            ValueError: The writer already finished.
        """
        ...

    def __enter__(self) -> "AtlasWriter": ...
    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        """Finishes the collection on a clean exit; abandons it on an exception."""
        ...

class DatasetWriter:
    """Builds one dataset inside a collection.

    Declare arrays with :meth:`define_array`, fill them with
    :meth:`write_array` in any order and any number of slabs, then
    :meth:`finish`. Nothing reaches the collection until then.
    """

    @property
    def name(self) -> str:
        """The dataset's name."""
        ...

    def list_arrays(self) -> list[str]:
        """Names of the arrays declared so far, in definition order."""
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
        """Declare an array.

        Args:
            name: Array name, unique within this dataset.
            dtype: One of ``bool``, ``int8``…``int64``, ``uint8``…``uint64``,
                ``float32``, ``float64``, ``string``, ``binary``,
                ``timestamp_nanoseconds``, ``list[<inner>]``, or
                ``fixed_size_list[<inner>,<n>]``. Short forms (``i32``, ``f64``,
                ``str``, ``datetime64[ns]``) are accepted.
            dims: Dimension names, one per axis.
            shape: Logical shape.
            chunk_shape: On-disk chunk shape. Defaults to ``shape``, storing the
                array as a single chunk.
            fill_value: What a read returns for elements never written. Checked
                against ``dtype``.

        Raises:
            FileExistsError: The array is already defined in this dataset.
            ValueError: The name or the dtype string is invalid.
            TypeError: ``fill_value`` does not match ``dtype``.
            OverflowError: ``fill_value`` is out of range for ``dtype``.
            NotImplementedError: The dtype is not writable from Python.
        """
        ...

    def write_array(self, name: str, start: Sequence[int], data: Any) -> None:
        """Write ``data`` into ``name`` with its origin at ``start``.

        The region may span chunks and need not be chunk-aligned. ``data`` must
        be a C-contiguous numpy array whose dtype matches the declared one. For
        a ``timestamp_nanoseconds`` array, pass ``arr.view(np.int64)``.

        Raises:
            KeyError: The array is not defined in this dataset.
            TypeError: ``data`` has the wrong dtype.
            ValueError: ``data`` is not C-contiguous.
        """
        ...

    def set_attribute(self, key: str, value: Any, dtype: Optional[str] = None) -> None:
        """Attach a dataset-level attribute. A repeated key replaces the value.

        ``dtype`` narrows how a Python scalar is stored, for example
        ``"int32"`` for a Python ``int``.
        """
        ...

    def set_array_attribute(
        self, array: str, key: str, value: Any, dtype: Optional[str] = None
    ) -> None:
        """Attach an attribute to one array, which must already be defined.

        Raises:
            KeyError: The array is not defined in this dataset.
        """
        ...

    def finish(self) -> None:
        """Commit the dataset into the collection.

        Raises:
            ValueError: This dataset, or the collection, already finished.
        """
        ...

    def abort(self) -> None:
        """Discard the dataset. It never enters the collection."""
        ...

    def __enter__(self) -> "DatasetWriter": ...
    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        """Commits on a clean exit; discards the dataset on an exception."""
        ...

class Atlas:
    """An open collection, read as metadata.

    Opening reads the container footer and the deletion mask, and nothing else,
    so listing datasets and inspecting schemas costs one range read however
    large the collection is.

    There is no array read here by design. Python writes collections; the Rust
    API reads their data.
    """

    @staticmethod
    def open(source: AtlasSource) -> "Atlas":
        """Open an existing collection.

        Args:
            source: A local directory path, or an obstore store handle.

        Raises:
            ValueError: There is no collection at ``source``, or it was written
                by an incompatible version.
            RuntimeError: The container or its deletion mask is damaged.
        """
        ...

    def list_datasets(self) -> list[str]:
        """Names of the live datasets, in write order. Deleted ones are omitted."""
        ...

    def list_arrays(self) -> list[str]:
        """Every distinct array name across the live datasets, sorted."""
        ...

    def dataset_exists(self, name: str) -> bool:
        """Whether a live dataset of this name exists."""
        ...

    def dataset_count(self) -> int:
        """How many datasets are live."""
        ...

    @property
    def created_unix_ms(self) -> int:
        """When the collection was written, in milliseconds since the epoch."""
        ...

    def dataset(self, name: str) -> "DatasetView":
        """A metadata view of one dataset.

        Raises:
            KeyError: No such dataset, or it has been deleted.
        """
        ...

    def coords(self, name: str) -> list[str]:
        """Names of the variables that were xarray coordinates in ``name``.

        Empty for a dataset that atlas did not write from xarray.
        """
        ...

    def attributes(self, name: str) -> dict[str, Any]:
        """Dataset-level attributes of ``name``, decoded back to Python values.

        Values that were JSON-encoded on the way in are restored. The reserved
        coordinate marker is omitted; see :meth:`coords`.
        """
        ...

    def array_attributes(self, name: str, array: str) -> dict[str, Any]:
        """Attributes of one array of ``name``, decoded back to Python values."""
        ...

    def delete_dataset(self, name: str) -> None:
        """Hide a dataset by adding it to the deletion mask.

        The container is not touched: the dataset's bytes stay where they are
        and no ordinal moves. Rewrite the collection to reclaim the space.

        Concurrent deletions are last-writer-wins; serialize them if that
        matters.

        Raises:
            KeyError: No such dataset, or it is already deleted.
        """
        ...

    def __contains__(self, name: str) -> bool: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[str]: ...

class DatasetView:
    """A read-only, metadata-only view of one dataset.

    Every method here is answered from the collection footer, which was already
    read when the collection was opened. None of them touch the store.
    """

    @property
    def name(self) -> str:
        """The dataset's name."""
        ...

    @property
    def ordinal(self) -> int:
        """The dataset's position in the collection.

        Stable for the life of the container: deleting a dataset does not
        renumber the others.
        """
        ...

    @property
    def segment_range(self) -> tuple[int, int]:
        """``(start, end)`` byte offsets of this dataset's segment in
        ``data.atlas``. Those bytes are a complete ``array-format`` file."""
        ...

    def list_arrays(self) -> list[str]:
        """Array names, in definition order."""
        ...

    def array_meta(self, array: str) -> Optional[dict[str, Any]]:
        """``{"dtype", "shape", "chunk_shape", "dimension_names", "fill_value"}``
        for ``array``, or ``None`` if this dataset does not declare it."""
        ...

    def array_fill_value(self, array: str) -> Optional[Any]:
        """What a read returns for elements never written, or ``None``."""
        ...

    def attributes(self) -> dict[str, Any]:
        """Dataset-level attributes, in the order they were set.

        Values are as stored. :meth:`Atlas.attributes` decodes them.
        """
        ...

    def get_attribute(self, key: str) -> Optional[Any]:
        """One dataset-level attribute, or ``None``."""
        ...

    def array_attributes(self, array: str) -> dict[str, Any]:
        """Attributes of one array, in the order they were set."""
        ...

    def get_array_attribute(self, array: str, key: str) -> Optional[Any]:
        """One attribute of one array, or ``None``."""
        ...

    def __contains__(self, array: str) -> bool: ...
    def __len__(self) -> int: ...
