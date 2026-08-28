# Read path

## What opening costs

```rust
let atlas = Atlas::open_path(dir).await?;
```

One range read of the container's tail, plus one for the deletion mask if it
exists. That is the whole open, for a collection of ten datasets or a million.

Everything below is then answered from memory, with no further I/O:

```rust
atlas.list_datasets();
atlas.list_arrays();
atlas.dataset_count();

let ds = atlas.dataset("jan_2024")?;   // no I/O
ds.schema();
ds.array_meta("temperature");
ds.attributes();
ds.array_attributes("temperature");
ds.array_fill_value("temperature");
```

`tests/integration.rs` asserts this with a request-counting `ObjectStore`:
opening a collection larger than the 64 KiB tail probe issues at most two reads
and transfers at most 64 KiB, and the metadata calls above issue none.

This is the property the whole format is arranged around, and it is why the
Python bindings expose metadata but not data — serving a catalogue of what a
collection holds never needs to touch array bytes.

## What reading data costs

```rust
let temp = ds.read_array::<f32>("temperature", vec![10, 20], vec![2, 2]).await?;
```

The first data read on a dataset opens its segment: two small range reads for
the `array-format` trailer and footer. That handle is cached per collection, so
it happens once per dataset regardless of how many arrays you then read.

The read itself fetches only the chunks the requested region overlaps. A 2×2
window out of a 32×64 array chunked 8×16 fetches one chunk, not sixteen. Pass
empty `start` and `shape` to read the whole array.

Elements that were never written are materialized from the fill value and cost
no I/O at all — an array declared and never written reads back as fill without
touching the container.

## SegmentStore

`array-format` opens a file through an `ObjectStore` and a path, but a segment
is a byte range inside a larger file. `SegmentStore` bridges that: it implements
`ObjectStore` over exactly one virtual object mapped to
`container[offset .. offset + len]`.

Range requests are translated and forwarded; nothing is buffered on the way
through. Three behaviours matter as much as the translation:

- **`list` returns nothing**, so sidecar discovery finds no delta layers. A
  segment is always a single compacted base.
- **Any other path is `NotFound`**, so the statistics probe for `seg<n>.stats`
  comes back empty instead of erroring out of the open.
- **A range ending past the segment is clamped**; a range starting past it is an
  error. Without the clamp a read could walk into the next dataset's bytes.

Every write method returns `NotSupported`. A collection is immutable; nothing
should ever try.

### The virtual name carries the ordinal

Segments are named `seg0.af`, `seg1.af`, and so on. That is not cosmetic. The
block cache is keyed by `(path, block_id)` and shared across every segment in a
collection — a name shared between segments would let one dataset's block 0
answer another dataset's read for block 0. It did, once, and the tests now catch
it.

## Caching

One `DeltaCache` per `Atlas` handle, shared by every segment: 256 MiB of
decompressed blocks and 64 MiB of raw I/O slabs. Reading a second array from a
dataset whose segment is already open costs strictly fewer requests than the
first, which `tests/integration.rs` also asserts.

## Type safety

`read_array::<T>` checks `T` against the dtype the footer records, and against
the dtype the segment records, and raises `CorruptCollection` if the two
disagree with each other. Asking for the wrong type is an error rather than a
reinterpretation of the bytes.

## Deletion

`delete_dataset` is the only operation on an open collection that writes
anything, and it writes only the mask:

1. Resolve the name to an ordinal, or `DatasetNotFound`.
2. Re-read the mask from the store, so a deletion made elsewhere since this
   handle opened is preserved rather than clobbered.
3. Insert the ordinal, write the whole mask back.
4. Update the in-memory set, so the change is visible on this handle at once.

Step 2 is why two handles can each delete a different dataset and both
deletions survive. It does not make concurrent deletes safe — two deletes that
interleave between read and write still lose one. See
[format.md](format.md#deletion-mask).

## Reading from Python

You cannot. `Atlas` in Python lists datasets, reports schemas, and reads
attributes; there is no `read_array`. Array data is read through the Rust API.

The split is deliberate. Python's job in this design is building collections
from xarray and serving their metadata; pulling array bytes through the GIL was
never the fast path. `tests/cross_fixture.rs` keeps the two honest — Python
writes a collection, Rust reads it back and checks every value.
