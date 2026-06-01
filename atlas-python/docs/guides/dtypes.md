# Supported dtypes

## Array dtypes

Pass these as the `dtype=` argument to
[`DatasetView.define_array`](../reference/dataset-view.md). The numpy type
on the right is what `write_array` / `read_array` expect and return.

| atlas dtype | numpy dtype | Range / notes |
|---|---|---|
| `"int8"`   | `np.int8`   | −128 … 127 |
| `"int16"`  | `np.int16`  | −32 768 … 32 767 |
| `"int32"`  | `np.int32`  | ±2 147 483 647 |
| `"int64"`  | `np.int64`  | ±9.2e18 |
| `"uint8"`  | `np.uint8`  | 0 … 255 |
| `"uint16"` | `np.uint16` | 0 … 65 535 |
| `"uint32"` | `np.uint32` | 0 … 4.3e9 |
| `"uint64"` | `np.uint64` | 0 … 1.8e19 |
| `"float32"` | `np.float32` | IEEE-754 single |
| `"float64"` | `np.float64` | IEEE-754 double |
| `"timestamp_nanoseconds"` (aliases: `"timestamp_ns"`, `"datetime64[ns]"`) | `np.datetime64[ns]` | int64 ns since the Unix epoch; round-trips bit-identically |
| `"string"` | `object` (Python `str`) | Variable-length UTF-8. `\|S<n>` / `\|U<n>` inputs are accepted and stored vlen. |

## Rules `write_array` enforces

- **Exact dtype match.** No silent widening. `int32` data into an `int64`
  array raises `TypeError`. Promote explicitly with `data.astype(np.int64)`.
- **C-contiguous buffer.** Pass `np.ascontiguousarray(data)` if the array
  came from a slice or a transpose.
- **`start + data.shape ≤ array.shape`** per axis. Out-of-bounds writes
  raise.
- **Strings.** `object` arrays of Python `str` (or `bytes`) are accepted.
  Fixed-size `\|S5` / `\|U10` arrays are converted to vlen UTF-8 on write
  and come back as Python `str`. Surrogate-escaped strings (common from
  netCDF backends) are sanitised on the way in.
- **`datetime64`.** Only the `[ns]` resolution is supported. Other
  resolutions raise — convert with `data.astype("datetime64[ns]")` first.

## Fill values

`fill_value=` on `define_array` must match the array dtype:

```python
ds.define_array("temp", dtype="float32", ..., fill_value=float("nan"))
ds.define_array("count", dtype="int32", ..., fill_value=-1)
ds.define_array("label", dtype="string", ..., fill_value="UNKNOWN")
```

- Integer / `timestamp_*` arrays: Python `int`, range-checked at the call
  site. Out-of-range raises `OverflowError`; wrong type raises `TypeError`.
- Float arrays: Python `float` (or `int`, which is coerced). `float("nan")`
  is allowed.
- String arrays: Python `str`.

Reading an **unwritten** cell returns the fill value. Any **written** cell
equal to the fill value is counted as a null in
[`array_stats`](stats.md) — this is how the null count works for
`NaN`-filled float arrays.

## 0-D scalar arrays

Every dtype above also works at `shape=[]` (and `chunk_shape=[]`,
implicitly). The numpy round-trip is a 0-D ndarray:

```python
ds.define_array("scale", dtype="float64", dims=[], shape=[])
ds.write_array("scale", start=[], data=np.array(3.14, dtype=np.float64))

scalar = ds.read_array("scale")     # -> np.ndarray, shape=(), dtype=float64
```

0-D arrays are useful for things like NetCDF `TRAJECTORY` identifiers or
single-value metadata that's logically *array data*, not an attribute.

## Reserved for a later release

| Type | Status |
|---|---|
| `bool` arrays | **Attribute-only** this release (limitation of the underlying `array-format` crate). |
| `binary` (variable-length bytes) | Reserved; not exposed yet. |
| `list[...]` (variable-length list of T) | Reserved; not exposed yet. |
| `fixed_size_list[..., N]` | Reserved; not exposed yet. |

If you need a packed bool array today, use `uint8` and document the
convention.

## Compatibility with numpy operations

The numpy arrays that come back from `read_array` are owned buffers, not
views into the atlas-managed memory. You can mutate / reshape / `astype`
them freely; the next `read_array` call returns a fresh buffer.
