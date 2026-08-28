# Supported dtypes

## Array dtypes

Pass these as the `dtype=` argument to
[`DatasetWriter.define_array`](../reference/dataset-writer.md). The numpy type
on the right is what `write_array` expects.

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

An unwritten cell reads back as the fill value and costs no bytes on disk.
`array_fill_value` reports it:

```python
collection.dataset("jan").array_fill_value("temp")   # nan
```

When ingesting via [`add_xarray_dataset`](xarray.md#fill-values-and-missing-data)
you don't pass these by hand — float arrays default to a `NaN` fill, datetimes
to `NaT`, and strings to `""`, so cells masked by `mask_and_scale=True` are
recorded as null automatically.

## 0-D scalar arrays

Every dtype above also works at `shape=[]` (and `chunk_shape=[]`,
implicitly). The numpy round-trip is a 0-D ndarray:

```python
ds.define_array("scale", dtype="float64", dims=[], shape=[])
ds.write_array("scale", start=[], data=np.array(3.14, dtype=np.float64))
```

0-D arrays are useful for things like NetCDF `TRAJECTORY` identifiers or
single-value metadata that's logically *array data*, not an attribute.

## Reserved for a later release

| Type | Status |
|---|---|
| `bool` arrays | **Attribute-only** — the underlying `array-format` crate does not support them as array elements. |
| `binary` (variable-length bytes) | Attribute-only; not exposed as an array type yet. |
| `list[...]` | Attribute-only; not exposed as an array type yet. |
| `fixed_size_list[..., N]` | Not exposed yet. |

All four work as [attribute](attributes.md) values. If you need a packed bool
array today, use `uint8` and document the convention.

## Reading values

Not from Python — array data is read through the Rust API. The dtype table
above still tells you what a Rust `read_array::<T>` will hand back:
`"float32"` is `f32`, `"timestamp_nanoseconds"` is `TimestampNs`, `"string"`
is `String`. See [Reading data](reading-data.md).
