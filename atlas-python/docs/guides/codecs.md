# Codecs

One knob, set when you start writing:

```python
atlas.AtlasWriter.create(path, codec="zstd")
```

| Value | When to pick it |
|---|---|
| `"zstd"` *(default)* | Best ratio at moderate CPU. Pick this unless you have a reason not to. |
| `"lz4"` | Larger files, faster to decompress. Worth it for scan-heavy reading where compressed bytes per second beats raw size. |
| `"none"` / `"uncompressed"` | Fastest write path, no size reduction. Tiny collections, or when you compress the whole file externally. |

## Readers are never told

Every block records the codec that produced it. `Atlas.open` takes no codec
argument, and none is needed — a reader decodes whatever it finds.

```python
with atlas.AtlasWriter.create(path, codec="lz4") as w:
    ...

atlas.Atlas.open(path)   # no codec argument, works
```

The value stored in the footer is informational only: it says what the writer
was configured with, not what any particular block used.

## Block size

```python
atlas.AtlasWriter.create(path, block_target_size=8 * 1024 * 1024)
```

Blocks are the unit of compression. Chunks smaller than the target share a
block; a larger chunk gets its own. The default of 8 MiB is a reasonable
trade-off between compression ratio (bigger blocks compress better) and read
amplification (a read must decompress a whole block).

Lower it if your chunks are small and reads are highly selective. Most callers
should leave it alone.

## What is not configurable

The collection footer has exactly one wire format — MessagePack,
zstd-compressed — pinned by the format version. There is no metadata-format or
metadata-compression knob: one decision fewer to make, one combination fewer to
test, and no way for a reader to guess wrong.

## Measuring

[`examples/05_codecs.py`](https://github.com/maris-development/atlas/blob/main/atlas-python/examples/05_codecs.py)
writes the same data under all three and prints file sizes. Use a realistic
field: smooth data exposes codec differences, while pure noise compresses
uniformly badly whatever you pick.
