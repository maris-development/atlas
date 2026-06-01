# Codecs and metadata

`Atlas.create` takes three independent knobs that control the on-disk
encoding:

```python
atlas.Atlas.create(
    path,
    codec="zstd",            # array codec — see below
    meta_format="json",      # metadata format
    meta_compression="none", # metadata compression
)
```

All three are auto-detected on `Atlas.open` from the on-disk filename, so
you never pass them when reopening.

## Array codec — `codec=`

| Value | When to pick it |
|---|---|
| `"zstd"` *(default)* | Best ratio at moderate CPU. Pick this unless you have a reason not to. |
| `"lz4"` | ~2× larger files but faster to decompress. Worth it for read-heavy scan loops where compressed-bytes-per-second beats raw size. |
| `"none"` / `"uncompressed"` | Fastest write path, no size reduction. Tiny stores, or when you'll compress the whole directory externally. |

The codec is recorded **per-array**, so reading is automatic regardless of
which codec the store was opened with. Existing blocks always decompress
with whichever codec wrote them; the `codec=` kwarg only affects *new*
blocks. This means you can switch codecs mid-life without rewriting.

See [`examples/05_codecs.py`](https://github.com/maris-development/atlas/blob/main/atlas-python/examples/05_codecs.py)
for a head-to-head on a smooth `float32` field (smooth data exposes codec
differences; pure noise compresses uniformly badly across all three).

## Metadata format — `meta_format=`

The metadata file is read on every `Atlas.open` and written on every
`flush`. It contains the dataset registry, every array schema, every
attribute, and every persisted stat — for stores with thousands of
datasets it can grow into tens of MB.

| Value | Filename | Notes |
|---|---|---|
| `"json"` *(default)* | `atlas.json` | Human-readable; greppable; backwards compatible. |
| `"msgpack"` / `"mp"` | `atlas.msgpack` | ~30–50% smaller, faster to parse, **not human-readable**. |

JSON is the default because the wins from msgpack only become visible on
stores big enough that the metadata file is a meaningful fraction of total
size. For small stores msgpack saves milliseconds you won't notice.

## Metadata compression — `meta_compression=`

Applied on top of the encoded metadata file:

| Value | Filename suffix | Notes |
|---|---|---|
| `"none"` / `"uncompressed"` *(default)* | (no suffix) | |
| `"zstd"` | `.zst` (e.g. `atlas.json.zst`) | Best ratio. |
| `"lz4"` | `.lz4` (e.g. `atlas.msgpack.lz4`) | Faster decode, larger file. |

Mostly useful for stores with thousands of datasets on a high-latency
object store: the metadata file is read in full on every open, so trimming
30–50% off the wire size pays for itself.

[`examples/04_meta_formats.py`](https://github.com/maris-development/atlas/blob/main/atlas-python/examples/04_meta_formats.py)
walks all six combinations on a 30-dataset × 4-array store and prints the
size ratios.

## Auto-detection on open

`Atlas.open(path)` looks for any of `atlas.json`, `atlas.json.zst`,
`atlas.json.lz4`, `atlas.msgpack`, `atlas.msgpack.zst`, `atlas.msgpack.lz4`
in the directory and picks the first one it finds. You **don't** pass
`meta_format=` / `meta_compression=` to `open` — the filename is the
source of truth.

```python
# Whatever combination the store was created with, this just works.
atlas = atlas.Atlas.open("/tmp/store")
```

## Picking a combination

For most use cases the defaults (`codec="zstd"`, `meta_format="json"`,
`meta_compression="none"`) are right. Reach for the alternatives when:

- **Lots of small datasets** (1000+ datasets, simple schemas) →
  `meta_format="msgpack"`, optionally `meta_compression="zstd"`. The
  metadata file shrinks dramatically; reads are unaffected because the
  array codec is independent.
- **Read-heavy scan loops** over already-warm chunks → `codec="lz4"`. The
  decompression rate is what matters.
- **Already compressing the whole directory** (e.g. tarballed and shipped
  through an object store with HTTP gzip) → `codec="none"` for the arrays
  and let the outer compression do the work.
