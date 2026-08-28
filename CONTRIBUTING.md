# Contributing to ATLAS

What the project is, how the pieces fit, and how to get a development
environment working.

For end-user docs see [README.md](README.md) (Rust) and
[atlas-python/README.md](atlas-python/README.md) (Python).

---

## What this is

**ATLAS** (Aggregated Tensor Large Array Store) keeps thousands of named
datasets in one immutable file. A dataset is a set of named N-dimensional
arrays with attributes — the shape a NetCDF file or an `xarray.Dataset` has.

The design goal is that *knowing what a collection holds* costs one request,
however many datasets it holds.

The repository is a Cargo workspace with two crates:

| Crate | Purpose |
|---|---|
| `atlas-rust` (workspace root, `[lib] name = "atlas"`) | The format, the writer, the reader |
| `atlas-python` ([atlas-python/](atlas-python/)) | PyO3 bindings + xarray integration |

---

## Architecture

### On-disk layout

```text
my_collection/
├── data.atlas      ATLS │ segment │ segment │ … │ footer │ trailer
└── deleted.mask    optional: ordinals of deleted datasets
```

One write-once file. Each dataset is a contiguous segment — itself a complete
[`array-format`](https://github.com/robinskil/array-format) file — and the
footer records every dataset's name, segment byte range, schema, and attributes.

Byte-level detail is in [docs/format.md](docs/format.md). The rest of
[docs/](docs/) covers the layers, the data model, and the two paths.

**The format is Rust only.** `atlas-python` holds no format knowledge — grep it
for `ATLS` and you get nothing. One implementation of the bytes, one place for a
bug to live. Keep it that way.

### Rust crate

| Path | Role |
|---|---|
| [src/lib.rs](src/lib.rs) | Public re-exports, `validate_name`, thread-safety asserts |
| [src/format/mod.rs](src/format/mod.rs) | Container framing: magic, header, trailer |
| [src/format/footer.rs](src/format/footer.rs) | `CollectionFooter`, `DatasetEntry`, `AttrS`, the interner |
| [src/format/mask.rs](src/format/mask.rs) | The deletion mask codec |
| [src/format/segment_store.rs](src/format/segment_store.rs) | `ObjectStore` adapter over one byte range |
| [src/writer/mod.rs](src/writer/mod.rs) | `AtlasWriter`, `DatasetWriter` |
| [src/reader/mod.rs](src/reader/mod.rs) | `Atlas`, `DatasetView` |
| [src/schema/](src/schema/) | `ArraySchema`, `DatasetSchema`, `Attr`, dtype serde |
| [src/config.rs](src/config.rs) | `Codec`, `WriterConfig` |
| [src/error.rs](src/error.rs) | `Error` / `Result` |

The API is async (tokio). Reads take no locks — the data is immutable. Writes
share one `tokio::sync::Mutex` over the output stream, held only for a dataset's
append.

### Python bindings

Mixed Python/Rust maturin layout:

```text
atlas-python/
├── Cargo.toml               cdylib named `_atlas`
├── pyproject.toml           maturin build backend
├── python/atlas/
│   ├── __init__.py          re-exports + accessor registration
│   ├── __init__.pyi         type stubs (PEP 561) — the authoritative contract
│   ├── py.typed             marker
│   ├── store.py             the Atlas / AtlasWriter facades
│   └── xarray.py            xarray write path + ds.atlas accessor
├── src/
│   ├── lib.rs               #[pymodule] wiring
│   ├── runtime.rs           shared OnceLock<tokio::Runtime>
│   ├── error.rs             atlas::Error → PyErr
│   ├── source.rs            AtlasSource (path | obstore handle), codec parsing
│   ├── dtype.rs             dtype string ⇄ DType
│   ├── attr.rs              Python value ⇄ Attr
│   ├── writer.rs            PyAtlasWriter, PyDatasetWriter
│   └── reader.rs            PyAtlas, PyDatasetView
├── tests/                   pytest, plus make_fixture.py
└── examples/                runnable scripts
```

Key points:

- **Python writes; Rust reads array data.** Python builds collections and reads
  their *metadata*. There is no `read_array` on the Python side. See
  [docs/read-path.md](docs/read-path.md) for the reasoning.
- **Sync Python API over a tokio runtime.** Each blocking call uses
  `py.detach(|| runtime().block_on(...))` so other Python threads keep running.
- **numpy zero-copy on the numeric path**, via the `numpy` crate. Strings are
  the exception — they are extracted element by element.
- **Type dispatch via macros.** `define_array` / `write_array` are generic over
  `T: ArrayElement`; the bindings dispatch at runtime through
  `numeric_dispatch!` in [atlas-python/src/writer.rs](atlas-python/src/writer.rs).

---

## Prerequisites

| Tool | Why |
|---|---|
| **Rust** stable, 1.85+ (edition 2024) | Build the workspace |
| **Python ≥ 3.10** | The wheel targets `abi3-py310` |
| **`maturin`** | Build the extension. `pip install maturin` |

---

## Rust: build and test

```bash
cargo build --workspace
cargo test -p atlas-rust
```

`cargo test --workspace` does not work: `atlas-python` links
`pyo3/extension-module`, so its test binary has no libpython to link against.
Test the core crate; test the bindings through pytest.

On macOS, `cargo build --workspace` also fails to link the cdylib for the same
reason. Use `cargo check --workspace` for type checking and `maturin develop`
for a real build.

Examples:

```bash
cargo run --example lifecycle
cargo run --example sensor_fleet
cargo run --example weather_store
```

---

## Python: build and test

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip maturin

maturin develop --extras test --manifest-path atlas-python/Cargo.toml
pytest atlas-python/tests -v
```

The install is editable, so changes under `python/atlas/*.py` take effect at
once. **Rust changes need `maturin develop` again** — otherwise pytest runs
against the previously built binary.

---

## Test fixtures

Two committed fixtures pin behaviour that a round-trip test would miss.

**`tests/fixtures/golden_v1/`** — a v1 container, read back by
[tests/golden.rs](tests/golden.rs) with every value asserted. If a change breaks
compatibility with an existing container, this catches it. Regenerate only when
you intend to break the format, which means bumping `FORMAT_VERSION`:

```bash
cargo test --test golden -- --ignored regenerate
```

The writer's output is deliberately not compared byte for byte — zstd makes no
promise of stable output across versions. Only the framing this crate produces
itself is asserted exactly.

**`tests/fixtures/from_python/`** — written by
[atlas-python/tests/make_fixture.py](atlas-python/tests/make_fixture.py) and read
back by [tests/cross_fixture.rs](tests/cross_fixture.rs). This is the only thing
verifying that the bytes the Python xarray layer writes are the bytes it meant,
now that pytest cannot read arrays. Regenerate after changing the write path:

```bash
python atlas-python/tests/make_fixture.py
```

---

## Common workflows

### Adding an array dtype

1. Confirm `array-format` supports it as an element type.
2. Add the dispatch arm in `numeric_dispatch!`
   ([atlas-python/src/writer.rs](atlas-python/src/writer.rs)), or an explicit
   branch alongside String / TimestampNs.
3. Extend [atlas-python/src/dtype.rs](atlas-python/src/dtype.rs) to parse the
   name.
4. Add it to `_NUMPY_TO_ATLAS` in
   [atlas-python/python/atlas/xarray.py](atlas-python/python/atlas/xarray.py) if
   it has a numpy equivalent.
5. Test in both suites, and extend the cross fixture if it is worth pinning.

### Exposing a new method to Python

1. Implement it on `Atlas` / `DatasetView` / the writers in `src/`.
2. Wrap it in [atlas-python/src/reader.rs](atlas-python/src/reader.rs) or
   [writer.rs](atlas-python/src/writer.rs), releasing the GIL with `py.detach`
   for anything that blocks.
3. Add the stub to
   [atlas-python/python/atlas/\_\_init\_\_.pyi](atlas-python/python/atlas/__init__.pyi)
   — that file is the contract, not `store.py`.
4. `maturin develop`, then write a test.

### Touching the on-disk format

Any change to the framing, the footer struct, or the mask is **breaking** — the
footer is compact MessagePack, so field order is part of the format.

1. Bump `FORMAT_VERSION` in [src/format/mod.rs](src/format/mod.rs).
2. Regenerate the golden fixture and update `tests/golden.rs` to the new
   version.
3. Update [docs/format.md](docs/format.md) — it is a specification, not a
   summary.
4. Note it in [docs/migration.md](docs/migration.md).

There is no in-place migration path by design. Collections are rewritten.

---

## Code style

- **Rust**: `cargo fmt` before committing, default rustfmt.
  `cargo clippy --all-targets` should be clean.
- **Python**: 4-space indentation, imports grouped stdlib / third-party / local.
  No formatter is enforced; match what is there.
- **Comments explain why, not what.** The code already shows what. A comment
  earns its place by recording a constraint, a trade-off, or a trap — the kind
  of thing the next reader would otherwise have to rediscover.
- **Tests are named as claims.** `an_end_past_the_segment_is_clamped_not_leaked`
  says what it protects; `test_read_3` does not.

---

## Pull requests

1. Branch from `main`.
2. `cargo test -p atlas-rust`, `cargo clippy --all-targets`, and
   `pytest atlas-python/tests` must pass.
3. If you change the format, the public API, or user-visible behaviour, update
   the relevant docs in the same PR.
4. Keep commits focused; squash fixups before review.
