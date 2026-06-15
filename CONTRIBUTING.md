# Contributing to ATLAS

Thanks for your interest. This document walks through what the project is, how the pieces fit, and how to set up a development environment.

For end-user docs see the top-level [README.md](README.md) (Rust crate) and [pyatlas/README.md](pyatlas/README.md) (Python bindings).

---

## What this is

**ATLAS** (Aggregated Tensor Large Array Store) is a directory-based store for many similarly-shaped, named, N-dimensional arrays. The design goal is fast cross-dataset variable scans — opening a single file to read variable `X` across N datasets, rather than N files.

The repository is a Cargo workspace containing two crates:

| Crate | Purpose |
| --- | --- |
| `atlas` (workspace root) | Rust library implementing the store and on-disk format. |
| `pyatlas` ([pyatlas/](pyatlas/)) | PyO3 Python bindings + xarray integration. |

---

## How it works (architecture)

### On-disk layout

```
my_store/
├── atlas.json               ← dataset registry + per-dataset attributes (JSON)
├── temperature/
│   └── data.af              ← ArrayFile: every dataset's "temperature" in one file
├── pressure/
│   └── data.af
└── time/
    └── data.af
```

Every variable name owns one physical `.af` file shared by all datasets that define it (variable-first). The `.af` format comes from the [`array-format`](https://github.com/robinskil/array-format) crate — it's a columnar, chunk-oriented binary container with per-block compression and persisted statistics.

`atlas.json` is the catalog. It records:
- Store version, store-level codec.
- Each dataset: its array schemas (dtype, shape, chunk shape, dimension names) and typed attributes.

### Rust crate (`atlas`)

| File | Role |
| --- | --- |
| [src/lib.rs](src/lib.rs) | Public re-exports; `validate_name`; thread-safety asserts. |
| [src/store.rs](src/store.rs) | The `Atlas` struct: open/create, dataset CRUD, path-based wrappers (`open_path` / `create_path`). |
| [src/dataset.rs](src/dataset.rs) | The `DatasetView` struct: define/read/write arrays, attributes, flush, compact. Also the shared `ArrayFile` cache. |
| [src/meta.rs](src/meta.rs) | `atlas.json` (de)serialisation. |
| [src/schema.rs](src/schema.rs) | `ArraySchema` and the typed `Attr` enum. |
| [src/config.rs](src/config.rs) | `StoreConfig`, `Codec` (`Zstd` / `Lz4` / `Uncompressed`). |
| [src/error.rs](src/error.rs) | `Error` / `Result`. |

The API is async (tokio). Each physical array file is guarded by a `tokio::sync::RwLock` — reads share, writes are exclusive. The cache map uses `parking_lot::RwLock` and is never held across an `await`.

### Python bindings (`pyatlas`)

Mixed Python/Rust maturin layout:

```
pyatlas/
├── Cargo.toml               ← cdylib named `_pyatlas`
├── pyproject.toml           ← maturin build backend
├── python/pyatlas/
│   ├── __init__.py          ← re-exports + xarray accessor registration
│   ├── __init__.pyi         ← type stubs (PEP 561)
│   ├── py.typed             ← marker
│   └── xarray.py            ← xarray integration + ds.atlas accessor
├── src/
│   ├── lib.rs               ← #[pymodule] _pyatlas wiring
│   ├── runtime.rs           ← shared OnceLock<tokio::Runtime>
│   ├── error.rs             ← atlas::Error → PyErr mapping
│   ├── dtype.rs             ← dtype string parsing & DType ↔ name
│   ├── attr.rs              ← Attr ↔ Python primitive conversion
│   ├── store.rs             ← PyAtlas (wraps atlas::Atlas)
│   └── dataset.rs           ← PyDatasetView (wraps atlas::DatasetView)
├── tests/                   ← pytest tests
└── examples/                ← runnable example scripts
```

Key design points:
- **Sync Python API, backed by an internal multi-threaded tokio runtime.** Each blocking call uses `py.allow_threads(|| RT.block_on(...))` so other Python threads can run.
- **Type dispatch via macros.** Atlas's read/write methods are generic over dtype (`T: ArrayElement`); the bindings dispatch at runtime via a `numeric_dispatch!` macro in [pyatlas/src/dataset.rs](pyatlas/src/dataset.rs).
- **`numpy`-zero-copy on the numeric path.** Python `np.ndarray` ↔ `ndarray::ArrayView` via the `numpy` crate.
- **xarray integration** lives in [pyatlas/python/pyatlas/xarray.py](pyatlas/python/pyatlas/xarray.py). The `Atlas.add_xarray_dataset` Rust pymethod and the `ds.atlas.write` accessor both delegate to the `_write_xarray_new_dataset` helper. dask-backed variables are streamed one chunk at a time via `arr.blocks[idx].compute()`.

---

## Prerequisites

| Tool | Why |
| --- | --- |
| **Rust** (stable, 1.85+ for edition 2024 in the atlas crate) | Build the workspace. |
| **Python ≥ 3.9** | Run / build pyatlas (wheel targets `abi3-py39`). |
| **`maturin`** | Build the Python extension. `pip install maturin`. |

Install Rust via [rustup.rs](https://rustup.rs). Python 3.13 is what the maintainers develop against.

---

## Building & testing the Rust crate

```bash
cargo build                       # compile the atlas crate + workspace
cargo test                        # run all tests (60 in atlas, plus pyatlas check-only)
cargo test -p atlas               # just the atlas crate (50 unit + 10 integration)
cargo run --example lifecycle     # try one of the Rust examples
```

Examples live in [examples/](examples/) — `lifecycle.rs`, `sensor_fleet.rs`, `weather_store.rs`.

---

## Building & installing the Python library

The Python extension links against libpython, so plain `cargo build -p pyatlas` will fail at link time. Use `maturin develop` (which sets the right flags and installs into the active virtualenv).

### One-time setup

```bash
# From the repo root
python3.13 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip maturin
```

`pyatlas` already lists `numpy`, `xarray`, and `dask` as required runtime deps, so `maturin develop` will pull them in automatically.

### Development build

```bash
cd pyatlas
maturin develop --release         # builds, then editable-installs into the active venv
```

After `maturin develop`, the package is editable: changes to `python/pyatlas/*.py` take effect immediately, but Rust changes require re-running `maturin develop`.

### Building a distributable wheel

```bash
cd pyatlas
maturin build --release           # produces pyatlas-0.1.0-*.whl in target/wheels/
```

### Verification

```bash
.venv/bin/python -c "import pyatlas; print(pyatlas.__version__)"
```

---

## Running the Python tests

```bash
.venv/bin/pip install pytest      # one-time
.venv/bin/pytest pyatlas/tests/ -v
```

The two test files:
- [pyatlas/tests/test_smoke.py](pyatlas/tests/test_smoke.py) — exercises the core pyatlas API.
- [pyatlas/tests/test_xarray.py](pyatlas/tests/test_xarray.py) — xarray accessor + dask streaming.

After every Rust change re-run `maturin develop` first, otherwise pytest will run against the previously-built binary.

---

## Running the Python examples

Self-contained scripts under [pyatlas/examples/](pyatlas/examples/) that write to temp directories:

```bash
.venv/bin/python pyatlas/examples/01_basics.py
.venv/bin/python pyatlas/examples/02_xarray.py
.venv/bin/python pyatlas/examples/03_dask_streaming.py
```

---

## Common dev workflows

### Adding a new array dtype

1. Add the variant to `atlas::DType` (depends on `array-format` supporting it).
2. Add the dispatch arm in [pyatlas/src/dataset.rs](pyatlas/src/dataset.rs)'s `numeric_dispatch!` (or the explicit String/Binary branches).
3. Extend [pyatlas/src/dtype.rs](pyatlas/src/dtype.rs) to parse the new dtype name.
4. Update [pyatlas/python/pyatlas/xarray.py](pyatlas/python/pyatlas/xarray.py)'s `_NUMPY_TO_ATLAS` map if it has a numpy equivalent.
5. Add a smoke test in [pyatlas/tests/test_smoke.py](pyatlas/tests/test_smoke.py).

### Exposing a new `DatasetView` / `Atlas` method to Python

1. Implement on the Rust side in [src/store.rs](src/store.rs) or [src/dataset.rs](src/dataset.rs).
2. Wrap as a pymethod in [pyatlas/src/store.rs](pyatlas/src/store.rs) / [pyatlas/src/dataset.rs](pyatlas/src/dataset.rs). Use `py.allow_threads(|| runtime().block_on(...))` for async calls.
3. Add the stub in [pyatlas/python/pyatlas/\_\_init\_\_.pyi](pyatlas/python/pyatlas/__init__.pyi).
4. `cd pyatlas && maturin develop --release`.
5. Write a test.

### Touching the on-disk format

Format changes are breaking — bump the `version` field in `StoreMeta` ([src/meta.rs](src/meta.rs)) and load both versions during `load_meta` until a migration path is clear.

---

## Code style

- Rust: `cargo fmt` before committing; we follow the default rustfmt configuration. `cargo clippy --all-targets` should be clean.
- Python: top-level scripts and tests use ordinary 4-space indentation and import ordering (stdlib, third-party, local). No formatter is enforced; match the existing style.
- Comments and docstrings explain *why*, not what — the code already shows what.

---

## Pull requests

1. Branch from `main`.
2. Make sure `cargo test`, `cargo clippy --all-targets`, and `pytest pyatlas/tests/` all pass.
3. If you change the on-disk format, the public API, or any behaviour visible to end users, update the relevant README.
4. Keep commits focused; squash trivial fixups before review.
