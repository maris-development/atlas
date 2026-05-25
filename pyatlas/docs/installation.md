# Installation

## Requirements

- **Python 3.10+**
- A working C/Rust toolchain *only* for source installs; wheels (when
  available) install with no compiler in scope.

`xarray` and `dask` are required runtime dependencies. They install
automatically with `pip install pyatlas` — there is no "without xarray"
build of `pyatlas`. The xarray accessor at `xr.Dataset.atlas` is registered
the moment you `import pyatlas`.

## From PyPI

```bash
pip install pyatlas
```

This pulls in `numpy>=1.23`, `xarray>=2023.1`, and `dask>=2023.1`.

## From source (development)

The Python module is built with [maturin](https://www.maturin.rs/); the
Rust crate lives in `pyatlas/`.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install maturin numpy
cd pyatlas
maturin develop --release    # builds the Rust extension and installs it editable
```

`maturin develop --release` is what every benchmark / test script in this
repo expects — the unoptimised debug build is correct but much slower for
large reads.

To run the test suite:

```bash
pytest pyatlas/tests/ -v
```

## Optional: cloud storage (S3, GCS, Azure)

To open or create atlas stores backed by S3, GCS, Azure Blob, or any
other [`object_store`](https://docs.rs/object_store)-supported backend,
install the `cloud` extra. This pulls in
[obstore](https://github.com/developmentseed/obstore):

```bash
pip install "pyatlas[cloud]"
```

Then construct an obstore handle and pass it where you'd otherwise pass
a path:

```python
import obstore as obs, pyatlas

store = obs.store.S3Store("my-bucket", prefix="stores/jan_2024", region="us-east-1")
atlas = pyatlas.Atlas.open(store)
```

See [Cloud storage (S3, GCS, Azure)](guides/cloud-storage.md) for the
full guide.

## Optional: benchmark dependencies

The cross-backend benchmark harness pulls in `zarr` and `netCDF4`:

```bash
pip install -e "pyatlas[bench]"
```

See [Benchmarks](benchmarks.md) for how to run them.

## Tracing / structured logging

`pyatlas.init_tracing()` enables tracing-subscriber-backed structured
logs from the Rust core to stderr. Pass an
[`env_filter`](https://docs.rs/tracing-subscriber/latest/tracing_subscriber/filter/struct.EnvFilter.html)
directive to override the default:

```python
import pyatlas
pyatlas.init_tracing("debug")           # everything
pyatlas.init_tracing("pyatlas=info")    # just pyatlas crate at info+
pyatlas.init_tracing()                  # re-read ATLAS_LOG / RUST_LOG env vars
```
