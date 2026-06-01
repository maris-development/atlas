# Installation

## Requirements

- **Python 3.10+**
- A working C/Rust toolchain *only* for source installs; wheels (when
  available) install with no compiler in scope.

`xarray` and `dask` are required runtime dependencies. They install
automatically with `pip install atlas-python` — there is no "without xarray"
build of `atlas`. The xarray accessor at `xr.Dataset.atlas` is registered
the moment you `import atlas`.

## From PyPI

```bash
pip install atlas-python
```

This pulls in `numpy>=1.23`, `xarray>=2023.1`, and `dask>=2023.1`.

## From source (development)

`atlas-python` is a thin [PyO3](https://pyo3.rs/) binding layer over the
**[`atlas-rust`](https://github.com/maris-development/atlas)** core crate — all
storage, compression, and I/O live in Rust. The binding crate lives in
`atlas-python/` and depends on the core crate at the repo root, both built with
[maturin](https://www.maturin.rs/).

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install maturin numpy
cd atlas-python
maturin develop --release    # builds the Rust extension and installs it editable
```

`maturin develop --release` is what every benchmark / test script in this
repo expects — the unoptimised debug build is correct but much slower for
large reads.

To run the test suite:

```bash
pytest atlas-python/tests/ -v
```

## Optional: cloud storage (S3, GCS, Azure)

To open or create atlas stores backed by S3, GCS, Azure Blob, or any
other [`object_store`](https://docs.rs/object_store)-supported backend,
install the `cloud` extra. This pulls in
[obstore](https://github.com/developmentseed/obstore):

```bash
pip install "atlas-python[cloud]"
```

Then construct an obstore handle and pass it where you'd otherwise pass
a path:

```python
import obstore as obs, atlas

store = obs.store.S3Store("my-bucket", prefix="stores/jan_2024", region="us-east-1")
atlas = atlas.Atlas.open(store)
```

See [Cloud storage (S3, GCS, Azure)](guides/cloud-storage.md) for the
full guide.

## Optional: benchmark dependencies

The cross-backend benchmark harness pulls in `zarr` and `netCDF4`:

```bash
pip install -e "atlas-python[bench]"
```

See [Benchmarks](benchmarks.md) for how to run them.

## Tracing / structured logging

`atlas.init_tracing()` enables tracing-subscriber-backed structured
logs from the Rust core to stderr. Pass an
[`env_filter`](https://docs.rs/tracing-subscriber/latest/tracing_subscriber/filter/struct.EnvFilter.html)
directive to override the default:

```python
import atlas
atlas.init_tracing("debug")           # everything
atlas.init_tracing("atlas_python=info")    # just atlas crate at info+
atlas.init_tracing()                  # re-read ATLAS_LOG / RUST_LOG env vars
```
