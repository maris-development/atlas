# Installation

## Requirements

- **Python 3.10+**
- A C and Rust toolchain, for a source install *only*. A wheel installs with no
  compiler, where a wheel exists.

`xarray` and `dask` are runtime dependencies. NetCDF is the one ingest route,
so there is no build without xarray.

## From PyPI

```bash
pip install atlas-python
```

This installs `numpy>=1.23`, `xarray>=2023.1`, and `dask>=2023.1`. It also puts
the `atlas` command on your PATH:

```bash
atlas --help
```

## From source (development)

`atlas-python` is a thin [PyO3](https://pyo3.rs/) binding layer over the
**[`atlas-rust`](https://github.com/maris-development/atlas)** core crate. All
storage, compression, and I/O live in Rust. The binding crate sits in
`atlas-python/`, and depends on the core crate at the repository root.
[maturin](https://www.maturin.rs/) builds both.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install maturin numpy
cd atlas-python
maturin develop --release    # build the Rust extension, and install it editable
```

Every benchmark and test script here expects `maturin develop --release`. The
debug build is correct, and much slower on a large read.

To run the test suite:

```bash
pytest atlas-python/tests/ -v
```

## Optional: cloud storage (S3, GCS, Azure)

Install the `cloud` extra to open or create an atlas store on S3, GCS, Azure
Blob, or any other backend
[`object_store`](https://docs.rs/object_store) supports. It installs
[obstore](https://github.com/developmentseed/obstore):

```bash
pip install "atlas-python[cloud]"
```

Then pass a URL where you would otherwise pass a path:

```bash
atlas ls s3://my-bucket/collections/2024 --region eu-west-1
```

```python
import atlas

atlas.list_datasets("s3://my-bucket/collections/2024", region="eu-west-1")
```

See [Cloud storage (S3, GCS, Azure)](guides/cloud-storage.md) for the
full guide.

## Tracing / structured logging

`atlas.init_tracing()` turns on structured logs from the Rust core to stderr.
They run through tracing-subscriber. Pass an
[`env_filter`](https://docs.rs/tracing-subscriber/latest/tracing_subscriber/filter/struct.EnvFilter.html)
directive to override the default:

```python
import atlas
atlas.init_tracing("debug")           # everything
atlas.init_tracing("atlas=info")      # the atlas crate alone, at info and above
atlas.init_tracing()                  # read ATLAS_LOG and RUST_LOG again
```
