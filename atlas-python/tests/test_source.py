"""Resolving a collection location.

The URL cases build a store object, and touch no network. An obstore handle
builds offline.
"""

import pytest

import atlas
from atlas import _source


@pytest.mark.parametrize(
    "path",
    ["/tmp/collection", "relative/path", "./here", "C:/data/collection"],
)
def test_a_plain_path_passes_through(path):
    """This covers a Windows drive letter, which urlparse reads as a scheme."""
    assert _source.resolve(path) == path


def test_a_file_url_becomes_a_path():
    assert _source.resolve("file:///tmp/collection") == "/tmp/collection"


@pytest.mark.parametrize(
    "url,expected",
    [
        ("s3://bucket/prefix", "S3Store"),
        ("s3://bucket", "S3Store"),
        ("gs://bucket/prefix", "GCSStore"),
        ("https://example.org/data", "HTTPStore"),
    ],
)
def test_urls_become_obstore_handles(url, expected):
    pytest.importorskip("obstore")
    options = {"region": "us-east-1"} if url.startswith("s3") else {}
    assert type(_source.resolve(url, **options)).__name__ == expected


def test_an_obstore_handle_passes_through(tmp_path):
    obstore = pytest.importorskip("obstore")
    store = obstore.store.LocalStore(str(tmp_path))
    assert _source.resolve(store) is store


def test_an_unsupported_scheme_is_rejected():
    with pytest.raises(atlas.SourceError, match="unsupported scheme"):
        _source.resolve("ftp://host/path")


def test_a_backend_failure_is_reported_as_a_source_error():
    """Azure needs an account name. An `az://` URL does not carry one."""
    pytest.importorskip("obstore")
    with pytest.raises(atlas.SourceError, match="could not open"):
        _source.resolve("az://container/prefix")


def test_store_options_reach_the_backend():
    pytest.importorskip("obstore")
    # The backend rejects an unknown option. That error reaches the caller as
    # ours.
    with pytest.raises(atlas.SourceError):
        _source.resolve("s3://bucket/p", not_a_real_option="x")


def test_every_operation_works_through_an_obstore_handle(netcdf_dir, tmp_path):
    """The whole lifecycle, against a store handle instead of a path."""
    obstore = pytest.importorskip("obstore")
    store = obstore.store.LocalStore(str(tmp_path / "remote"), mkdir=True)

    created = atlas.create(netcdf_dir, store)
    assert created["dataset_count"] == 3

    assert atlas.list_datasets(store) == ["2024-01.nc", "2024-02.nc", "2024-03.nc"]

    described = atlas.describe(store, "2024-02.nc")
    assert described["dimensions"] == {"lat": 4, "lon": 6}
    arrays = {a["name"]: a for a in described["arrays"]}
    assert arrays["temperature"]["stats"]["row_count"] == 24

    assert atlas.remove(store, ["2024-01.nc"])["remaining"] == 2
    assert atlas.info(store)["deleted_count"] == 1
