"""Resolving a collection location, local or remote.

Every operation takes a `source`. That is a local path, a URL, or an obstore
handle. This module hands each URL to obstore. Nothing above it needs to know
the backend.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

# Schemes obstore builds a store from. This module handles `file` itself, as a
# local path, so the common case needs no obstore.
REMOTE_SCHEMES = (
    "s3",
    "s3a",
    "gs",
    "gcs",
    "az",
    "adl",
    "abfs",
    "abfss",
    "azure",
    "http",
    "https",
)


class SourceError(ValueError):
    """The source could not be resolved to a store."""


def _scheme(text: str) -> str:
    """The URL scheme of `text`, or `""` if it is a plain path.

    A single-letter scheme is a Windows drive, not a scheme.
    """
    scheme = urlparse(text).scheme
    return "" if len(scheme) <= 1 else scheme


def resolve(source: Any, **store_options: Any) -> Any:
    """Turn `source` into something the bindings accept.

    Accepts a local path, a URL (`s3://`, `gs://`, `az://`, `http(s)://`, or
    `file://`), or an already-constructed obstore handle, which passes through
    untouched.

    Extra keyword arguments go to obstore: `region`, `endpoint`,
    `skip_signature`, and the rest. Credentials belong to obstore. Atlas never
    sees them.
    """
    # An obstore handle, or anything else that is no string. Pass it on, and
    # let the bindings decide.
    if not isinstance(source, (str, bytes)):
        return source

    text = source.decode() if isinstance(source, bytes) else source
    scheme = _scheme(text)

    if scheme == "":
        return text
    if scheme == "file":
        return urlparse(text).path

    if scheme not in REMOTE_SCHEMES:
        raise SourceError(
            f"unsupported scheme {scheme!r} in {text!r}; expected a local path "
            f"or one of: {', '.join(REMOTE_SCHEMES)}"
        )

    try:
        import obstore
    except ImportError as exc:
        raise SourceError(
            f"{text!r} needs the obstore package: "
            'pip install "atlas-python[cloud]"'
        ) from exc

    # obstore parses the URL itself, with every backend quirk. An Azure
    # account in the host part, a virtual-hosted S3 URL, and the rest.
    try:
        return obstore.store.from_url(text, **store_options)
    except Exception as exc:
        raise SourceError(f"could not open {text!r}: {exc}") from exc


def describe(source: Any) -> str:
    """A short label for `source`, for error messages and headings."""
    if isinstance(source, (str, bytes)):
        return source.decode() if isinstance(source, bytes) else source
    return repr(source)
