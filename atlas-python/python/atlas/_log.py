"""Where atlas sends its errors and warnings.

Every module here logs to the ``atlas`` logger. That logger carries no handler
of its own, so a library user sees nothing until they add one. Attach your own
handler, or call :func:`log_to_file`.

The Rust core logs through ``tracing``, not through this. `atlas.init_tracing`
sends that stream to stderr.
"""

from __future__ import annotations

import logging
import os
from typing import Union

LOGGER_NAME = "atlas"

# One line per record: when it happened, how bad it is, and what it was.
FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"

# A library must not print on its own. The null handler keeps it silent.
logging.getLogger(LOGGER_NAME).addHandler(logging.NullHandler())


def log_to_file(
    path: Union[str, os.PathLike[str]],
    *,
    level: int = logging.INFO,
    capture_warnings: bool = True,
) -> logging.Handler:
    """Writes the atlas log to `path`. Returns the handler it attached.

    The file gets every record at `level` and above. That covers each file an
    ingest skips, each array it skips, and every error an operation reports.

    With `capture_warnings`, a Python warning goes to the same file. That also
    moves it off stderr, because `logging.captureWarnings` is process-wide.

    The file opens in append mode, so a second run adds to it.

    A second call for a path that is already attached returns the handler it
    already has. One line therefore never lands twice.

    Remove the handler to stop:

    ```python
    handler = atlas.log_to_file("ingest.log")
    ...
    logging.getLogger("atlas").removeHandler(handler)
    ```
    """
    target = logging.getLogger(LOGGER_NAME)
    wanted = os.path.abspath(os.fspath(path))
    for attached in target.handlers:
        if isinstance(attached, logging.FileHandler):
            if attached.baseFilename == wanted:
                return attached

    handler = logging.FileHandler(os.fspath(path), encoding="utf-8")
    handler.setFormatter(logging.Formatter(FORMAT))
    handler.setLevel(level)

    target.addHandler(handler)
    # The logger defaults to NOTSET, which defers to root at WARNING. Without
    # this an INFO record never reaches the file.
    target.setLevel(level)

    if capture_warnings:
        logging.captureWarnings(True)
        logging.getLogger("py.warnings").addHandler(handler)
    return handler


def get_logger(name: str) -> logging.Logger:
    """The child logger of one module, under the `atlas` root."""
    return logging.getLogger("%s.%s" % (LOGGER_NAME, name))


def describe_exception(exc: BaseException) -> str:
    """A one-line label for an exception, for a log message."""
    return "%s: %s" % (type(exc).__name__, exc)
