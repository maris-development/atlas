"""Runs the command as ``python -m atlas``.

The ``atlas`` console script needs its directory on ``PATH``. This entry point
does not. It works in any environment that can import the package, which makes
it the reliable form inside a container, a CI job, or a virtual environment
nobody activated.
"""

import sys

from ._cli import main

if __name__ == "__main__":
    sys.exit(main())
