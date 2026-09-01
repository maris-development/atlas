"""Runs the command as ``python -m atlas``.

The ``atlas`` console script needs its directory on ``PATH``. This entry point
does not. It works in any environment that can import the package. That makes
it the reliable form inside a container, in a CI job, and in a virtual
environment nobody activated.
"""

import sys

from ._cli import main

if __name__ == "__main__":
    sys.exit(main())
