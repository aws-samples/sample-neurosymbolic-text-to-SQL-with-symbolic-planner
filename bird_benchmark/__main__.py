"""Entry point for ``python -m bird_benchmark``.

The CLI implementation lives in :mod:`bird_benchmark.cli`; this shim
just calls its :func:`main` function and propagates the integer exit
code through :func:`sys.exit` so the operator's shell sees the right
status. ``main`` is imported lazily so any future expensive top-level
import in ``cli.py`` does not run when this module is merely imported
(e.g. for entry-point discovery).
"""

from __future__ import annotations

import sys


def main() -> None:
    from bird_benchmark.cli import main as cli_main

    sys.exit(cli_main())


if __name__ == "__main__":
    main()
