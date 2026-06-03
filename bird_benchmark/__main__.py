"""Entry point for ``python -m bird_benchmark``.

The CLI implementation lives in :mod:`bird_benchmark.cli` and is wired in by
task 14.1. The import is deferred to call time so this entry point can be
loaded before ``cli.py`` exists.
"""

from __future__ import annotations


def main() -> None:
    from bird_benchmark.cli import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
