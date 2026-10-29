"""Command-line entry point for ``python -m xai_framework``.

The module delegates argument parsing and execution to
:func:`xai_framework.cli.main` and converts its return value into a process
exit status.
"""

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())
