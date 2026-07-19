"""Private command-line entry point."""

from __future__ import annotations

import signal

from ._commands import cli
from ._failures import TerminationRequested


def _terminate(_signum: int, _frame: object) -> None:
    raise TerminationRequested


def main() -> None:
    """Run the Batchwork command-line interface."""
    previous = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, _terminate)
    try:
        cli()
    finally:
        signal.signal(signal.SIGTERM, previous)
