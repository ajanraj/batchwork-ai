"""Private command-line entry point."""

from __future__ import annotations

import os
import signal
import sys

from ._commands import cli
from ._failures import InterruptionRequested, QuietBrokenPipe, TerminationRequested

_received_signal: int | None = None


def _request_stop(signum: int, _frame: object) -> None:
    global _received_signal
    if _received_signal is not None:
        os._exit(128 + signum)
    _received_signal = signum
    if signum == signal.SIGINT:
        raise InterruptionRequested
    raise TerminationRequested


def _close_broken_stdout() -> None:
    try:
        stdout_fd = sys.stdout.fileno()
    except (AttributeError, OSError, ValueError):
        stdout_fd = None
    if stdout_fd is None:
        return
    try:
        replacement = os.open(os.devnull, os.O_WRONLY)
        os.dup2(replacement, stdout_fd)
        os.close(replacement)
    except (AttributeError, OSError, ValueError):
        pass


def main() -> None:
    """Run the Batchwork command-line interface."""
    global _received_signal
    previous_interrupt = signal.getsignal(signal.SIGINT)
    previous_termination = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)
    try:
        cli()
        sys.stdout.flush()
    except (BrokenPipeError, QuietBrokenPipe):
        _close_broken_stdout()
    except SystemExit as error:
        if _received_signal is not None:
            sys.stdout.flush()
            sys.stderr.flush()
            code = error.code if isinstance(error.code, int) else 1
            os._exit(code)
        raise
    finally:
        _received_signal = None
        signal.signal(signal.SIGINT, previous_interrupt)
        signal.signal(signal.SIGTERM, previous_termination)
