"""Private command-line entry point."""

from ._commands import cli


def main() -> None:
    """Run the Batchwork command-line interface."""
    cli()
