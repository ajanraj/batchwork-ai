"""Click parsing boundary for the Batchwork CLI."""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import click

from batchwork.types import BatchProvider

from ._state import OutputMode, RootOptions

CommandFunction = TypeVar("CommandFunction", bound=Callable[..., object])

CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}
PROVIDER = click.Choice([provider.value for provider in BatchProvider], case_sensitive=True)
FORMAT = click.Choice(["json", "jsonl", "csv", "text"], case_sensitive=True)
DURATION_PATTERN = re.compile(r"^(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?[smhd]$")

CLI_HELP_PATHS = (
    (),
    ("submit",),
    ("submit", "text"),
    ("submit", "embeddings"),
    ("submit", "images"),
    ("run",),
    ("run", "text"),
    ("run", "embeddings"),
    ("run", "images"),
    ("status",),
    ("wait",),
    ("results",),
    ("cancel",),
    ("list",),
    ("forget",),
    ("prune",),
    ("config",),
    ("config", "path"),
    ("config", "validate"),
    ("config", "show"),
    ("registry",),
    ("registry", "check"),
    ("registry", "reset"),
)


class PositiveFiniteFloat(click.ParamType):
    name = "positive finite number"

    def convert(
        self,
        value: object,
        param: click.Parameter | None,
        ctx: click.Context | None,
    ) -> float:
        try:
            converted = float(str(value))
        except ValueError:
            self.fail(f"{value!r} is not a number", param, ctx)
        if not math.isfinite(converted) or converted <= 0:
            self.fail(f"{value!r} is not a positive finite number", param, ctx)
        return converted


class Duration(click.ParamType):
    name = "duration"

    def convert(
        self,
        value: object,
        param: click.Parameter | None,
        ctx: click.Context | None,
    ) -> str:
        duration = str(value)
        number = float(duration[:-1]) if DURATION_PATTERN.fullmatch(duration) else 0
        if not math.isfinite(number) or number <= 0:
            self.fail(
                f"{value!r} must be a positive finite number followed by s, m, h, or d",
                param,
                ctx,
            )
        return duration


POSITIVE_FINITE_FLOAT = PositiveFiniteFloat()
DURATION = Duration()


def _foundation_only() -> None:
    raise click.UsageError(
        "This development build provides CLI help and schema contracts only; "
        "use --help to inspect the available command surface."
    )


def _creation_options(function: CommandFunction) -> CommandFunction:
    options = (
        click.option("--model", metavar="PROVIDER/MODEL", help="Provider-qualified model."),
        click.option("--format", "input_format", type=FORMAT, help="Input transport format."),
        click.option("--name", help="Local registry alias."),
        click.option(
            "--batch-metadata",
            metavar="KEY=VALUE",
            multiple=True,
            help="Provider batch metadata; do not include secrets.",
        ),
        click.option("--provider-options", metavar="JSON_OBJECT"),
        click.option(
            "--provider-options-file",
            type=click.Path(path_type=Path, dir_okay=False),
        ),
        click.option("--allow-large-batch", is_flag=True),
        click.option("--base-url", metavar="URL"),
        click.option("--api-key-env", metavar="VARIABLE"),
        click.option("--header", metavar="NAME=VALUE", multiple=True),
        click.option("--header-env", metavar="NAME=VARIABLE", multiple=True),
    )
    for option in reversed(options):
        function = option(function)
    return function


def _text_options(function: CommandFunction) -> CommandFunction:
    options = (
        click.option("--system", metavar="TEXT"),
        click.option("--max-output-tokens", type=click.IntRange(min=1)),
        click.option("--temperature", type=float),
        click.option("--top-p", type=float),
        click.option("--top-k", type=click.IntRange(min=1)),
        click.option("--seed", type=int),
        click.option("--frequency-penalty", type=float),
        click.option("--presence-penalty", type=float),
        click.option("--stop", metavar="TEXT", multiple=True),
        click.option(
            "--tool-choice",
            type=click.Choice(["auto", "none", "required"], case_sensitive=True),
        ),
        click.option(
            "--endpoint",
            type=click.Choice(
                ["chat-completions", "responses", "completions"], case_sensitive=True
            ),
        ),
    )
    for option in reversed(options):
        function = option(function)
    return function


def _embedding_options(function: CommandFunction) -> CommandFunction:
    return click.option("--dimensions", type=click.IntRange(min=1))(function)


def _image_options(function: CommandFunction) -> CommandFunction:
    options = (
        click.option("--n", type=click.IntRange(min=1)),
        click.option("--aspect-ratio", metavar="WIDTH:HEIGHT"),
        click.option("--seed", type=int),
        click.option("--size", metavar="WIDTHxHEIGHT"),
    )
    for option in reversed(options):
        function = option(function)
    return function


def _wait_options(function: CommandFunction) -> CommandFunction:
    function = click.option("--timeout", type=DURATION, metavar="DURATION")(function)
    return click.option("--poll-interval", type=POSITIVE_FINITE_FLOAT)(function)


def _direct_routing_options(function: CommandFunction) -> CommandFunction:
    options = (
        click.option("--base-url", metavar="URL"),
        click.option("--api-key-env", metavar="VARIABLE"),
        click.option("--header", metavar="NAME=VALUE", multiple=True),
        click.option("--header-env", metavar="NAME=VARIABLE", multiple=True),
        click.option("--provider", type=PROVIDER),
        click.option("--save", is_flag=True),
        click.option("--name"),
    )
    for option in reversed(options):
        function = option(function)
    return function


def _selected_output_mode(human: bool, json_output: bool, jsonl: bool) -> OutputMode | None:
    selected = [
        mode
        for enabled, mode in (
            (human, OutputMode.HUMAN),
            (json_output, OutputMode.JSON),
            (jsonl, OutputMode.JSONL),
        )
        if enabled
    ]
    if len(selected) > 1:
        raise click.UsageError("--human, --json, and --jsonl are mutually exclusive.")
    return selected[0] if selected else None


@click.group(context_settings=CONTEXT_SETTINGS, no_args_is_help=True)
@click.option("--config", type=click.Path(path_type=Path, dir_okay=False))
@click.option("--registry", type=click.Path(path_type=Path, dir_okay=False))
@click.option("--profile", metavar="NAME")
@click.option("--human", is_flag=True, help="Force human-readable output.")
@click.option("--json", "json_output", is_flag=True, help="Emit one buffered JSON document.")
@click.option("--jsonl", is_flag=True, help="Emit newline-delimited JSON records.")
@click.option("--quiet", is_flag=True, help="Suppress progress and non-essential warnings.")
@click.option("--progress", is_flag=True, help="Force wait/run progress on redirected stderr.")
@click.option("--color", "color_enabled", is_flag=True, help="Force human-output color.")
@click.option("--no-color", is_flag=True, help="Disable human-output color.")
@click.version_option(package_name="batchwork-ai", prog_name="batchwork")
@click.pass_context
def cli(
    context: click.Context,
    config: Path | None,
    registry: Path | None,
    profile: str | None,
    human: bool,
    json_output: bool,
    jsonl: bool,
    quiet: bool,
    progress: bool,
    color_enabled: bool,
    no_color: bool,
) -> None:
    """Submit and manage provider-native AI batch jobs."""
    if color_enabled and no_color:
        raise click.UsageError("--color and --no-color are mutually exclusive.")
    color = True if color_enabled else False if no_color else None
    context.obj = RootOptions(
        config=config,
        registry=registry,
        profile=profile,
        output_mode=_selected_output_mode(human, json_output, jsonl),
        quiet=quiet,
        progress=progress and not quiet,
        color=color,
    )


@cli.group(no_args_is_help=True)
def submit() -> None:
    """Submit one provider-native batch and return immediately."""


@submit.command("text")
@click.argument("source", type=click.Path(path_type=Path, dir_okay=False, allow_dash=True))
@_creation_options
@_text_options
def submit_text(**_: object) -> None:
    """Submit canonical text requests from SOURCE."""
    _foundation_only()


@submit.command("embeddings")
@click.argument("source", type=click.Path(path_type=Path, dir_okay=False, allow_dash=True))
@_creation_options
@_embedding_options
def submit_embeddings(**_: object) -> None:
    """Submit canonical embedding requests from SOURCE."""
    _foundation_only()


@submit.command("images")
@click.argument("source", type=click.Path(path_type=Path, dir_okay=False, allow_dash=True))
@_creation_options
@_image_options
def submit_images(**_: object) -> None:
    """Submit canonical image requests from SOURCE."""
    _foundation_only()


@cli.group(no_args_is_help=True)
def run() -> None:
    """Submit, wait for, and retrieve one provider-native batch."""


@run.command("text")
@click.argument("source", type=click.Path(path_type=Path, dir_okay=False, allow_dash=True))
@_creation_options
@_text_options
@_wait_options
def run_text(**_: object) -> None:
    """Run canonical text requests from SOURCE."""
    _foundation_only()


@run.command("embeddings")
@click.argument("source", type=click.Path(path_type=Path, dir_okay=False, allow_dash=True))
@_creation_options
@_embedding_options
@_wait_options
def run_embeddings(**_: object) -> None:
    """Run canonical embedding requests from SOURCE."""
    _foundation_only()


@run.command("images")
@click.argument("source", type=click.Path(path_type=Path, dir_okay=False, allow_dash=True))
@click.option("--output-dir", type=click.Path(path_type=Path, file_okay=False))
@_creation_options
@_image_options
@_wait_options
def run_images(**_: object) -> None:
    """Run canonical image requests from SOURCE."""
    _foundation_only()


@cli.command()
@click.argument("job")
@_direct_routing_options
def status(**_: object) -> None:
    """Fetch one current snapshot for JOB."""
    _foundation_only()


@cli.command()
@click.argument("job")
@_direct_routing_options
@_wait_options
def wait(**_: object) -> None:
    """Wait locally until JOB reaches a terminal state."""
    _foundation_only()


@cli.command()
@click.argument("job")
@click.option("--output-dir", type=click.Path(path_type=Path, file_okay=False))
@_direct_routing_options
def results(**_: object) -> None:
    """Retrieve available terminal results for JOB without waiting."""
    _foundation_only()


@cli.command()
@click.argument("job")
@_direct_routing_options
def cancel(**_: object) -> None:
    """Request cancellation once, unless JOB is already terminal."""
    _foundation_only()


@cli.command("list")
@click.option("--provider", type=PROVIDER)
@click.option("--modality", type=click.Choice(["text", "embeddings", "images"]))
@click.option("--status", "statuses", multiple=True)
@click.option("--name")
@click.option("--limit", type=click.IntRange(min=1))
def list_jobs(**_: object) -> None:
    """List cached local registry records without provider scans."""
    _foundation_only()


@cli.command()
@click.argument("job")
def forget(**_: object) -> None:
    """Remove one local record without changing the remote job."""
    _foundation_only()


@cli.command()
@click.option("--older-than", required=True, type=DURATION, metavar="DURATION")
@click.option("--yes", is_flag=True, help="Commit the displayed local deletion plan.")
def prune(**_: object) -> None:
    """Preview or commit deletion of old terminal local records."""
    _foundation_only()


@cli.group(no_args_is_help=True)
def config() -> None:
    """Inspect non-secret configuration state."""


@config.command("path")
def config_path() -> None:
    """Show resolved configuration and registry paths."""
    _foundation_only()


@config.command("validate")
def config_validate() -> None:
    """Validate configuration without reading credential values."""
    _foundation_only()


@config.command("show")
def config_show() -> None:
    """Show normalized non-secret effective configuration."""
    _foundation_only()


@cli.group(no_args_is_help=True)
def registry() -> None:
    """Inspect and recover the local job registry."""


@registry.command("check")
def registry_check() -> None:
    """Check registry schema and integrity."""
    _foundation_only()


@registry.command("reset")
@click.option("--backup", is_flag=True, required=True)
def registry_reset(**_: object) -> None:
    """Back up the registry recovery set, then create a fresh registry."""
    _foundation_only()
