"""Click parsing boundary for the Batchwork CLI."""

from __future__ import annotations

import asyncio
import math
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import click

from batchwork.types import BatchProvider, BatchResult

from ._config import API_KEY_ENV, ConfigError, load_config, registry_path, select_profile
from ._config import config_path as resolve_config_path
from ._contract import (
    ConfigProviderView,
    ConfigValidationEnvelope,
    ConfigViewEnvelope,
    ErrorDetail,
    ErrorEnvelope,
    PathsEnvelope,
    PathState,
    ResultEnvelope,
    RunEnvelope,
    SnapshotEnvelope,
    serialize_envelope,
)
from ._input import INPUT_FORMATS
from ._lifecycle import (
    LifecycleFailure,
    LifecycleOptions,
    LifecycleResult,
    ResolvedJob,
    cancel_job,
    duration_seconds,
    render_lifecycle_error,
    render_results,
    render_snapshot,
    results_job,
    status_job,
    unsuccessful,
    wait_job,
)
from ._state import OutputMode, RootOptions
from ._submit_text import SubmissionResult, SubmitTextOptions, render_error, render_job
from ._submit_text import submit_text as execute_submit_text

CommandFunction = TypeVar("CommandFunction", bound=Callable[..., object])

CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}
PROVIDER = click.Choice([provider.value for provider in BatchProvider], case_sensitive=True)
FORMAT = click.Choice(INPUT_FORMATS, case_sensitive=True)
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


class ConfigAwareGroup(click.Group):
    def invoke(self, ctx: click.Context) -> object:
        try:
            return super().invoke(ctx)
        except ConfigError as error:
            root = ctx.obj
            if isinstance(root, RootOptions) and root.output_mode in {
                OutputMode.JSON,
                OutputMode.JSONL,
            }:
                click.echo(
                    serialize_envelope(
                        ErrorEnvelope(
                            error=ErrorDetail(
                                code=error.code,
                                category="configuration",
                                message=error.message,
                                exit_code=3,
                                retryable=False,
                                operation="configuration",
                            )
                        )
                    ),
                    nl=False,
                    err=True,
                )
            else:
                click.echo(f"Error: {error.message}", err=True)
            raise click.exceptions.Exit(3) from error


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


def _output_mode(root: RootOptions, *, streaming: bool = False) -> OutputMode:
    if root.output_mode is not None:
        return root.output_mode
    if sys.stdout.isatty():
        return OutputMode.HUMAN
    return OutputMode.JSONL if streaming else OutputMode.JSON


def _lifecycle_options(
    job: str,
    base_url: str | None,
    api_key_env: str | None,
    header: tuple[str, ...],
    header_env: tuple[str, ...],
    provider: str | None,
    save: bool,
    name: str | None,
) -> LifecycleOptions:
    return LifecycleOptions(job, base_url, api_key_env, header, header_env, provider, save, name)


def _fail_lifecycle(failure: LifecycleFailure, mode: OutputMode) -> None:
    click.echo(render_lifecycle_error(failure, mode), nl=False, err=True)
    raise click.exceptions.Exit(failure.exit_code)


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


@click.group(cls=ConfigAwareGroup, context_settings=CONTEXT_SETTINGS, no_args_is_help=True)
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
@click.pass_obj
def submit_text(
    root: RootOptions,
    source: Path,
    model: str | None,
    input_format: str | None,
    name: str | None,
    batch_metadata: tuple[str, ...],
    provider_options: str | None,
    provider_options_file: Path | None,
    allow_large_batch: bool,
    base_url: str | None,
    api_key_env: str | None,
    header: tuple[str, ...],
    header_env: tuple[str, ...],
    system: str | None,
    max_output_tokens: int | None,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    seed: int | None,
    frequency_penalty: float | None,
    presence_penalty: float | None,
    stop: tuple[str, ...],
    tool_choice: str | None,
    endpoint: str | None,
) -> None:
    """Submit canonical text requests from SOURCE."""
    result = asyncio.run(
        execute_submit_text(
            root,
            SubmitTextOptions(
                source=source,
                model=model,
                input_format=input_format,
                name=name,
                batch_metadata=batch_metadata,
                provider_options=provider_options,
                provider_options_file=provider_options_file,
                allow_large_batch=allow_large_batch,
                base_url=base_url,
                api_key_env=api_key_env,
                header=header,
                header_env=header_env,
                system=system,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                seed=seed,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stop=stop,
                tool_choice=tool_choice,
                endpoint=endpoint,
            ),
        )
    )
    mode = root.output_mode or (OutputMode.HUMAN if sys.stdout.isatty() else OutputMode.JSON)
    click.echo(render_job(result.job, mode), nl=False)
    if result.error is not None:
        click.echo(render_error(result.error, mode), nl=False, err=True)
        raise click.exceptions.Exit(8)


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
@click.pass_obj
def run_text(
    root: RootOptions,
    source: Path,
    model: str | None,
    input_format: str | None,
    name: str | None,
    batch_metadata: tuple[str, ...],
    provider_options: str | None,
    provider_options_file: Path | None,
    allow_large_batch: bool,
    base_url: str | None,
    api_key_env: str | None,
    header: tuple[str, ...],
    header_env: tuple[str, ...],
    system: str | None,
    max_output_tokens: int | None,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    seed: int | None,
    frequency_penalty: float | None,
    presence_penalty: float | None,
    stop: tuple[str, ...],
    tool_choice: str | None,
    endpoint: str | None,
    timeout: str | None,
    poll_interval: float | None,
) -> None:
    """Run canonical text requests from SOURCE."""
    mode = _output_mode(root, streaming=True)
    submit_options = SubmitTextOptions(
        source=source,
        model=model,
        input_format=input_format,
        name=name,
        batch_metadata=batch_metadata,
        provider_options=provider_options,
        provider_options_file=provider_options_file,
        allow_large_batch=allow_large_batch,
        base_url=base_url,
        api_key_env=api_key_env,
        header=header,
        header_env=header_env,
        system=system,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seed=seed,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        stop=stop,
        tool_choice=tool_choice,
        endpoint=endpoint,
    )

    async def execute() -> tuple[SubmissionResult, LifecycleResult, LifecycleResult]:
        submission = await execute_submit_text(root, submit_options)
        if mode is not OutputMode.JSON or submission.error is not None:
            click.echo(render_job(submission.job, mode), nl=False)
        if submission.error is not None:
            click.echo(render_error(submission.error, mode), nl=False, err=True)
            raise click.exceptions.Exit(8)
        selector = submission.job.record_id or submission.job.provider_reference
        lifecycle = LifecycleOptions(selector, None, None, (), (), None, False, None)
        waited = await wait_job(
            root,
            lifecycle,
            poll_interval=poll_interval or 15.0,
            timeout_seconds=duration_seconds(timeout),
        )
        if mode is OutputMode.JSONL:
            click.echo(
                serialize_envelope(
                    SnapshotEnvelope(job=waited.resolved.machine_job, snapshot=waited.snapshot)
                ),
                nl=False,
            )

        def emit_result(_resolved: ResolvedJob, item: BatchResult) -> None:
            if mode is OutputMode.JSONL:
                click.echo(
                    serialize_envelope(
                        ResultEnvelope(job=waited.resolved.machine_job, result=item)
                    ),
                    nl=False,
                )

        collected = await results_job(
            root,
            lifecycle,
            on_result=emit_result if mode is OutputMode.JSONL else None,
        )
        return submission, waited, collected

    try:
        submission, waited, collected = asyncio.run(execute())
    except LifecycleFailure as failure:
        _fail_lifecycle(failure, mode)
        return
    if mode is OutputMode.JSON:
        click.echo(
            serialize_envelope(
                RunEnvelope(
                    job=submission.job,
                    snapshot=waited.snapshot,
                    results=collected.results or [],
                )
            ),
            nl=False,
        )
    elif mode is OutputMode.HUMAN:
        click.echo(render_snapshot(waited, mode), nl=False)
        click.echo(render_results(collected, mode), nl=False)
    if unsuccessful(collected):
        raise click.exceptions.Exit(6)


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
@click.pass_obj
def status(
    root: RootOptions,
    job: str,
    base_url: str | None,
    api_key_env: str | None,
    header: tuple[str, ...],
    header_env: tuple[str, ...],
    provider: str | None,
    save: bool,
    name: str | None,
) -> None:
    """Fetch one current snapshot for JOB."""
    mode = _output_mode(root)
    options = _lifecycle_options(
        job, base_url, api_key_env, header, header_env, provider, save, name
    )
    result = asyncio.run(status_job(root, options))
    click.echo(render_snapshot(result, mode), nl=False)


@cli.command()
@click.argument("job")
@_direct_routing_options
@_wait_options
@click.pass_obj
def wait(
    root: RootOptions,
    job: str,
    base_url: str | None,
    api_key_env: str | None,
    header: tuple[str, ...],
    header_env: tuple[str, ...],
    provider: str | None,
    save: bool,
    name: str | None,
    timeout: str | None,
    poll_interval: float | None,
) -> None:
    """Wait locally until JOB reaches a terminal state."""
    mode = _output_mode(root)
    options = _lifecycle_options(
        job, base_url, api_key_env, header, header_env, provider, save, name
    )
    try:
        result = asyncio.run(
            wait_job(
                root,
                options,
                poll_interval=poll_interval or 15.0,
                timeout_seconds=duration_seconds(timeout),
            )
        )
    except LifecycleFailure as failure:
        _fail_lifecycle(failure, mode)
        return
    click.echo(render_snapshot(result, mode), nl=False)
    if unsuccessful(result):
        raise click.exceptions.Exit(6)


@cli.command()
@click.argument("job")
@click.option("--output-dir", type=click.Path(path_type=Path, file_okay=False))
@_direct_routing_options
@click.pass_obj
def results(
    root: RootOptions,
    job: str,
    output_dir: Path | None,
    base_url: str | None,
    api_key_env: str | None,
    header: tuple[str, ...],
    header_env: tuple[str, ...],
    provider: str | None,
    save: bool,
    name: str | None,
) -> None:
    """Retrieve available terminal results for JOB without waiting."""
    if output_dir is not None:
        raise click.UsageError("--output-dir is available for image jobs only.")
    mode = _output_mode(root, streaming=True)
    options = _lifecycle_options(
        job, base_url, api_key_env, header, header_env, provider, save, name
    )

    def emit_result(resolved: ResolvedJob, item: BatchResult) -> None:
        machine_job = resolved.machine_job
        fingerprint = resolved.machine_fingerprint
        click.echo(
            serialize_envelope(
                ResultEnvelope(
                    job=machine_job,
                    routing_fingerprint=fingerprint,
                    result=item,
                )
            ),
            nl=False,
        )

    try:
        result = asyncio.run(
            results_job(
                root,
                options,
                on_result=emit_result if mode is OutputMode.JSONL else None,
            )
        )
    except LifecycleFailure as failure:
        _fail_lifecycle(failure, mode)
        return
    if mode is not OutputMode.JSONL:
        click.echo(render_results(result, mode), nl=False)
    if unsuccessful(result):
        raise click.exceptions.Exit(6)


@cli.command()
@click.argument("job")
@_direct_routing_options
@click.pass_obj
def cancel(
    root: RootOptions,
    job: str,
    base_url: str | None,
    api_key_env: str | None,
    header: tuple[str, ...],
    header_env: tuple[str, ...],
    provider: str | None,
    save: bool,
    name: str | None,
) -> None:
    """Request cancellation once, unless JOB is already terminal."""
    mode = _output_mode(root)
    options = _lifecycle_options(
        job, base_url, api_key_env, header, header_env, provider, save, name
    )
    result = asyncio.run(cancel_job(root, options))
    click.echo(render_snapshot(result, mode), nl=False)


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
@click.pass_obj
def config_path(root: RootOptions) -> None:
    """Show resolved configuration and registry paths."""
    selected_config, _ = resolve_config_path(root.config)
    selected_registry = registry_path(root.registry)
    envelope = PathsEnvelope(
        config=PathState(path=str(selected_config), exists=selected_config.is_file()),
        registry=PathState(path=str(selected_registry), exists=selected_registry.is_file()),
    )
    mode = _output_mode(root)
    if mode in {OutputMode.JSON, OutputMode.JSONL}:
        click.echo(serialize_envelope(envelope), nl=False)
    else:
        click.echo(f"Config: {selected_config}\nRegistry: {selected_registry}")


@config.command("validate")
@click.pass_obj
def config_validate(root: RootOptions) -> None:
    """Validate configuration without reading credential values."""
    loaded = load_config(root.config)
    document = loaded.document
    envelope = ConfigValidationEnvelope(
        path=str(loaded.path),
        exists=loaded.exists,
        valid=True,
        config_schema_version=document.schema_version,
        profiles=sorted(document.profiles),
        default_profile=document.default_profile,
    )
    mode = _output_mode(root)
    if mode in {OutputMode.JSON, OutputMode.JSONL}:
        click.echo(serialize_envelope(envelope), nl=False)
    else:
        state = "valid" if loaded.exists else "valid (default file absent)"
        click.echo(f"Config: {loaded.path}\nStatus: {state}")


@config.command("show")
@click.pass_obj
def config_show(root: RootOptions) -> None:
    """Show normalized non-secret effective configuration."""
    loaded = load_config(root.config)
    profile_name, profile = select_profile(loaded, root.profile)
    models = {} if profile is None else profile.models
    providers = (
        {}
        if profile is None
        else {
            provider.value: ConfigProviderView(
                api_key_env=settings.api_key_env or API_KEY_ENV[provider],
                base_url=settings.base_url,
                headers=settings.headers,
                header_env=settings.header_env,
            )
            for provider, settings in sorted(
                profile.providers.items(), key=lambda item: item[0].value
            )
        }
    )
    envelope = ConfigViewEnvelope(
        path=str(loaded.path), profile=profile_name, models=models, providers=providers
    )
    mode = _output_mode(root)
    if mode in {OutputMode.JSON, OutputMode.JSONL}:
        click.echo(serialize_envelope(envelope), nl=False)
    else:
        click.echo(f"Config: {loaded.path}\nProfile: {profile_name or 'none'}")


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
