"""Click parsing boundary for the Batchwork CLI."""

from __future__ import annotations

import asyncio
import math
import re
import sqlite3
import sys
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import NoReturn, TypeVar

import click

from batchwork._text_validation import _TEXT_ENDPOINTS
from batchwork.types import BatchProvider, BatchResult, BatchSnapshot, BatchStatus

from ._config import API_KEY_ENV, ConfigError, load_config, registry_path, select_profile
from ._contract import (
    ConfigProviderView,
    ConfigValidationEnvelope,
    ConfigViewEnvelope,
    ErrorDetail,
    ErrorEnvelope,
    JobEnvelope,
    JobListEnvelope,
    KnownErrorCode,
    Modality,
    PathsEnvelope,
    PathState,
    RegistryChangeEnvelope,
    RegistryCheckEnvelope,
    RegistryPrunePlanEnvelope,
    ResultEnvelope,
    ResultListEnvelope,
    RunEnvelope,
    SnapshotEnvelope,
    serialize_envelope,
)
from ._failures import (
    CliFailure,
    CliUsageError,
    FailureContext,
    InterruptionRequested,
    QuietBrokenPipe,
    TerminationRequested,
    configuration_failure,
    internal_failure,
    job_state_failure,
    output_failure,
    usage_failure,
)
from ._human import ProgressReporter, human_error, human_table, terminal_color
from ._input import INPUT_FORMATS
from ._lifecycle import (
    LifecycleFailure,
    LifecycleOptions,
    LifecycleResult,
    ResolvedJob,
    _recovery_command,
    cancel_job,
    duration_seconds,
    render_lifecycle_error,
    render_results,
    render_snapshot,
    resolve_job,
    results_job,
    status_job,
    unsuccessful,
    wait_job,
)
from ._materialize import ImageMaterializer, prepare_output_directory
from ._output import JsonResultSpool
from ._registry import (
    CURRENT_SCHEMA_VERSION,
    RegistryIntegrityError,
    RegistrySchemaError,
    check_registry,
    forget_job,
    is_job_name,
    is_record_id,
    list_registry_jobs,
    prune_jobs,
    reset_registry,
)
from ._state import OutputMode, RootOptions
from ._submit_text import (
    SubmissionResult,
    SubmitEmbeddingOptions,
    SubmitImageOptions,
    SubmitTextOptions,
    render_error,
    render_job,
)
from ._submit_text import submit_embeddings as execute_submit_embeddings
from ._submit_text import submit_images as execute_submit_images
from ._submit_text import submit_text as execute_submit_text

CommandFunction = TypeVar("CommandFunction", bound=Callable[..., object])
SubmitOptions = TypeVar(
    "SubmitOptions", SubmitTextOptions, SubmitEmbeddingOptions, SubmitImageOptions
)
SubmitFunction = Callable[[RootOptions, SubmitOptions], Awaitable[SubmissionResult]]

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
    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        try:
            return super().parse_args(ctx, args)
        except click.UsageError as error:
            mode = (
                OutputMode.JSONL
                if "--jsonl" in args
                else OutputMode.JSON
                if "--json" in args or not sys.stdout.isatty()
                else OutputMode.HUMAN
            )
            _emit_failure(usage_failure(error, "cli"), mode)

    def invoke(self, ctx: click.Context) -> object:
        try:
            return super().invoke(ctx)
        except click.exceptions.Exit:
            raise
        except CliFailure as failure:
            _emit_failure(failure, _context_output_mode(ctx))
        except ConfigError as error:
            root = ctx.obj if isinstance(ctx.obj, RootOptions) else None
            failure = configuration_failure(
                error,
                FailureContext(
                    operation=_context_operation(ctx),
                    profile=root.profile if root is not None else None,
                    config_path=str(root.config) if root is not None and root.config else None,
                ),
            )
            _emit_failure(failure, _context_output_mode(ctx))
        except click.UsageError as error:
            _emit_failure(usage_failure(error, _context_operation(ctx)), _context_output_mode(ctx))
        except BrokenPipeError:
            raise QuietBrokenPipe from None
        except InterruptionRequested:
            from ._lifecycle import active_signal_failure
            from ._submit_text import active_submission_signal_failure

            failure = (
                active_submission_signal_failure(interrupted=True)
                or active_signal_failure(interrupted=True)
                or CliFailure(
                    ErrorEnvelope(
                        error=ErrorDetail(
                            code="interrupted",
                            category="interrupted",
                            message=(
                                "Batchwork was interrupted; no remote cancellation was requested."
                            ),
                            exit_code=130,
                            retryable=False,
                            operation=_context_operation(ctx),
                        )
                    )
                )
            )
            _emit_failure(failure, _context_output_mode(ctx))
        except (KeyboardInterrupt, click.Abort):
            failure = CliFailure(
                ErrorEnvelope(
                    error=ErrorDetail(
                        code="interrupted",
                        category="interrupted",
                        message="Batchwork was interrupted; no remote cancellation was requested.",
                        exit_code=130,
                        retryable=False,
                        operation=_context_operation(ctx),
                    )
                )
            )
            _emit_failure(failure, _context_output_mode(ctx))
        except TerminationRequested:
            from ._lifecycle import active_signal_failure
            from ._submit_text import active_submission_signal_failure

            failure = (
                active_submission_signal_failure(interrupted=False)
                or active_signal_failure(interrupted=False)
                or CliFailure(
                    ErrorEnvelope(
                        error=ErrorDetail(
                            code="terminated",
                            category="terminated",
                            message=(
                                "Batchwork was terminated; no remote cancellation was requested."
                            ),
                            exit_code=143,
                            retryable=False,
                            operation=_context_operation(ctx),
                        )
                    )
                )
            )
            _emit_failure(failure, _context_output_mode(ctx))
        except OSError:
            _emit_failure(
                output_failure(FailureContext(operation=_context_operation(ctx))),
                _context_output_mode(ctx),
            )
        except Exception as error:
            _emit_failure(internal_failure(_context_operation(ctx)), _context_output_mode(ctx))
            raise AssertionError("unreachable") from error


def _context_operation(ctx: click.Context) -> str:
    return ctx.invoked_subcommand or "cli"


def _context_output_mode(ctx: click.Context) -> OutputMode:
    root = ctx.obj
    if isinstance(root, RootOptions):
        return _output_mode(root)
    return OutputMode.HUMAN if sys.stdout.isatty() else OutputMode.JSON


def _emit_failure(failure: CliFailure, mode: OutputMode) -> NoReturn:
    if mode in {OutputMode.JSON, OutputMode.JSONL}:
        output = serialize_envelope(failure.envelope)
    else:
        output = human_error(failure.envelope.error)
    click.echo(output, nl=False, err=True)
    raise click.exceptions.Exit(failure.exit_code) from failure


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
    modality: Modality | None,
    operation: str,
) -> LifecycleOptions:
    return LifecycleOptions(
        job,
        base_url,
        api_key_env,
        header,
        header_env,
        provider,
        save,
        name,
        modality,
        operation,
    )


def _fail_lifecycle(failure: LifecycleFailure, mode: OutputMode) -> None:
    click.echo(render_lifecycle_error(failure, mode), nl=False, err=True)
    raise click.exceptions.Exit(failure.exit_code)


def _fail_output(
    error: OSError,
    mode: OutputMode,
    operation: str,
    resolved: ResolvedJob | None = None,
    *,
    records_emitted: int = 0,
    materializer: ImageMaterializer | None = None,
) -> NoReturn:
    if isinstance(error, BrokenPipeError):
        raise QuietBrokenPipe from None
    failure = output_failure(
        FailureContext(
            operation=operation,
            provider=resolved.provider if resolved is not None else None,
            job=resolved.machine_job if resolved is not None else None,
            routing_fingerprint=(resolved.machine_fingerprint if resolved is not None else None),
        ),
        records_emitted=records_emitted,
    )
    _emit_failure(
        CliFailure(
            _partial_error_envelope(
                failure,
                materializer=materializer,
                records_emitted=records_emitted,
            )
        ),
        mode,
    )


def _partial_error_envelope(
    failure: CliFailure,
    *,
    materializer: ImageMaterializer | None,
    records_emitted: int = 0,
) -> ErrorEnvelope:
    detail = failure.envelope.error
    emitted = max(records_emitted, detail.records_emitted or 0)
    updates: dict[str, object] = {}
    if emitted:
        updates["records_emitted"] = emitted
    if materializer is not None:
        updates["materialized_images"] = len(materializer.entries)
        updates["materialized_bytes"] = materializer.byte_count
    if detail.partial_output or emitted or (materializer is not None and materializer.entries):
        updates["partial_output"] = True
    return ErrorEnvelope(error=detail.model_copy(update=updates))


def _fail_unsuccessful(
    result: LifecycleResult,
    operation: str,
    mode: OutputMode,
    *,
    materializer: ImageMaterializer | None = None,
    records_emitted: int = 0,
) -> None:
    if not unsuccessful(result):
        return
    resolved = result.resolved
    failure = job_state_failure(
        result.snapshot.status,
        FailureContext(
            operation=operation,
            provider=resolved.provider,
            job=resolved.machine_job,
            routing_fingerprint=resolved.machine_fingerprint,
        ),
        item_successes=result.item_successes,
        item_failures=result.item_failures,
        recovery_command=_recovery_command("status", resolved),
    )
    _emit_failure(
        CliFailure(
            _partial_error_envelope(
                failure,
                materializer=materializer,
                records_emitted=records_emitted,
            )
        ),
        mode,
    )


def _fail_local_state(
    root: RootOptions,
    *,
    code: KnownErrorCode,
    message: str,
    operation: str,
    path: Path,
) -> None:
    mode = _output_mode(root)
    if mode in {OutputMode.JSON, OutputMode.JSONL}:
        click.echo(
            serialize_envelope(
                ErrorEnvelope(
                    error=ErrorDetail(
                        code=code,
                        category="local_state",
                        message=message,
                        exit_code=8,
                        retryable=False,
                        operation=operation,
                        registry_path=str(path),
                    )
                )
            ),
            nl=False,
            err=True,
        )
    else:
        click.echo(f"Error: {message}", err=True)
    raise click.exceptions.Exit(8)


def _registry_error_code(error: BaseException) -> KnownErrorCode:
    if isinstance(error, RegistrySchemaError):
        return "registry_schema_unsupported"
    if isinstance(error, RegistryIntegrityError):
        return "registry_integrity_failed"
    return "registry_unavailable"


def _creation_options(function: CommandFunction) -> CommandFunction:
    options = (
        click.option("--model", metavar="PROVIDER/MODEL", help="Provider-qualified model."),
        click.option("--format", "input_format", type=FORMAT, help="Input transport format."),
        click.option("--name", metavar="NAME", help="Save a shell-safe local alias."),
        click.option(
            "--batch-metadata",
            metavar="KEY=VALUE",
            multiple=True,
            help="Provider batch metadata; do not include secrets.",
        ),
        click.option(
            "--provider-options",
            metavar="JSON_OBJECT",
            help="Selected-provider options; exact keys: https://batchwork.ajanraj.com/docs/providers/",
        ),
        click.option(
            "--provider-options-file",
            type=click.Path(path_type=Path, dir_okay=False),
            help="Read selected-provider options JSON; docs: https://batchwork.ajanraj.com/docs/providers/",
        ),
        click.option(
            "--allow-large-batch",
            is_flag=True,
            help="Authorize work above the soft volume gate.",
        ),
        click.option("--base-url", metavar="URL", help="Use an explicit provider endpoint."),
        click.option(
            "--api-key-env",
            metavar="ENV_VAR",
            help="Read the provider credential from this variable.",
        ),
        click.option(
            "--header",
            metavar="NAME=VALUE",
            multiple=True,
            help="Repeatable non-secret literal provider header.",
        ),
        click.option(
            "--header-env",
            metavar="NAME=ENV_VAR",
            multiple=True,
            help="Repeatable secret provider header by variable name.",
        ),
    )
    for option in reversed(options):
        function = option(function)
    return function


def _text_options(function: CommandFunction) -> CommandFunction:
    options = (
        click.option("--system", metavar="TEXT", help="Default system instruction."),
        click.option(
            "--max-output-tokens",
            type=click.IntRange(min=1),
            help="Default positive output-token limit.",
        ),
        click.option("--temperature", type=float, help="Default sampling temperature."),
        click.option("--top-p", type=float, help="Default nucleus-sampling value."),
        click.option(
            "--top-k",
            type=click.IntRange(min=1),
            help="Default positive top-k sampling value.",
        ),
        click.option("--seed", type=int, help="Default deterministic seed when supported."),
        click.option(
            "--frequency-penalty",
            type=float,
            help="Default frequency penalty when supported.",
        ),
        click.option(
            "--presence-penalty",
            type=float,
            help="Default presence penalty when supported.",
        ),
        click.option("--stop", metavar="TEXT", multiple=True, help="Repeatable stop sequence."),
        click.option(
            "--tool-choice",
            type=click.Choice(["auto", "none", "required"], case_sensitive=True),
            help="Default tool-selection behavior.",
        ),
        click.option(
            "--endpoint",
            type=click.Choice(_TEXT_ENDPOINTS, case_sensitive=True),
            help="Select the provider text endpoint.",
        ),
    )
    for option in reversed(options):
        function = option(function)
    return function


def _embedding_options(function: CommandFunction) -> CommandFunction:
    return click.option(
        "--dimensions",
        type=click.IntRange(min=1),
        help="Default positive embedding dimensions.",
    )(function)


def _image_options(function: CommandFunction) -> CommandFunction:
    options = (
        click.option("--n", type=click.IntRange(min=1), help="Default images per request."),
        click.option(
            "--aspect-ratio",
            metavar="WIDTH:HEIGHT",
            help="Default generated-image aspect ratio.",
        ),
        click.option("--seed", type=int, help="Default deterministic seed when supported."),
        click.option(
            "--size",
            metavar="WIDTHxHEIGHT",
            help="Default generated-image dimensions.",
        ),
    )
    for option in reversed(options):
        function = option(function)
    return function


def _wait_options(function: CommandFunction) -> CommandFunction:
    function = click.option(
        "--timeout",
        type=DURATION,
        metavar="DURATION",
        help="Stop waiting locally after a positive s/m/h/d duration.",
    )(function)
    return click.option(
        "--poll-interval",
        type=POSITIVE_FINITE_FLOAT,
        metavar="SECONDS",
        help="Use this positive polling interval.",
    )(function)


def _direct_routing_options(function: CommandFunction) -> CommandFunction:
    options = (
        click.option("--base-url", metavar="URL", help="Use an explicit provider endpoint."),
        click.option(
            "--api-key-env",
            metavar="ENV_VAR",
            help="Read the provider credential from this variable.",
        ),
        click.option(
            "--header",
            metavar="NAME=VALUE",
            multiple=True,
            help="Repeatable non-secret literal provider header.",
        ),
        click.option(
            "--header-env",
            metavar="NAME=ENV_VAR",
            multiple=True,
            help="Repeatable secret provider header by variable name.",
        ),
        click.option("--provider", type=PROVIDER, help="Qualify a bare provider job ID."),
        click.option("--save", is_flag=True, help="Adopt a successful direct operation locally."),
        click.option("--name", metavar="NAME", help="Local alias to create with --save."),
        click.option(
            "--modality",
            type=click.Choice(["text", "embeddings", "images"], case_sensitive=True),
            help="Record the modality when adopting a direct job.",
        ),
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
@click.option(
    "--config",
    type=click.Path(path_type=Path, dir_okay=False),
    help="Use this configuration file.",
)
@click.option(
    "--registry",
    type=click.Path(path_type=Path, dir_okay=False),
    help="Use this local job registry.",
)
@click.option("--profile", metavar="NAME", help="Select a configuration profile.")
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
    """Submit and manage provider-native AI batch jobs.

    Interactive stdout uses human output. Redirected stdout uses JSON for
    bounded commands and JSONL for streaming results and run.

    Lifecycle JOB selectors may be a local alias, a bw_ record ID, a
    provider:provider-job-id reference, or a bare ID with --provider.
    """
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


def _show_submission(root: RootOptions, result: SubmissionResult) -> None:
    mode = _output_mode(root)
    color = terminal_color(root, mode, sys.stdout)
    click.echo(render_job(result.job, mode, color=color), nl=False, color=color)
    if result.error is not None:
        click.echo(render_error(result.error, mode), nl=False, err=True)
        raise click.exceptions.Exit(result.error.error.exit_code)


def _run_submission(
    root: RootOptions,
    submit_options: SubmitOptions,
    submit: SubmitFunction[SubmitOptions],
    *,
    timeout: str | None,
    poll_interval: float | None,
    output_dir: Path | None = None,
) -> None:
    mode = _output_mode(root, streaming=True)
    materializer = (
        ImageMaterializer(
            prepare_output_directory(output_dir, operation="run"),
            operation="run",
        )
        if output_dir is not None
        else None
    )
    spool: JsonResultSpool | None = None
    prepared_resolved: ResolvedJob | None = None
    records_emitted = 0

    async def execute() -> tuple[LifecycleResult, LifecycleResult]:
        nonlocal spool, prepared_resolved, records_emitted
        submission = await submit(root, submit_options)
        color = terminal_color(root, mode, sys.stdout)
        if mode is not OutputMode.JSON or submission.error is not None:
            click.echo(render_job(submission.job, mode, color=color), nl=False, color=color)
            records_emitted += 1
        if submission.error is not None:
            click.echo(render_error(submission.error, mode), nl=False, err=True)
            raise click.exceptions.Exit(submission.error.error.exit_code)
        selector = submission.job.record_id or submission.job.provider_reference
        lifecycle = LifecycleOptions(selector, None, None, (), (), None, False, None)
        progress = ProgressReporter(root, mode)
        try:
            waited = await wait_job(
                root,
                lifecycle,
                poll_interval=poll_interval or 15.0,
                timeout_seconds=duration_seconds(timeout),
                on_progress=progress.update,
            )
        finally:
            progress.close()
        if mode is OutputMode.JSONL:
            click.echo(
                serialize_envelope(
                    SnapshotEnvelope(job=waited.resolved.machine_job, snapshot=waited.snapshot)
                ),
                nl=False,
            )
            records_emitted += 1
        elif mode is OutputMode.JSON:
            prepared_resolved = waited.resolved
            spool = JsonResultSpool(
                RunEnvelope(job=submission.job, snapshot=waited.snapshot, results=[])
            )

        human_results: list[BatchResult] = []

        async def emit_result(_resolved: ResolvedJob, item: BatchResult) -> None:
            nonlocal records_emitted
            materialization = (
                await materializer.materialize_result(
                    waited.resolved.machine_job,
                    waited.resolved.machine_fingerprint,
                    item,
                )
                if materializer is not None
                else None
            )
            if mode is OutputMode.JSONL:
                click.echo(
                    serialize_envelope(
                        ResultEnvelope(
                            job=waited.resolved.machine_job,
                            routing_fingerprint=waited.resolved.machine_fingerprint,
                            result=item,
                            materialization=materialization,
                        )
                    ),
                    nl=False,
                )
                records_emitted += 1
            elif mode is OutputMode.JSON:
                if spool is None:
                    raise RuntimeError("batchwork: JSON run spool was not prepared")
                spool.append(item)
            else:
                human_results.append(item)

        def reset_buffer() -> None:
            if spool is not None:
                spool.reset()
            human_results.clear()

        collected = await results_job(
            root,
            lifecycle,
            on_result=(
                emit_result
                if mode in {OutputMode.JSON, OutputMode.JSONL} or materializer is not None
                else None
            ),
            on_retry=reset_buffer if mode is OutputMode.JSON else None,
            output_is_streaming=mode is OutputMode.JSONL,
            initial_records_emitted=records_emitted if mode is OutputMode.JSONL else 0,
            restart_after_result=materializer is None,
        )
        if mode is OutputMode.HUMAN and materializer is not None:
            collected = LifecycleResult(
                collected.resolved,
                collected.snapshot,
                human_results,
                collected.item_failed,
                collected.item_successes,
                collected.item_failures,
            )
        return waited, collected

    try:
        waited, collected = asyncio.run(execute())
    except LifecycleFailure as failure:
        if spool is not None:
            spool.close()
        failure = LifecycleFailure(
            _partial_error_envelope(
                failure,
                materializer=materializer,
                records_emitted=records_emitted,
            )
        )
        _fail_lifecycle(failure, mode)
        return
    except CliFailure as failure:
        if spool is not None:
            spool.close()
        raise CliFailure(
            _partial_error_envelope(
                failure,
                materializer=materializer,
                records_emitted=records_emitted,
            )
        ) from failure
    except OSError as error:
        if spool is not None:
            spool.close()
        _fail_output(
            error,
            mode,
            "run",
            prepared_resolved,
            records_emitted=records_emitted,
            materializer=materializer,
        )
    if mode is OutputMode.JSON:
        if spool is None:
            raise RuntimeError("batchwork: JSON run spool was not prepared")
        try:
            if materializer is not None:
                spool.set_materialization(materializer.summary())
            spool.publish()
        except OSError as error:
            _fail_output(error, mode, "run", waited.resolved, materializer=materializer)
        finally:
            spool.close()
    elif mode is OutputMode.HUMAN:
        color = terminal_color(root, mode, sys.stdout)
        click.echo(
            render_snapshot(
                waited,
                mode,
                title=f"Job {waited.snapshot.status.value}",
                color=color,
            ),
            nl=False,
            color=color,
        )
        click.echo(
            render_results(
                collected,
                mode,
                materialization=materializer.summary() if materializer is not None else None,
                color=color,
            ),
            nl=False,
            color=color,
        )
    _fail_unsuccessful(
        collected,
        "run",
        mode,
        materializer=materializer,
        records_emitted=records_emitted,
    )


@cli.group(no_args_is_help=True)
def submit() -> None:
    """Submit one provider-native batch and return without polling."""


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
    """Submit text or message requests from SOURCE.

    SOURCE is one regular file or "-" for explicit stdin. Stdin and unknown
    extensions require --format.
    """
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
    _show_submission(root, result)


@submit.command("embeddings")
@click.argument("source", type=click.Path(path_type=Path, dir_okay=False, allow_dash=True))
@_creation_options
@_embedding_options
@click.pass_obj
def submit_embeddings(
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
    dimensions: int | None,
) -> None:
    """Submit embedding requests from SOURCE."""
    result = asyncio.run(
        execute_submit_embeddings(
            root,
            SubmitEmbeddingOptions(
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
                dimensions=dimensions,
            ),
        )
    )
    _show_submission(root, result)


@submit.command("images")
@click.argument("source", type=click.Path(path_type=Path, dir_okay=False, allow_dash=True))
@_creation_options
@_image_options
@click.pass_obj
def submit_images(
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
    n: int | None,
    aspect_ratio: str | None,
    seed: int | None,
    size: str | None,
) -> None:
    """Submit image-generation requests from SOURCE."""
    result = asyncio.run(
        execute_submit_images(
            root,
            SubmitImageOptions(
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
                n=n,
                aspect_ratio=aspect_ratio,
                seed=seed,
                size=size,
            ),
        )
    )
    _show_submission(root, result)


@cli.group(no_args_is_help=True)
def run() -> None:
    """Submit one batch, wait for it, then retrieve available results.

    Local timeout or interruption never cancels the provider job.
    """


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
    _run_submission(
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
        execute_submit_text,
        timeout=timeout,
        poll_interval=poll_interval,
    )


@run.command("embeddings")
@click.argument("source", type=click.Path(path_type=Path, dir_okay=False, allow_dash=True))
@_creation_options
@_embedding_options
@_wait_options
@click.pass_obj
def run_embeddings(
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
    dimensions: int | None,
    timeout: str | None,
    poll_interval: float | None,
) -> None:
    """Run canonical embedding requests from SOURCE."""
    _run_submission(
        root,
        SubmitEmbeddingOptions(
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
            dimensions=dimensions,
        ),
        execute_submit_embeddings,
        timeout=timeout,
        poll_interval=poll_interval,
    )


@run.command("images")
@click.argument("source", type=click.Path(path_type=Path, dir_okay=False, allow_dash=True))
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path, file_okay=False),
    help="Materialize images into an absent or empty directory.",
)
@_creation_options
@_image_options
@_wait_options
@click.pass_obj
def run_images(
    root: RootOptions,
    source: Path,
    output_dir: Path | None,
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
    n: int | None,
    aspect_ratio: str | None,
    seed: int | None,
    size: str | None,
    timeout: str | None,
    poll_interval: float | None,
) -> None:
    """Run canonical image requests from SOURCE."""
    _run_submission(
        root,
        SubmitImageOptions(
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
            n=n,
            aspect_ratio=aspect_ratio,
            seed=seed,
            size=size,
        ),
        execute_submit_images,
        timeout=timeout,
        poll_interval=poll_interval,
        output_dir=output_dir,
    )


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
    modality: Modality | None,
) -> None:
    """Refresh and show one current snapshot for JOB."""
    mode = _output_mode(root)
    options = _lifecycle_options(
        job,
        base_url,
        api_key_env,
        header,
        header_env,
        provider,
        save,
        name,
        modality,
        "status",
    )
    try:
        result = asyncio.run(status_job(root, options))
    except LifecycleFailure as failure:
        _fail_lifecycle(failure, mode)
        return
    color = terminal_color(root, mode, sys.stdout)
    click.echo(
        render_snapshot(result, mode, title="Job status", color=color),
        nl=False,
        color=color,
    )


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
    modality: Modality | None,
    timeout: str | None,
    poll_interval: float | None,
) -> None:
    """Wait locally until JOB reaches a terminal state.

    Timeout or interruption never cancels remote work; errors include a
    copyable recovery command.
    """
    mode = _output_mode(root)
    options = _lifecycle_options(
        job,
        base_url,
        api_key_env,
        header,
        header_env,
        provider,
        save,
        name,
        modality,
        "wait",
    )
    progress = ProgressReporter(root, mode)
    try:
        result = asyncio.run(
            wait_job(
                root,
                options,
                poll_interval=poll_interval or 15.0,
                timeout_seconds=duration_seconds(timeout),
                on_progress=progress.update,
            )
        )
    except LifecycleFailure as failure:
        _fail_lifecycle(failure, mode)
        return
    finally:
        progress.close()
    color = terminal_color(root, mode, sys.stdout)
    click.echo(
        render_snapshot(
            result,
            mode,
            title=f"Job {result.snapshot.status.value}",
            color=color,
        ),
        nl=False,
        color=color,
    )
    _fail_unsuccessful(result, "wait", mode)


@cli.command()
@click.argument("job")
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path, file_okay=False),
    help="Materialize image results into an absent or empty directory.",
)
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
    modality: Modality | None,
) -> None:
    """Retrieve available terminal results for JOB once, without waiting."""
    mode = _output_mode(root, streaming=True)
    options = _lifecycle_options(
        job,
        base_url,
        api_key_env,
        header,
        header_env,
        provider,
        save,
        name,
        modality,
        "results",
    )
    if output_dir is not None:
        if (":" in job or provider is not None) and modality != "images":
            raise CliUsageError("--output-dir with a direct JOB requires --modality images.")
        preview = resolve_job(root, options)
        known_modality = (
            preview.record.job.modality if preview.record is not None else options.modality
        )
        if known_modality is not None and known_modality != "images":
            raise CliUsageError("--output-dir is available for image jobs only.")
        if known_modality is None and modality != "images":
            raise CliUsageError(
                "--output-dir with a job of unknown modality requires --modality images."
            )
    materializer = (
        ImageMaterializer(
            prepare_output_directory(output_dir, operation="results"),
            operation="results",
        )
        if output_dir is not None
        else None
    )
    spool: JsonResultSpool | None = None
    prepared_resolved: ResolvedJob | None = None
    records_emitted = 0
    retrieval_complete = False
    human_results: list[BatchResult] = []

    def prepare_buffer(resolved: ResolvedJob, _snapshot: BatchSnapshot) -> None:
        nonlocal spool, prepared_resolved
        prepared_resolved = resolved
        if mode is OutputMode.JSON and spool is None:
            spool = JsonResultSpool(
                ResultListEnvelope(
                    job=resolved.machine_job,
                    routing_fingerprint=resolved.machine_fingerprint,
                    results=[],
                )
            )

    def reset_buffer() -> None:
        if spool is not None:
            spool.reset()
        human_results.clear()

    async def emit_result(resolved: ResolvedJob, item: BatchResult) -> None:
        nonlocal records_emitted
        materialization = (
            await materializer.materialize_result(
                resolved.machine_job,
                resolved.machine_fingerprint,
                item,
            )
            if materializer is not None
            else None
        )
        if mode is OutputMode.JSON:
            if spool is None:
                raise RuntimeError("batchwork: JSON result spool was not prepared")
            spool.append(item)
        elif mode is OutputMode.JSONL:
            click.echo(
                serialize_envelope(
                    ResultEnvelope(
                        job=resolved.machine_job,
                        routing_fingerprint=resolved.machine_fingerprint,
                        result=item,
                        materialization=materialization,
                    )
                ),
                nl=False,
            )
            records_emitted += 1
        else:
            human_results.append(item)

    try:
        result = asyncio.run(
            results_job(
                root,
                options,
                on_result=(
                    emit_result
                    if mode in {OutputMode.JSON, OutputMode.JSONL} or materializer is not None
                    else None
                ),
                on_snapshot=prepare_buffer if mode is OutputMode.JSON else None,
                on_retry=reset_buffer if mode is OutputMode.JSON else None,
                output_is_streaming=mode is OutputMode.JSONL,
                restart_after_result=materializer is None,
            )
        )
        retrieval_complete = True
    except LifecycleFailure as failure:
        failure = LifecycleFailure(
            _partial_error_envelope(
                failure,
                materializer=materializer,
                records_emitted=records_emitted,
            )
        )
        _fail_lifecycle(failure, mode)
        return
    except CliFailure as failure:
        raise CliFailure(
            _partial_error_envelope(
                failure,
                materializer=materializer,
                records_emitted=records_emitted,
            )
        ) from failure
    except OSError as error:
        _fail_output(
            error,
            mode,
            "results",
            prepared_resolved,
            records_emitted=records_emitted,
            materializer=materializer,
        )
    finally:
        if spool is not None and not retrieval_complete:
            spool.close()
    if mode is OutputMode.JSON:
        if spool is None:
            raise RuntimeError("batchwork: JSON result spool was not prepared")
        try:
            if materializer is not None:
                spool.set_materialization(materializer.summary())
            spool.publish()
        except OSError as error:
            _fail_output(error, mode, "results", result.resolved, materializer=materializer)
        finally:
            spool.close()
    elif mode is not OutputMode.JSONL:
        if materializer is not None:
            result = LifecycleResult(
                result.resolved,
                result.snapshot,
                human_results,
                result.item_failed,
                result.item_successes,
                result.item_failures,
            )
        color = terminal_color(root, mode, sys.stdout)
        click.echo(
            render_results(
                result,
                mode,
                materialization=materializer.summary() if materializer is not None else None,
                color=color,
            ),
            nl=False,
            color=color,
        )
    _fail_unsuccessful(
        result,
        "results",
        mode,
        materializer=materializer,
        records_emitted=records_emitted,
    )


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
    modality: Modality | None,
) -> None:
    """Request cancellation once, unless JOB is already terminal."""
    mode = _output_mode(root)
    options = _lifecycle_options(
        job,
        base_url,
        api_key_env,
        header,
        header_env,
        provider,
        save,
        name,
        modality,
        "cancel",
    )
    try:
        result = asyncio.run(cancel_job(root, options))
    except LifecycleFailure as failure:
        _fail_lifecycle(failure, mode)
        return
    color = terminal_color(root, mode, sys.stdout)
    title = (
        "Cancellation requested"
        if result.snapshot.status in {BatchStatus.CANCELLING, BatchStatus.CANCELLED}
        else f"Job {result.snapshot.status.value}"
    )
    click.echo(
        render_snapshot(result, mode, title=title, color=color),
        nl=False,
        color=color,
    )


@cli.command("list")
@click.option("--provider", type=PROVIDER, help="Filter by provider.")
@click.option(
    "--modality",
    type=click.Choice(["text", "embeddings", "images"]),
    help="Filter by modality.",
)
@click.option(
    "--status",
    "statuses",
    type=click.Choice([status.value for status in BatchStatus]),
    multiple=True,
    help="Repeatable status filter; selected statuses are ORed.",
)
@click.option("--name", help="Filter by exact local alias.")
@click.option("--limit", type=click.IntRange(min=1), help="Return at most this many records.")
@click.pass_obj
def list_jobs(
    root: RootOptions,
    provider: str | None,
    modality: str | None,
    statuses: tuple[str, ...],
    name: str | None,
    limit: int | None,
) -> None:
    """List locally recorded jobs without provider scans."""
    selected_registry = registry_path(root.registry)
    try:
        records = list_registry_jobs(
            selected_registry,
            provider=BatchProvider(provider) if provider is not None else None,
            modality=modality,
            statuses=tuple(BatchStatus(status) for status in statuses),
            name=name,
            limit=limit,
        )
    except (OSError, sqlite3.Error) as error:
        _fail_local_state(
            root,
            code=_registry_error_code(error),
            message=(
                f"Could not read local registry: {error}. No records were changed; check "
                "the registry path, permissions, and integrity, then retry."
            ),
            operation="list",
            path=selected_registry,
        )
        return
    mode = _output_mode(root)
    if mode is OutputMode.JSONL:
        for record in records:
            click.echo(serialize_envelope(JobEnvelope(job=record.job)), nl=False)
    elif mode is OutputMode.JSON:
        click.echo(
            serialize_envelope(JobListEnvelope(jobs=[record.job for record in records])),
            nl=False,
        )
    else:
        click.echo(human_table(records), nl=False)


@cli.command()
@click.argument("job")
@click.pass_obj
def forget(root: RootOptions, job: str) -> None:
    """Remove one local record without changing the remote job."""
    if ":" in job or (not is_record_id(job) and not is_job_name(job)):
        raise CliUsageError("JOB must be a local alias or record ID.", code="invalid_job_selector")
    selected_registry = registry_path(root.registry)
    try:
        record = forget_job(selected_registry, job)
    except (OSError, sqlite3.Error) as error:
        _fail_local_state(
            root,
            code=_registry_error_code(error),
            message=(
                f"Could not forget the local record: {error}. The remote job was not changed; "
                "check registry integrity, then retry."
            ),
            operation="forget",
            path=selected_registry,
        )
        return
    if record is None:
        _fail_local_state(
            root,
            code="local_job_not_found",
            message=(
                f'Local JOB "{job}" was not found. Nothing was removed and no remote job was '
                'changed; run "batchwork list" to find a local selector.'
            ),
            operation="forget",
            path=selected_registry,
        )
        return
    envelope = RegistryChangeEnvelope(
        operation="forget",
        path=str(selected_registry),
        changed_records=1,
        record_id=record.job.record_id,
        provider_reference=record.job.provider_reference,
    )
    mode = _output_mode(root)
    if mode in {OutputMode.JSON, OutputMode.JSONL}:
        click.echo(serialize_envelope(envelope), nl=False)
    else:
        click.echo(f"Forgot {record.job.record_id}; remote job unchanged.")


@cli.command()
@click.option("--older-than", required=True, type=DURATION, metavar="DURATION")
@click.option("--yes", is_flag=True, help="Commit the displayed local deletion plan.")
@click.pass_obj
def prune(root: RootOptions, older_than: str, yes: bool) -> None:
    """Preview or remove old terminal records; remote jobs are unchanged."""
    seconds = duration_seconds(older_than)
    if seconds is None:
        raise click.UsageError("--older-than is required.")
    cutoff_at = datetime.now(UTC) - timedelta(seconds=seconds)
    selected_registry = registry_path(root.registry)
    try:
        record_count = prune_jobs(selected_registry, cutoff_at, commit=yes)
    except (OSError, sqlite3.Error) as error:
        _fail_local_state(
            root,
            code=_registry_error_code(error),
            message=(
                f"Could not {'update' if yes else 'read'} the local prune set: {error}. "
                "No remote jobs were changed; check registry integrity, then retry."
            ),
            operation="prune",
            path=selected_registry,
        )
        return
    mode = _output_mode(root)
    if yes:
        envelope = RegistryChangeEnvelope(
            operation="prune",
            path=str(selected_registry),
            changed_records=record_count,
            older_than=older_than,
        )
        human = f"Pruned {record_count} local terminal record(s); remote jobs unchanged."
    else:
        envelope = RegistryPrunePlanEnvelope(
            path=str(selected_registry),
            older_than=older_than,
            cutoff_at=cutoff_at,
            candidate_records=record_count,
        )
        human = f"Would prune {record_count} local terminal record(s); rerun with --yes."
    if mode in {OutputMode.JSON, OutputMode.JSONL}:
        click.echo(serialize_envelope(envelope), nl=False)
    else:
        click.echo(human)


@cli.group(no_args_is_help=True)
def config() -> None:
    """Inspect non-secret configuration state."""


@config.command("path")
@click.pass_obj
def config_path(root: RootOptions) -> None:
    """Show resolved configuration and registry paths."""
    loaded = load_config(root.config)
    selected_config = loaded.path
    selected_registry = registry_path(root.registry)
    envelope = PathsEnvelope(
        config=PathState(path=str(selected_config), exists=loaded.exists),
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
        lines = [
            "Effective configuration",
            f"  Path      {loaded.path}",
            f"  Profile   {profile_name or 'none'}",
        ]
        for modality, model in sorted(models.items()):
            lines.append(f"  Model     {modality}: {model}")
        for provider, settings in providers.items():
            lines.append(f"  Provider  {provider}")
            lines.append(f"    Credential variable  {settings.api_key_env}")
            if settings.base_url is not None:
                lines.append(f"    Base URL             {settings.base_url}")
            if settings.headers:
                lines.append(f"    Literal headers      {', '.join(sorted(settings.headers))}")
            if settings.header_env:
                variables = ", ".join(
                    f"{name}={variable}" for name, variable in sorted(settings.header_env.items())
                )
                lines.append(f"    Header variables     {variables}")
        lines.append("Credential values were not read.")
        click.echo("\n".join(lines))


@cli.group(no_args_is_help=True)
def registry() -> None:
    """Inspect and recover the local job registry."""


@registry.command("check")
@click.pass_obj
def registry_check(root: RootOptions) -> None:
    """Check registry schema and integrity."""
    selected_registry = registry_path(root.registry)
    report = check_registry(selected_registry)
    if not report.ok:
        code = (
            "registry_schema_unsupported"
            if report.integrity in {"unsupported_schema", "schema_missing"}
            else "registry_unavailable"
            if report.integrity.startswith("open_failed:")
            else "registry_integrity_failed"
        )
        _fail_local_state(
            root,
            code=code,
            message=(
                "The local registry check failed. No records or remote jobs were changed; "
                "preserve the registry and run batchwork registry reset --backup if needed."
            ),
            operation="registry",
            path=selected_registry,
        )
        return
    envelope = RegistryCheckEnvelope(
        path=str(selected_registry),
        ok=report.ok,
        user_version=report.user_version,
        integrity=report.integrity,
    )
    mode = _output_mode(root)
    if mode in {OutputMode.JSON, OutputMode.JSONL}:
        click.echo(serialize_envelope(envelope), nl=False)
    else:
        state = "ok" if report.ok else "failed"
        click.echo(
            f"Registry: {selected_registry}\nSchema: {report.user_version}\n"
            f"Integrity: {report.integrity} ({state})"
        )


@registry.command("reset")
@click.option("--backup", is_flag=True, required=True)
@click.pass_obj
def registry_reset(root: RootOptions, **_: object) -> None:
    """Back up the registry recovery set, then create a fresh registry."""
    selected_registry = registry_path(root.registry)
    try:
        result = reset_registry(selected_registry)
    except (OSError, sqlite3.Error) as error:
        _fail_local_state(
            root,
            code="registry_unavailable",
            message=(
                f"Could not preserve and reset the local registry: {error}. "
                "No remote jobs were changed; inspect the registry path and recovery files."
            ),
            operation="registry",
            path=selected_registry,
        )
        return
    envelope = RegistryChangeEnvelope(
        operation="reset",
        path=str(selected_registry),
        changed_records=result.records_count,
        backup_path=str(result.backup_path) if result.backup_path is not None else None,
        records_count_known=result.records_count is not None,
        user_version=CURRENT_SCHEMA_VERSION,
    )
    mode = _output_mode(root)
    if mode in {OutputMode.JSON, OutputMode.JSONL}:
        click.echo(serialize_envelope(envelope), nl=False)
    else:
        backup = str(result.backup_path) if result.backup_path is not None else "not needed"
        click.echo(f"Registry reset: {selected_registry}\nRecovery set: {backup}")
