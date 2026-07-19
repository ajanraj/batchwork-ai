"""Provider-neutral CLI lifecycle operations."""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import click

from batchwork.client import Batchwork
from batchwork.types import (
    BatchProvider,
    BatchRef,
    BatchResult,
    BatchResultStatus,
    BatchSnapshot,
    BatchStatus,
    is_terminal_status,
)

from ._config import ConfigError, ProviderConfig, load_config, registry_path, select_profile
from ._contract import (
    ErrorDetail,
    ErrorEnvelope,
    Recovery,
    ResultEnvelope,
    ResultListEnvelope,
    SnapshotEnvelope,
    serialize_envelope,
)
from ._registry import RegistryJob, adopt_job, get_job, update_job, update_job_profile
from ._state import OutputMode, RootOptions
from ._submit_text import (
    ResolvedRoute,
    _resolve_route,
    resolve_registered_route,
    resolve_route_descriptor,
)

_JOB_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


@dataclass(frozen=True, slots=True)
class LifecycleOptions:
    job: str
    base_url: str | None
    api_key_env: str | None
    header: Sequence[str]
    header_env: Sequence[str]
    provider: str | None
    save: bool
    name: str | None


@dataclass(frozen=True, slots=True)
class ResolvedJob:
    selector: str
    provider: BatchProvider
    provider_job_id: str
    route: ResolvedRoute
    registry_path: Path | None = None
    record: RegistryJob | None = None
    selected_profile: str | None = None

    @property
    def machine_job(self) -> str:
        if self.record is not None and self.record.job.record_id is not None:
            return self.record.job.record_id
        return f"{self.provider.value}:{self.provider_job_id}"

    @property
    def machine_fingerprint(self) -> str | None:
        return None if self.record is not None else self.route.registry.fingerprint


@dataclass(frozen=True, slots=True)
class LifecycleResult:
    resolved: ResolvedJob
    snapshot: BatchSnapshot
    results: list[BatchResult] | None = None
    item_failed: bool = False


class LifecycleFailure(Exception):
    def __init__(self, envelope: ErrorEnvelope) -> None:
        super().__init__(envelope.error.message)
        self.envelope = envelope

    @property
    def exit_code(self) -> int:
        return self.envelope.error.exit_code


def duration_seconds(duration: str | None) -> float | None:
    if duration is None:
        return None
    factors = {"s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}
    return float(duration[:-1]) * factors[duration[-1]]


def _direct_flags(options: LifecycleOptions) -> bool:
    return any(
        (
            options.base_url,
            options.api_key_env,
            options.header,
            options.header_env,
        )
    )


def resolve_job(root: RootOptions, options: LifecycleOptions) -> ResolvedJob:
    loaded = load_config(root.config)
    if options.name is not None and not options.save:
        raise click.UsageError("--name requires --save.")
    if options.name is not None and not _JOB_NAME.fullmatch(options.name):
        raise click.UsageError("--name must be 1-64 shell-safe characters.")

    provider: BatchProvider | None = None
    provider_job_id: str | None = None
    if ":" in options.job:
        provider_name, provider_job_id = options.job.split(":", 1)
        if options.provider is not None:
            raise click.UsageError("--provider cannot be used with a provider-qualified JOB.")
        try:
            provider = BatchProvider(provider_name)
        except ValueError as error:
            raise click.UsageError(f'Unknown provider in JOB: "{provider_name}".') from error
        if not provider_job_id:
            raise click.UsageError("Provider-qualified JOB must include a provider job ID.")
    elif options.provider is not None:
        provider = BatchProvider(options.provider)
        provider_job_id = options.job

    if provider is not None and provider_job_id is not None:
        if root.profile is not None and _direct_flags(options):
            raise click.UsageError("--profile cannot be combined with direct routing flags.")
        profile_name, profile = select_profile(loaded, root.profile)
        route = _resolve_route(
            provider,
            api_key_env=options.api_key_env,
            base_url=options.base_url,
            header=options.header,
            header_env=options.header_env,
            profile=profile.providers.get(provider) if profile else None,
        )
        return ResolvedJob(
            options.job,
            provider,
            provider_job_id,
            route,
            selected_profile=profile_name,
        )

    if options.save or _direct_flags(options):
        raise click.UsageError(
            "Direct routing options require provider:provider-job-id or --provider."
        )
    selected_registry_path = registry_path(root.registry)
    record = get_job(selected_registry_path, options.job)
    if record is None:
        raise click.UsageError(
            f'Local JOB "{options.job}" was not found; use provider:provider-job-id or '
            "a bare ID with --provider."
        )
    selected_profile = None
    if root.profile is not None:
        selected_profile, profile = select_profile(loaded, root.profile, ambient=False)
        settings = (
            profile.providers.get(record.job.provider, ProviderConfig())
            if profile is not None
            else ProviderConfig()
        )
        candidate = resolve_route_descriptor(
            record.job.provider,
            api_key_env=None,
            base_url=None,
            header=(),
            header_env=(),
            profile=settings,
        )
        if candidate.fingerprint != record.route.fingerprint:
            direct = record.job.provider_reference
            raise ConfigError(
                f'Profile "{selected_profile}" routing fingerprint does not match local '
                f'JOB "{options.job}". Use "{direct}" with --profile '
                f'"{selected_profile}" --save to create a distinct record.'
            )
    route = resolve_registered_route(record.job.provider, record.route)
    return ResolvedJob(
        options.job,
        record.job.provider,
        record.job.provider_job_id,
        route,
        selected_registry_path,
        record,
        selected_profile,
    )


def _ref(resolved: ResolvedJob) -> BatchRef:
    return BatchRef(
        id=resolved.provider_job_id,
        provider=resolved.provider,
        api_key=resolved.route.api_key,
        base_url=resolved.route.base_url,
        headers=resolved.route.headers,
    )


def _persist(resolved: ResolvedJob, snapshot: BatchSnapshot) -> None:
    if (
        resolved.registry_path is not None
        and resolved.record is not None
        and resolved.record.job.record_id is not None
    ):
        update_job(
            resolved.registry_path,
            resolved.record.job.record_id,
            snapshot,
            datetime.now(UTC),
        )
        if (
            resolved.selected_profile is not None
            and resolved.record.job.profile != resolved.selected_profile
        ):
            update_job_profile(
                resolved.registry_path,
                resolved.record.job.record_id,
                resolved.selected_profile,
            )


def _recovery_command(operation: str, resolved: ResolvedJob) -> list[str]:
    command = ["batchwork", operation, resolved.machine_job]
    if resolved.record is not None:
        return command
    route = resolved.route.registry
    command.extend(["--api-key-env", route.api_key_env])
    if route.base_url:
        command.extend(["--base-url", route.base_url])
    for name, value in sorted(route.headers.items()):
        command.extend(["--header", f"{name}={value}"])
    for name, variable in sorted(route.header_env.items()):
        command.extend(["--header-env", f"{name}={variable}"])
    return command


def _adopt_if_requested(
    root: RootOptions,
    options: LifecycleOptions,
    resolved: ResolvedJob,
    snapshot: BatchSnapshot,
) -> ResolvedJob:
    if not options.save:
        return resolved
    selected_registry_path = registry_path(root.registry)
    record = adopt_job(
        selected_registry_path,
        name=options.name,
        profile=resolved.selected_profile,
        route=resolved.route.registry,
        snapshot=snapshot,
        registered_at=datetime.now(UTC),
    )
    return ResolvedJob(
        resolved.selector,
        resolved.provider,
        resolved.provider_job_id,
        resolved.route,
        selected_registry_path,
        record,
        resolved.selected_profile,
    )


async def status_job(root: RootOptions, options: LifecycleOptions) -> LifecycleResult:
    resolved = resolve_job(root, options)
    async with Batchwork() as client:
        job = await client.get_batch(_ref(resolved))
    _persist(resolved, job.snapshot)
    resolved = _adopt_if_requested(root, options, resolved, job.snapshot)
    return LifecycleResult(resolved, job.snapshot)


async def wait_job(
    root: RootOptions,
    options: LifecycleOptions,
    *,
    poll_interval: float = 15.0,
    timeout_seconds: float | None = None,
) -> LifecycleResult:
    resolved = resolve_job(root, options)
    started = time.monotonic()
    try:
        async with asyncio.timeout(timeout_seconds):
            async with Batchwork() as client:
                job = await client.get_batch(_ref(resolved))
                snapshot = job.snapshot
                _persist(resolved, snapshot)
                while not is_terminal_status(snapshot.status):
                    if timeout_seconds is None:
                        sleep_for = poll_interval
                    else:
                        remaining = timeout_seconds - (time.monotonic() - started)
                        sleep_for = min(poll_interval, max(0.0, remaining))
                    await asyncio.sleep(sleep_for)
                    snapshot = await job.poll()
                    _persist(resolved, snapshot)
    except TimeoutError:
        raise LifecycleFailure(
            ErrorEnvelope(
                error=ErrorDetail(
                    code="wait_timeout",
                    category="wait_timeout",
                    message=(
                        f'Local wait timed out; remote job "{resolved.machine_job}" is unchanged.'
                    ),
                    exit_code=7,
                    retryable=True,
                    operation="wait",
                    provider=resolved.provider,
                    job=resolved.machine_job,
                    routing_fingerprint=resolved.machine_fingerprint,
                    recovery=Recovery(
                        action="resume_wait",
                        command=_recovery_command("wait", resolved),
                    ),
                )
            )
        ) from None
    resolved = _adopt_if_requested(root, options, resolved, snapshot)
    return LifecycleResult(resolved, snapshot)


def _nonterminal_failure(resolved: ResolvedJob, snapshot: BatchSnapshot) -> LifecycleFailure:
    return LifecycleFailure(
        ErrorEnvelope(
            error=ErrorDetail(
                code="results_not_terminal",
                category="job_state",
                message=(
                    f'Results are unavailable while job "{resolved.machine_job}" is '
                    f"{snapshot.status.value}."
                ),
                exit_code=6,
                retryable=True,
                operation="results",
                provider=resolved.provider,
                job=resolved.machine_job,
                routing_fingerprint=resolved.machine_fingerprint,
                recovery=Recovery(
                    action="wait_for_terminal_state",
                    command=_recovery_command("wait", resolved),
                ),
            )
        )
    )


async def results_job(
    root: RootOptions,
    options: LifecycleOptions,
    *,
    on_result: Callable[[ResolvedJob, BatchResult], None] | None = None,
) -> LifecycleResult:
    resolved = resolve_job(root, options)
    async with Batchwork() as client:
        job = await client.get_batch(_ref(resolved))
        _persist(resolved, job.snapshot)
        if not is_terminal_status(job.status):
            raise _nonterminal_failure(resolved, job.snapshot)
        results: list[BatchResult] | None = [] if on_result is None else None
        item_failed = False
        async for item in job._results_from_current_snapshot():
            item_failed = item_failed or item.status is not BatchResultStatus.SUCCEEDED
            if on_result is None:
                if results is None:
                    raise RuntimeError("batchwork: missing buffered result collection")
                results.append(item)
            else:
                on_result(resolved, item)
    resolved = _adopt_if_requested(root, options, resolved, job.snapshot)
    return LifecycleResult(resolved, job.snapshot, results, item_failed)


async def cancel_job(root: RootOptions, options: LifecycleOptions) -> LifecycleResult:
    resolved = resolve_job(root, options)
    async with Batchwork() as client:
        job = await client.get_batch(_ref(resolved))
        if not is_terminal_status(job.status):
            snapshot = await job.cancel()
        else:
            snapshot = job.snapshot
    _persist(resolved, snapshot)
    resolved = _adopt_if_requested(root, options, resolved, snapshot)
    return LifecycleResult(resolved, snapshot)


def unsuccessful(result: LifecycleResult) -> bool:
    return (
        result.snapshot.status is not BatchStatus.COMPLETED
        or result.item_failed
        or any(item.status is not BatchResultStatus.SUCCEEDED for item in (result.results or ()))
    )


def render_snapshot(result: LifecycleResult, mode: OutputMode) -> str:
    if mode in {OutputMode.JSON, OutputMode.JSONL}:
        return serialize_envelope(
            SnapshotEnvelope(
                job=result.resolved.machine_job,
                routing_fingerprint=result.resolved.machine_fingerprint,
                snapshot=result.snapshot,
            )
        )
    counts = result.snapshot.request_counts
    return (
        f"Job: {result.resolved.machine_job}\n"
        f"Provider: {result.snapshot.provider.value}\n"
        f"Status: {result.snapshot.status.value}\n"
        f"Requests: {counts.completed}/{counts.total} complete, {counts.failed} failed\n"
    )


def render_results(result: LifecycleResult, mode: OutputMode) -> str:
    items = result.results or []
    if mode is OutputMode.JSON:
        return serialize_envelope(
            ResultListEnvelope(
                job=result.resolved.machine_job,
                routing_fingerprint=result.resolved.machine_fingerprint,
                results=items,
            )
        )
    if mode is OutputMode.JSONL:
        return "".join(
            serialize_envelope(
                ResultEnvelope(
                    job=result.resolved.machine_job,
                    routing_fingerprint=result.resolved.machine_fingerprint,
                    result=item,
                )
            )
            for item in items
        )
    lines = [f"Results for {result.resolved.machine_job}: {len(items)}"]
    for item in items:
        preview = f" — {item.text[:80]}" if item.text else ""
        lines.append(f"{item.custom_id}: {item.status.value}{preview}")
    return "\n".join(lines) + "\n"


def render_lifecycle_error(failure: LifecycleFailure, mode: OutputMode) -> str:
    if mode in {OutputMode.JSON, OutputMode.JSONL}:
        return serialize_envelope(failure.envelope)
    error = failure.envelope.error
    command = error.recovery.command if error.recovery else None
    recovery = f"\nRecovery: {' '.join(command)}" if command else ""
    return f"Error: {error.message}{recovery}\n"
