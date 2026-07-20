"""Provider-neutral CLI lifecycle operations."""

from __future__ import annotations

import asyncio
import inspect
import random
import sqlite3
import ssl
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TypeVar

import httpx

from batchwork._provider_failure import ProviderFailureError, ProviderFailureKind
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
    Materialization,
    Modality,
    Recovery,
    ResultEnvelope,
    ResultListEnvelope,
    SnapshotEnvelope,
    serialize_envelope,
)
from ._failures import (
    CliFailure,
    CliUsageError,
    FailureContext,
    InterruptionRequested,
    TerminationRequested,
    job_state_failure,
    provider_failure,
)
from ._human import human_error, human_results, human_snapshot
from ._registry import (
    RegistryIntegrityError,
    RegistryJob,
    RegistryNameConflict,
    RegistrySchemaError,
    adopt_job,
    get_job,
    is_job_name,
    is_record_id,
    update_job,
)
from ._state import OutputMode, RootOptions
from ._submit_text import (
    ResolvedRoute,
    _resolve_route,
    resolve_registered_route,
    resolve_route_descriptor,
)

_READ_ATTEMPTS = 3
_RETRYABLE_HTTP_STATUSES = frozenset({408, 429, 500, 502, 503, 504})
_MAX_RETRY_DELAY_SECONDS = 60.0
_ReadResult = TypeVar("_ReadResult")


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
    modality: Modality | None = None
    operation: str = "status"


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
    item_successes: int = 0
    item_failures: int = 0


class LifecycleFailure(CliFailure):
    pass


@dataclass(slots=True)
class _ActiveLifecycle:
    operation: str
    resolved: ResolvedJob
    snapshot: BatchSnapshot | None = None
    records_emitted: int = 0
    item_successes: int = 0
    item_failures: int = 0


_active_lifecycle: _ActiveLifecycle | None = None


def _activate(operation: str, resolved: ResolvedJob) -> _ActiveLifecycle:
    global _active_lifecycle
    active = _ActiveLifecycle(operation, resolved)
    _active_lifecycle = active
    return active


def active_signal_failure(*, interrupted: bool) -> LifecycleFailure | None:
    active = _active_lifecycle
    if active is None:
        return None
    error = InterruptionRequested() if interrupted else TerminationRequested()
    return _signal_failure(
        error,
        active.operation,
        active.resolved,
        active.snapshot,
        records_emitted=active.records_emitted,
        item_successes=active.item_successes,
        item_failures=active.item_failures,
    )


def _cause_chain(error: BaseException) -> list[BaseException]:
    causes: list[BaseException] = []
    current: BaseException | None = error
    while current is not None and current not in causes:
        causes.append(current)
        current = current.__cause__ or current.__context__
    return causes


def _retryable_read_failure(error: BaseException) -> bool:
    if not isinstance(error, ProviderFailureError):
        return False
    failure = error.failure
    if failure.kind is ProviderFailureKind.UNAVAILABLE:
        return failure.status_code in _RETRYABLE_HTTP_STATUSES
    if failure.kind is not ProviderFailureKind.TRANSPORT:
        return False
    causes = _cause_chain(error)
    if any(isinstance(cause, ssl.SSLError) for cause in causes):
        return False
    return any(
        isinstance(
            cause,
            (
                httpx.ConnectError,
                httpx.ConnectTimeout,
                httpx.ReadError,
                httpx.ReadTimeout,
            ),
        )
        for cause in causes
    )


def _retry_delay(error: BaseException, retry_index: int) -> float:
    if isinstance(error, ProviderFailureError):
        retry_after = error.failure.retry_after_seconds
        if retry_after is not None:
            return min(float(retry_after), _MAX_RETRY_DELAY_SECONDS)
    ceiling = min(0.25 * (2**retry_index), _MAX_RETRY_DELAY_SECONDS)
    return random.uniform(ceiling / 2, ceiling)


async def _retry_read(
    read: Callable[[], Awaitable[_ReadResult]],
    *,
    deadline: float | None = None,
) -> _ReadResult:
    for attempt in range(_READ_ATTEMPTS):
        try:
            return await read()
        except Exception as error:
            if attempt + 1 == _READ_ATTEMPTS or not _retryable_read_failure(error):
                raise
            delay = _retry_delay(error, attempt)
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError from None
                if delay >= remaining:
                    await asyncio.sleep(remaining)
                    raise TimeoutError from None
            await asyncio.sleep(delay)
    raise RuntimeError("batchwork: exhausted retry loop")


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
        raise CliUsageError("--name requires --save.")
    expected_image_results = options.operation == "results" and options.modality == "images"
    if options.modality is not None and not options.save and not expected_image_results:
        raise CliUsageError("--modality requires --save.")
    if options.name is not None and not is_job_name(options.name):
        raise CliUsageError("--name must be 1-64 shell-safe characters and cannot be a record ID.")

    provider: BatchProvider | None = None
    provider_job_id: str | None = None
    if ":" in options.job:
        provider_name, provider_job_id = options.job.split(":", 1)
        if options.provider is not None:
            raise CliUsageError("--provider cannot be used with a provider-qualified JOB.")
        try:
            provider = BatchProvider(provider_name)
        except ValueError as error:
            raise CliUsageError(
                f'Unknown provider in JOB: "{provider_name}".', code="invalid_job_selector"
            ) from error
        if not provider_job_id:
            raise CliUsageError(
                "Provider-qualified JOB must include a provider job ID.",
                code="invalid_job_selector",
            )
    elif options.provider is not None:
        provider = BatchProvider(options.provider)
        provider_job_id = options.job

    if provider is not None and provider_job_id is not None:
        if root.profile is not None and _direct_flags(options):
            raise CliUsageError("--profile cannot be combined with direct routing flags.")
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
        raise CliUsageError(
            "Direct routing options require provider:provider-job-id or --provider."
        )
    if not is_record_id(options.job) and not is_job_name(options.job):
        raise CliUsageError(
            "JOB must be a local alias, record ID, provider reference, or bare ID with --provider.",
            code="invalid_job_selector",
        )
    selected_registry_path = registry_path(root.registry)
    try:
        record = get_job(selected_registry_path, options.job)
    except (OSError, sqlite3.Error) as error:
        code = (
            "registry_schema_unsupported"
            if isinstance(error, RegistrySchemaError)
            else "registry_integrity_failed"
            if isinstance(error, RegistryIntegrityError)
            else "registry_unavailable"
        )
        raise LifecycleFailure(
            ErrorEnvelope(
                error=ErrorDetail(
                    code=code,
                    category="local_state",
                    message=(
                        f"Could not read local registry: {error}. No provider operation was "
                        "attempted; check registry integrity or use a direct provider reference."
                    ),
                    exit_code=8,
                    retryable=False,
                    operation=options.operation,
                    registry_path=str(selected_registry_path),
                )
            )
        ) from error
    if record is None:
        raise LifecycleFailure(
            ErrorEnvelope(
                error=ErrorDetail(
                    code="local_job_not_found",
                    category="local_state",
                    message=(
                        f'Local JOB "{options.job}" was not found. No provider operation was '
                        "attempted; use a direct provider reference or run batchwork list."
                    ),
                    exit_code=8,
                    retryable=False,
                    operation=options.operation,
                    registry_path=str(selected_registry_path),
                )
            )
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


def _failure_context(operation: str, resolved: ResolvedJob) -> FailureContext:
    return FailureContext(
        operation=operation,
        provider=resolved.provider,
        job=resolved.machine_job,
        routing_fingerprint=resolved.machine_fingerprint,
        profile=resolved.selected_profile,
        registry_path=(str(resolved.registry_path) if resolved.registry_path is not None else None),
    )


def _lifecycle_provider_failure(
    error: BaseException,
    operation: str,
    resolved: ResolvedJob,
    *,
    result_stream: bool = False,
    records_emitted: int = 0,
    item_successes: int = 0,
    item_failures: int = 0,
    cancellation_refresh: bool = False,
) -> LifecycleFailure | None:
    failure = provider_failure(
        error,
        _failure_context(operation, resolved),
        result_stream=result_stream,
        records_emitted=records_emitted,
        item_successes=item_successes,
        item_failures=item_failures,
        cancellation_refresh=cancellation_refresh,
    )
    if failure is None:
        return None
    if cancellation_refresh:
        detail = failure.envelope.error.model_copy(
            update={
                "recovery": Recovery(
                    action="check_status",
                    command=_recovery_command("status", resolved),
                )
            }
        )
        return LifecycleFailure(ErrorEnvelope(error=detail))
    return LifecycleFailure(failure.envelope)


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
            profile=(
                resolved.selected_profile
                if resolved.selected_profile is not None
                and resolved.record.job.profile != resolved.selected_profile
                else None
            ),
        )


def _recovery_command(operation: str, resolved: ResolvedJob) -> list[str]:
    command = ["batchwork", operation, resolved.machine_job]
    if resolved.record is not None:
        return command
    route = resolved.route.registry
    command.extend(["--api-key-env", route.api_key_env])
    if route.base_url:
        command.extend(["--base-url", route.base_url])
    for name, variable in sorted(route.header_env.items()):
        command.extend(["--header-env", f"{name}={variable}"])
    return command


def _signal_failure(
    error: InterruptionRequested | TerminationRequested,
    operation: str,
    resolved: ResolvedJob,
    snapshot: BatchSnapshot | None = None,
    *,
    records_emitted: int = 0,
    item_successes: int = 0,
    item_failures: int = 0,
) -> LifecycleFailure:
    interrupted = isinstance(error, InterruptionRequested)
    if snapshot is not None and not records_emitted and operation != "results":
        item_successes = snapshot.request_counts.completed
        item_failures = snapshot.request_counts.failed
    detail = ErrorDetail(
        code="interrupted" if interrupted else "terminated",
        category="interrupted" if interrupted else "terminated",
        message=(
            f'Batchwork was interrupted; remote job "{resolved.machine_job}" was not cancelled.'
            if interrupted
            else f'Batchwork was terminated; remote job "{resolved.machine_job}" was not cancelled.'
        ),
        exit_code=130 if interrupted else 143,
        retryable=False,
        operation=operation,
        provider=resolved.provider,
        job=resolved.machine_job,
        routing_fingerprint=resolved.machine_fingerprint,
        partial_output=True if records_emitted else None,
        records_emitted=records_emitted if records_emitted else None,
        item_successes=item_successes,
        item_failures=item_failures,
        recovery=Recovery(
            action="resume_operation",
            command=_recovery_command(operation, resolved),
        ),
    )
    return LifecycleFailure(ErrorEnvelope(error=detail))


def _persist_or_fail(operation: str, resolved: ResolvedJob, snapshot: BatchSnapshot) -> None:
    try:
        _persist(resolved, snapshot)
    except (OSError, sqlite3.Error) as error:
        provider_reference = f"{resolved.provider.value}:{resolved.provider_job_id}"
        direct = ResolvedJob(
            provider_reference,
            resolved.provider,
            resolved.provider_job_id,
            resolved.route,
        )
        raise LifecycleFailure(
            ErrorEnvelope(
                error=ErrorDetail(
                    code="registry_unavailable",
                    category="local_state",
                    message=(
                        "The provider operation succeeded, but Batchwork could not update the "
                        "registry. The remote job is unchanged."
                    ),
                    exit_code=8,
                    retryable=False,
                    operation=operation,
                    provider=resolved.provider,
                    job=provider_reference,
                    routing_fingerprint=resolved.route.registry.fingerprint,
                    registry_path=(
                        str(resolved.registry_path) if resolved.registry_path is not None else None
                    ),
                    recovery=Recovery(
                        action="resume_with_direct_reference",
                        command=_recovery_command(operation, direct),
                    ),
                )
            )
        ) from error


def _adopt_if_requested(
    root: RootOptions,
    options: LifecycleOptions,
    resolved: ResolvedJob,
    snapshot: BatchSnapshot,
    operation: str,
) -> ResolvedJob:
    if not options.save:
        return resolved
    selected_registry_path = registry_path(root.registry)
    try:
        record = adopt_job(
            selected_registry_path,
            name=options.name,
            profile=resolved.selected_profile,
            route=resolved.route.registry,
            snapshot=snapshot,
            registered_at=datetime.now(UTC),
            modality=options.modality,
        )
    except (OSError, sqlite3.Error) as error:
        conflict = isinstance(error, RegistryNameConflict)
        direct_recovery = _recovery_command("status", resolved)
        message = (
            f'Local name "{options.name}" is already in use; the provider operation '
            "succeeded but the registry was unchanged."
            if conflict
            else "The provider operation succeeded, but Batchwork could not update the registry."
        )
        raise LifecycleFailure(
            ErrorEnvelope(
                error=ErrorDetail(
                    code="registry_unavailable",
                    category="local_state",
                    message=message,
                    exit_code=8,
                    retryable=False,
                    operation=operation,
                    provider=resolved.provider,
                    job=f"{resolved.provider.value}:{resolved.provider_job_id}",
                    routing_fingerprint=resolved.route.registry.fingerprint,
                    registry_path=str(selected_registry_path),
                    recovery=Recovery(
                        action="retry_adoption_without_name",
                        command=[
                            "batchwork",
                            "--registry",
                            str(selected_registry_path),
                            *direct_recovery[1:],
                            "--save",
                        ],
                    ),
                )
            )
        ) from error
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
    active = _activate("status", resolved)
    try:
        async with Batchwork() as client:
            job = await _retry_read(lambda: client.get_batch(_ref(resolved)))
    except (InterruptionRequested, TerminationRequested) as error:
        raise _signal_failure(error, "status", resolved) from None
    except Exception as error:
        failure = _lifecycle_provider_failure(error, "status", resolved)
        if failure is None:
            raise
        raise failure from error
    active.snapshot = job.snapshot
    _persist_or_fail("status", resolved, job.snapshot)
    resolved = _adopt_if_requested(root, options, resolved, job.snapshot, "status")
    return LifecycleResult(resolved, job.snapshot)


async def wait_job(
    root: RootOptions,
    options: LifecycleOptions,
    *,
    poll_interval: float = 15.0,
    timeout_seconds: float | None = None,
    on_progress: Callable[[BatchSnapshot], None] | None = None,
) -> LifecycleResult:
    resolved = resolve_job(root, options)
    active = _activate("wait", resolved)
    started = time.monotonic()
    deadline = started + timeout_seconds if timeout_seconds is not None else None
    snapshot: BatchSnapshot | None = None
    try:
        try:
            async with asyncio.timeout(timeout_seconds):
                async with Batchwork() as client:
                    job = await _retry_read(
                        lambda: client.get_batch(_ref(resolved)), deadline=deadline
                    )
                    snapshot = job.snapshot
                    active.snapshot = snapshot
                    _persist_or_fail("wait", resolved, snapshot)
                    if on_progress is not None:
                        on_progress(snapshot)
                    while not is_terminal_status(snapshot.status):
                        if timeout_seconds is None:
                            sleep_for = poll_interval
                        else:
                            remaining = timeout_seconds - (time.monotonic() - started)
                            sleep_for = min(poll_interval, max(0.0, remaining))
                        await asyncio.sleep(sleep_for)
                        snapshot = await _retry_read(job.poll, deadline=deadline)
                        active.snapshot = snapshot
                        _persist_or_fail("wait", resolved, snapshot)
                        if on_progress is not None:
                            on_progress(snapshot)
        except (InterruptionRequested, TerminationRequested) as error:
            raise _signal_failure(error, "wait", resolved, snapshot) from None
        except Exception as error:
            failure = _lifecycle_provider_failure(error, "wait", resolved)
            if failure is None:
                raise
            raise failure from error
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
                    item_successes=(
                        snapshot.request_counts.completed if snapshot is not None else None
                    ),
                    item_failures=(
                        snapshot.request_counts.failed if snapshot is not None else None
                    ),
                    recovery=Recovery(
                        action="resume_wait",
                        command=_recovery_command("wait", resolved),
                    ),
                )
            )
        ) from None
    if snapshot is None:
        raise RuntimeError("batchwork: wait completed without a snapshot")
    resolved = _adopt_if_requested(root, options, resolved, snapshot, "wait")
    return LifecycleResult(resolved, snapshot)


def _nonterminal_failure(resolved: ResolvedJob, snapshot: BatchSnapshot) -> LifecycleFailure:
    return LifecycleFailure(
        ErrorEnvelope(
            error=ErrorDetail(
                code="results_not_ready",
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
    on_result: Callable[[ResolvedJob, BatchResult], None | Awaitable[None]] | None = None,
    on_snapshot: Callable[[ResolvedJob, BatchSnapshot], None] | None = None,
    on_retry: Callable[[], None] | None = None,
    output_is_streaming: bool = False,
    initial_records_emitted: int = 0,
    restart_after_result: bool = True,
) -> LifecycleResult:
    resolved = resolve_job(root, options)
    active = _activate("results", resolved)
    active.records_emitted = initial_records_emitted
    results: list[BatchResult] | None = [] if on_result is None else None
    item_successes = 0
    item_failures = 0
    snapshot: BatchSnapshot | None = None
    job = None
    try:
        async with Batchwork() as client:
            for attempt in range(_READ_ATTEMPTS):
                try:
                    job = await client.get_batch(_ref(resolved))
                    snapshot = job.snapshot
                    active.snapshot = snapshot
                    _persist_or_fail("results", resolved, job.snapshot)
                    if not is_terminal_status(job.status):
                        raise _nonterminal_failure(resolved, job.snapshot)
                    if on_snapshot is not None:
                        on_snapshot(resolved, snapshot)
                    async for item in job._results_from_current_snapshot():
                        if on_result is None:
                            if results is None:
                                raise RuntimeError("batchwork: missing buffered result collection")
                            results.append(item)
                        else:
                            emitted = on_result(resolved, item)
                            if inspect.isawaitable(emitted):
                                await emitted
                        if item.status is BatchResultStatus.SUCCEEDED:
                            item_successes += 1
                        else:
                            item_failures += 1
                        active.records_emitted = initial_records_emitted + (
                            item_successes + item_failures if output_is_streaming else 0
                        )
                        active.item_successes = item_successes
                        active.item_failures = item_failures
                except (InterruptionRequested, TerminationRequested):
                    raise
                except Exception as error:
                    consumed_items = item_successes + item_failures
                    can_restart = consumed_items == 0 or (
                        not output_is_streaming and restart_after_result
                    )
                    if (
                        attempt + 1 < _READ_ATTEMPTS
                        and can_restart
                        and _retryable_read_failure(error)
                    ):
                        await asyncio.sleep(_retry_delay(error, attempt))
                        item_successes = 0
                        item_failures = 0
                        active.records_emitted = initial_records_emitted
                        active.item_successes = 0
                        active.item_failures = 0
                        if results is not None:
                            results.clear()
                        if on_retry is not None:
                            on_retry()
                        continue
                    raise
                break
    except (InterruptionRequested, TerminationRequested) as error:
        records_emitted = initial_records_emitted + (
            item_successes + item_failures if output_is_streaming else 0
        )
        raise _signal_failure(
            error,
            "results",
            resolved,
            snapshot,
            records_emitted=records_emitted,
            item_successes=item_successes,
            item_failures=item_failures,
        ) from None
    except Exception as error:
        if (
            snapshot is not None
            and is_terminal_status(snapshot.status)
            and snapshot.status is not BatchStatus.COMPLETED
        ):
            mapped = provider_failure(error, _failure_context("results", resolved))
            if mapped is not None:
                state = job_state_failure(
                    snapshot.status,
                    _failure_context("results", resolved),
                    item_successes=item_successes,
                    item_failures=item_failures,
                    secondary_retrieval_failed=True,
                    recovery_command=_recovery_command("status", resolved),
                )
                raise LifecycleFailure(state.envelope) from error
        failure = _lifecycle_provider_failure(
            error,
            "results",
            resolved,
            result_stream=True,
            records_emitted=initial_records_emitted
            + (item_successes + item_failures if output_is_streaming else 0),
            item_successes=item_successes,
            item_failures=item_failures,
        )
        if failure is None:
            raise
        raise failure from error
    if job is None:
        raise RuntimeError("batchwork: results completed without a provider job")
    resolved = _adopt_if_requested(root, options, resolved, job.snapshot, "results")
    return LifecycleResult(
        resolved,
        job.snapshot,
        results,
        bool(item_failures),
        item_successes,
        item_failures,
    )


async def cancel_job(root: RootOptions, options: LifecycleOptions) -> LifecycleResult:
    resolved = resolve_job(root, options)
    active = _activate("cancel", resolved)
    snapshot: BatchSnapshot | None = None
    try:
        async with Batchwork() as client:
            job = await client.get_batch(_ref(resolved))
            if not is_terminal_status(job.status):
                await job._request_cancel()
                try:
                    snapshot = await job.poll()
                except Exception as error:
                    failure = _lifecycle_provider_failure(
                        error, "cancel", resolved, cancellation_refresh=True
                    )
                    if failure is None:
                        raise
                    raise failure from error
            else:
                snapshot = job.snapshot
            active.snapshot = snapshot
    except (InterruptionRequested, TerminationRequested) as error:
        raise _signal_failure(error, "cancel", resolved, snapshot) from None
    except Exception as error:
        if isinstance(error, LifecycleFailure):
            raise
        failure = _lifecycle_provider_failure(error, "cancel", resolved)
        if failure is None:
            raise
        raise failure from error
    if snapshot is None:
        raise RuntimeError("batchwork: cancel completed without a snapshot")
    _persist_or_fail("cancel", resolved, snapshot)
    resolved = _adopt_if_requested(root, options, resolved, snapshot, "cancel")
    return LifecycleResult(resolved, snapshot)


def unsuccessful(result: LifecycleResult) -> bool:
    return (
        result.snapshot.status is not BatchStatus.COMPLETED
        or result.item_failed
        or any(item.status is not BatchResultStatus.SUCCEEDED for item in (result.results or ()))
    )


def render_snapshot(
    result: LifecycleResult,
    mode: OutputMode,
    *,
    title: str = "Job status",
    color: bool = False,
) -> str:
    if mode in {OutputMode.JSON, OutputMode.JSONL}:
        return serialize_envelope(
            SnapshotEnvelope(
                job=result.resolved.machine_job,
                routing_fingerprint=result.resolved.machine_fingerprint,
                snapshot=result.snapshot,
            )
        )
    return human_snapshot(result, title=title, color=color)


def render_results(
    result: LifecycleResult,
    mode: OutputMode,
    *,
    materialization: Materialization | None = None,
    color: bool = False,
) -> str:
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
    return human_results(
        result,
        materialization=materialization,
        color=color,
    )


def render_lifecycle_error(failure: LifecycleFailure, mode: OutputMode) -> str:
    if mode in {OutputMode.JSON, OutputMode.JSONL}:
        return serialize_envelope(failure.envelope)
    return human_error(failure.envelope.error)
