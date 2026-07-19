"""Safe conversion of expected CLI failures into the schema-v1 contract."""

from __future__ import annotations

from dataclasses import dataclass

import click
from pydantic import ValidationError

from batchwork._provider_failure import (
    ProviderFailure,
    ProviderFailureError,
    ProviderFailureKind,
)
from batchwork.errors import BatchworkError
from batchwork.types import BatchProvider, BatchStatus

from ._config import ConfigError
from ._contract import (
    ErrorCategory,
    ErrorDetail,
    ErrorEnvelope,
    ExitCode,
    KnownErrorCode,
    Recovery,
)


@dataclass(frozen=True, slots=True)
class FailureContext:
    operation: str
    provider: BatchProvider | None = None
    job: str | None = None
    routing_fingerprint: str | None = None
    profile: str | None = None
    config_path: str | None = None
    registry_path: str | None = None


class CliFailure(Exception):
    """Expected process failure carrying its complete public envelope."""

    def __init__(self, envelope: ErrorEnvelope) -> None:
        super().__init__(envelope.error.message)
        self.envelope = envelope

    @property
    def exit_code(self) -> int:
        return self.envelope.error.exit_code


class TerminationRequested(Exception):
    """Raised by the installed CLI's SIGTERM handler."""


class CliUsageError(click.UsageError):
    """Click usage failure with a stable symbolic code."""

    def __init__(self, message: str, *, code: KnownErrorCode = "usage_error") -> None:
        super().__init__(message)
        self.code = code


def usage_failure(error: click.UsageError, operation: str) -> CliFailure:
    code = error.code if isinstance(error, CliUsageError) else "usage_error"
    return CliFailure(
        ErrorEnvelope(
            error=ErrorDetail(
                code=code,
                category="usage",
                message=error.format_message(),
                exit_code=2,
                retryable=False,
                operation=operation,
            )
        )
    )


def configuration_failure(error: ConfigError, context: FailureContext) -> CliFailure:
    return CliFailure(
        ErrorEnvelope(
            error=ErrorDetail(
                code=error.code,
                category="configuration",
                message=error.message,
                exit_code=3,
                retryable=False,
                operation=context.operation,
                provider=context.provider,
                profile=context.profile,
                config_path=context.config_path,
            )
        )
    )


def provider_failure(
    error: BaseException,
    context: FailureContext,
    *,
    submission: bool = False,
    result_stream: bool = False,
    records_emitted: int = 0,
    item_successes: int = 0,
    item_failures: int = 0,
    cancellation_refresh: bool = False,
) -> CliFailure | None:
    failure = _provider_metadata(error)
    if failure is None:
        return None
    code, category, exit_code, message = _provider_mapping(failure)
    retryable = category == "provider_availability" and not submission
    submission_outcome = None
    recovery = None
    if submission:
        if code == "provider_job_not_found":
            code = "provider_rejected"
            message = "The provider rejected the batch submission endpoint or request."
        if category in {"configuration", "provider_rejection"} or (
            failure.status_code is not None and failure.status_code < 500
        ):
            submission_outcome = "rejected"
        else:
            code = "provider_unavailable"
            message = (
                "Batch submission outcome is unknown. Do not resubmit blindly; inspect the "
                "provider account first because resubmission may duplicate work or cost."
            )
            submission_outcome = "unknown"
            recovery = Recovery(action="inspect_provider_account")
        retryable = False
    if result_stream and category == "provider_availability":
        code = "result_stream_failed"
        message = (
            "Provider result retrieval ended before the complete result set was received. "
            "Replay starts from the beginning; deduplicate by job identity and custom_id."
        )
    if cancellation_refresh:
        code = "cancellation_refresh_failed"
        category = "provider_availability"
        exit_code = 5
        retryable = True
        message = (
            "The cancellation request was sent, but Batchwork could not confirm the latest "
            "remote state."
        )
        recovery = Recovery(
            action="check_status",
            command=["batchwork", "status", context.job] if context.job is not None else None,
        )
    return CliFailure(
        ErrorEnvelope(
            error=ErrorDetail(
                code=code,
                category=category,
                message=message,
                exit_code=exit_code,
                retryable=retryable,
                operation=context.operation,
                provider=context.provider,
                job=context.job,
                routing_fingerprint=context.routing_fingerprint,
                profile=context.profile,
                http_status=failure.status_code,
                request_id=failure.request_id,
                retry_after_seconds=failure.retry_after_seconds,
                submission_outcome=submission_outcome,
                partial_output=True if records_emitted else None,
                records_emitted=records_emitted if records_emitted else None,
                item_successes=item_successes if records_emitted else None,
                item_failures=item_failures if records_emitted else None,
                cancel_requested=True if cancellation_refresh else None,
                recovery=recovery,
            )
        )
    )


def job_state_failure(
    status: BatchStatus,
    context: FailureContext,
    *,
    item_successes: int = 0,
    item_failures: int = 0,
    secondary_retrieval_failed: bool = False,
    recovery_command: list[str] | None = None,
) -> CliFailure:
    code: KnownErrorCode
    if status is BatchStatus.COMPLETED:
        code = "completed_with_item_failures"
        message = "The job completed, but one or more result items failed."
    elif item_successes or item_failures or secondary_retrieval_failed:
        code = "terminal_partial_results"
        message = (
            f'The remote job ended with status "{status.value}"; only provider-available '
            "partial results could be retrieved."
        )
    else:
        state_codes: dict[BatchStatus, KnownErrorCode] = {
            BatchStatus.FAILED: "job_failed",
            BatchStatus.EXPIRED: "job_expired",
            BatchStatus.CANCELLED: "job_cancelled",
        }
        code = state_codes.get(status, "terminal_partial_results")
        message = f'The remote job ended with unsuccessful status "{status.value}".'
    return CliFailure(
        ErrorEnvelope(
            error=ErrorDetail(
                code=code,
                category="job_state",
                message=message,
                exit_code=6,
                retryable=False,
                operation=context.operation,
                provider=context.provider,
                job=context.job,
                routing_fingerprint=context.routing_fingerprint,
                partial_output=True if item_successes or item_failures else None,
                item_successes=item_successes if item_successes or item_failures else None,
                item_failures=item_failures if item_successes or item_failures else None,
                recovery=(
                    Recovery(
                        action="check_status",
                        command=recovery_command or ["batchwork", "status", context.job],
                    )
                    if context.job is not None
                    else None
                ),
            )
        )
    )


def internal_failure(operation: str) -> CliFailure:
    return CliFailure(
        ErrorEnvelope(
            error=ErrorDetail(
                code="internal_error",
                category="internal",
                message="Batchwork encountered an unexpected internal error.",
                exit_code=1,
                retryable=False,
                operation=operation,
            )
        )
    )


def _provider_metadata(error: BaseException) -> ProviderFailure | None:
    if isinstance(error, ProviderFailureError):
        return error.failure
    if isinstance(error, ValidationError):
        return ProviderFailure(ProviderFailureKind.PROTOCOL)
    if isinstance(error, BatchworkError):
        return ProviderFailure(ProviderFailureKind.PROTOCOL)
    return None


def _provider_mapping(
    failure: ProviderFailure,
) -> tuple[KnownErrorCode, ErrorCategory, ExitCode, str]:
    if failure.kind is ProviderFailureKind.AUTHENTICATION:
        return (
            "authentication_failed",
            "configuration",
            3,
            "The provider rejected the configured credentials.",
        )
    if failure.kind is ProviderFailureKind.AUTHORIZATION:
        return (
            "authorization_failed",
            "configuration",
            3,
            "The provider denied this operation for the configured credentials.",
        )
    if failure.kind is ProviderFailureKind.NOT_FOUND:
        return (
            "provider_job_not_found",
            "provider_rejection",
            4,
            "The provider did not recognize the requested job.",
        )
    if failure.kind is ProviderFailureKind.REJECTED:
        return (
            "provider_rejected",
            "provider_rejection",
            4,
            "The provider rejected the requested operation.",
        )
    if failure.kind is ProviderFailureKind.PROTOCOL:
        return (
            "provider_protocol_error",
            "provider_availability",
            5,
            "The provider returned an invalid lifecycle response.",
        )
    if failure.kind is ProviderFailureKind.TRANSPORT:
        return (
            "transport_failed",
            "provider_availability",
            5,
            "The provider request failed before a valid response was received.",
        )
    return (
        "provider_unavailable",
        "provider_availability",
        5,
        "The provider is temporarily unavailable.",
    )
