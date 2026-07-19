"""Safe, structured failures raised by provider HTTP helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from enum import StrEnum
from typing import Final

import httpx

from .errors import BatchworkError

_REQUEST_ID_HEADERS: Final = (
    "x-request-id",
    "request-id",
    "anthropic-request-id",
    "x-goog-request-id",
    "x-amzn-requestid",
    "x-amz-request-id",
)
_MAX_REQUEST_ID_LENGTH: Final = 256
_MAX_RETRY_AFTER_SECONDS: Final = 3600
_RETRYABLE_CLIENT_STATUSES: Final = frozenset({408, 425, 429})


class ProviderFailureKind(StrEnum):
    """Failure categories safe for caller policy decisions."""

    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    NOT_FOUND = "not_found"
    REJECTED = "rejected"
    UNAVAILABLE = "unavailable"
    TRANSPORT = "transport"
    PROTOCOL = "protocol"


@dataclass(frozen=True, slots=True)
class ProviderFailure:
    """Provider failure metadata with no request, response body, or headers."""

    kind: ProviderFailureKind
    status_code: int | None = None
    request_id: str | None = None
    retry_after_seconds: int | None = None


class ProviderFailureError(BatchworkError):
    """A compatibility-preserving ``BatchworkError`` with safe failure metadata."""

    def __init__(self, message: str, failure: ProviderFailure) -> None:
        super().__init__(message)
        self.failure = failure


def http_failure(response: httpx.Response) -> ProviderFailure:
    """Classify a non-success response without retaining it."""
    status_code = response.status_code
    if status_code == 401:
        kind = ProviderFailureKind.AUTHENTICATION
    elif status_code == 403:
        kind = ProviderFailureKind.AUTHORIZATION
    elif status_code == 404:
        kind = ProviderFailureKind.NOT_FOUND
    elif status_code in _RETRYABLE_CLIENT_STATUSES or status_code >= 500:
        kind = ProviderFailureKind.UNAVAILABLE
    else:
        kind = ProviderFailureKind.REJECTED
    return ProviderFailure(
        kind=kind,
        status_code=status_code,
        request_id=_request_id(response),
        retry_after_seconds=_retry_after_seconds(response),
    )


def transport_failure() -> ProviderFailure:
    return ProviderFailure(kind=ProviderFailureKind.TRANSPORT)


def protocol_failure(response: httpx.Response) -> ProviderFailure:
    return ProviderFailure(
        kind=ProviderFailureKind.PROTOCOL,
        status_code=response.status_code,
        request_id=_request_id(response),
    )


def _request_id(response: httpx.Response) -> str | None:
    for name in _REQUEST_ID_HEADERS:
        value = response.headers.get(name)
        if value is not None and _safe_request_id(value):
            return value
    return None


def _safe_request_id(value: str) -> bool:
    return 0 < len(value) <= _MAX_REQUEST_ID_LENGTH and value.isascii() and value.isprintable()


def _retry_after_seconds(response: httpx.Response) -> int | None:
    value = response.headers.get("retry-after")
    if value is None:
        return None
    try:
        seconds = int(value)
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=UTC)
        seconds = max(0, int((retry_at - datetime.now(UTC)).total_seconds()))
    if seconds < 0:
        return None
    return min(seconds, _MAX_RETRY_AFTER_SECONDS)
