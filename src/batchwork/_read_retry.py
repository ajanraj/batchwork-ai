"""Provider-neutral retry policy for safe read operations."""

from __future__ import annotations

import asyncio
import random
import ssl
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

import httpx

from ._provider_failure import ProviderFailureError, ProviderFailureKind

_READ_ATTEMPTS = 3
_RETRYABLE_HTTP_STATUSES = frozenset({408, 429, 500, 502, 503, 504})
_MAX_RETRY_DELAY_SECONDS = 60.0
_ReadResult = TypeVar("_ReadResult")


class _ReadRetryDeadlineExceeded(TimeoutError):
    """The local read-retry deadline expired before another attempt."""


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
                    raise _ReadRetryDeadlineExceeded from None
                if delay >= remaining:
                    await asyncio.sleep(remaining)
                    raise _ReadRetryDeadlineExceeded from None
            await asyncio.sleep(delay)
    raise RuntimeError("batchwork: exhausted retry loop")
