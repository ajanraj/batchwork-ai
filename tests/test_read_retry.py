from __future__ import annotations

import ssl

import httpx
import pytest

from batchwork._provider_failure import ProviderFailure, ProviderFailureError, ProviderFailureKind
from batchwork._read_retry import (
    _ReadRetryDeadlineExceeded,
    _retry_delay,
    _retry_read,
    _retryable_read_failure,
)


def _failure(
    kind: ProviderFailureKind,
    *,
    status_code: int | None = None,
    cause: BaseException | None = None,
    retry_after_seconds: int | None = None,
) -> ProviderFailureError:
    error = ProviderFailureError(
        "safe",
        ProviderFailure(
            kind,
            status_code=status_code,
            retry_after_seconds=retry_after_seconds,
        ),
    )
    error.__cause__ = cause
    return error


@pytest.mark.parametrize("status_code", [408, 429, 500, 502, 503, 504])
def test_retryable_read_failure_accepts_exact_http_allowlist(status_code: int) -> None:
    assert _retryable_read_failure(
        _failure(ProviderFailureKind.UNAVAILABLE, status_code=status_code)
    )


@pytest.mark.parametrize("status_code", [None, 409, 425, 501])
def test_retryable_read_failure_rejects_other_unavailable_statuses(
    status_code: int | None,
) -> None:
    assert not _retryable_read_failure(
        _failure(ProviderFailureKind.UNAVAILABLE, status_code=status_code)
    )


@pytest.mark.parametrize(
    "cause",
    [
        httpx.ConnectError("connection failed"),
        httpx.ConnectTimeout("connection timed out"),
        httpx.ReadError("read failed"),
        httpx.ReadTimeout("read timed out"),
    ],
)
def test_retryable_read_failure_accepts_exact_transport_allowlist(
    cause: httpx.HTTPError,
) -> None:
    assert _retryable_read_failure(_failure(ProviderFailureKind.TRANSPORT, cause=cause))


def test_retryable_read_failure_rejects_tls_in_cause_chain() -> None:
    cause = httpx.ReadError("read failed")
    cause.__cause__ = ssl.SSLError("certificate validation failed")

    assert not _retryable_read_failure(_failure(ProviderFailureKind.TRANSPORT, cause=cause))


@pytest.mark.parametrize(
    ("retry_after_seconds", "expected"),
    [(0, 0.0), (30, 30.0), (3600, 60.0)],
)
def test_retry_delay_honors_retry_after_with_sixty_second_cap(
    retry_after_seconds: int,
    expected: float,
) -> None:
    error = _failure(
        ProviderFailureKind.UNAVAILABLE,
        status_code=503,
        retry_after_seconds=retry_after_seconds,
    )

    assert _retry_delay(error, 0) == expected


def test_retry_delay_uses_half_to_full_jitter(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[float, float]] = []

    def uniform(lower: float, upper: float) -> float:
        seen.append((lower, upper))
        return upper

    monkeypatch.setattr("batchwork._read_retry.random.uniform", uniform)

    assert _retry_delay(RuntimeError("read failed"), 1) == 0.5
    assert seen == [(0.25, 0.5)]


@pytest.mark.asyncio
async def test_retry_read_preserves_final_error_after_three_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    errors = [_failure(ProviderFailureKind.UNAVAILABLE, status_code=503) for _ in range(3)]
    attempts = 0

    async def read() -> object:
        nonlocal attempts
        error = errors[attempts]
        attempts += 1
        raise error

    async def sleep(delay: float) -> None:
        assert delay >= 0

    monkeypatch.setattr("batchwork._read_retry.asyncio.sleep", sleep)

    with pytest.raises(ProviderFailureError) as caught:
        await _retry_read(read)

    assert attempts == 3
    assert caught.value is errors[-1]


@pytest.mark.asyncio
async def test_retry_read_recovers_within_three_attempt_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0

    async def read() -> object:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise _failure(ProviderFailureKind.UNAVAILABLE, status_code=503)
        return "recovered"

    delays: list[float] = []

    async def sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr("batchwork._read_retry.asyncio.sleep", sleep)
    monkeypatch.setattr("batchwork._read_retry._retry_delay", lambda *_: 0.25)

    assert await _retry_read(read) == "recovered"
    assert attempts == 3
    assert delays == [0.25, 0.25]


@pytest.mark.asyncio
async def test_retry_read_does_not_retry_nonretryable_failure() -> None:
    error = RuntimeError("read failed")
    attempts = 0

    async def read() -> object:
        nonlocal attempts
        attempts += 1
        raise error

    with pytest.raises(RuntimeError) as caught:
        await _retry_read(read)

    assert caught.value is error
    assert attempts == 1


@pytest.mark.asyncio
async def test_retry_read_truncates_sleep_at_deadline_then_raises_sentinel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0

    async def read() -> object:
        nonlocal attempts
        attempts += 1
        raise _failure(
            ProviderFailureKind.UNAVAILABLE,
            status_code=503,
            retry_after_seconds=1,
        )

    delays: list[float] = []

    async def sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr("batchwork._read_retry.time.monotonic", lambda: 10.0)
    monkeypatch.setattr("batchwork._read_retry.asyncio.sleep", sleep)

    with pytest.raises(_ReadRetryDeadlineExceeded):
        await _retry_read(read, deadline=10.2)

    assert attempts == 1
    assert delays == pytest.approx([0.2])
