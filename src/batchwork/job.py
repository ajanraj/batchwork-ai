"""Stateful handle for a submitted or rehydrated batch."""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Protocol

from .errors import BatchTimeoutError
from .types import (
    BatchRequestCounts,
    BatchResult,
    BatchSnapshot,
    BatchStatus,
    ProviderCredentials,
    is_terminal_status,
)


class _Adapter(Protocol):
    async def retrieve(self, id: str, credentials: ProviderCredentials) -> BatchSnapshot: ...
    def results(self, id: str, credentials: ProviderCredentials) -> AsyncIterator[BatchResult]: ...
    async def cancel(self, id: str, credentials: ProviderCredentials) -> None: ...


PollCallback = Callable[[BatchSnapshot], object | Awaitable[object]]


async def _call(callback: PollCallback | None, snapshot: BatchSnapshot) -> None:
    if callback is None:
        return
    result = callback(snapshot)
    if inspect.isawaitable(result):
        await result


class BatchJob:
    """Poll, wait for, cancel, and collect one provider batch."""

    def __init__(
        self,
        adapter: _Adapter,
        credentials: ProviderCredentials,
        snapshot: BatchSnapshot,
        *,
        ensure_open: Callable[[], None] | None = None,
    ) -> None:
        self._adapter = adapter
        self._credentials = credentials
        self._snapshot = snapshot
        self._ensure_open = ensure_open or (lambda: None)
        self.id = snapshot.id
        self.provider = snapshot.provider

    @property
    def status(self) -> BatchStatus:
        return self._snapshot.status

    @property
    def request_counts(self) -> BatchRequestCounts:
        return self._snapshot.request_counts

    @property
    def snapshot(self) -> BatchSnapshot:
        return self._snapshot

    async def poll(self) -> BatchSnapshot:
        self._ensure_open()
        self._snapshot = await self._adapter.retrieve(self.id, self._credentials)
        return self._snapshot

    def wait(
        self,
        *,
        poll_interval: float = 15.0,
        timeout: float | None = None,
        on_poll: PollCallback | None = None,
    ) -> Awaitable[BatchSnapshot]:
        return self._wait(
            poll_interval=poll_interval,
            timeout_seconds=timeout,
            on_poll=on_poll,
        )

    async def _wait(
        self,
        *,
        poll_interval: float,
        timeout_seconds: float | None,
        on_poll: PollCallback | None,
    ) -> BatchSnapshot:
        if poll_interval < 0:
            raise ValueError("poll_interval must be non-negative")
        if timeout_seconds is not None and timeout_seconds < 0:
            raise ValueError("timeout must be non-negative")
        deadline = None if timeout_seconds is None else time.monotonic() + timeout_seconds

        while True:
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                raise BatchTimeoutError(f'batchwork: timed out waiting for batch "{self.id}"')

            if remaining is None:
                snapshot = await self.poll()
                await _call(on_poll, snapshot)
            else:
                poll_timeout = asyncio.timeout(remaining)
                try:
                    async with poll_timeout:
                        snapshot = await self.poll()
                        await _call(on_poll, snapshot)
                except TimeoutError:
                    if not poll_timeout.expired():
                        raise
                    raise BatchTimeoutError(
                        f'batchwork: timed out waiting for batch "{self.id}"'
                    ) from None

            if is_terminal_status(snapshot.status):
                return snapshot

            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                raise BatchTimeoutError(f'batchwork: timed out waiting for batch "{self.id}"')
            sleep_for = poll_interval if remaining is None else min(poll_interval, remaining)
            await asyncio.sleep(sleep_for)

    async def results(self) -> AsyncIterator[BatchResult]:
        self._ensure_open()
        async for result in self._adapter.results(self.id, self._credentials):
            self._ensure_open()
            yield result

    async def collect(self) -> list[BatchResult]:
        return [result async for result in self.results()]

    async def cancel(self) -> BatchSnapshot:
        self._ensure_open()
        await self._adapter.cancel(self.id, self._credentials)
        return await self.poll()


__all__ = ["BatchJob", "PollCallback", "is_terminal_status"]
