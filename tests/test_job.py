import asyncio
from collections.abc import AsyncIterator

import pytest

from batchwork.errors import BatchClosedError, BatchTimeoutError
from batchwork.job import BatchJob
from batchwork.types import (
    BatchProvider,
    BatchRequestCounts,
    BatchResult,
    BatchResultStatus,
    BatchSnapshot,
    BatchStatus,
    ProviderCredentials,
)


def snapshot(status: BatchStatus) -> BatchSnapshot:
    return BatchSnapshot(
        id="batch_1",
        provider=BatchProvider.OPENAI,
        status=status,
        request_counts=BatchRequestCounts(
            total=1,
            completed=1 if status is BatchStatus.COMPLETED else 0,
            failed=0,
        ),
    )


class FakeAdapter:
    def __init__(self, statuses: list[BatchStatus]) -> None:
        self.statuses = statuses
        self.cancelled = False

    async def retrieve(self, id: str, credentials: ProviderCredentials) -> BatchSnapshot:
        return snapshot(self.statuses.pop(0) if self.statuses else BatchStatus.COMPLETED)

    async def results(
        self, id: str, credentials: ProviderCredentials
    ) -> AsyncIterator[BatchResult]:
        yield BatchResult(custom_id="a", status=BatchResultStatus.SUCCEEDED, text="hello")

    async def cancel(self, id: str, credentials: ProviderCredentials) -> None:
        self.cancelled = True


class SnapshotResultsAdapter(FakeAdapter):
    def __init__(self) -> None:
        super().__init__([BatchStatus.COMPLETED])
        self.retrieve_calls = 0
        self.result_calls = 0
        self.result_snapshot: BatchSnapshot | None = None

    async def retrieve(self, id: str, credentials: ProviderCredentials) -> BatchSnapshot:
        self.retrieve_calls += 1
        return await super().retrieve(id, credentials)

    async def results(
        self, id: str, credentials: ProviderCredentials
    ) -> AsyncIterator[BatchResult]:
        self.result_calls += 1
        raise AssertionError("snapshot-aware lifecycle path must not call results()")
        yield BatchResult(custom_id="unused", status=BatchResultStatus.SUCCEEDED)

    async def results_from_snapshot(
        self, value: BatchSnapshot, credentials: ProviderCredentials
    ) -> AsyncIterator[BatchResult]:
        self.result_snapshot = value
        yield BatchResult(custom_id="a", status=BatchResultStatus.SUCCEEDED, text="hello")


class BlockingRetrieveAdapter(FakeAdapter):
    def __init__(self, retrieves_before_block: int) -> None:
        super().__init__([BatchStatus.IN_PROGRESS] * retrieves_before_block)
        self.retrieves_before_block = retrieves_before_block
        self.retrieve_calls = 0
        self.retrieve_cancelled = False

    async def retrieve(self, id: str, credentials: ProviderCredentials) -> BatchSnapshot:
        self.retrieve_calls += 1
        if self.retrieve_calls <= self.retrieves_before_block:
            return await super().retrieve(id, credentials)
        try:
            await asyncio.Event().wait()
        finally:
            self.retrieve_cancelled = True


class TimeoutRetrieveAdapter(FakeAdapter):
    async def retrieve(self, id: str, credentials: ProviderCredentials) -> BatchSnapshot:
        raise TimeoutError("provider timeout")


@pytest.mark.asyncio
async def test_wait_polls_immediately_and_calls_sync_or_async_callback() -> None:
    adapter = FakeAdapter([BatchStatus.IN_PROGRESS, BatchStatus.COMPLETED])
    seen: list[BatchStatus] = []

    async def on_poll(value: BatchSnapshot) -> None:
        seen.append(value.status)

    job = BatchJob(adapter, ProviderCredentials(), snapshot(BatchStatus.COMPLETED))
    completed = await job.wait(poll_interval=0, timeout=1, on_poll=on_poll)
    assert completed.status is BatchStatus.COMPLETED
    assert seen == [BatchStatus.IN_PROGRESS, BatchStatus.COMPLETED]


@pytest.mark.asyncio
async def test_wait_calls_sync_callback_within_timeout() -> None:
    seen: list[BatchStatus] = []

    def on_poll(value: BatchSnapshot) -> None:
        seen.append(value.status)

    job = BatchJob(
        FakeAdapter([BatchStatus.COMPLETED]),
        ProviderCredentials(),
        snapshot(BatchStatus.IN_PROGRESS),
    )
    await job.wait(timeout=1, on_poll=on_poll)
    assert seen == [BatchStatus.COMPLETED]


@pytest.mark.asyncio
async def test_wait_uses_timeout() -> None:
    adapter = FakeAdapter([BatchStatus.IN_PROGRESS] * 10)
    job = BatchJob(adapter, ProviderCredentials(), snapshot(BatchStatus.IN_PROGRESS))
    with pytest.raises(BatchTimeoutError, match="timed out"):
        await job.wait(poll_interval=0.01, timeout=0)


@pytest.mark.asyncio
@pytest.mark.parametrize("retrieves_before_block", [0, 1])
async def test_wait_timeout_bounds_blocked_retrieve(retrieves_before_block: int) -> None:
    adapter = BlockingRetrieveAdapter(retrieves_before_block)
    job = BatchJob(adapter, ProviderCredentials(), snapshot(BatchStatus.IN_PROGRESS))

    with pytest.raises(
        BatchTimeoutError,
        match='batchwork: timed out waiting for batch "batch_1"',
    ):
        async with asyncio.timeout(0.5):
            await job.wait(poll_interval=0, timeout=0.01)

    assert adapter.retrieve_calls == retrieves_before_block + 1
    assert adapter.retrieve_cancelled


@pytest.mark.asyncio
async def test_wait_timeout_bounds_stalled_callback() -> None:
    callback_cancelled = asyncio.Event()

    async def on_poll(value: BatchSnapshot) -> None:
        try:
            await asyncio.Event().wait()
        finally:
            callback_cancelled.set()

    job = BatchJob(
        FakeAdapter([BatchStatus.COMPLETED]),
        ProviderCredentials(),
        snapshot(BatchStatus.IN_PROGRESS),
    )

    with pytest.raises(
        BatchTimeoutError,
        match=r'^batchwork: timed out waiting for batch "batch_1"$',
    ):
        async with asyncio.timeout(0.5):
            await job.wait(timeout=0.01, on_poll=on_poll)

    await asyncio.wait_for(callback_cancelled.wait(), timeout=0.1)


@pytest.mark.asyncio
async def test_wait_preserves_callback_timeout_error() -> None:
    async def on_poll(value: BatchSnapshot) -> None:
        raise TimeoutError("callback timeout")

    job = BatchJob(
        FakeAdapter([BatchStatus.COMPLETED]),
        ProviderCredentials(),
        snapshot(BatchStatus.IN_PROGRESS),
    )

    with pytest.raises(TimeoutError, match="callback timeout"):
        await job.wait(timeout=1, on_poll=on_poll)


@pytest.mark.asyncio
async def test_wait_preserves_provider_timeout_error() -> None:
    job = BatchJob(
        TimeoutRetrieveAdapter([]),
        ProviderCredentials(),
        snapshot(BatchStatus.IN_PROGRESS),
    )

    with pytest.raises(TimeoutError, match="provider timeout"):
        await job.wait(timeout=1)


@pytest.mark.asyncio
async def test_wait_cancellation_cancels_callback() -> None:
    callback_started = asyncio.Event()
    callback_cancelled = asyncio.Event()

    async def on_poll(value: BatchSnapshot) -> None:
        callback_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            callback_cancelled.set()

    job = BatchJob(
        FakeAdapter([BatchStatus.COMPLETED]),
        ProviderCredentials(),
        snapshot(BatchStatus.IN_PROGRESS),
    )
    wait_task = asyncio.create_task(job.wait(timeout=1, on_poll=on_poll))
    await callback_started.wait()
    wait_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await wait_task

    await asyncio.wait_for(callback_cancelled.wait(), timeout=0.1)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status",
    [BatchStatus.IN_PROGRESS, BatchStatus.FAILED, BatchStatus.EXPIRED, BatchStatus.CANCELLED],
)
async def test_results_delegate_for_every_batch_state(status: BatchStatus) -> None:
    adapter = FakeAdapter([])
    job = BatchJob(adapter, ProviderCredentials(), snapshot(status))
    assert [result.text for result in await job.collect()] == ["hello"]


@pytest.mark.asyncio
async def test_results_refresh_once_and_reuse_snapshot_for_output() -> None:
    adapter = SnapshotResultsAdapter()
    job = BatchJob(adapter, ProviderCredentials(), snapshot(BatchStatus.IN_PROGRESS))

    assert [result.text for result in await job.collect()] == ["hello"]
    assert adapter.retrieve_calls == 1
    assert adapter.result_calls == 0
    assert adapter.result_snapshot is job.snapshot
    assert job.status is BatchStatus.COMPLETED


@pytest.mark.asyncio
async def test_collect_cancel_and_owner_close() -> None:
    adapter = FakeAdapter([BatchStatus.COMPLETED, BatchStatus.CANCELLING])
    open_state = True

    def ensure_open() -> None:
        if not open_state:
            raise BatchClosedError("closed")

    job = BatchJob(
        adapter,
        ProviderCredentials(),
        snapshot(BatchStatus.COMPLETED),
        ensure_open=ensure_open,
    )
    assert [result.text for result in await job.collect()] == ["hello"]
    assert (await job.cancel()).status is BatchStatus.CANCELLING
    assert adapter.cancelled
    open_state = False
    with pytest.raises(BatchClosedError):
        await job.poll()
