from __future__ import annotations

from collections.abc import Mapping

import pytest

from batchwork.errors import BatchworkError
from batchwork.server import (
    BatchPoller,
    MemoryWebhookReplayStore,
    TrackedBatch,
    TrackTarget,
    sign_webhook,
    verify_webhook,
)
from batchwork.server import poller as poller_module
from batchwork.stores import MemoryBatchStore
from batchwork.types import (
    BatchProvider,
    BatchRequestCounts,
    BatchSnapshot,
    BatchStatus,
)


class Adapter:
    def __init__(self, snapshots: dict[str, BatchSnapshot]) -> None:
        self.snapshots = snapshots

    async def retrieve(self, batch_id: str, _credentials: object) -> BatchSnapshot:
        return self.snapshots[batch_id]


class FailingOnceAdapter(Adapter):
    def __init__(self, snapshots: dict[str, BatchSnapshot]) -> None:
        super().__init__(snapshots)
        self.attempts = 0

    async def retrieve(self, batch_id: str, credentials: object) -> BatchSnapshot:
        self.attempts += 1
        if self.attempts == 1:
            raise RuntimeError("provider unavailable")
        return await super().retrieve(batch_id, credentials)


class SequencedAdapter:
    def __init__(self, snapshots: list[BatchSnapshot]) -> None:
        self.snapshots = snapshots
        self.attempts = 0

    async def retrieve(self, _batch_id: str, _credentials: object) -> BatchSnapshot:
        snapshot = self.snapshots[self.attempts]
        self.attempts += 1
        return snapshot


class RecordingTransport:
    def __init__(self, statuses: list[int]) -> None:
        self.statuses = statuses
        self.requests: list[tuple[str, bytes, dict[str, str]]] = []

    async def post(self, url: str, body: bytes, headers: Mapping[str, str]) -> int:
        self.requests.append((url, body, dict(headers)))
        return self.statuses.pop(0)


def snapshot(batch_id: str, status: BatchStatus) -> BatchSnapshot:
    return BatchSnapshot(
        id=batch_id,
        provider=BatchProvider.OPENAI,
        status=status,
        request_counts=BatchRequestCounts(total=1, completed=1, failed=0),
    )


def install_adapter(monkeypatch: pytest.MonkeyPatch, *snapshots: BatchSnapshot) -> None:
    adapter = Adapter({item.id: item for item in snapshots})
    monkeypatch.setattr(poller_module, "get_adapter", lambda *_args, **_kwargs: adapter)


async def allow_test_url(_url: object) -> None:
    return None


async def test_track_without_delivery_sink_leaves_store_unchanged() -> None:
    store = MemoryBatchStore()
    poller = BatchPoller(store)

    with pytest.raises(BatchworkError, match="has no webhook_url to deliver to"):
        await poller.track(TrackTarget(id="missing-sink", provider=BatchProvider.OPENAI))

    assert await store.list() == []
    await poller.aclose()


async def test_track_without_webhook_url_allows_completion_sink() -> None:
    store = MemoryBatchStore()
    poller = BatchPoller(store, on_complete=lambda *_args: None)

    record = await poller.track(TrackTarget(id="completion-sink", provider=BatchProvider.OPENAI))

    assert await store.get(record.id) == record
    assert record.webhook_url is None
    await poller.aclose()


async def test_tick_persists_status_and_delivers_terminal_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_adapter(
        monkeypatch,
        snapshot("open", BatchStatus.FINALIZING),
        snapshot("done", BatchStatus.COMPLETED),
    )
    store = MemoryBatchStore()
    transport = RecordingTransport([200])
    poller = BatchPoller(
        store,
        validate_webhook_url=allow_test_url,
        webhook_transport=transport,
    )
    await poller.track(
        TrackTarget(id="open", provider=BatchProvider.OPENAI),
        webhook_url="https://hooks.example.test/batch",
    )
    await poller.track(
        TrackTarget(id="done", provider=BatchProvider.OPENAI),
        webhook_url="https://hooks.example.test/batch",
        secret="secret",
    )
    result = await poller.tick()
    assert result.checked == 2
    assert result.delivered == ("done",)
    open_record = await store.get("open")
    done_record = await store.get("done")
    assert open_record is not None
    assert done_record is not None
    assert open_record.status is BatchStatus.FINALIZING
    assert done_record.delivered_at is not None
    await poller.aclose()


async def test_failed_delivery_retries_with_unique_attempt_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_adapter(monkeypatch, snapshot("retry", BatchStatus.COMPLETED))
    store = MemoryBatchStore()
    transport = RecordingTransport([500, 200])
    poller = BatchPoller(
        store,
        validate_webhook_url=allow_test_url,
        webhook_transport=transport,
    )
    await poller.track(
        TrackTarget(id="retry", provider=BatchProvider.OPENAI),
        webhook_url="https://hooks.example.test/batch",
        secret="secret",
    )
    with pytest.raises(BatchworkError, match=r"failed \(500\)"):
        await poller.tick()
    pending = await store.get("retry")
    assert pending is not None
    assert pending.delivered_at is None
    assert (await poller.tick()).delivered == ("retry",)

    first = transport.requests[0]
    second = transport.requests[1]
    assert first[2]["webhook-event-id"] == second[2]["webhook-event-id"] == "retry"
    assert first[2]["webhook-id"] != second[2]["webhook-id"]
    receiver_store = MemoryWebhookReplayStore()
    await verify_webhook(first[2], first[1], "secret", replay_store=receiver_store)
    await verify_webhook(second[2], second[1], "secret", replay_store=receiver_store)
    await poller.aclose()


async def test_on_error_reports_and_continues(monkeypatch: pytest.MonkeyPatch) -> None:
    install_adapter(
        monkeypatch,
        snapshot("bad", BatchStatus.COMPLETED),
        snapshot("good", BatchStatus.COMPLETED),
    )
    store = MemoryBatchStore()
    seen: list[str] = []

    async def sink(record: TrackedBatch, _snapshot: BatchSnapshot) -> None:
        if record.id == "bad":
            raise RuntimeError("receiver unavailable")

    async def on_error(record: TrackedBatch, _error: Exception) -> None:
        seen.append(record.id)

    poller = BatchPoller(store, on_complete=sink, on_error=on_error)
    await poller.track(TrackTarget(id="bad", provider=BatchProvider.OPENAI))
    await poller.track(TrackTarget(id="good", provider=BatchProvider.OPENAI))
    result = await poller.tick()
    assert seen == ["bad"]
    assert result.delivered == ("good",)
    assert result.failed[0].error == "receiver unavailable"
    await poller.aclose()


async def test_openai_webhook_portable_responses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_adapter(monkeypatch, snapshot("native", BatchStatus.IN_PROGRESS))
    poller = BatchPoller(MemoryBatchStore(), on_complete=lambda *_args: None)
    secret = "secret"
    ignored_body = b'{"type":"thread.run","data":{"id":"x"}}'
    ignored = await poller.handle_openai_webhook(
        sign_webhook(secret, "evt-ignored", ignored_body), ignored_body, secret
    )
    assert ignored.status_code == 202

    missing_body = b'{"type":"batch.completed","data":{}}'
    missing = await poller.handle_openai_webhook(
        sign_webhook(secret, "evt-missing", missing_body), missing_body, secret
    )
    assert missing.status_code == 400
    invalid = await poller.handle_openai_webhook({}, b"{}", secret)
    assert invalid.status_code == 400
    await poller.aclose()


async def test_openai_webhook_retries_same_delivery_after_retrieve_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = snapshot("native-retrieve", BatchStatus.COMPLETED)
    adapter = FailingOnceAdapter({batch.id: batch})
    monkeypatch.setattr(poller_module, "get_adapter", lambda *_args, **_kwargs: adapter)
    store = MemoryBatchStore()
    delivered: list[str] = []

    def sink(record: TrackedBatch, _snapshot: BatchSnapshot) -> None:
        delivered.append(record.id)

    poller = BatchPoller(store, on_complete=sink)
    await poller.track(TrackTarget(id=batch.id, provider=BatchProvider.OPENAI))
    secret = "secret"
    body = b'{"type":"batch.completed","data":{"id":"native-retrieve"}}'
    headers = sign_webhook(secret, "evt-retrieve", body, delivery_id="msg-retrieve")
    replay_store = MemoryWebhookReplayStore()

    with pytest.raises(RuntimeError, match="provider unavailable"):
        await poller.handle_openai_webhook(headers, body, secret, replay_store=replay_store)
    retried = await poller.handle_openai_webhook(headers, body, secret, replay_store=replay_store)
    replay = await poller.handle_openai_webhook(headers, body, secret, replay_store=replay_store)

    assert retried.status_code == 200
    assert replay.status_code == 400
    assert delivered == [batch.id]
    assert adapter.attempts == 2
    await poller.aclose()


async def test_openai_webhook_retries_when_retrieve_lags_terminal_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch_id = "native-lagging"
    adapter = SequencedAdapter(
        [
            snapshot(batch_id, BatchStatus.FINALIZING),
            snapshot(batch_id, BatchStatus.COMPLETED),
        ]
    )
    monkeypatch.setattr(poller_module, "get_adapter", lambda *_args, **_kwargs: adapter)
    store = MemoryBatchStore()
    delivered: list[str] = []
    poller = BatchPoller(store, on_complete=lambda record, _snapshot: delivered.append(record.id))
    await poller.track(TrackTarget(id=batch_id, provider=BatchProvider.OPENAI))
    secret = "secret"
    body = b'{"type":"batch.completed","data":{"id":"native-lagging"}}'
    headers = sign_webhook(secret, "evt-lagging", body, delivery_id="msg-lagging")
    replay_store = MemoryWebhookReplayStore()

    pending = await poller.handle_openai_webhook(headers, body, secret, replay_store=replay_store)
    retried = await poller.handle_openai_webhook(headers, body, secret, replay_store=replay_store)
    replay = await poller.handle_openai_webhook(headers, body, secret, replay_store=replay_store)

    assert pending.status_code == 503
    assert pending.headers == {"retry-after": "1"}
    assert retried.status_code == 200
    assert replay.status_code == 400
    assert delivered == [batch_id]
    assert adapter.attempts == 2
    await poller.aclose()


async def test_openai_webhook_retries_same_delivery_after_sink_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = snapshot("native-deliver", BatchStatus.COMPLETED)
    install_adapter(monkeypatch, batch)
    store = MemoryBatchStore()
    attempts = 0

    def sink(_record: TrackedBatch, _snapshot: BatchSnapshot) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("receiver unavailable")

    poller = BatchPoller(store, on_complete=sink)
    await poller.track(TrackTarget(id=batch.id, provider=BatchProvider.OPENAI))
    secret = "secret"
    body = b'{"type":"batch.completed","data":{"id":"native-deliver"}}'
    headers = sign_webhook(secret, "evt-deliver", body, delivery_id="msg-deliver")
    replay_store = MemoryWebhookReplayStore()

    with pytest.raises(RuntimeError, match="receiver unavailable"):
        await poller.handle_openai_webhook(headers, body, secret, replay_store=replay_store)
    retried = await poller.handle_openai_webhook(headers, body, secret, replay_store=replay_store)
    replay = await poller.handle_openai_webhook(headers, body, secret, replay_store=replay_store)

    assert retried.status_code == 200
    assert replay.status_code == 400
    assert attempts == 2
    await poller.aclose()
