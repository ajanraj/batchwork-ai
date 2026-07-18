from __future__ import annotations

import asyncio

import pytest

from batchwork.errors import BatchworkError
from batchwork.server import (
    MemoryWebhookReplayStore,
    sign_webhook,
    verify_batch_webhook,
    verify_webhook,
)

SECRET = "whsec_dGVzdHNlY3JldA=="


async def test_signing_round_trip_preserves_event_and_delivery_id() -> None:
    body = (
        '{"type":"batch.completed","id":"b1","provider":"openai",'
        '"requestCounts":{"total":1,"completed":1,"failed":0}}'
    )
    headers = sign_webhook(SECRET, "b1", body, 1_000, delivery_id="msg_attempt_1")
    verified = await verify_webhook(
        headers, body, SECRET, replay_store=MemoryWebhookReplayStore(), now=1_001
    )
    assert verified.id == "msg_attempt_1"
    assert verified.event_id == "b1"
    event = await verify_batch_webhook(
        sign_webhook(SECRET, "b1", body, 1_000, delivery_id="msg_attempt_2"),
        body,
        SECRET,
        replay_store=MemoryWebhookReplayStore(),
        now=1_001,
    )
    assert event.id == "b1"


async def test_signing_generates_unique_delivery_attempt_ids() -> None:
    body = b'{"id":"stable-event"}'
    first = sign_webhook(SECRET, "stable-event", body, 1_000)
    second = sign_webhook(SECRET, "stable-event", body, 1_000)

    assert first["webhook-event-id"] == second["webhook-event-id"] == "stable-event"
    assert first["webhook-id"] != second["webhook-id"]

    store = MemoryWebhookReplayStore()
    await verify_webhook(first, body, SECRET, replay_store=store, now=1_001)
    await verify_webhook(second, body, SECRET, replay_store=store, now=1_001)


async def test_failed_custom_handler_can_release_verified_claim() -> None:
    body = b'{"id":"retryable-event"}'
    headers = sign_webhook(
        SECRET,
        "retryable-event",
        body,
        1_000,
        delivery_id="retryable-delivery",
    )
    store = MemoryWebhookReplayStore()
    verified = await verify_webhook(headers, body, SECRET, replay_store=store, now=1_001)

    await verified.release()

    retried = await verify_webhook(headers, body, SECRET, replay_store=store, now=1_001)
    assert retried.id == "retryable-delivery"


async def test_unsigned_event_header_cannot_change_idempotency_identity() -> None:
    body = b'{"id":"trusted-event"}'
    headers = sign_webhook(
        SECRET,
        "trusted-event",
        body,
        1_000,
        delivery_id="trusted-delivery",
    )
    headers["webhook-event-id"] = "intermediary-event"

    verified = await verify_webhook(
        headers,
        body,
        SECRET,
        replay_store=MemoryWebhookReplayStore(),
        now=1_001,
    )

    assert verified.id == "trusted-delivery"
    assert verified.event_id == "trusted-event"


async def test_replay_and_tampering_are_rejected() -> None:
    body = b"{}"
    headers = sign_webhook(SECRET, "event", body)
    store = MemoryWebhookReplayStore()
    await verify_webhook(headers, body, SECRET, replay_store=store)
    with pytest.raises(BatchworkError, match="replay detected"):
        await verify_webhook(headers, body, SECRET, replay_store=store)
    with pytest.raises(BatchworkError, match="verification failed"):
        await verify_webhook(
            sign_webhook(SECRET, "other", body, delivery_id="different"),
            b'{"tampered":true}',
            SECRET,
            replay_store=store,
        )


async def test_atomic_store_rejects_concurrent_replay() -> None:
    body = b"{}"
    headers = sign_webhook(SECRET, "event-race", body)
    store = MemoryWebhookReplayStore()
    outcomes = await asyncio.gather(
        verify_webhook(headers, body, SECRET, replay_store=store),
        verify_webhook(headers, body, SECRET, replay_store=store),
        return_exceptions=True,
    )
    assert sum(not isinstance(outcome, Exception) for outcome in outcomes) == 1
    assert sum(isinstance(outcome, BatchworkError) for outcome in outcomes) == 1


async def test_missing_and_stale_headers_are_rejected() -> None:
    with pytest.raises(BatchworkError, match="missing webhook signature"):
        await verify_webhook({}, b"{}", SECRET)
    headers = sign_webhook(SECRET, "old", b"{}", 1)
    with pytest.raises(BatchworkError, match="outside tolerance"):
        await verify_webhook(headers, b"{}", SECRET, now=10_000)
