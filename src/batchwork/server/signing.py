"""Standard Webhooks-compatible signing and replay protection."""

from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import hmac
import json
import math
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from batchwork.errors import BatchworkError
from batchwork.server.models import BatchWebhookEvent

SIGNATURE_VERSION = "v1"
TOLERANCE_SECONDS = 300
SECRET_PREFIX = "whsec_"


@runtime_checkable
class WebhookReplayStore(Protocol):
    """Persistence for consumed delivery-attempt identifiers."""

    async def get(self, delivery_id: str) -> float | None: ...

    async def set(self, delivery_id: str, expires_at: float) -> None: ...


class AtomicWebhookReplayStore(WebhookReplayStore, Protocol):
    """Replay store with an atomic claim operation for distributed receivers."""

    async def claim(self, delivery_id: str, expires_at: float, now: float) -> bool: ...


class MemoryWebhookReplayStore:
    """Process-local atomic replay store."""

    def __init__(self) -> None:
        self._entries: dict[str, float] = {}
        self._lock = asyncio.Lock()

    async def get(self, delivery_id: str) -> float | None:
        async with self._lock:
            return self._entries.get(delivery_id)

    async def set(self, delivery_id: str, expires_at: float) -> None:
        async with self._lock:
            self._entries[delivery_id] = expires_at

    async def claim(self, delivery_id: str, expires_at: float, now: float) -> bool:
        async with self._lock:
            expired = [key for key, expiry in self._entries.items() if expiry <= now]
            for key in expired:
                self._entries.pop(key, None)
            existing = self._entries.get(delivery_id)
            if existing is not None and existing > now:
                return False
            self._entries[delivery_id] = expires_at
            return True


@dataclass(frozen=True, slots=True)
class VerifiedWebhook:
    """Authenticated webhook payload and its two idempotency identities."""

    id: str
    event_id: str
    timestamp: int
    body: bytes
    _replay_store: WebhookReplayStore | None = field(default=None, repr=False, compare=False)

    async def release(self) -> None:
        """Release this delivery claim after handling fails."""

        if self._replay_store is None:
            raise BatchworkError("batchwork: verified webhook has no replay claim to release.")
        await self._replay_store.set(self.id, 0.0)


_default_replay_store = MemoryWebhookReplayStore()
_fallback_locks: dict[int, asyncio.Lock] = {}


def _resolve_replay_store(replay_store: WebhookReplayStore | None) -> WebhookReplayStore:
    return replay_store if replay_store is not None else _default_replay_store


def _secret_bytes(secret: str) -> bytes:
    if not secret.startswith(SECRET_PREFIX):
        return secret.encode()
    try:
        return base64.b64decode(secret.removeprefix(SECRET_PREFIX), validate=True)
    except (binascii.Error, ValueError) as error:
        raise BatchworkError("batchwork: invalid webhook signing secret.") from error


def _body_bytes(body: str | bytes) -> bytes:
    return body.encode() if isinstance(body, str) else body


def _header_identity(value: str, label: str) -> str:
    if not value or any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise BatchworkError(f"batchwork: {label} must be a valid HTTP header value.")
    try:
        value.encode("ascii")
    except UnicodeEncodeError as error:
        raise BatchworkError(f"batchwork: {label} must be a valid HTTP header value.") from error
    return value


def _signature(secret: str, content: bytes) -> str:
    digest = hmac.new(_secret_bytes(secret), content, hashlib.sha256).digest()
    return base64.b64encode(digest).decode("ascii")


def sign_webhook(
    secret: str,
    event_id: str,
    body: str | bytes,
    timestamp_seconds: float | None = None,
    *,
    delivery_id: str | None = None,
) -> dict[str, str]:
    """Sign one attempt; ``event_id`` remains stable across delivery retries."""

    timestamp = math.floor(time.time() if timestamp_seconds is None else timestamp_seconds)
    event_id = _header_identity(event_id, "webhook event id")
    attempt_id = _header_identity(
        delivery_id if delivery_id is not None else f"msg_{uuid.uuid4().hex}",
        "webhook delivery id",
    )
    payload = b".".join((attempt_id.encode(), str(timestamp).encode(), _body_bytes(body)))
    return {
        "webhook-event-id": event_id,
        "webhook-id": attempt_id,
        "webhook-signature": f"{SIGNATURE_VERSION},{_signature(secret, payload)}",
        "webhook-timestamp": str(timestamp),
    }


def _headers(headers: Mapping[str, str]) -> dict[str, str]:
    return {key.lower(): value for key, value in headers.items()}


def _authenticated_event_id(body: bytes, delivery_id: str) -> str:
    """Derive idempotency identity only from signed material."""

    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return delivery_id
    event_id = payload.get("id") if isinstance(payload, dict) else None
    return event_id if isinstance(event_id, str) and event_id else delivery_id


async def _claim(
    store: WebhookReplayStore,
    delivery_id: str,
    expires_at: float,
    now: float,
) -> None:
    claim = getattr(store, "claim", None)
    if claim is not None:
        if not await claim(delivery_id, expires_at, now):
            raise BatchworkError("batchwork: webhook replay detected.")
        return

    # Serializes get/set for legacy stores within this process. Distributed
    # receivers must provide atomic ``claim`` to guarantee race protection.
    lock = _fallback_locks.setdefault(id(store), asyncio.Lock())
    async with lock:
        existing = await store.get(delivery_id)
        if existing is not None and existing > now:
            raise BatchworkError("batchwork: webhook replay detected.")
        await store.set(delivery_id, expires_at)


async def verify_webhook(
    headers: Mapping[str, str],
    body: str | bytes,
    secret: str,
    *,
    replay_store: WebhookReplayStore | None = None,
    now: float | None = None,
) -> VerifiedWebhook:
    """Authenticate and consume a signed webhook delivery attempt."""

    normalized = _headers(headers)
    delivery_id = normalized.get("webhook-id")
    raw_timestamp = normalized.get("webhook-timestamp")
    raw_signatures = normalized.get("webhook-signature")
    if not delivery_id or not raw_timestamp or not raw_signatures:
        raise BatchworkError("batchwork: missing webhook signature headers.")

    current = time.time() if now is None else now
    try:
        timestamp_value = float(raw_timestamp)
    except ValueError as error:
        raise BatchworkError("batchwork: webhook timestamp outside tolerance.") from error
    if not math.isfinite(timestamp_value) or abs(current - timestamp_value) > TOLERANCE_SECONDS:
        raise BatchworkError("batchwork: webhook timestamp outside tolerance.")

    raw_body = _body_bytes(body)
    content = b".".join((delivery_id.encode(), raw_timestamp.encode(), raw_body))
    expected = _signature(secret, content)
    signatures = []
    for item in raw_signatures.split():
        version, separator, signature = item.partition(",")
        if separator and version == SIGNATURE_VERSION:
            signatures.append(signature)
    if not any(hmac.compare_digest(expected, candidate) for candidate in signatures):
        raise BatchworkError("batchwork: webhook signature verification failed.")

    resolved_replay_store = _resolve_replay_store(replay_store)
    await _claim(
        resolved_replay_store,
        delivery_id,
        timestamp_value + TOLERANCE_SECONDS,
        current,
    )
    return VerifiedWebhook(
        id=delivery_id,
        event_id=_authenticated_event_id(raw_body, delivery_id),
        timestamp=math.floor(timestamp_value),
        body=raw_body,
        _replay_store=resolved_replay_store,
    )


async def verify_batch_webhook(
    headers: Mapping[str, str],
    body: str | bytes,
    secret: str,
    *,
    replay_store: WebhookReplayStore | None = None,
    now: float | None = None,
) -> BatchWebhookEvent:
    """Verify and parse a unified batchwork webhook event."""

    verified = await verify_webhook(headers, body, secret, replay_store=replay_store, now=now)
    try:
        payload = json.loads(verified.body)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise BatchworkError("batchwork: webhook body is not valid JSON.") from error
    return BatchWebhookEvent.model_validate(payload)


__all__ = [
    "AtomicWebhookReplayStore",
    "MemoryWebhookReplayStore",
    "VerifiedWebhook",
    "WebhookReplayStore",
    "sign_webhook",
    "verify_batch_webhook",
    "verify_webhook",
]
