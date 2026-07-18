"""Managed polling and portable native webhook handling."""

from __future__ import annotations

import json
import uuid
from collections.abc import Awaitable, Callable, Mapping
from datetime import datetime

import httpx

from batchwork.errors import BatchworkError
from batchwork.providers import get_adapter
from batchwork.server.models import (
    BatchWebhookEvent,
    TickFailure,
    TickResult,
    TrackedBatch,
    TrackTarget,
    WebhookEventType,
    WebhookResponse,
)
from batchwork.server.signing import (
    WebhookReplayStore,
    sign_webhook,
    verify_webhook,
)
from batchwork.server.transport import (
    PinnedWebhookTransport,
    WebhookTransport,
    WebhookUrlValidator,
    parse_webhook_url,
)
from batchwork.server.transport import (
    validate_webhook_url as default_validate_webhook_url,
)
from batchwork.stores.base import BatchStore
from batchwork.types import BatchProvider, BatchSnapshot, BatchStatus, ProviderCredentials

CredentialResolver = (
    ProviderCredentials
    | Callable[[BatchProvider], ProviderCredentials | Awaitable[ProviderCredentials]]
)
CompletionSink = Callable[[TrackedBatch, BatchSnapshot], None | Awaitable[None]]
ErrorHandler = Callable[[TrackedBatch, Exception], None | Awaitable[None]]

_TERMINAL = {"cancelled", "completed", "expired", "failed"}
_EVENT_BY_STATUS: dict[str, WebhookEventType] = {
    "cancelled": "batch.cancelled",
    "completed": "batch.completed",
    "expired": "batch.expired",
    "failed": "batch.failed",
}


def _status_value(status: BatchStatus | str) -> str:
    return status.value if isinstance(status, BatchStatus) else status


def _event(provider: BatchProvider, snapshot: BatchSnapshot) -> BatchWebhookEvent:
    status = _status_value(snapshot.status)
    event_type = _EVENT_BY_STATUS.get(status, "batch.completed")
    return BatchWebhookEvent(
        type=event_type,
        id=snapshot.id,
        provider=provider,
        request_counts=snapshot.request_counts,
        created_at=snapshot.created_at,
        completed_at=snapshot.completed_at,
    )


class BatchPoller:
    """Track provider batches and deliver terminal events at least once."""

    def __init__(
        self,
        store: BatchStore,
        *,
        credentials: CredentialResolver | None = None,
        on_complete: CompletionSink | None = None,
        validate_webhook_url: WebhookUrlValidator | None = None,
        on_error: ErrorHandler | None = None,
        webhook_transport: WebhookTransport | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._store = store
        self._credentials = credentials
        self._on_complete = on_complete
        self._validator = validate_webhook_url or default_validate_webhook_url
        self._on_error = on_error
        self._webhook_transport = webhook_transport or PinnedWebhookTransport()
        self._http_client = http_client or httpx.AsyncClient(
            timeout=httpx.Timeout(300.0, connect=10.0, pool=10.0)
        )
        self._owns_http_client = http_client is None
        self._closed = False

    async def __aenter__(self) -> BatchPoller:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._owns_http_client:
            await self._http_client.aclose()

    def _assert_open(self) -> None:
        if self._closed:
            raise BatchworkError("batchwork: batch poller is closed.")

    async def _validate_url(self, raw_url: str) -> str:
        parsed = parse_webhook_url(raw_url)
        outcome = self._validator(parsed)
        if outcome is not None:
            await outcome
        return parsed.geturl()

    async def _resolve_credentials(self, provider: BatchProvider) -> ProviderCredentials:
        resolver = self._credentials
        if resolver is None:
            return ProviderCredentials()
        if isinstance(resolver, ProviderCredentials):
            return resolver
        resolved = resolver(provider)
        if isinstance(resolved, ProviderCredentials):
            return resolved
        return await resolved

    async def track(
        self,
        target: TrackTarget,
        *,
        webhook_url: str | None = None,
        secret: str | None = None,
    ) -> TrackedBatch:
        """Upsert a watched batch and its optional delivery target."""

        self._assert_open()
        if webhook_url is None and self._on_complete is None:
            raise BatchworkError("batchwork: tracked batch has no webhook_url to deliver to.")
        validated_url = webhook_url
        if webhook_url is not None and self._on_complete is None:
            validated_url = await self._validate_url(webhook_url)
        record = TrackedBatch(
            id=target.id,
            provider=target.provider,
            status=target.status,
            webhook_url=validated_url,
            webhook_secret=secret,
        )
        await self._store.set(record)
        return record

    async def deliver(self, record: TrackedBatch, snapshot: BatchSnapshot) -> None:
        """Run the side effect before persisting delivery success."""

        self._assert_open()
        if self._on_complete is not None:
            outcome = self._on_complete(record, snapshot)
            if outcome is not None:
                await outcome
        else:
            await self._deliver_webhook(record, snapshot)
        await self._store.set(
            record.model_copy(update={"delivered_at": _utc_now(), "status": snapshot.status})
        )

    async def _deliver_webhook(self, record: TrackedBatch, snapshot: BatchSnapshot) -> None:
        if record.webhook_url is None:
            raise BatchworkError("batchwork: tracked batch has no webhook_url to deliver to.")
        webhook_url = await self._validate_url(record.webhook_url)
        body = (
            _event(record.provider, snapshot)
            .model_dump_json(by_alias=True, exclude_none=True)
            .encode()
        )
        headers = {"content-type": "application/json"}
        if record.webhook_secret:
            # The event identity is stable for consumer idempotency; every
            # network attempt gets a replay-safe delivery identity.
            headers.update(
                sign_webhook(
                    record.webhook_secret,
                    record.id,
                    body,
                    delivery_id=f"msg_{uuid.uuid4().hex}",
                )
            )
        status = await self._webhook_transport.post(webhook_url, body, headers)
        if 300 <= status < 400:
            raise BatchworkError(
                f"batchwork: webhook delivery to {webhook_url} redirected ({status})."
            )
        if not 200 <= status < 300:
            raise BatchworkError(f"batchwork: webhook delivery to {webhook_url} failed ({status}).")

    async def _retrieve(self, record: TrackedBatch) -> BatchSnapshot:
        adapter = get_adapter(record.provider, http_client=self._http_client)
        credentials = await self._resolve_credentials(record.provider)
        return await adapter.retrieve(record.id, credentials)

    async def _process(self, record: TrackedBatch, delivered: list[str]) -> None:
        snapshot = await self._retrieve(record)
        if _status_value(snapshot.status) in _TERMINAL:
            await self.deliver(record, snapshot)
            delivered.append(record.id)
        elif snapshot.status != record.status:
            await self._store.set(record.model_copy(update={"status": snapshot.status}))

    async def tick(self) -> TickResult:
        """Poll undelivered records serially to avoid provider bursts."""

        self._assert_open()
        pending = await self._store.list(delivered=False)
        delivered: list[str] = []
        failed: list[TickFailure] = []
        for record in pending:
            try:
                await self._process(record, delivered)
            except Exception as error:
                if self._on_error is None:
                    raise
                outcome = self._on_error(record, error)
                if outcome is not None:
                    await outcome
                failed.append(TickFailure(id=record.id, error=str(error)))
        return TickResult(checked=len(pending), delivered=tuple(delivered), failed=tuple(failed))

    async def handle_openai_webhook(
        self,
        headers: Mapping[str, str],
        body: str | bytes,
        signing_secret: str,
        *,
        replay_store: WebhookReplayStore | None = None,
    ) -> WebhookResponse:
        """Verify and process an OpenAI webhook without a framework dependency."""

        self._assert_open()
        try:
            verified = await verify_webhook(
                headers, body, signing_secret, replay_store=replay_store
            )
        except BatchworkError:
            return WebhookResponse(status_code=400, body="invalid signature")
        try:
            payload = json.loads(verified.body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return WebhookResponse(status_code=400, body="invalid payload")
        event_type = payload.get("type") if isinstance(payload, dict) else None
        if not isinstance(event_type, str) or not event_type.startswith("batch."):
            return WebhookResponse(status_code=202, body="ignored")
        data = payload.get("data")
        batch_id = data.get("id") if isinstance(data, dict) else None
        if not isinstance(batch_id, str) or not batch_id:
            return WebhookResponse(status_code=400, body="missing batch id")
        try:
            record = await self._store.get(batch_id)
            if record is None or record.delivered_at is not None:
                return WebhookResponse(status_code=200, body="ok")
            snapshot = await self._retrieve(record)
            if _status_value(snapshot.status) in _TERMINAL:
                await self.deliver(record, snapshot)
            else:
                await verified.release()
                return WebhookResponse(
                    status_code=503,
                    body="batch status not terminal",
                    headers={"retry-after": "1"},
                )
        except BaseException:
            # Verification claims before side effects, preserving atomic replay
            # rejection. Failed handling releases only that claim so a provider
            # retry with the same signed delivery ID can run again.
            await verified.release()
            raise
        return WebhookResponse(status_code=200, body="ok")


def _utc_now() -> datetime:
    # Local import keeps the hot-path namespace small and produces aware UTC.
    from batchwork.server.models import utc_now

    return utc_now()


def create_batch_poller(
    store: BatchStore,
    *,
    credentials: CredentialResolver | None = None,
    on_complete: CompletionSink | None = None,
    validate_webhook_url: WebhookUrlValidator | None = None,
    on_error: ErrorHandler | None = None,
    webhook_transport: WebhookTransport | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> BatchPoller:
    """Create a poller through the functional public API."""

    return BatchPoller(
        store,
        credentials=credentials,
        on_complete=on_complete,
        validate_webhook_url=validate_webhook_url,
        on_error=on_error,
        webhook_transport=webhook_transport,
        http_client=http_client,
    )


__all__ = [
    "BatchPoller",
    "CompletionSink",
    "CredentialResolver",
    "ErrorHandler",
    "create_batch_poller",
]
