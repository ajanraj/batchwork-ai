"""Portable models used by batch tracking and webhook delivery."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import Field

from batchwork.types import (
    BatchProvider,
    BatchRequestCounts,
    BatchStatus,
    BatchworkModel,
)

WebhookEventType = Literal[
    "batch.completed",
    "batch.failed",
    "batch.expired",
    "batch.cancelled",
]


def utc_now() -> datetime:
    """Return an aware UTC timestamp."""

    return datetime.now(UTC)


class TrackedBatch(BatchworkModel):
    """A provider batch watched by a poller."""

    id: str
    provider: BatchProvider
    status: BatchStatus = BatchStatus.IN_PROGRESS
    created_at: datetime = Field(default_factory=utc_now)
    webhook_url: str | None = None
    webhook_secret: str | None = Field(default=None, repr=False)
    delivered_at: datetime | None = None


class BatchWebhookEvent(BatchworkModel):
    """Provider-independent terminal batch event."""

    type: WebhookEventType
    id: str
    provider: BatchProvider
    request_counts: BatchRequestCounts
    created_at: datetime | None = None
    completed_at: datetime | None = None


class TrackTarget(BatchworkModel):
    """Batch identity accepted by :meth:`BatchPoller.track`."""

    id: str
    provider: BatchProvider
    status: BatchStatus = BatchStatus.IN_PROGRESS


class TickFailure(BatchworkModel):
    """A batch that could not be processed during a tolerant tick."""

    id: str
    error: str


class TickResult(BatchworkModel):
    """Summary returned after one serial polling pass."""

    checked: int
    delivered: tuple[str, ...] = ()
    failed: tuple[TickFailure, ...] = ()


class WebhookResponse(BatchworkModel):
    """Framework-neutral HTTP response from an inbound webhook handler."""

    status_code: int
    body: str
    headers: dict[str, str] = Field(default_factory=dict)


__all__ = [
    "BatchWebhookEvent",
    "TickFailure",
    "TickResult",
    "TrackTarget",
    "TrackedBatch",
    "WebhookEventType",
    "WebhookResponse",
]
