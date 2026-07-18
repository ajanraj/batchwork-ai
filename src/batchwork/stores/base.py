"""Persistence contract for watched batches."""

from __future__ import annotations

import builtins
from typing import Protocol, runtime_checkable

from batchwork.server.models import TrackedBatch


@runtime_checkable
class BatchStore(Protocol):
    """Async upsert store used by :class:`batchwork.server.BatchPoller`."""

    async def get(self, batch_id: str) -> TrackedBatch | None: ...

    async def set(self, record: TrackedBatch) -> None: ...

    async def delete(self, batch_id: str) -> None: ...

    async def list(self, delivered: bool | None = None) -> builtins.list[TrackedBatch]: ...


__all__ = ["BatchStore"]
