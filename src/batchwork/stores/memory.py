"""Single-process batch store."""

from __future__ import annotations

import asyncio
import builtins

from batchwork.server.models import TrackedBatch


class MemoryBatchStore:
    """In-memory store for development, tests, and one-process services."""

    def __init__(self) -> None:
        self._records: dict[str, TrackedBatch] = {}
        self._lock = asyncio.Lock()

    async def get(self, batch_id: str) -> TrackedBatch | None:
        async with self._lock:
            return self._records.get(batch_id)

    async def set(self, record: TrackedBatch) -> None:
        async with self._lock:
            self._records[record.id] = record

    async def delete(self, batch_id: str) -> None:
        async with self._lock:
            self._records.pop(batch_id, None)

    async def list(self, delivered: bool | None = None) -> builtins.list[TrackedBatch]:
        async with self._lock:
            records = builtins.list(self._records.values())
        if delivered is None:
            return records
        return [record for record in records if (record.delivered_at is not None) is delivered]


def create_memory_store() -> MemoryBatchStore:
    """Return a fresh in-memory store."""

    return MemoryBatchStore()


__all__ = ["MemoryBatchStore", "create_memory_store"]
