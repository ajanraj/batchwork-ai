from __future__ import annotations

import json

import pytest

from batchwork.errors import MissingDependencyError
from batchwork.server import TrackedBatch
from batchwork.stores import MemoryBatchStore, RedisBatchStore
from batchwork.stores.redis import create_redis_store
from batchwork.types import BatchProvider, BatchStatus


class FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.sets: dict[str, set[str]] = {}

    def get(self, key: str) -> str | None:
        return self.values.get(key)

    def set(self, key: str, value: str) -> None:
        self.values[key] = value

    def delete(self, key: str) -> None:
        self.values.pop(key, None)

    def sadd(self, key: str, *values: str) -> None:
        self.sets.setdefault(key, set()).update(values)

    def srem(self, key: str, *values: str) -> None:
        self.sets.setdefault(key, set()).difference_update(values)

    def smembers(self, key: str) -> list[str]:
        return sorted(self.sets.get(key, set()))

    def mget(self, *keys: str) -> list[str | None]:
        return [self.values.get(key) for key in keys]


def record(batch_id: str, *, delivered: bool = False) -> TrackedBatch:
    values: dict[str, object] = {
        "id": batch_id,
        "provider": BatchProvider.OPENAI,
        "status": BatchStatus.COMPLETED if delivered else BatchStatus.IN_PROGRESS,
    }
    if delivered:
        values["delivered_at"] = "2026-07-16T00:00:00Z"
    return TrackedBatch.model_validate(values)


@pytest.mark.parametrize(
    "store",
    [MemoryBatchStore(), RedisBatchStore(FakeRedis())],
    ids=["memory", "redis"],
)
async def test_store_contract(store: MemoryBatchStore | RedisBatchStore) -> None:
    assert await store.get("missing") is None
    await store.set(record("open"))
    await store.set(record("done", delivered=True))
    assert {item.id for item in await store.list()} == {"open", "done"}
    assert [item.id for item in await store.list(delivered=False)] == ["open"]
    assert [item.id for item in await store.list(delivered=True)] == ["done"]

    await store.set(record("open", delivered=True))
    updated = await store.get("open")
    assert updated is not None
    assert updated.delivered_at is not None
    await store.delete("open")
    assert await store.get("open") is None


async def test_redis_store_namespace_and_object_values() -> None:
    redis = FakeRedis()
    first = RedisBatchStore(redis, prefix="one")
    second = RedisBatchStore(redis, prefix="two")
    expected = record("b1")
    await first.set(expected)
    assert await first.get("b1") == expected
    assert await second.get("b1") is None

    # Upstash may auto-decode JSON depending on client configuration.
    redis.values["one:batch:b1"] = json.dumps(expected.model_dump(mode="json"))
    assert await first.get("b1") == expected


def test_redis_store_reports_missing_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins

    original_import = builtins.__import__

    def reject_upstash(name: str, *args: object, **kwargs: object) -> object:
        if name.startswith("upstash_redis"):
            raise ImportError(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_upstash)
    with pytest.raises(MissingDependencyError, match=r"batchwork-ai\[redis\]"):
        create_redis_store()
