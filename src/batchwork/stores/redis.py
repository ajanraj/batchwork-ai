"""Optional Upstash Redis batch store."""

from __future__ import annotations

import builtins
import json
from collections.abc import Awaitable, Sequence
from typing import Protocol

from batchwork.errors import MissingDependencyError
from batchwork.server.models import TrackedBatch

DEFAULT_PREFIX = "batchwork"
RedisResult = object | Awaitable[object]


class UpstashRedis(Protocol):
    """Minimal sync-or-async Upstash client surface used by this package."""

    def get(self, key: str) -> RedisResult: ...
    def set(self, key: str, value: str) -> RedisResult: ...
    def delete(self, key: str) -> RedisResult: ...
    def sadd(self, key: str, *values: str) -> RedisResult: ...
    def srem(self, key: str, *values: str) -> RedisResult: ...
    def smembers(self, key: str) -> RedisResult: ...
    def mget(self, *keys: str) -> RedisResult: ...


async def _resolve(value: RedisResult) -> object:
    if isinstance(value, Awaitable):
        return await value
    return value


def _coerce(value: object) -> TrackedBatch | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        value = json.loads(value)
    return TrackedBatch.model_validate(value)


class RedisBatchStore:
    """Redis store compatible with ``upstash-redis`` sync and async clients."""

    def __init__(self, redis: UpstashRedis, *, prefix: str = DEFAULT_PREFIX) -> None:
        if not prefix:
            raise ValueError("batchwork: Redis prefix must not be empty.")
        self._redis = redis
        self._prefix = prefix

    def _batch_key(self, batch_id: str) -> str:
        return f"{self._prefix}:batch:{batch_id}"

    @property
    def _index_key(self) -> str:
        return f"{self._prefix}:batches"

    async def get(self, batch_id: str) -> TrackedBatch | None:
        return _coerce(await _resolve(self._redis.get(self._batch_key(batch_id))))

    async def set(self, record: TrackedBatch) -> None:
        value = record.model_dump_json()
        await _resolve(self._redis.set(self._batch_key(record.id), value))
        await _resolve(self._redis.sadd(self._index_key, record.id))

    async def delete(self, batch_id: str) -> None:
        delete = getattr(self._redis, "delete", None)
        if delete is None:
            delete = getattr(self._redis, "del")
        await _resolve(delete(self._batch_key(batch_id)))
        await _resolve(self._redis.srem(self._index_key, batch_id))

    async def list(self, delivered: bool | None = None) -> builtins.list[TrackedBatch]:
        raw_ids = await _resolve(self._redis.smembers(self._index_key))
        if not isinstance(raw_ids, Sequence) or isinstance(raw_ids, (str, bytes, bytearray)):
            return []
        ids = [item.decode() if isinstance(item, bytes) else str(item) for item in raw_ids]
        if not ids:
            return []
        raw_records = await _resolve(
            self._redis.mget(*(self._batch_key(batch_id) for batch_id in ids))
        )
        if not isinstance(raw_records, Sequence) or isinstance(
            raw_records, (str, bytes, bytearray)
        ):
            return []
        records = [record for item in raw_records if (record := _coerce(item))]
        if delivered is None:
            return records
        return [record for record in records if (record.delivered_at is not None) is delivered]


def create_redis_store(
    redis: UpstashRedis | None = None, *, prefix: str = DEFAULT_PREFIX
) -> RedisBatchStore:
    """Create a store from an injected client or lazily from Upstash environment values."""

    if redis is None:
        try:
            from upstash_redis.asyncio import Redis
        except ImportError as error:
            raise MissingDependencyError("the Upstash Redis store", "redis") from error
        redis = Redis.from_env()
    return RedisBatchStore(redis, prefix=prefix)


__all__ = ["DEFAULT_PREFIX", "RedisBatchStore", "UpstashRedis", "create_redis_store"]
