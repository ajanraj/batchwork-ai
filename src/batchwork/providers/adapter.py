"""Provider adapter contract."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from typing import Protocol

from batchwork.body import BuiltRequest
from batchwork.types import (
    BatchLimits,
    BatchProvider,
    BatchResult,
    BatchSnapshot,
    ProviderCredentials,
)


class BatchAdapter(Protocol):
    id: BatchProvider

    async def submit(
        self,
        *,
        built: Sequence[BuiltRequest],
        credentials: ProviderCredentials,
        endpoint: str,
        model_id: str,
        metadata: Mapping[str, str] | None = None,
        limits: BatchLimits | None = None,
        validate_upload: Callable[[int], None] | None = None,
    ) -> BatchSnapshot: ...

    async def retrieve(self, id: str, credentials: ProviderCredentials) -> BatchSnapshot: ...

    def results(self, id: str, credentials: ProviderCredentials) -> AsyncIterator[BatchResult]: ...

    async def cancel(self, id: str, credentials: ProviderCredentials) -> None: ...
