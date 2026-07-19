"""Mistral batch jobs adapter."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping, Sequence

import httpx

from batchwork.body import BuiltRequest
from batchwork.errors import BatchworkError
from batchwork.types import (
    BatchLimits,
    BatchProvider,
    BatchResult,
    BatchSnapshot,
    ProviderCredentials,
)

from .ids import simple_provider_id
from .shared import (
    api_key,
    base_url,
    encode_jsonl,
    merge_headers,
    request_json,
    stream_result_file,
    timestamp,
    upload_file,
)


class MistralAdapter:
    id = BatchProvider.MISTRAL

    def __init__(self, http_client: httpx.AsyncClient | None) -> None:
        self._client = http_client

    def _base(self, credentials: ProviderCredentials) -> str:
        return base_url(credentials, "https://api.mistral.ai/v1")

    def _headers(self, credentials: ProviderCredentials) -> dict[str, str]:
        key = api_key(credentials, ["MISTRAL_API_KEY"], "Mistral")
        return merge_headers({"Authorization": f"Bearer {key}"}, credentials)

    def _snapshot(self, raw: Mapping[str, object]) -> BatchSnapshot:
        statuses = {
            "QUEUED": "validating",
            "SUCCESS": "completed",
            "FAILED": "failed",
            "TIMEOUT_EXCEEDED": "expired",
            "CANCELLATION_REQUESTED": "cancelling",
            "CANCELLED": "cancelled",
        }
        raw_succeeded = raw.get("succeeded_requests")
        raw_failed = raw.get("failed_requests")
        raw_total = raw.get("total_requests")
        succeeded = raw_succeeded if isinstance(raw_succeeded, int) else 0
        failed = raw_failed if isinstance(raw_failed, int) else 0
        total = raw_total if isinstance(raw_total, int) else succeeded + failed
        raw_id = raw.get("id")
        raw_status = raw.get("status")
        return BatchSnapshot.model_validate(
            {
                "id": simple_provider_id("Mistral job id", raw_id)
                if isinstance(raw_id, str)
                else "",
                "provider": self.id,
                "status": statuses.get(raw_status, "in_progress")
                if isinstance(raw_status, str)
                else "in_progress",
                "raw": dict(raw),
                "created_at": timestamp(raw.get("created_at")),
                "completed_at": timestamp(raw.get("completed_at")),
                "request_counts": {"completed": succeeded, "failed": failed, "total": total},
            }
        )

    async def submit(
        self,
        *,
        built: Sequence[BuiltRequest],
        credentials: ProviderCredentials,
        endpoint: str,
        model_id: str,
        metadata: Mapping[str, str] | None = None,
        limits: BatchLimits | None = None,
    ) -> BatchSnapshot:
        lines = []
        for item in built:
            body = {
                key: value for key, value in item.body.items() if key not in {"model", "stream"}
            }
            lines.append({"custom_id": item.custom_id, "body": body})
        payload = encode_jsonl(lines, limits)
        url = self._base(credentials)
        headers = self._headers(credentials)
        file_id = await upload_file(self._client, url, headers, payload)
        create_body: dict[str, object] = {
            "endpoint": endpoint,
            "input_files": [file_id],
            "model": model_id,
        }
        if metadata is not None:
            create_body["metadata"] = dict(metadata)
        raw = await request_json(
            self._client,
            "POST",
            f"{url}/batch/jobs",
            headers={**headers, "content-type": "application/json"},
            content=json.dumps(create_body),
        )
        return self._snapshot(raw)

    async def retrieve(self, id: str, credentials: ProviderCredentials) -> BatchSnapshot:
        job_id = simple_provider_id("Mistral job id", id)
        raw = await request_json(
            self._client,
            "GET",
            f"{self._base(credentials)}/batch/jobs/{job_id}",
            headers=self._headers(credentials),
        )
        return self._snapshot(raw)

    async def results(
        self, id: str, credentials: ProviderCredentials
    ) -> AsyncIterator[BatchResult]:
        snapshot = await self.retrieve(id, credentials)
        async for result in self.results_from_snapshot(snapshot, credentials):
            yield result

    async def results_from_snapshot(
        self, snapshot: BatchSnapshot, credentials: ProviderCredentials
    ) -> AsyncIterator[BatchResult]:
        output = snapshot.raw.get("output_file") if isinstance(snapshot.raw, Mapping) else None
        error = snapshot.raw.get("error_file") if isinstance(snapshot.raw, Mapping) else None
        if not isinstance(output, str) and not isinstance(error, str):
            raise BatchworkError(
                f'batchwork: results are not ready for batch "{snapshot.id}" '
                f"(status: {snapshot.status})."
            )
        url = self._base(credentials)
        headers = self._headers(credentials)
        if isinstance(output, str):
            safe = simple_provider_id("Mistral output file id", output)
            async for item in stream_result_file(self._client, url, headers, safe):
                yield item
        if isinstance(error, str):
            safe = simple_provider_id("Mistral error file id", error)
            async for item in stream_result_file(self._client, url, headers, safe):
                yield item

    async def cancel(self, id: str, credentials: ProviderCredentials) -> None:
        job_id = simple_provider_id("Mistral job id", id)
        await request_json(
            self._client,
            "POST",
            f"{self._base(credentials)}/batch/jobs/{job_id}/cancel",
            headers=self._headers(credentials),
        )
