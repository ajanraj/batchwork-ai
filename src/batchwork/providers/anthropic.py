"""Anthropic Message Batches adapter."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from urllib.parse import SplitResult, urlsplit

import httpx

from batchwork._typing import is_string_mapping
from batchwork.body import BuiltRequest
from batchwork.errors import BatchworkError
from batchwork.types import (
    BatchLimits,
    BatchProvider,
    BatchResult,
    BatchSnapshot,
    BatchUsage,
    ProviderCredentials,
)

from ._capabilities import validate_batch_metadata
from .ids import simple_provider_id
from .shared import (
    api_key,
    base_url,
    encode_json,
    jsonl,
    merge_headers,
    request_json,
    stream_request,
    timestamp,
)


def _origin(url: SplitResult) -> tuple[str, str, int | None] | None:
    hostname = url.hostname
    if hostname is None:
        return None
    try:
        port = url.port
    except ValueError:
        return None
    if port is None:
        port = {"http": 80, "https": 443}.get(url.scheme.lower())
    return url.scheme.lower(), hostname.lower(), port


class AnthropicAdapter:
    id = BatchProvider.ANTHROPIC

    def __init__(self, http_client: httpx.AsyncClient | None) -> None:
        self._client = http_client

    def _base(self, credentials: ProviderCredentials) -> str:
        return base_url(credentials, "https://api.anthropic.com")

    def _headers(self, credentials: ProviderCredentials) -> dict[str, str]:
        key = api_key(credentials, ["ANTHROPIC_API_KEY"], "Anthropic")
        return merge_headers(
            {
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            credentials,
        )

    def _snapshot(self, raw: Mapping[str, object]) -> BatchSnapshot:
        raw_counts = raw.get("request_counts")
        counts = raw_counts if is_string_mapping(raw_counts) else {}
        succeeded = counts.get("succeeded", 0)
        failed = counts.get("errored", 0)
        processing = counts.get("processing", 0)
        canceled = counts.get("canceled", 0)
        expired = counts.get("expired", 0)
        numeric = [
            item if isinstance(item, int) else 0
            for item in (succeeded, failed, processing, canceled, expired)
        ]
        status_value = raw.get("processing_status")
        status = (
            "completed"
            if status_value == "ended"
            else "cancelling"
            if status_value == "canceling"
            else "in_progress"
        )
        return BatchSnapshot.model_validate(
            {
                "id": raw.get("id") if isinstance(raw.get("id"), str) else "",
                "provider": self.id,
                "status": status,
                "raw": dict(raw),
                "created_at": timestamp(raw.get("created_at")),
                "completed_at": timestamp(raw.get("ended_at")),
                "expires_at": timestamp(raw.get("expires_at")),
                "request_counts": {
                    "completed": numeric[0],
                    "failed": numeric[1],
                    "processing": numeric[2],
                    "canceled": numeric[3],
                    "expired": numeric[4],
                    "total": sum(numeric),
                },
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
        validate_upload: Callable[[int], None] | None = None,
    ) -> BatchSnapshot:
        validate_batch_metadata(self.id, metadata)
        del endpoint, model_id
        requests = [
            {
                "custom_id": item.custom_id,
                "params": {key: value for key, value in item.body.items() if key != "stream"},
            }
            for item in built
        ]
        content = encode_json({"requests": requests}, limits, validate_upload=validate_upload)
        raw = await request_json(
            self._client,
            "POST",
            f"{self._base(credentials)}/v1/messages/batches",
            headers=self._headers(credentials),
            content=content,
        )
        return self._snapshot(raw)

    async def retrieve(self, id: str, credentials: ProviderCredentials) -> BatchSnapshot:
        batch_id = simple_provider_id("Anthropic batch id", id)
        raw = await request_json(
            self._client,
            "GET",
            f"{self._base(credentials)}/v1/messages/batches/{batch_id}",
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
        raw_url = snapshot.raw.get("results_url") if is_string_mapping(snapshot.raw) else None
        if not isinstance(raw_url, str):
            raise BatchworkError(
                f'batchwork: results are not ready for batch "{snapshot.id}" '
                f"(status: {snapshot.status})."
            )
        target = urlsplit(raw_url)
        expected = urlsplit(self._base(credentials))
        target_origin = _origin(target)
        expected_origin = _origin(expected)
        if (
            target.username is not None
            or target.password is not None
            or expected.username is not None
            or expected.password is not None
            or target_origin is None
            or target_origin != expected_origin
        ):
            raise BatchworkError(
                "batchwork: Anthropic results_url must match the configured API origin."
            )
        async with stream_request(
            self._client, "GET", raw_url, headers=self._headers(credentials)
        ) as response:
            async for item in jsonl(response):
                yield self._result(item)

    def _result(self, item: Mapping[str, object]) -> BatchResult:
        custom_id = item.get("custom_id") if isinstance(item.get("custom_id"), str) else ""
        raw_result = item.get("result")
        result = raw_result if is_string_mapping(raw_result) else {}
        kind = result.get("type")
        if kind == "succeeded":
            raw_message = result.get("message")
            message = raw_message if is_string_mapping(raw_message) else {}
            content = message.get("content")
            text = ""
            if isinstance(content, Sequence):
                for block in content:
                    if (
                        is_string_mapping(block)
                        and block.get("type") == "text"
                        and isinstance(block.get("text"), str)
                    ):
                        block_text = block.get("text")
                        if isinstance(block_text, str):
                            text += block_text
            usage_raw = message.get("usage")
            usage = usage_raw if is_string_mapping(usage_raw) else {}
            input_tokens = usage.get("input_tokens")
            output_tokens = usage.get("output_tokens")
            normalized_usage = None
            if isinstance(input_tokens, int) or isinstance(output_tokens, int):
                normalized_usage = BatchUsage(
                    input_tokens=input_tokens if isinstance(input_tokens, int) else None,
                    output_tokens=output_tokens if isinstance(output_tokens, int) else None,
                    total_tokens=(input_tokens if isinstance(input_tokens, int) else 0)
                    + (output_tokens if isinstance(output_tokens, int) else 0),
                )
            return BatchResult.model_validate(
                {
                    "custom_id": custom_id,
                    "status": "succeeded",
                    "response": raw_message,
                    "text": text or None,
                    "usage": normalized_usage,
                }
            )
        if kind == "errored":
            raw_error = result.get("error")
            error = raw_error if is_string_mapping(raw_error) else {}
            nested = error.get("error")
            source = nested if is_string_mapping(nested) else error
            return BatchResult.model_validate(
                {
                    "custom_id": custom_id,
                    "status": "errored",
                    "response": raw_error,
                    "error": {
                        "message": source.get("message")
                        if isinstance(source.get("message"), str)
                        else "Request errored.",
                        "type": source.get("type") if isinstance(source.get("type"), str) else None,
                    },
                }
            )
        return BatchResult.model_validate(
            {"custom_id": custom_id, "status": "expired" if kind == "expired" else "canceled"}
        )

    async def cancel(self, id: str, credentials: ProviderCredentials) -> None:
        batch_id = simple_provider_id("Anthropic batch id", id)
        await request_json(
            self._client,
            "POST",
            f"{self._base(credentials)}/v1/messages/batches/{batch_id}/cancel",
            headers=self._headers(credentials),
        )
