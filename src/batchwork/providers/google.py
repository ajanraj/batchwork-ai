"""Google Gemini inline batch adapter."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Mapping, Sequence

import httpx

from batchwork._typing import is_string_mapping
from batchwork.body import BuiltRequest
from batchwork.errors import BatchworkError
from batchwork.types import (
    BatchImage,
    BatchLimits,
    BatchProvider,
    BatchResult,
    BatchSnapshot,
    BatchUsage,
    ProviderCredentials,
)

from ._capabilities import validate_batch_metadata
from .ids import prefixed_provider_id
from .shared import (
    api_key,
    base_url,
    encode_json,
    max_upload_bytes,
    merge_headers,
    number_array,
    request_json,
)

_INLINE_BATCH_MAX_BYTES = 20 * 1024 * 1024


def _inline_upload_limits(limits: BatchLimits | None) -> BatchLimits:
    return BatchLimits(max_upload_bytes=min(max_upload_bytes(limits), _INLINE_BATCH_MAX_BYTES))


def _inline(raw: Mapping[str, object]) -> list[Mapping[str, object]]:
    for container_name in ("response", "dest"):
        raw_container = raw.get(container_name)
        container = raw_container if is_string_mapping(raw_container) else {}
        for outer_fields in (
            ("inlinedResponses", "inlined_responses"),
            ("inlinedEmbedContentResponses", "inlined_embed_content_responses"),
        ):
            for outer_field in outer_fields:
                value = container.get(outer_field)
                if isinstance(value, Sequence):
                    items = [item for item in value if is_string_mapping(item)]
                elif is_string_mapping(value):
                    items = []
                    for inner_field in ("inlinedResponses", "inlined_responses"):
                        nested = value.get(inner_field)
                        if isinstance(nested, Sequence):
                            items = [item for item in nested if is_string_mapping(item)]
                            break
                else:
                    continue
                if items:
                    return items
    return []


def _count(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, str) and value.isascii() and value.isdecimal():
        return int(value)
    return None


def _status(raw: Mapping[str, object]) -> str:
    state_value = raw.get("state")
    if is_string_mapping(state_value):
        state_value = state_value.get("name")
    metadata = raw.get("metadata")
    if not isinstance(state_value, str) and is_string_mapping(metadata):
        state_value = metadata.get("state")
    if isinstance(state_value, str):
        for suffix, normalized in (
            ("SUCCEEDED", "completed"),
            ("FAILED", "failed"),
            ("CANCELLED", "cancelled"),
            ("EXPIRED", "expired"),
            ("PENDING", "validating"),
            ("RUNNING", "in_progress"),
        ):
            if state_value.endswith(suffix):
                return normalized
    return "completed" if raw.get("done") is True else "in_progress"


class GoogleAdapter:
    id = BatchProvider.GOOGLE

    def __init__(self, http_client: httpx.AsyncClient | None) -> None:
        self._client = http_client

    def _base(self, credentials: ProviderCredentials) -> str:
        return base_url(credentials, "https://generativelanguage.googleapis.com/v1beta")

    def _headers(self, credentials: ProviderCredentials) -> dict[str, str]:
        key = api_key(
            credentials, ["GOOGLE_GENERATIVE_AI_API_KEY", "GEMINI_API_KEY"], "Google Gemini"
        )
        return merge_headers(
            {"x-goog-api-key": key, "content-type": "application/json"}, credentials
        )

    def _snapshot(self, raw: Mapping[str, object]) -> BatchSnapshot:
        items = _inline(raw)
        inline_failed = sum(1 for item in items if item.get("error"))
        metadata = raw.get("metadata")
        raw_stats = metadata.get("batchStats") if is_string_mapping(metadata) else None
        stats = raw_stats if is_string_mapping(raw_stats) else {}
        total = _count(stats.get("requestCount"))
        completed = _count(stats.get("successfulRequestCount"))
        failed = _count(stats.get("failedRequestCount"))
        processing = _count(stats.get("pendingRequestCount"))
        request_counts = {
            "completed": len(items) - inline_failed if completed is None else completed,
            "failed": inline_failed if failed is None else failed,
            "total": len(items) if total is None else total,
        }
        if processing is not None:
            request_counts["processing"] = processing
        raw_id = raw.get("name")
        id = (
            prefixed_provider_id("Google operation id", raw_id, "batches")
            if isinstance(raw_id, str)
            else ""
        )
        return BatchSnapshot.model_validate(
            {
                "id": id,
                "provider": self.id,
                "status": _status(raw),
                "raw": dict(raw),
                "request_counts": request_counts,
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
        embedding = "embedcontent" in endpoint.lower()
        method = "asyncBatchEmbedContent" if embedding else "batchGenerateContent"
        requests: list[dict[str, object]] = []
        for item in built:
            payload = {key: value for key, value in item.body.items() if key != "stream"}
            if embedding:
                config = {
                    key: payload.pop(key)
                    for key in ("outputDimensionality", "taskType", "title")
                    if key in payload
                }
                if config:
                    payload["embedContentConfig"] = config
            requests.append({"metadata": {"key": item.custom_id}, "request": payload})
        content = encode_json(
            {
                "batch": {
                    "display_name": "batchwork",
                    "input_config": {"requests": {"requests": requests}},
                }
            },
            _inline_upload_limits(limits),
            validate_upload=validate_upload,
        )
        raw = await request_json(
            self._client,
            "POST",
            f"{self._base(credentials)}/models/{model_id}:{method}",
            headers=self._headers(credentials),
            content=content,
        )
        return self._snapshot(raw)

    async def retrieve(self, id: str, credentials: ProviderCredentials) -> BatchSnapshot:
        operation = prefixed_provider_id("Google operation id", id, "batches")
        raw = await request_json(
            self._client,
            "GET",
            f"{self._base(credentials)}/{operation}",
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
        raw = snapshot.raw
        response = raw.get("response") if is_string_mapping(raw) else None
        dest = raw.get("dest") if is_string_mapping(raw) else None
        for container in (response, dest):
            if is_string_mapping(container) and any(
                field in container
                for field in ("responsesFile", "responses_file", "fileName", "file_name")
            ):
                raise BatchworkError(
                    f'batchwork: batch "{snapshot.id}" returned file-mode results, '
                    "which are not supported yet."
                )
        items = _inline(raw) if is_string_mapping(raw) else []
        if not items:
            raise BatchworkError(
                f'batchwork: results are not ready for batch "{snapshot.id}" '
                f"(status: {snapshot.status})."
            )
        for item in items:
            yield self._result(item)

    def _result(self, item: Mapping[str, object]) -> BatchResult:
        metadata = item.get("metadata")
        key = (
            metadata.get("key")
            if is_string_mapping(metadata)
            else item.get("key", item.get("custom_id"))
        )
        custom_id = key if isinstance(key, str) else ""
        raw_error = item.get("error")
        if is_string_mapping(raw_error):
            return BatchResult.model_validate(
                {
                    "custom_id": custom_id,
                    "status": "errored",
                    "response": raw_error,
                    "error": {
                        "message": raw_error.get("message", "Request errored."),
                        "type": raw_error.get("status"),
                        "code": raw_error.get("code"),
                    },
                }
            )
        raw_response = item.get("response")
        response = raw_response if is_string_mapping(raw_response) else {}
        embedding: list[float] | None = None
        raw_embedding = response.get("embedding")
        if is_string_mapping(raw_embedding):
            embedding = number_array(raw_embedding.get("values"))
        text_parts: list[str] = []
        images: list[BatchImage] = []
        candidates = response.get("candidates")
        if isinstance(candidates, Sequence) and candidates and is_string_mapping(candidates[0]):
            content = candidates[0].get("content")
            parts = content.get("parts") if is_string_mapping(content) else None
            if isinstance(parts, Sequence):
                for part in parts:
                    if not is_string_mapping(part):
                        continue
                    part_text = part.get("text")
                    if isinstance(part_text, str):
                        text_parts.append(part_text)
                    inline = part.get("inlineData", part.get("inline_data"))
                    if is_string_mapping(inline) and isinstance(inline.get("data"), str):
                        media_type = inline.get("mimeType", inline.get("mime_type"))
                        image_data = inline.get("data")
                        if (
                            isinstance(image_data, str)
                            and isinstance(media_type, str)
                            and media_type.startswith("image/")
                        ):
                            images.append(BatchImage(data=image_data, media_type=media_type))
        raw_usage = response.get("usageMetadata")
        usage = raw_usage if is_string_mapping(raw_usage) else {}
        input_tokens = usage.get("promptTokenCount")
        output_tokens = usage.get("candidatesTokenCount")
        total_tokens = usage.get("totalTokenCount")
        normalized_usage = None
        if any(isinstance(value, int) for value in (input_tokens, output_tokens, total_tokens)):
            normalized_usage = BatchUsage(
                input_tokens=input_tokens if isinstance(input_tokens, int) else None,
                output_tokens=output_tokens if isinstance(output_tokens, int) else None,
                total_tokens=total_tokens
                if isinstance(total_tokens, int)
                else (input_tokens if isinstance(input_tokens, int) else 0)
                + (output_tokens if isinstance(output_tokens, int) else 0),
            )
        return BatchResult.model_validate(
            {
                "custom_id": custom_id,
                "status": "succeeded",
                "response": raw_response,
                "text": "".join(text_parts) or None,
                "embedding": embedding,
                "images": images or None,
                "usage": normalized_usage,
            }
        )

    async def cancel(self, id: str, credentials: ProviderCredentials) -> None:
        operation = prefixed_provider_id("Google operation id", id, "batches")
        await request_json(
            self._client,
            "POST",
            f"{self._base(credentials)}/{operation}:cancel",
            headers=self._headers(credentials),
        )
