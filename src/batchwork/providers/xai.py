"""xAI proprietary batch lifecycle adapter."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from urllib.parse import urlencode

import httpx

from batchwork._typing import is_string_mapping
from batchwork.body import BuiltRequest
from batchwork.types import (
    BatchImage,
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
    text_from_body,
    timestamp,
    upload_file,
    usage_from_body,
)


class XAIAdapter:
    id = BatchProvider.XAI

    def __init__(self, http_client: httpx.AsyncClient | None) -> None:
        self._client = http_client

    def _base(self, credentials: ProviderCredentials) -> str:
        return base_url(credentials, "https://api.x.ai/v1")

    def _headers(self, credentials: ProviderCredentials) -> dict[str, str]:
        key = api_key(credentials, ["XAI_API_KEY"], "xAI")
        return merge_headers({"Authorization": f"Bearer {key}"}, credentials)

    def _snapshot(self, raw: Mapping[str, object]) -> BatchSnapshot:
        raw_state = raw.get("state")
        state = raw_state if is_string_mapping(raw_state) else {}
        pending = state.get("num_pending")
        total = state.get("num_requests") if isinstance(state.get("num_requests"), int) else 0
        cancelled = state.get("num_cancelled") if isinstance(state.get("num_cancelled"), int) else 0
        if not isinstance(pending, int) or total == 0 or pending > 0:
            status = "in_progress"
        elif cancelled == total:
            status = "cancelled"
        else:
            status = "completed"
        raw_id = raw.get("batch_id", raw.get("id"))
        id = simple_provider_id("xAI batch id", raw_id) if isinstance(raw_id, str) else ""
        return BatchSnapshot.model_validate(
            {
                "id": id,
                "provider": self.id,
                "status": status,
                "raw": dict(raw),
                "created_at": timestamp(raw.get("create_time", raw.get("created_at"))),
                "completed_at": timestamp(raw.get("finish_time", raw.get("completed_at"))),
                "expires_at": timestamp(raw.get("expire_time", raw.get("expires_at"))),
                "request_counts": {
                    "completed": state.get("num_success", 0),
                    "failed": state.get("num_error", 0),
                    "processing": state.get("num_pending", 0),
                    "canceled": cancelled,
                    "total": total,
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
        del model_id, metadata
        lines = [
            {
                "body": {key: value for key, value in item.body.items() if key != "stream"},
                "custom_id": item.custom_id,
                "method": "POST",
                "url": endpoint,
            }
            for item in built
        ]
        payload = encode_jsonl(lines, limits, validate_upload=validate_upload)
        url = self._base(credentials)
        headers = self._headers(credentials)
        file_id = await upload_file(self._client, url, headers, payload, purpose=None)
        raw = await request_json(
            self._client,
            "POST",
            f"{url}/batches",
            headers={**headers, "content-type": "application/json"},
            content=json.dumps({"input_file_id": file_id, "name": "batchwork"}),
        )
        return self._snapshot(raw)

    async def retrieve(self, id: str, credentials: ProviderCredentials) -> BatchSnapshot:
        batch_id = simple_provider_id("xAI batch id", id)
        raw = await request_json(
            self._client,
            "GET",
            f"{self._base(credentials)}/batches/{batch_id}",
            headers=self._headers(credentials),
        )
        return self._snapshot(raw)

    async def results(
        self, id: str, credentials: ProviderCredentials
    ) -> AsyncIterator[BatchResult]:
        batch_id = simple_provider_id("xAI batch id", id)
        token: str | None = None
        while True:
            query: dict[str, object] = {"limit": 100}
            if token:
                query["pagination_token"] = token
            raw = await request_json(
                self._client,
                "GET",
                f"{self._base(credentials)}/batches/{batch_id}/results?{urlencode(query)}",
                headers=self._headers(credentials),
            )
            items = raw.get("results")
            if isinstance(items, Sequence):
                for item in items:
                    if is_string_mapping(item):
                        yield self._result(item)
            raw_token = raw.get("pagination_token")
            token = raw_token if isinstance(raw_token, str) and raw_token else None
            if token is None:
                return

    def _result(self, item: Mapping[str, object]) -> BatchResult:
        raw_id = item.get("batch_request_id")
        custom_id = raw_id if isinstance(raw_id, str) else ""
        raw_batch = item.get("batch_result")
        batch = raw_batch if is_string_mapping(raw_batch) else {}
        raw_error = batch.get("error")
        message = item.get("error_message")
        error_message = (
            message
            if isinstance(message, str)
            else raw_error
            if isinstance(raw_error, str)
            else raw_error.get("message")
            if is_string_mapping(raw_error)
            else None
        )
        if isinstance(error_message, str):
            return BatchResult.model_validate(
                {
                    "custom_id": custom_id,
                    "status": "errored",
                    "response": dict(item),
                    "error": {
                        "message": error_message,
                        "type": raw_error.get("type") if is_string_mapping(raw_error) else None,
                    },
                }
            )
        raw_response = batch.get("response")
        response = raw_response if is_string_mapping(raw_response) else {}
        completion = response.get("chat_get_completion")
        if completion is None and response:
            completion = next(iter(response.values()))
        images: list[BatchImage] = []
        if is_string_mapping(completion):
            raw_data = completion.get("data")
            sources = raw_data if isinstance(raw_data, Sequence) else [completion]
            for source in sources:
                if not is_string_mapping(source):
                    continue
                data = source.get("base64", source.get("b64_json"))
                url = source.get("url")
                if isinstance(data, str) or isinstance(url, str):
                    images.append(
                        BatchImage(
                            data=data if isinstance(data, str) else None,
                            url=url if isinstance(url, str) else None,
                        )
                    )
        return BatchResult.model_validate(
            {
                "custom_id": custom_id,
                "status": "succeeded",
                "response": completion,
                "text": text_from_body(completion),
                "images": images or None,
                "usage": usage_from_body(completion),
            }
        )

    async def cancel(self, id: str, credentials: ProviderCredentials) -> None:
        batch_id = simple_provider_id("xAI batch id", id)
        await request_json(
            self._client,
            "POST",
            f"{self._base(credentials)}/batches/{batch_id}:cancel",
            headers=self._headers(credentials),
        )
