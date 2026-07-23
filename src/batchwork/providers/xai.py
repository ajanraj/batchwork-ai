"""xAI proprietary batch lifecycle adapter."""

from __future__ import annotations

import hashlib
import json
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from urllib.parse import urlencode

import httpx

from batchwork._provider_failure import ProviderFailure, ProviderFailureError, ProviderFailureKind
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

from ._capabilities import validate_batch_metadata
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

_RESULT_PAGE_SIZE = 100
_MAX_RESULT_PAGES = 10_000
_TOKEN_FINGERPRINT_LENGTH = 12


def _token_fingerprint(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()[:_TOKEN_FINGERPRINT_LENGTH]


def _pagination_failure(message: str) -> ProviderFailureError:
    return ProviderFailureError(message, ProviderFailure(ProviderFailureKind.PROTOCOL))


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
        cancel_message = raw.get("cancel_by_xai_message")
        has_cancellation_marker = timestamp(raw.get("cancel_time")) is not None or (
            isinstance(cancel_message, str) and bool(cancel_message)
        )
        if has_cancellation_marker and isinstance(pending, int) and pending > 0:
            status = "cancelling"
        elif has_cancellation_marker:
            status = "cancelled"
        elif not isinstance(pending, int) or total == 0 or pending > 0:
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
        validate_batch_metadata(self.id, metadata)
        del model_id
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
        observed_tokens: set[str] = set()
        emitted_ids: set[str] = set()
        page_count = 0
        emitted_count = 0
        while True:
            query: dict[str, object] = {"limit": _RESULT_PAGE_SIZE}
            if token:
                query["pagination_token"] = token
            raw = await request_json(
                self._client,
                "GET",
                f"{self._base(credentials)}/batches/{batch_id}/results?{urlencode(query)}",
                headers=self._headers(credentials),
            )
            page_count += 1
            raw_token = raw.get("pagination_token")
            next_token = raw_token if isinstance(raw_token, str) and raw_token else None
            if next_token is not None and next_token in observed_tokens:
                last_safe_token = token if token is not None else next_token
                raise _pagination_failure(
                    "batchwork: xAI result pagination repeated a continuation token on page "
                    f"{page_count}; stopped before emitting that page after {emitted_count} "
                    "records (last safe token fingerprint: "
                    f"sha256:{_token_fingerprint(last_safe_token)})."
                )
            if next_token is not None:
                observed_tokens.add(next_token)
            items = raw.get("results")
            if isinstance(items, Sequence):
                for item in items:
                    if is_string_mapping(item):
                        raw_id = item.get("batch_request_id")
                        if isinstance(raw_id, str) and raw_id:
                            if raw_id in emitted_ids:
                                raise _pagination_failure(
                                    "batchwork: xAI result pagination returned a duplicate result "
                                    f"id after {emitted_count} records; stopped before emitting "
                                    "the duplicate."
                                )
                            emitted_ids.add(raw_id)
                        result = self._result(item)
                        emitted_count += 1
                        yield result
            if next_token is None:
                return
            if page_count >= _MAX_RESULT_PAGES:
                raise _pagination_failure(
                    "batchwork: xAI result pagination exceeded the "
                    f"{_MAX_RESULT_PAGES:,}-page safety limit after "
                    f"{emitted_count:,} records; stopped before requesting another page."
                )
            token = next_token

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
