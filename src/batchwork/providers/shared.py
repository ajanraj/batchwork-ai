"""HTTP, JSONL, credential, and result normalization helpers."""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from datetime import UTC, datetime

import httpx

from batchwork._provider_failure import (
    ProviderFailureError,
    http_failure,
    protocol_failure,
    transport_failure,
)
from batchwork._typing import is_string_mapping
from batchwork.errors import BatchworkError
from batchwork.types import BatchImage, BatchLimits, BatchResult, BatchUsage, ProviderCredentials


def api_key(credentials: ProviderCredentials, env_vars: Sequence[str], label: str) -> str:
    if credentials.api_key:
        return credentials.api_key
    for name in env_vars:
        value = os.getenv(name)
        if value:
            return value
    joined = " (or ".join(env_vars)
    suffix = ")" if len(env_vars) > 1 else ""
    raise BatchworkError(
        f"batchwork: missing {label} API key. Set {joined}{suffix} or pass `api_key`."
    )


def base_url(credentials: ProviderCredentials, default: str) -> str:
    return (credentials.base_url or default).rstrip("/")


def merge_headers(defaults: Mapping[str, str], credentials: ProviderCredentials) -> dict[str, str]:
    return {**defaults, **credentials.headers}


def max_upload_bytes(limits: BatchLimits | None) -> int:
    if limits is None or limits.max_upload_bytes is None:
        return 200 * 1024 * 1024
    return limits.max_upload_bytes


def encode_jsonl(items: Sequence[Mapping[str, object]], limits: BatchLimits | None) -> bytes:
    payload = b"".join(
        json.dumps(item, ensure_ascii=False, separators=(",", ":")).encode() + b"\n"
        for item in items
    )
    maximum = max_upload_bytes(limits)
    if len(payload) > maximum:
        raise BatchworkError(
            f"batchwork: batch upload JSONL is {len(payload)} bytes, "
            f"exceeding the {maximum} byte limit."
        )
    return payload


def encode_json(value: Mapping[str, object], limits: BatchLimits | None) -> bytes:
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
    maximum = max_upload_bytes(limits)
    if len(payload) > maximum:
        raise BatchworkError(
            f"batchwork: batch upload payload is {len(payload)} bytes, "
            f"exceeding the {maximum} byte limit."
        )
    return payload


async def request(
    client: httpx.AsyncClient | None,
    method: str,
    url: str,
    *,
    headers: Mapping[str, str] | None = None,
    content: bytes | str | None = None,
    files: Mapping[str, tuple[str, bytes, str]] | None = None,
    data: Mapping[str, str] | None = None,
) -> httpx.Response:
    try:
        if client is None:
            async with httpx.AsyncClient() as owned:
                response = await owned.request(
                    method,
                    url,
                    headers=headers,
                    content=content,
                    files=files,
                    data=data,
                    follow_redirects=False,
                )
        else:
            response = await client.request(
                method,
                url,
                headers=headers,
                content=content,
                files=files,
                data=data,
                follow_redirects=False,
            )
    except httpx.HTTPError as error:
        raise ProviderFailureError(
            "batchwork: provider request failed during transport.", transport_failure()
        ) from error
    if not 200 <= response.status_code < 300:
        raise ProviderFailureError(
            f"batchwork: {method} {url} failed with {response.status_code}", http_failure(response)
        )
    return response


@asynccontextmanager
async def stream_request(
    client: httpx.AsyncClient | None,
    method: str,
    url: str,
    *,
    headers: Mapping[str, str] | None = None,
) -> AsyncIterator[httpx.Response]:
    try:
        if client is None:
            async with httpx.AsyncClient() as owned:
                async with owned.stream(
                    method,
                    url,
                    headers=headers,
                    follow_redirects=False,
                ) as response:
                    _raise_for_provider_status(response, method, url)
                    yield response
            return

        async with client.stream(
            method,
            url,
            headers=headers,
            follow_redirects=False,
        ) as response:
            _raise_for_provider_status(response, method, url)
            yield response
    except httpx.HTTPError as error:
        raise ProviderFailureError(
            "batchwork: provider request failed during transport.", transport_failure()
        ) from error


async def request_json(
    client: httpx.AsyncClient | None,
    method: str,
    url: str,
    *,
    headers: Mapping[str, str] | None = None,
    content: bytes | str | None = None,
) -> dict[str, object]:
    response = await request(client, method, url, headers=headers, content=content)
    return response_json(response, method, url)


def response_json(response: httpx.Response, method: str, url: str) -> dict[str, object]:
    try:
        value = response.json()
    except ValueError as error:
        raise ProviderFailureError(
            f"batchwork: {method} {url} returned invalid JSON.", protocol_failure(response)
        ) from error
    if not isinstance(value, dict):
        raise ProviderFailureError(
            f"batchwork: {method} {url} returned a non-object JSON response.",
            protocol_failure(response),
        )
    return value


def _raise_for_provider_status(response: httpx.Response, method: str, url: str) -> None:
    if not 200 <= response.status_code < 300:
        raise ProviderFailureError(
            f"batchwork: {method} {url} failed with {response.status_code}", http_failure(response)
        )


async def jsonl(response: httpx.Response) -> AsyncIterator[dict[str, object]]:
    number = 0
    async for raw in response.aiter_lines():
        number += 1
        if not raw.strip():
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as error:
            raise BatchworkError(f"batchwork: malformed JSONL at line {number}.") from error
        if not isinstance(value, dict):
            raise BatchworkError(
                f"batchwork: malformed JSONL at line {number}: expected an object."
            )
        yield value


def timestamp(value: object) -> datetime | None:
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, UTC)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)
        except ValueError:
            return None
    return None


def _number(value: object) -> int | float | None:
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def text_from_body(value: object) -> str | None:
    if not is_string_mapping(value):
        return None
    choices = value.get("choices")
    if isinstance(choices, Sequence) and choices and is_string_mapping(choices[0]):
        choice = choices[0]
        message = choice.get("message")
        if is_string_mapping(message) and isinstance(message.get("content"), str):
            return str(message["content"])
        if isinstance(choice.get("text"), str):
            return str(choice["text"])
    output_text = value.get("output_text")
    if isinstance(output_text, str):
        return output_text
    output = value.get("output")
    if isinstance(output, Sequence):
        parts: list[str] = []
        for item in output:
            if not is_string_mapping(item):
                continue
            content = item.get("content")
            if not isinstance(content, Sequence):
                continue
            for part in content:
                if is_string_mapping(part) and isinstance(part.get("text"), str):
                    part_text = part.get("text")
                    if isinstance(part_text, str):
                        parts.append(part_text)
        return "".join(parts) or None
    return None


def number_array(value: object) -> list[float] | None:
    if not isinstance(value, list) or not value:
        return None
    numbers: list[float] = []
    for item in value:
        if not isinstance(item, (int, float)) or isinstance(item, bool):
            return None
        numbers.append(float(item))
    return numbers


def embedding_from_body(value: object) -> list[float] | None:
    if not is_string_mapping(value):
        return None
    data = value.get("data")
    if not isinstance(data, Sequence) or not data or not is_string_mapping(data[0]):
        return None
    raw = data[0].get("embedding")
    return number_array(raw)


def images_from_body(value: object) -> list[BatchImage] | None:
    if not is_string_mapping(value):
        return None
    raw = value.get("data")
    if not isinstance(raw, Sequence):
        return None
    media_type = f"image/{value.get('output_format', 'png')}"
    images: list[BatchImage] = []
    for item in raw:
        if not is_string_mapping(item):
            continue
        data = item.get("b64_json", item.get("base64"))
        url = item.get("url")
        if isinstance(data, str) or isinstance(url, str):
            images.append(
                BatchImage(
                    data=data if isinstance(data, str) else None,
                    media_type=media_type if isinstance(data, str) else None,
                    url=url if isinstance(url, str) else None,
                )
            )
    return images or None


def usage_from_body(value: object) -> BatchUsage | None:
    if not is_string_mapping(value) or not is_string_mapping(value.get("usage")):
        return None
    usage = value.get("usage")
    if not is_string_mapping(usage):
        return None
    input_tokens = _number(usage.get("prompt_tokens", usage.get("input_tokens")))
    output_tokens = _number(usage.get("completion_tokens", usage.get("output_tokens")))
    total_tokens = _number(usage.get("total_tokens"))
    if input_tokens is None and output_tokens is None and total_tokens is None:
        return None
    return BatchUsage(
        input_tokens=int(input_tokens) if input_tokens is not None else None,
        output_tokens=int(output_tokens) if output_tokens is not None else None,
        total_tokens=int(total_tokens)
        if total_tokens is not None
        else int(input_tokens or 0) + int(output_tokens or 0),
    )


def normalize_openai_result(line: Mapping[str, object]) -> BatchResult:
    custom_id = line.get("custom_id") if isinstance(line.get("custom_id"), str) else ""
    top_error = line.get("error")
    if top_error:
        error = top_error if is_string_mapping(top_error) else {}
        nested_error = error.get("error")
        source = nested_error if is_string_mapping(nested_error) else error
        return BatchResult.model_validate(
            {
                "custom_id": custom_id,
                "status": "errored",
                "error": {
                    "message": source.get("message")
                    if isinstance(source.get("message"), str)
                    else "Request errored.",
                    "type": source.get("type") if isinstance(source.get("type"), str) else None,
                    "code": source.get("code"),
                },
                "response": top_error,
            }
        )
    response = line.get("response")
    response_map = response if is_string_mapping(response) else {}
    status_code = response_map.get("status_code")
    body = response_map.get("body")
    if isinstance(status_code, int) and 200 <= status_code < 300:
        return BatchResult.model_validate(
            {
                "custom_id": custom_id,
                "status": "succeeded",
                "response": body,
                "text": text_from_body(body),
                "embedding": embedding_from_body(body),
                "images": images_from_body(body),
                "usage": usage_from_body(body),
            }
        )
    body_map = body if is_string_mapping(body) else {}
    nested = body_map.get("error")
    source = nested if is_string_mapping(nested) else body_map
    return BatchResult.model_validate(
        {
            "custom_id": custom_id,
            "status": "errored",
            "response": body,
            "error": {
                "message": source.get("message")
                if isinstance(source.get("message"), str)
                else f"Request failed with status {status_code or 0}.",
                "type": source.get("type") if isinstance(source.get("type"), str) else None,
                "code": source.get("code"),
            },
        }
    )


async def upload_file(
    client: httpx.AsyncClient | None,
    url: str,
    headers: Mapping[str, str],
    payload: bytes,
    *,
    purpose: str | None = "batch",
    endpoint: str = "/files",
    file_name: str | None = None,
) -> str:
    data: dict[str, str] = {}
    if purpose is not None:
        data["purpose"] = purpose
    if file_name is not None:
        data["file_name"] = file_name
    response = await request(
        client,
        "POST",
        f"{url}{endpoint}",
        headers=headers,
        files={"file": ("batchwork.jsonl", payload, "application/jsonl")},
        data=data or None,
    )
    value = response_json(response, "POST", f"{url}{endpoint}")
    if not is_string_mapping(value) or not isinstance(value.get("id"), str):
        raise BatchworkError("batchwork: provider file upload returned no file id.")
    return str(value["id"])


async def stream_result_file(
    client: httpx.AsyncClient | None,
    url: str,
    headers: Mapping[str, str],
    file_id: str,
) -> AsyncIterator[BatchResult]:
    async with stream_request(
        client, "GET", f"{url}/files/{file_id}/content", headers=headers
    ) as response:
        async for line in jsonl(response):
            yield normalize_openai_result(line)
