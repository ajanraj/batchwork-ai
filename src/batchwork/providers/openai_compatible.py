"""OpenAI-shaped Files and Batches adapters."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from urllib.parse import urlsplit

import httpx

from batchwork._network import (
    AddressResolutionFailureReason,
    ResolvedAddresses,
    resolve_public_addresses,
)
from batchwork._provider_failure import (
    ProviderFailureError,
    http_failure,
    protocol_failure,
    transport_failure,
)
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

_TOGETHER_INPUT_FILE_NAME = "batchwork.jsonl"
_FileUploader = Callable[
    [httpx.AsyncClient | None, str, Mapping[str, str], bytes, str], Awaitable[str]
]


def _together_upload_location(location: str) -> tuple[httpx.URL, str, int]:
    if any(ord(character) < 32 for character in location):
        raise BatchworkError("batchwork: Together upload Location must be a valid URL.")
    try:
        parsed = urlsplit(location)
        _ = parsed.port
    except ValueError as error:
        raise BatchworkError("batchwork: Together upload Location must be a valid URL.") from error
    if parsed.scheme != "https":
        raise BatchworkError("batchwork: Together upload Location must use https.")
    if not parsed.hostname:
        raise BatchworkError("batchwork: Together upload Location must be a valid URL.")
    if parsed.username or parsed.password:
        raise BatchworkError("batchwork: Together upload Location must not include credentials.")
    upload_url = httpx.URL(location)
    host = upload_url.host
    if not host:
        raise BatchworkError("batchwork: Together upload Location must be a valid URL.")
    return upload_url, host, parsed.port or 443


async def _resolve_together_upload_addresses(host: str, port: int) -> tuple[str, ...]:
    result = await resolve_public_addresses(host, port)
    if isinstance(result, ResolvedAddresses):
        return result.addresses
    if result.reason is AddressResolutionFailureReason.LOOKUP:
        raise ProviderFailureError(
            "batchwork: Together upload hostname could not be resolved.", transport_failure()
        ) from None
    if result.reason is AddressResolutionFailureReason.EMPTY:
        raise ProviderFailureError(
            "batchwork: Together upload hostname resolved to no addresses.", transport_failure()
        )
    if result.reason is AddressResolutionFailureReason.INVALID:
        raise BatchworkError(
            f"batchwork: Together upload hostname {host!r} returned an invalid address."
        ) from result.cause
    if result.reason is AddressResolutionFailureReason.NON_GLOBAL:
        raise BatchworkError(
            "batchwork: Together upload Location must not target localhost or private networks."
        )
    raise RuntimeError("batchwork: unexpected Together address resolution failure")


async def _put_together_upload(client: httpx.AsyncClient, location: str, payload: bytes) -> int:
    upload_url, host, port = _together_upload_location(location)
    addresses = await _resolve_together_upload_addresses(host, port)
    authority = upload_url.netloc.decode("ascii")
    for address in addresses:
        request = httpx.Request(
            "PUT",
            upload_url.copy_with(host=address),
            headers={"host": authority},
            content=payload,
            extensions={"sni_hostname": host},
        )
        try:
            response = await client.send(
                request,
                stream=True,
                auth=None,
                follow_redirects=False,
            )
            try:
                if not 200 <= response.status_code < 300:
                    raise ProviderFailureError(
                        f"batchwork: Together file upload failed ({response.status_code}).",
                        http_failure(response),
                    )
                return response.status_code
            finally:
                await response.aclose()
        except (httpx.ConnectError, httpx.ConnectTimeout):
            continue
        except httpx.HTTPError:
            raise ProviderFailureError(
                "batchwork: Together file upload failed during transport.", transport_failure()
            ) from None
    raise ProviderFailureError(
        "batchwork: Together file upload could not connect to any resolved address.",
        transport_failure(),
    )


async def _upload_together_file_with_client(
    client: httpx.AsyncClient,
    base: str,
    headers: Mapping[str, str],
    payload: bytes,
    purpose: str,
) -> str:
    metadata = {
        "purpose": (None, purpose),
        "file_name": (None, _TOGETHER_INPUT_FILE_NAME),
        "file_type": (None, "jsonl"),
    }
    try:
        async with client.stream(
            "POST",
            f"{base}/files",
            headers=headers,
            files=metadata,
            follow_redirects=False,
        ) as initiated:
            status = initiated.status_code
            location = initiated.headers.get("location")
            file_id = initiated.headers.get("x-together-file-id")
            if status != httpx.codes.FOUND or not location or not file_id:
                failure = (
                    http_failure(initiated)
                    if status != httpx.codes.FOUND
                    else protocol_failure(initiated)
                )
                raise ProviderFailureError(
                    f"batchwork: Together upload could not be initiated ({status}).", failure
                )
    except httpx.HTTPError:
        raise ProviderFailureError(
            "batchwork: Together upload initiation failed during transport.", transport_failure()
        ) from None

    await _put_together_upload(client, location, payload)

    safe_file_id = simple_provider_id("Together file id", file_id)
    await request_json(
        client,
        "POST",
        f"{base}/files/{safe_file_id}/preprocess",
        headers=headers,
    )
    return safe_file_id


async def _upload_together_file(
    client: httpx.AsyncClient | None,
    base: str,
    headers: Mapping[str, str],
    payload: bytes,
    purpose: str,
) -> str:
    if client is not None:
        return await _upload_together_file_with_client(client, base, headers, payload, purpose)
    async with httpx.AsyncClient() as owned:
        return await _upload_together_file_with_client(owned, base, headers, payload, purpose)


def _status(value: object) -> str:
    if isinstance(value, str) and value.lower() in {
        "validating",
        "in_progress",
        "finalizing",
        "completed",
        "failed",
        "expired",
        "cancelling",
        "cancelled",
    }:
        return value.lower()
    return "in_progress"


class OpenAICompatibleAdapter:
    def __init__(
        self,
        provider: BatchProvider,
        *,
        http_client: httpx.AsyncClient | None,
        default_base_url: str,
        env_var: str,
        label: str,
        line_format: str = "method-url",
        completion_window: str = "24h",
        file_purpose: str = "batch",
        file_endpoint: str = "/files",
        file_name: str | None = None,
        file_uploader: _FileUploader | None = None,
        normalize_endpoint: Callable[[str], str] | None = None,
    ) -> None:
        self.id = provider
        self._client = http_client
        self._default_base_url = default_base_url
        self._env_var = env_var
        self._label = label
        self._line_format = line_format
        self._completion_window = completion_window
        self._file_purpose = file_purpose
        self._file_endpoint = file_endpoint
        self._file_name = file_name
        self._file_uploader = file_uploader
        self._normalize_endpoint = normalize_endpoint

    def _base(self, credentials: ProviderCredentials) -> str:
        return base_url(credentials, self._default_base_url)

    def _headers(self, credentials: ProviderCredentials) -> dict[str, str]:
        key = api_key(credentials, [self._env_var], self._label)
        return merge_headers({"Authorization": f"Bearer {key}"}, credentials)

    def _snapshot(self, raw: Mapping[str, object]) -> BatchSnapshot:
        nested = raw.get("job")
        source = nested if isinstance(nested, Mapping) else raw
        counts_value = source.get("request_counts")
        counts = counts_value if isinstance(counts_value, Mapping) else {}
        return BatchSnapshot.model_validate(
            {
                "id": source.get("id") if isinstance(source.get("id"), str) else "",
                "provider": self.id,
                "status": _status(source.get("status")),
                "raw": dict(source),
                "created_at": timestamp(source.get("created_at")),
                "completed_at": timestamp(source.get("completed_at")),
                "expires_at": timestamp(source.get("expires_at")),
                "request_counts": {
                    "completed": counts.get("completed", 0),
                    "failed": counts.get("failed", 0),
                    "total": counts.get("total", 0),
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
        del model_id
        normalized = self._normalize_endpoint(endpoint) if self._normalize_endpoint else endpoint
        lines: list[dict[str, object]] = []
        for item in built:
            body = {key: value for key, value in item.body.items() if key != "stream"}
            if self._line_format == "body-only":
                lines.append({"custom_id": item.custom_id, "body": body})
            else:
                lines.append(
                    {"custom_id": item.custom_id, "method": "POST", "url": normalized, "body": body}
                )
        payload = encode_jsonl(lines, limits, validate_upload=validate_upload)
        url = self._base(credentials)
        headers = self._headers(credentials)
        if self._file_uploader is None:
            input_file_id = await upload_file(
                self._client,
                url,
                headers,
                payload,
                purpose=self._file_purpose,
                endpoint=self._file_endpoint,
                file_name=self._file_name,
            )
        else:
            input_file_id = await self._file_uploader(
                self._client, url, headers, payload, self._file_purpose
            )
        create_body: dict[str, object] = {
            "completion_window": self._completion_window,
            "endpoint": normalized,
            "input_file_id": input_file_id,
        }
        if metadata is not None:
            create_body["metadata"] = dict(metadata)
        raw = await request_json(
            self._client,
            "POST",
            f"{url}/batches",
            headers={**headers, "content-type": "application/json"},
            content=json.dumps(create_body),
        )
        return self._snapshot(raw)

    async def retrieve(self, id: str, credentials: ProviderCredentials) -> BatchSnapshot:
        batch_id = simple_provider_id(f"{self.id.value} batch id", id)
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
        snapshot = await self.retrieve(id, credentials)
        async for result in self.results_from_snapshot(snapshot, credentials):
            yield result

    async def results_from_snapshot(
        self, snapshot: BatchSnapshot, credentials: ProviderCredentials
    ) -> AsyncIterator[BatchResult]:
        raw = snapshot.raw
        output = raw.get("output_file_id") if isinstance(raw, Mapping) else None
        error = raw.get("error_file_id") if isinstance(raw, Mapping) else None
        if not isinstance(output, str) and not isinstance(error, str):
            raise BatchworkError(
                f'batchwork: results are not ready for batch "{snapshot.id}" '
                f"(status: {snapshot.status})."
            )
        url = self._base(credentials)
        headers = self._headers(credentials)
        if isinstance(output, str):
            safe = simple_provider_id(f"{self.id.value} output file id", output)
            async for result in stream_result_file(self._client, url, headers, safe):
                yield result
        if isinstance(error, str):
            safe = simple_provider_id(f"{self.id.value} error file id", error)
            async for result in stream_result_file(self._client, url, headers, safe):
                yield result

    async def cancel(self, id: str, credentials: ProviderCredentials) -> None:
        batch_id = simple_provider_id(f"{self.id.value} batch id", id)
        await request_json(
            self._client,
            "POST",
            f"{self._base(credentials)}/batches/{batch_id}/cancel",
            headers=self._headers(credentials),
        )


def openai_adapter(http_client: httpx.AsyncClient | None) -> OpenAICompatibleAdapter:
    return OpenAICompatibleAdapter(
        BatchProvider.OPENAI,
        http_client=http_client,
        default_base_url="https://api.openai.com/v1",
        env_var="OPENAI_API_KEY",
        label="OpenAI",
    )


def groq_adapter(http_client: httpx.AsyncClient | None) -> OpenAICompatibleAdapter:
    return OpenAICompatibleAdapter(
        BatchProvider.GROQ,
        http_client=http_client,
        default_base_url="https://api.groq.com/openai/v1",
        env_var="GROQ_API_KEY",
        label="Groq",
        normalize_endpoint=lambda endpoint: endpoint.removeprefix("/openai"),
    )


def together_adapter(http_client: httpx.AsyncClient | None) -> OpenAICompatibleAdapter:
    return OpenAICompatibleAdapter(
        BatchProvider.TOGETHER,
        http_client=http_client,
        default_base_url="https://api.together.xyz/v1",
        env_var="TOGETHER_API_KEY",
        label="Together AI",
        line_format="body-only",
        file_purpose="batch-api",
        file_uploader=_upload_together_file,
    )
