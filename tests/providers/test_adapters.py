from __future__ import annotations

import asyncio
import json
import socket

import httpx
import pytest

import batchwork.providers.openai_compatible as compatible_module
import batchwork.providers.shared as shared_module
from batchwork._base_url import BaseUrlError
from batchwork._network import AddressResolutionFailure, AddressResolutionFailureReason
from batchwork._provider_failure import ProviderFailureError, ProviderFailureKind
from batchwork.body import BuiltRequest
from batchwork.errors import BatchworkError
from batchwork.providers import get_adapter
from batchwork.providers.shared import (
    embedding_from_body,
    jsonl,
    normalize_openai_result,
    request,
    request_json,
)
from batchwork.types import BatchLimits, BatchProvider, ProviderCredentials

_GOOGLE_INLINE_BATCH_MAX_BYTES = 20 * 1024 * 1024


@pytest.mark.asyncio
async def test_adapter_revalidates_base_url_before_network_dispatch() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200)

    credentials = ProviderCredentials.model_construct(
        api_key="secret",
        base_url="http://169.254.169.254/latest",
        headers={},
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        adapter = get_adapter(BatchProvider.OPENAI, http_client=client)
        with pytest.raises(BaseUrlError, match="absolute HTTPS"):
            await adapter.retrieve("batch_1", credentials)

    assert requests == []


@pytest.mark.asyncio
async def test_provider_response_content_length_is_rejected_before_body_read() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"Content-Length": str(320 * 1024 * 1024 + 1)},
            stream=httpx.ByteStream(b""),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ProviderFailureError) as caught:
            await request_json(client, "GET", "https://example.test/batch")

    assert caught.value.failure.kind is ProviderFailureKind.PROTOCOL


@pytest.mark.asyncio
async def test_provider_gzip_response_is_decoded_once() -> None:
    import gzip

    payload = gzip.compress(b'{"ok":true}')

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={
                "Content-Encoding": "gzip",
                "Content-Length": str(len(payload)),
                "Content-Type": "application/json",
            },
            stream=httpx.ByteStream(payload),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        body = await request_json(client, "GET", "https://example.test/batch")

    assert body == {"ok": True}


@pytest.mark.asyncio
async def test_result_record_overflow_preserves_prior_complete_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(shared_module, "MAX_RESULT_RECORD_BYTES", 7)
    response = httpx.Response(200, content=b'{"a":1}\n{"long":2}\n')
    records: list[dict[str, object]] = []

    with pytest.raises(BatchworkError, match="record at line 2"):
        async for record in jsonl(response):
            records.append(record)

    assert records == [{"a": 1}]


@pytest.mark.asyncio
async def test_result_aggregate_overflow_stops_after_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(shared_module, "MAX_AGGREGATE_RESULTS_BYTES", 9)
    response = httpx.Response(200, stream=httpx.ByteStream(b'{"a":1}\n{}\n'))
    records: list[dict[str, object]] = []

    with pytest.raises(BatchworkError, match="transport exceeded"):
        async for record in jsonl(response):
            records.append(record)

    assert records == [{"a": 1}]


@pytest.mark.asyncio
async def test_result_content_length_overflow_stops_before_body_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(shared_module, "MAX_AGGREGATE_RESULTS_BYTES", 9)
    response = httpx.Response(
        200,
        headers={"Content-Length": "10"},
        stream=httpx.ByteStream(b'{"a":1}\n{}'),
    )
    records: list[dict[str, object]] = []

    with pytest.raises(BatchworkError, match="transport exceeded"):
        async for record in jsonl(response):
            records.append(record)

    assert records == []


def _google_built_request_for_payload_size(payload_size: int) -> BuiltRequest:
    body: dict[str, object] = {"contents": [{"parts": [{"text": ""}]}]}
    envelope = {
        "batch": {
            "display_name": "batchwork",
            "input_config": {
                "requests": {"requests": [{"metadata": {"key": "a"}, "request": body}]}
            },
        }
    }
    empty_size = len(json.dumps(envelope, ensure_ascii=False, separators=(",", ":")).encode())
    text_size = payload_size - empty_size
    if text_size < 0:
        raise ValueError(f"payload size must be at least {empty_size} bytes")
    return BuiltRequest(
        body={"contents": [{"parts": [{"text": "x" * text_size}]}]},
        custom_id="a",
        endpoint="/v1beta/models/gemini:generateContent",
    )


def test_openai_result_normalizes_all_current_text_shapes() -> None:
    bodies = [
        {"choices": [{"message": {"content": "chat"}}]},
        {"choices": [{"text": "completion"}]},
        {"output": [{"content": [{"type": "output_text", "text": "response"}]}]},
    ]
    assert [
        normalize_openai_result(
            {"custom_id": str(index), "response": {"status_code": 200, "body": body}}
        ).text
        for index, body in enumerate(bodies)
    ] == ["chat", "completion", "response"]


def test_openai_result_normalizes_image_data_and_usage() -> None:
    result = normalize_openai_result(
        {
            "custom_id": "image-1",
            "response": {
                "status_code": 200,
                "body": {
                    "data": [{"b64_json": "aW1n"}],
                    "output_format": "webp",
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 20,
                        "total_tokens": 30,
                    },
                },
            },
        }
    )

    assert result.images is not None
    assert result.images[0].data == "aW1n"
    assert result.images[0].media_type == "image/webp"
    assert result.usage is not None
    assert result.usage.model_dump() == {
        "input_tokens": 10,
        "output_tokens": 20,
        "total_tokens": 30,
    }


def test_openai_result_unwraps_nested_error_and_rejects_malformed_embedding() -> None:
    result = normalize_openai_result(
        {
            "custom_id": "a",
            "error": {"error": {"message": "nested", "type": "invalid", "code": "bad"}},
        }
    )
    assert result.error is not None
    assert (result.error.message, result.error.type, result.error.code) == (
        "nested",
        "invalid",
        "bad",
    )
    assert embedding_from_body({"data": [{"embedding": [1, "bad", 2]}]}) is None
    assert embedding_from_body({"data": [{"embedding": [1, True, 2]}]}) is None


@pytest.mark.asyncio
async def test_openai_uploads_jsonl_then_creates_batch() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/files"):
            return httpx.Response(200, json={"id": "file-in"})
        if request.url.path.endswith("/batches"):
            return httpx.Response(
                200,
                json={
                    "id": "batch_1",
                    "status": "validating",
                    "request_counts": {"completed": 0, "failed": 0, "total": 1},
                },
            )
        raise AssertionError(f"unexpected request: {request.method} {request.url}")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        adapter = get_adapter(BatchProvider.OPENAI, http_client=client)
        snapshot = await adapter.submit(
            built=[
                BuiltRequest(
                    body={"model": "gpt", "messages": [{"role": "user", "content": "hi"}]},
                    custom_id="a",
                    endpoint="/v1/chat/completions",
                )
            ],
            credentials=ProviderCredentials(api_key="secret"),
            endpoint="/v1/chat/completions",
            model_id="gpt",
        )

    assert snapshot.id == "batch_1"
    assert [request.url.path for request in requests] == ["/v1/files", "/v1/batches"]
    create = json.loads(requests[1].content)
    assert create["input_file_id"] == "file-in"
    assert create["completion_window"] == "24h"


@pytest.mark.asyncio
async def test_http_error_does_not_include_provider_body() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="secret prompt and provider diagnostics")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(BatchworkError) as caught:
            await request(client, "GET", "https://example.test/failure")

    assert "secret prompt" not in str(caught.value)
    assert str(caught.value) == "batchwork: GET https://example.test/failure failed with 400"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "kind"),
    [
        (401, ProviderFailureKind.AUTHENTICATION),
        (403, ProviderFailureKind.AUTHORIZATION),
        (404, ProviderFailureKind.NOT_FOUND),
        (400, ProviderFailureKind.REJECTED),
        (429, ProviderFailureKind.UNAVAILABLE),
        (503, ProviderFailureKind.UNAVAILABLE),
    ],
)
async def test_http_error_exposes_only_safe_structured_metadata(
    status: int, kind: ProviderFailureKind
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status,
            headers={
                "x-request-id": "req_123",
                "retry-after": "7200",
                "x-provider-secret": "must-not-be-retained",
            },
            text="secret body",
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ProviderFailureError) as caught:
            await request(client, "GET", "https://example.test/failure")

    assert caught.value.failure.kind is kind
    assert caught.value.failure.status_code == status
    assert caught.value.failure.request_id == "req_123"
    assert caught.value.failure.retry_after_seconds == 3600
    assert not hasattr(caught.value.failure, "response")


@pytest.mark.asyncio
async def test_transport_error_is_structured_without_transport_details() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("secret headers and URL", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ProviderFailureError) as caught:
            await request(client, "GET", "https://secret.example.test/failure")

    assert caught.value.failure.kind is ProviderFailureKind.TRANSPORT
    assert caught.value.failure.status_code is None
    assert str(caught.value) == "batchwork: provider request failed during transport."


@pytest.mark.asyncio
@pytest.mark.parametrize("body", ["not json", "[]"])
async def test_successful_invalid_json_is_structured_protocol_failure(body: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=body, headers={"request-id": "req_protocol"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ProviderFailureError) as caught:
            await request_json(client, "GET", "https://example.test/json")

    assert caught.value.failure.kind is ProviderFailureKind.PROTOCOL
    assert caught.value.failure.status_code == 200
    assert caught.value.failure.request_id == "req_protocol"


@pytest.mark.asyncio
async def test_anthropic_rejects_cross_origin_results_url() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/batches/batch_1"):
            return httpx.Response(
                200,
                json={
                    "id": "batch_1",
                    "processing_status": "ended",
                    "results_url": "https://attacker.example/results",
                    "request_counts": {},
                },
            )
        raise AssertionError("cross-origin request must not be made")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        adapter = get_adapter(BatchProvider.ANTHROPIC, http_client=client)
        with pytest.raises(BatchworkError, match="must match"):
            _ = [
                item
                async for item in adapter.results("batch_1", ProviderCredentials(api_key="secret"))
            ]


@pytest.mark.asyncio
async def test_anthropic_text_submit_uses_exact_batch_contract() -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            json={
                "id": "batch_1",
                "processing_status": "in_progress",
                "request_counts": {"processing": 1},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        snapshot = await get_adapter(BatchProvider.ANTHROPIC, http_client=client).submit(
            built=[
                BuiltRequest(
                    body={
                        "model": "claude-test",
                        "messages": [{"role": "user", "content": "hello"}],
                        "max_tokens": 32,
                    },
                    custom_id="a",
                    endpoint="/v1/messages",
                )
            ],
            credentials=ProviderCredentials(api_key="secret"),
            endpoint="/v1/messages",
            model_id="claude-test",
        )

    assert snapshot.id == "batch_1"
    assert captured[0].url.path == "/v1/messages/batches"
    assert captured[0].headers["x-api-key"] == "secret"
    assert captured[0].headers["anthropic-version"] == "2023-06-01"
    assert json.loads(captured[0].content) == {
        "requests": [
            {
                "custom_id": "a",
                "params": {
                    "model": "claude-test",
                    "messages": [{"role": "user", "content": "hello"}],
                    "max_tokens": 32,
                },
            }
        ]
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider", "upload_path", "captured_endpoint"),
    [
        (BatchProvider.GROQ, "/openai/v1/files", "/openai/v1/chat/completions"),
    ],
)
async def test_openai_compatible_provider_specific_upload_and_line_shape(
    provider: BatchProvider, upload_path: str, captured_endpoint: str
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == upload_path:
            return httpx.Response(200, json={"id": "file-in"})
        if request.url.path.endswith("/batches"):
            return httpx.Response(
                200,
                json={
                    "id": "batch_1",
                    "status": "validating",
                    "request_counts": {"completed": 0, "failed": 0, "total": 1},
                },
            )
        raise AssertionError(f"unexpected request: {request.url}")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        adapter = get_adapter(provider, http_client=client)
        await adapter.submit(
            built=[
                BuiltRequest(
                    body={"model": "model", "messages": []},
                    custom_id="a",
                    endpoint=captured_endpoint,
                )
            ],
            credentials=ProviderCredentials(api_key="secret"),
            endpoint=captured_endpoint,
            model_id="model",
        )

    upload = requests[0].content
    create = json.loads(requests[1].content)
    assert b'"url":"/v1/chat/completions"' in upload
    assert create["endpoint"] == "/v1/chat/completions"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider", [BatchProvider.OPENAI, BatchProvider.GROQ, BatchProvider.TOGETHER]
)
async def test_openai_compatible_lifecycle_uses_provider_route(provider: BatchProvider) -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        if request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "id": "batch_1",
                    "status": "completed",
                    "request_counts": {"total": 1, "completed": 1, "failed": 0},
                },
            )
        return httpx.Response(200, json={})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        adapter = get_adapter(provider, http_client=client)
        credentials = ProviderCredentials(api_key="secret")
        snapshot = await adapter.retrieve("batch_1", credentials)
        await adapter.cancel("batch_1", credentials)

    assert snapshot.status == "completed"
    assert [request.url.path for request in captured] == [
        ("/openai/v1/batches/batch_1" if provider is BatchProvider.GROQ else "/v1/batches/batch_1"),
        (
            "/openai/v1/batches/batch_1/cancel"
            if provider is BatchProvider.GROQ
            else "/v1/batches/batch_1/cancel"
        ),
    ]
    assert all(request.headers["authorization"] == "Bearer secret" for request in captured)


@pytest.mark.asyncio
async def test_together_uses_pinned_presigned_upload_protocol_before_batch_create(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[httpx.Request] = []
    storage_url = "https://storage.example/presigned-put?signature=signed"

    async def resolve(host: str, port: int) -> tuple[str, ...]:
        assert (host, port) == ("storage.example", 443)
        return ("8.8.8.8",)

    monkeypatch.setattr(compatible_module, "_resolve_together_upload_addresses", resolve)

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/v1/files":
            return httpx.Response(
                302,
                headers={"Location": storage_url, "X-Together-File-Id": "file-in"},
            )
        if request.method == "PUT":
            return httpx.Response(200)
        if request.url.path == "/v1/files/file-in/preprocess":
            return httpx.Response(200, json={"id": "file-in"})
        if request.url.path == "/v1/batches":
            return httpx.Response(
                200,
                json={
                    "job": {
                        "id": "batch_t",
                        "status": "VALIDATING",
                        "request_counts": {"completed": 0, "failed": 0, "total": 1},
                    }
                },
            )
        raise AssertionError(f"unexpected request: {request.method} {request.url}")

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        auth=("injected-client-user", "injected-client-password"),
        headers={
            "authorization": "Bearer injected-client-secret",
            "x-client-secret": "injected-client-secret",
        },
        cookies={"session": "injected-cookie-secret"},
    ) as client:
        adapter = get_adapter(BatchProvider.TOGETHER, http_client=client)
        snapshot = await adapter.submit(
            built=[
                BuiltRequest(
                    body={"model": "model", "messages": [], "stream": True},
                    custom_id="a",
                    endpoint="/v1/chat/completions",
                )
            ],
            credentials=ProviderCredentials(api_key="secret"),
            endpoint="/v1/chat/completions",
            model_id="model",
        )
        assert not client.is_closed

    assert snapshot.id == "batch_t"
    assert snapshot.status == "validating"
    assert [(request.method, request.url.path) for request in requests] == [
        ("POST", "/v1/files"),
        ("PUT", "/presigned-put"),
        ("POST", "/v1/files/file-in/preprocess"),
        ("POST", "/v1/batches"),
    ]

    init = requests[0]
    assert b'name="purpose"' in init.content and b"batch-api" in init.content
    assert b'name="file_name"' in init.content and b"batchwork.jsonl" in init.content
    assert b'name="file_type"' in init.content and b"jsonl" in init.content
    assert b'filename="' not in init.content
    assert b'"custom_id"' not in init.content

    upload = requests[1]
    assert "authorization" not in upload.headers
    assert "x-client-secret" not in upload.headers
    assert "cookie" not in upload.headers
    assert upload.url.host == "8.8.8.8"
    assert upload.url.query == b"signature=signed"
    assert upload.headers["host"] == "storage.example"
    assert upload.headers["content-length"] == str(len(upload.content))
    assert upload.extensions["sni_hostname"] == "storage.example"
    assert upload.content == b'{"custom_id":"a","body":{"model":"model","messages":[]}}\n'

    create = json.loads(requests[3].content)
    assert create["input_file_id"] == "file-in"
    assert create["endpoint"] == "/v1/chat/completions"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "location",
    [
        "https://[::ffff:127.0.0.1]/presigned-put",
        "https://127.0.0.1/presigned-put",
    ],
)
async def test_together_rejects_non_global_upload_literals(location: str) -> None:
    attempted: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempted.append(request)
        return httpx.Response(200)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(BatchworkError, match="localhost or private networks"):
            await compatible_module._put_together_upload(client, location, b"payload")

    assert attempted == []


@pytest.mark.asyncio
async def test_together_rejects_entire_dns_result_if_any_address_is_private(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = asyncio.get_running_loop()
    attempted: list[httpx.Request] = []

    async def getaddrinfo(
        host: str,
        port: int,
        *,
        family: int,
        type: int,
    ) -> list[tuple[int, int, int, str, tuple[str, int]]]:
        assert (host, port, family, type) == (
            "storage.example",
            443,
            socket.AF_UNSPEC,
            socket.SOCK_STREAM,
        )
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 443)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443)),
        ]

    def handler(request: httpx.Request) -> httpx.Response:
        attempted.append(request)
        return httpx.Response(200)

    monkeypatch.setattr(loop, "getaddrinfo", getaddrinfo)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(BatchworkError, match="localhost or private networks"):
            await compatible_module._put_together_upload(
                client, "https://storage.example/presigned-put", b"payload"
            )

    assert attempted == []


@pytest.mark.asyncio
async def test_together_rejects_invalid_dns_address(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = asyncio.get_running_loop()

    async def getaddrinfo(
        _host: str,
        _port: int,
        *,
        family: int,
        type: int,
    ) -> list[tuple[int, int, int, str, tuple[str, int]]]:
        assert (family, type) == (socket.AF_UNSPEC, socket.SOCK_STREAM)
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("invalid-address", 443))]

    monkeypatch.setattr(loop, "getaddrinfo", getaddrinfo)

    with pytest.raises(BatchworkError, match="returned an invalid address"):
        await compatible_module._resolve_together_upload_addresses("storage.example", 443)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reason", "message"),
    [
        (AddressResolutionFailureReason.LOOKUP, "could not be resolved"),
        (AddressResolutionFailureReason.EMPTY, "resolved to no addresses"),
    ],
)
async def test_together_dns_availability_failures_are_structured_transport_errors(
    monkeypatch: pytest.MonkeyPatch,
    reason: AddressResolutionFailureReason,
    message: str,
) -> None:
    async def resolve(_host: str, _port: int) -> AddressResolutionFailure:
        return AddressResolutionFailure(reason, cause=OSError("secret resolver details"))

    monkeypatch.setattr(compatible_module, "resolve_public_addresses", resolve)
    with pytest.raises(ProviderFailureError, match=message) as caught:
        await compatible_module._resolve_together_upload_addresses("storage.example", 443)

    assert caught.value.failure.kind is ProviderFailureKind.TRANSPORT
    assert caught.value.failure.status_code is None
    assert "secret resolver details" not in str(caught.value)


@pytest.mark.asyncio
async def test_together_upload_tries_each_validated_address(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts: list[tuple[str, str, object]] = []

    async def resolve(host: str, port: int) -> tuple[str, ...]:
        assert (host, port) == ("storage.example", 443)
        return ("8.8.8.8", "1.1.1.1")

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(
            (request.url.host, request.headers["host"], request.extensions.get("sni_hostname"))
        )
        if request.url.host == "8.8.8.8":
            raise httpx.ConnectError("first address unavailable", request=request)
        return httpx.Response(200)

    monkeypatch.setattr(compatible_module, "_resolve_together_upload_addresses", resolve)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        status = await compatible_module._put_together_upload(
            client, "https://storage.example/presigned-put", b"payload"
        )
        assert not client.is_closed

    assert status == 200
    assert attempts == [
        ("8.8.8.8", "storage.example", "storage.example"),
        ("1.1.1.1", "storage.example", "storage.example"),
    ]


@pytest.mark.asyncio
async def test_together_init_failure_does_not_consume_provider_body() -> None:
    class UnreadableStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            raise AssertionError("failed response body must not be consumed")
            yield b""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, stream=UnreadableStream())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        adapter = get_adapter(BatchProvider.TOGETHER, http_client=client)
        with pytest.raises(
            ProviderFailureError, match=r"Together upload could not be initiated \(503\)"
        ) as caught:
            await adapter.submit(
                built=[
                    BuiltRequest(
                        body={"model": "model", "messages": []},
                        custom_id="a",
                        endpoint="/v1/chat/completions",
                    )
                ],
                credentials=ProviderCredentials(api_key="secret"),
                endpoint="/v1/chat/completions",
                model_id="model",
            )

    assert caught.value.failure.kind is ProviderFailureKind.UNAVAILABLE
    assert caught.value.failure.status_code == 503


@pytest.mark.asyncio
async def test_together_presigned_upload_failure_is_structured_without_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def resolve(host: str, port: int) -> tuple[str, ...]:
        assert (host, port) == ("storage.example", 443)
        return ("8.8.8.8",)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            headers={"x-request-id": "req_together", "retry-after": "30"},
            text="provider diagnostics",
        )

    monkeypatch.setattr(compatible_module, "_resolve_together_upload_addresses", resolve)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ProviderFailureError) as caught:
            await compatible_module._put_together_upload(
                client, "https://storage.example/presigned-put", b"payload"
            )

    assert str(caught.value) == "batchwork: Together file upload failed (429)."
    assert caught.value.failure.kind is ProviderFailureKind.UNAVAILABLE
    assert caught.value.failure.status_code == 429
    assert caught.value.failure.request_id == "req_together"
    assert caught.value.failure.retry_after_seconds == 30


@pytest.mark.asyncio
async def test_google_embedding_submit_and_inline_result() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "POST":
            return httpx.Response(
                200,
                json={"name": "batches/1", "metadata": {"state": "JOB_STATE_PENDING"}},
            )
        return httpx.Response(
            200,
            json={
                "name": "batches/1",
                "done": True,
                "metadata": {"state": "JOB_STATE_SUCCEEDED"},
                "response": {
                    "inlinedResponses": {
                        "inlinedResponses": [
                            {
                                "metadata": {"key": "a"},
                                "response": {"embedding": {"values": [0.1, 0.2]}},
                            }
                        ]
                    }
                },
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        adapter = get_adapter(BatchProvider.GOOGLE, http_client=client)
        snapshot = await adapter.submit(
            built=[
                BuiltRequest(
                    body={
                        "model": "models/embed",
                        "content": {"parts": [{"text": "hello"}]},
                        "outputDimensionality": 256,
                        "taskType": "RETRIEVAL_DOCUMENT",
                        "title": "Knowledge base entry",
                    },
                    custom_id="a",
                    endpoint="/v1beta/models/embed:embedContent",
                )
            ],
            credentials=ProviderCredentials(api_key="secret"),
            endpoint="/v1beta/models/embed:embedContent",
            model_id="embed",
        )
        results = [
            item
            async for item in adapter.results(snapshot.id, ProviderCredentials(api_key="secret"))
        ]

    assert requests[0].url.path.endswith(":asyncBatchEmbedContent")
    sent = json.loads(requests[0].content)
    request_body = sent["batch"]["input_config"]["requests"]["requests"][0]["request"]
    assert request_body["embedContentConfig"] == {
        "outputDimensionality": 256,
        "taskType": "RETRIEVAL_DOCUMENT",
        "title": "Knowledge base entry",
    }
    assert results[0].embedding == [0.1, 0.2]


@pytest.mark.asyncio
async def test_google_text_submit_uses_native_batch_endpoint_and_headers() -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            json={"name": "batches/1", "metadata": {"state": "JOB_STATE_PENDING"}},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await get_adapter(BatchProvider.GOOGLE, http_client=client).submit(
            built=[
                BuiltRequest(
                    body={
                        "contents": [{"parts": [{"text": "hello"}]}],
                        "generationConfig": {"temperature": 0.2},
                    },
                    custom_id="a",
                    endpoint="/v1beta/models/gemini-test:generateContent",
                )
            ],
            credentials=ProviderCredentials(api_key="secret"),
            endpoint="/v1beta/models/gemini-test:generateContent",
            model_id="gemini-test",
        )

    assert captured[0].url.path == "/v1beta/models/gemini-test:batchGenerateContent"
    assert captured[0].headers["x-goog-api-key"] == "secret"
    sent = json.loads(captured[0].content)
    request = sent["batch"]["input_config"]["requests"]["requests"][0]
    assert request == {
        "metadata": {"key": "a"},
        "request": {
            "contents": [{"parts": [{"text": "hello"}]}],
            "generationConfig": {"temperature": 0.2},
        },
    }


@pytest.mark.asyncio
async def test_google_inline_submit_accepts_payload_at_provider_limit() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={"name": "batches/1", "metadata": {"state": "JOB_STATE_PENDING"}},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await get_adapter(BatchProvider.GOOGLE, http_client=client).submit(
            built=[_google_built_request_for_payload_size(_GOOGLE_INLINE_BATCH_MAX_BYTES)],
            credentials=ProviderCredentials(api_key="secret"),
            endpoint="/v1beta/models/gemini:generateContent",
            model_id="gemini",
        )

    assert len(requests) == 1
    assert len(requests[0].content) == _GOOGLE_INLINE_BATCH_MAX_BYTES


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("limits", "payload_size", "effective_limit"),
    [
        (None, _GOOGLE_INLINE_BATCH_MAX_BYTES + 1, _GOOGLE_INLINE_BATCH_MAX_BYTES),
        (BatchLimits(max_upload_bytes=512), 513, 512),
    ],
)
async def test_google_inline_submit_rejects_oversized_payload_before_request(
    limits: BatchLimits | None, payload_size: int, effective_limit: int
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(500)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(BatchworkError) as caught:
            await get_adapter(BatchProvider.GOOGLE, http_client=client).submit(
                built=[_google_built_request_for_payload_size(payload_size)],
                credentials=ProviderCredentials(api_key="secret"),
                endpoint="/v1beta/models/gemini:generateContent",
                model_id="gemini",
                limits=limits,
            )

    assert str(caught.value) == (
        f"batchwork: batch upload payload is at least {payload_size} bytes, "
        f"exceeding the {effective_limit} byte limit."
    )
    assert requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("outer_field", "inner_field"),
    [
        ("inlinedEmbedContentResponses", "inlinedResponses"),
        ("inlined_embed_content_responses", "inlined_responses"),
    ],
)
async def test_google_embedding_results_use_embedding_destination(
    outer_field: str, inner_field: str
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "name": "batches/1",
                "done": True,
                "metadata": {"state": "JOB_STATE_SUCCEEDED"},
                "dest": {
                    outer_field: {
                        inner_field: [
                            {
                                "metadata": {"key": "embedded"},
                                "response": {"embedding": {"values": [0.3, 0.4]}},
                            }
                        ]
                    }
                },
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        results = [
            item
            async for item in get_adapter(BatchProvider.GOOGLE, http_client=client).results(
                "batches/1", ProviderCredentials(api_key="secret")
            )
        ]

    assert len(results) == 1
    assert results[0].custom_id == "embedded"
    assert results[0].embedding == [0.3, 0.4]


@pytest.mark.asyncio
async def test_google_snapshot_uses_pending_batch_stats_before_inline_results() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "name": "batches/1",
                "metadata": {
                    "state": "JOB_STATE_PENDING",
                    "batchStats": {
                        "requestCount": "4",
                        "successfulRequestCount": "0",
                        "failedRequestCount": "0",
                        "pendingRequestCount": "4",
                    },
                },
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        snapshot = await get_adapter(BatchProvider.GOOGLE, http_client=client).retrieve(
            "batches/1", ProviderCredentials(api_key="secret")
        )

    assert snapshot.request_counts.model_dump(exclude_none=True) == {
        "total": 4,
        "completed": 0,
        "failed": 0,
        "processing": 4,
    }
    assert snapshot.status == "validating"


@pytest.mark.asyncio
async def test_google_snapshot_falls_back_to_completed_inline_results() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "name": "batches/1",
                "done": True,
                "metadata": {"state": "JOB_STATE_SUCCEEDED"},
                "response": {
                    "inlinedResponses": {
                        "inlinedResponses": [
                            {"metadata": {"key": "ok"}, "response": {}},
                            {
                                "metadata": {"key": "bad"},
                                "error": {"code": 400, "message": "bad request"},
                            },
                        ]
                    }
                },
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        snapshot = await get_adapter(BatchProvider.GOOGLE, http_client=client).retrieve(
            "batches/1", ProviderCredentials(api_key="secret")
        )

    assert snapshot.request_counts.model_dump(exclude_none=True) == {
        "total": 2,
        "completed": 1,
        "failed": 1,
    }


@pytest.mark.asyncio
async def test_mistral_job_shape_and_output_results() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/v1/files":
            return httpx.Response(200, json={"id": "file-in"})
        if request.url.path == "/v1/batch/jobs" and request.method == "POST":
            return httpx.Response(
                200,
                json={"id": "job_1", "status": "QUEUED", "total_requests": 1},
            )
        if request.url.path == "/v1/batch/jobs/job_1":
            return httpx.Response(
                200,
                json={
                    "id": "job_1",
                    "status": "SUCCESS",
                    "total_requests": 1,
                    "succeeded_requests": 1,
                    "output_file": "file-out",
                },
            )
        if request.url.path == "/v1/files/file-out/content":
            return httpx.Response(
                200,
                text='{"custom_id":"a","response":{"status_code":200,"body":{"choices":[{"text":"ok"}]}}}\n',
            )
        if request.url.path == "/v1/batch/jobs/job_1/cancel":
            return httpx.Response(200, json={})
        raise AssertionError(f"unexpected request: {request.url}")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        adapter = get_adapter(BatchProvider.MISTRAL, http_client=client)
        await adapter.submit(
            built=[
                BuiltRequest(
                    body={
                        "model": "mistral",
                        "input": ["hello"],
                        "encoding_format": "float",
                    },
                    custom_id="a",
                    endpoint="/v1/embeddings",
                )
            ],
            credentials=ProviderCredentials(api_key="secret"),
            endpoint="/v1/embeddings",
            model_id="mistral",
        )
        results = [
            item async for item in adapter.results("job_1", ProviderCredentials(api_key="secret"))
        ]
        await adapter.cancel("job_1", ProviderCredentials(api_key="secret"))

    assert b'"body":{"input":["hello"],"encoding_format":"float"}' in requests[0].content
    create = json.loads(requests[1].content)
    assert create == {
        "endpoint": "/v1/embeddings",
        "input_files": ["file-in"],
        "model": "mistral",
    }
    assert results[0].text == "ok"
    assert requests[-1].url.path == "/v1/batch/jobs/job_1/cancel"


@pytest.mark.asyncio
async def test_xai_paginated_image_results_and_cancel() -> None:
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(str(request.url))
        if request.url.path.endswith(":cancel"):
            return httpx.Response(200, json={})
        token = request.url.params.get("pagination_token")
        if token == "next":
            return httpx.Response(
                200,
                json={"results": [{"batch_request_id": "b", "error_message": "failed"}]},
            )
        return httpx.Response(
            200,
            json={
                "pagination_token": "next",
                "results": [
                    {
                        "batch_request_id": "a",
                        "batch_result": {
                            "response": {
                                "image_response": {
                                    "data": [
                                        {
                                            "base64": "aW1hZ2U=",
                                            "url": "https://example.test/image.png",
                                        }
                                    ]
                                }
                            }
                        },
                    }
                ],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        adapter = get_adapter(BatchProvider.XAI, http_client=client)
        credentials = ProviderCredentials(api_key="secret")
        results = [item async for item in adapter.results("batch_1", credentials)]
        await adapter.cancel("batch_1", credentials)

    assert results[0].images is not None and results[0].images[0].data == "aW1hZ2U="
    assert results[0].images[0].url == "https://example.test/image.png"
    assert results[1].error is not None and results[1].error.message == "failed"
    assert any("pagination_token=next" in path for path in paths)
    assert paths[-1].endswith("/batches/batch_1:cancel")


@pytest.mark.asyncio
async def test_xai_text_submit_uploads_responses_jsonl_then_creates_batch() -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        if request.url.path == "/v1/files":
            return httpx.Response(200, json={"id": "file-in"})
        if request.url.path == "/v1/batches":
            return httpx.Response(
                200,
                json={
                    "batch_id": "batch_1",
                    "state": {"num_requests": 1, "num_pending": 1},
                },
            )
        raise AssertionError(f"unexpected request: {request.url}")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        snapshot = await get_adapter(BatchProvider.XAI, http_client=client).submit(
            built=[
                BuiltRequest(
                    body={"model": "grok-test", "input": "hello"},
                    custom_id="a",
                    endpoint="/v1/responses",
                )
            ],
            credentials=ProviderCredentials(api_key="secret"),
            endpoint="/v1/responses",
            model_id="grok-test",
        )

    assert snapshot.id == "batch_1"
    assert [request.url.path for request in captured] == ["/v1/files", "/v1/batches"]
    assert all(request.headers["authorization"] == "Bearer secret" for request in captured)
    assert b'"url":"/v1/responses"' in captured[0].content
    assert b'"custom_id":"a"' in captured[0].content
    assert json.loads(captured[1].content) == {"input_file_id": "file-in", "name": "batchwork"}


@pytest.mark.asyncio
async def test_xai_snapshot_maps_lifecycle_timestamps() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "batch_id": "batch_1",
                "create_time": 1_700_000_000,
                "finish_time": 1_700_000_100,
                "expire_time": 1_700_003_600,
                "state": {
                    "num_requests": 1,
                    "num_pending": 0,
                    "num_success": 1,
                    "num_error": 0,
                    "num_cancelled": 0,
                },
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        snapshot = await get_adapter(BatchProvider.XAI, http_client=client).retrieve(
            "batch_1", ProviderCredentials(api_key="secret")
        )
    assert snapshot.created_at is not None
    assert snapshot.completed_at is not None
    assert snapshot.expires_at is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", [BatchProvider.ANTHROPIC, BatchProvider.GOOGLE])
async def test_bodyless_cancel_matches_provider_wire_contract(
    provider: BatchProvider,
) -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await get_adapter(provider, http_client=client).cancel(
            "batch_1" if provider is BatchProvider.ANTHROPIC else "batches/batch_1",
            ProviderCredentials(api_key="secret"),
        )
    assert captured[0].method == "POST"
    assert captured[0].content == b""
