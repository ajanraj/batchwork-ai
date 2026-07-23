from collections.abc import AsyncIterator, Mapping, Sequence

import httpx
import pytest
from pydantic import ValidationError

import batchwork.client as client_module
from batchwork import Batchwork
from batchwork.body import BuiltRequest
from batchwork.errors import BatchClosedError, BatchworkError, MediaResolutionError
from batchwork.media import ResolvedMedia
from batchwork.types import (
    BatchDefaults,
    BatchEmbeddingDefaults,
    BatchEmbeddingRequest,
    BatchImageRequest,
    BatchLimits,
    BatchProvider,
    BatchRef,
    BatchRequest,
    BatchRequestCounts,
    BatchResult,
    BatchResultStatus,
    BatchSnapshot,
    BatchStatus,
    FilePart,
    ImagePart,
    ProviderCredentials,
    UserMessage,
)


class FakeAdapter:
    id = BatchProvider.OPENAI

    def __init__(self) -> None:
        self.credentials: ProviderCredentials | None = None
        self.built: Sequence[BuiltRequest] = []
        self.retrieve_calls = 0
        self.result_calls = 0

    async def submit(
        self,
        *,
        built: Sequence[BuiltRequest],
        credentials: ProviderCredentials,
        endpoint: str,
        model_id: str,
        metadata: Mapping[str, str] | None = None,
        limits=None,
    ) -> BatchSnapshot:
        self.credentials = credentials
        self.built = built
        return BatchSnapshot(
            id="batch_1",
            provider=BatchProvider.OPENAI,
            status=BatchStatus.IN_PROGRESS,
            request_counts=BatchRequestCounts(total=len(built), completed=0, failed=0),
        )

    async def retrieve(self, id: str, credentials: ProviderCredentials) -> BatchSnapshot:
        self.retrieve_calls += 1
        return BatchSnapshot(
            id=id,
            provider=BatchProvider.OPENAI,
            status=BatchStatus.COMPLETED,
            request_counts=BatchRequestCounts(total=1, completed=1, failed=0),
        )

    async def results(
        self, id: str, credentials: ProviderCredentials
    ) -> AsyncIterator[BatchResult]:
        self.result_calls += 1
        yield BatchResult(custom_id="a", status=BatchResultStatus.SUCCEEDED, text="ok")

    async def cancel(self, id: str, credentials: ProviderCredentials) -> None:
        return None


@pytest.mark.asyncio
async def test_client_routes_submission_and_merges_credentials(monkeypatch) -> None:
    adapter = FakeAdapter()
    monkeypatch.setattr(client_module, "_get_adapter", lambda provider, client: adapter)
    monkeypatch.setenv("OPENAI_API_KEY", "environment")
    client = Batchwork(
        credentials={
            "openai": ProviderCredentials(
                api_key="configured", headers={"x-base": "1", "x-replace": "base"}
            )
        }
    )
    job = await client.batch(
        model="openai/gpt-5",
        requests=[BatchRequest(prompt="hello")],
        api_key="per-call",
        headers={"x-replace": "call"},
    )
    assert job.id == "batch_1"
    assert adapter.credentials is not None
    assert adapter.credentials.api_key == "per-call"
    assert adapter.credentials.headers == {"x-base": "1", "x-replace": "call"}
    await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("batch_request", "defaults", "message"),
    (
        (
            BatchRequest(prompt="hello"),
            BatchDefaults(top_k=7),
            'canonical setting "top_k" is unsupported',
        ),
        (
            BatchRequest(prompt="hello", top_k=7),
            None,
            'canonical setting "top_k" is unsupported',
        ),
        (
            BatchRequest(prompt="hello"),
            BatchDefaults(provider_options={"openai": {"unknownOption": True}}),
            'provider option "unknownOption" is unsupported',
        ),
        (
            BatchRequest(prompt="hello"),
            BatchDefaults(
                max_output_tokens=32,
                provider_options={"openai": {"maxCompletionTokens": 64}},
            ),
            "max_output_tokens conflicts with OpenAI provider option maxCompletionTokens",
        ),
    ),
)
async def test_client_strictly_validates_text_requests_before_submission(
    monkeypatch,
    batch_request: BatchRequest,
    defaults: BatchDefaults | None,
    message: str,
) -> None:
    adapter = FakeAdapter()
    monkeypatch.setattr(client_module, "_get_adapter", lambda provider, client: adapter)
    client = Batchwork()

    with pytest.raises(BatchworkError, match=message):
        await client.batch(
            model="openai/gpt-4.1",
            requests=[batch_request],
            defaults=defaults,
        )

    assert adapter.built == []
    await client.aclose()


@pytest.mark.asyncio
async def test_client_strictly_validates_embedding_options_before_submission(monkeypatch) -> None:
    adapter = FakeAdapter()
    monkeypatch.setattr(client_module, "_get_adapter", lambda provider, client: adapter)
    client = Batchwork()

    with pytest.raises(BatchworkError, match='provider option "unknownOption"'):
        await client.batch_embeddings(
            model="mistral/mistral-embed",
            requests=[BatchEmbeddingRequest(value="hello")],
            defaults=BatchEmbeddingDefaults(provider_options={"mistral": {"unknownOption": True}}),
        )

    assert adapter.built == []
    await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("modality", ["text", "embeddings", "images"])
async def test_client_rejects_unsupported_batch_metadata_before_adapter_lookup(
    monkeypatch, modality: str
) -> None:
    def unexpected_adapter_lookup(provider, client):
        raise AssertionError("metadata validation must precede adapter lookup")

    monkeypatch.setattr(client_module, "_get_adapter", unexpected_adapter_lookup)
    client = Batchwork()

    with pytest.raises(
        BatchworkError,
        match='provider "google" does not support submission-level batch metadata',
    ):
        if modality == "text":
            await client.batch(
                model="google/gemini-2.5-flash",
                requests=[BatchRequest(prompt="hello")],
                metadata={"purpose": "test"},
            )
        elif modality == "embeddings":
            await client.batch_embeddings(
                model="google/gemini-embedding-001",
                requests=[BatchEmbeddingRequest(value="hello")],
                metadata={"purpose": "test"},
            )
        else:
            await client.batch_images(
                model="google/gemini-3-pro-image-preview",
                requests=[BatchImageRequest(prompt="hello")],
                metadata={"purpose": "test"},
            )

    await client.aclose()


@pytest.mark.asyncio
async def test_client_routes_openai_image_generation(monkeypatch) -> None:
    adapter = FakeAdapter()
    monkeypatch.setattr(client_module, "_get_adapter", lambda provider, client: adapter)
    client = Batchwork()

    job = await client.batch_images(
        model="openai/gpt-image-2",
        requests=[
            BatchImageRequest(
                custom_id="image-1",
                prompt="A red bicycle.",
                size="1536x1024",
            )
        ],
        api_key="secret",
    )

    assert job.provider is BatchProvider.OPENAI
    assert len(adapter.built) == 1
    assert adapter.built[0].endpoint == "/v1/images/generations"
    assert adapter.built[0].body == {
        "model": "gpt-image-2",
        "prompt": "A red bicycle.",
        "n": 1,
        "size": "1536x1024",
    }
    await client.aclose()


@pytest.mark.asyncio
async def test_injected_http_client_remains_caller_owned(monkeypatch) -> None:
    adapter = FakeAdapter()
    monkeypatch.setattr(client_module, "_get_adapter", lambda provider, client: adapter)
    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200))
    )
    client = Batchwork(http_client=http_client)
    await client.aclose()
    assert not http_client.is_closed
    with pytest.raises(BatchClosedError):
        await client.batch(model="openai/gpt-5", requests=[BatchRequest(prompt="hello")])
    await http_client.aclose()


@pytest.mark.asyncio
async def test_client_preserves_zero_timeout() -> None:
    client = Batchwork(timeout=0)

    assert client._http_client.timeout == httpx.Timeout(0)
    await client.aclose()


@pytest.mark.asyncio
async def test_empty_requests_fail_before_adapter_lookup() -> None:
    client = Batchwork()
    with pytest.raises(BatchworkError, match="must not be empty"):
        await client.batch(model="openai/gpt-5", requests=[])
    await client.aclose()


@pytest.mark.asyncio
async def test_environment_base_url_is_rejected_before_network_dispatch(monkeypatch) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200)

    monkeypatch.setenv("OPENAI_BASE_URL", "http://169.254.169.254/latest")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = Batchwork(http_client=http_client)
        with pytest.raises(ValidationError, match="absolute HTTPS"):
            await client.batch(model="openai/gpt-5", requests=[BatchRequest(prompt="hello")])
        await client.aclose()

    assert requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "arguments",
    (
        {
            "api_key": "super-secret",
            "base_url": "https://user:super-secret@example.com/v1",
        },
        {
            "credentials": {
                "api_key": "super-secret",
                "base_url": "https://user:super-secret@example.com/v1",
            }
        },
    ),
)
async def test_per_call_base_url_is_rejected_before_credential_dispatch(
    arguments: dict[str, object],
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = Batchwork(http_client=http_client)
        with pytest.raises(ValidationError, match="absolute HTTPS") as caught:
            await client.batch(
                model="openai/gpt-5",
                requests=[BatchRequest(prompt="hello")],
                **arguments,
            )
        await client.aclose()

    for rendered in (
        str(caught.value),
        repr(caught.value),
        repr(caught.value.errors()),
        caught.value.json(),
    ):
        assert "super-secret" not in rendered
        assert "https://user:super-secret@example.com/v1" not in rendered
    assert caught.value.errors()[0]["input"] == "<redacted>"
    assert requests == []


def test_constructor_credentials_reject_unsafe_base_url() -> None:
    with pytest.raises(ValidationError, match="absolute HTTPS"):
        Batchwork(
            credentials={
                "openai": {
                    "api_key": "super-secret",
                    "base_url": "http://not-loopback.example/v1",
                }
            }
        )


@pytest.mark.asyncio
async def test_batch_ref_lifecycle_rejects_unsafe_base_url_before_dispatch() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200)

    ref = BatchRef.model_construct(
        id="batch_1",
        provider=BatchProvider.OPENAI,
        model=None,
        api_key="super-secret",
        base_url="http://not-loopback.example/v1",
        headers={},
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = Batchwork(http_client=http_client)
        with pytest.raises(ValidationError, match="absolute HTTPS"):
            await client.get_batch(ref)
        with pytest.raises(ValidationError, match="absolute HTTPS"):
            _ = [item async for item in client.get_batch_results(ref)]
        with pytest.raises(ValidationError, match="absolute HTTPS"):
            await client.cancel_batch(ref)
        await client.aclose()

    assert requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "media_url", ["http://example.com/image.png", "https://example.com/image.png"]
)
async def test_client_preserves_provider_supported_remote_media(
    monkeypatch, media_url: str
) -> None:
    adapter = FakeAdapter()
    monkeypatch.setattr(client_module, "_get_adapter", lambda provider, client: adapter)

    class Resolver:
        async def resolve(self, source, *, media_type=None, max_bytes):
            raise AssertionError(f"unexpected media download: {source}, {media_type}, {max_bytes}")

    client = Batchwork(media_resolver=Resolver())
    await client.batch(
        model="openai/gpt-4.1",
        requests=[BatchRequest(messages=[UserMessage(content=[ImagePart(image=media_url)])])],
    )
    assert adapter.built
    assert media_url in str(adapter.built[0].body)
    await client.aclose()


@pytest.mark.asyncio
async def test_client_enforces_aggregate_decoded_media_limit(monkeypatch) -> None:
    adapter = FakeAdapter()
    monkeypatch.setattr(client_module, "_get_adapter", lambda provider, client: adapter)

    class Resolver:
        async def resolve(self, source, *, media_type=None, max_bytes):
            return ResolvedMedia(b"123456", "image/png")

    client = Batchwork(media_resolver=Resolver())
    with pytest.raises(BatchworkError, match="aggregate decoded media"):
        await client.batch(
            model="openai/gpt-4.1",
            requests=[
                BatchRequest(
                    messages=[
                        UserMessage(
                            content=[
                                ImagePart(image=b"first"),
                                ImagePart(image=b"second"),
                            ]
                        )
                    ]
                )
            ],
            limits=BatchLimits(max_request_bytes=10, max_upload_bytes=10),
        )
    assert adapter.built == []
    await client.aclose()


@pytest.mark.asyncio
async def test_client_enforces_fixed_media_limits_when_batch_limits_are_higher(
    monkeypatch,
) -> None:
    adapter = FakeAdapter()
    monkeypatch.setattr(client_module, "_get_adapter", lambda provider, client: adapter)
    monkeypatch.setattr(client_module, "MAX_DECODED_MEDIA_BYTES", 5)
    monkeypatch.setattr(client_module, "MAX_AGGREGATE_MEDIA_BYTES", 10)
    observed_limits: list[int] = []

    class Resolver:
        async def resolve(self, source, *, media_type=None, max_bytes):
            observed_limits.append(max_bytes)
            return ResolvedMedia(b"123456", "text/plain")

    client = Batchwork(media_resolver=Resolver())
    with pytest.raises(BatchworkError, match="aggregate decoded media"):
        await client.batch(
            model="together/model",
            requests=[
                BatchRequest(
                    messages=[UserMessage(content=[FilePart(data="eA==", media_type="text/plain")])]
                ),
                BatchRequest(
                    messages=[UserMessage(content=[FilePart(data="eQ==", media_type="text/plain")])]
                ),
            ],
            limits=BatchLimits(max_request_bytes=100, max_upload_bytes=100),
        )

    assert observed_limits == [5, 5]
    assert adapter.built == []
    await client.aclose()


@pytest.mark.asyncio
async def test_client_completes_local_media_preflight_before_remote_fetch(monkeypatch) -> None:
    adapter = FakeAdapter()
    monkeypatch.setattr(client_module, "_get_adapter", lambda provider, client: adapter)
    resolved_sources: list[object] = []

    class Resolver:
        async def resolve(self, source, *, media_type=None, max_bytes):
            resolved_sources.append(source)
            if str(source).startswith("https://"):
                raise AssertionError("remote media fetched before local preflight completed")
            raise MediaResolutionError("batchwork: invalid local media")

    client = Batchwork(media_resolver=Resolver())
    with pytest.raises(MediaResolutionError, match="invalid local media"):
        await client.batch(
            model="together/model",
            requests=[
                BatchRequest(
                    messages=[
                        UserMessage(
                            content=[
                                FilePart(
                                    data="https://example.com/file.txt", media_type="text/plain"
                                )
                            ]
                        )
                    ]
                ),
                BatchRequest(
                    messages=[
                        UserMessage(content=[FilePart(data="invalid", media_type="text/plain")])
                    ]
                ),
            ],
        )

    assert resolved_sources == ["invalid"]
    assert adapter.built == []
    await client.aclose()


@pytest.mark.asyncio
async def test_client_validates_locally_expanded_body_before_remote_fetch(monkeypatch) -> None:
    adapter = FakeAdapter()
    monkeypatch.setattr(client_module, "_get_adapter", lambda provider, client: adapter)
    resolved_sources: list[object] = []

    class Resolver:
        async def resolve(self, source, *, media_type=None, max_bytes):
            resolved_sources.append(source)
            if str(source).lower().startswith("https://"):
                raise AssertionError("remote media fetched before local body validation")
            return ResolvedMedia(b"x" * 80, "text/plain")

    client = Batchwork(media_resolver=Resolver())
    with pytest.raises(BatchworkError, match="request-1"):
        await client.batch(
            model="together/model",
            requests=[
                BatchRequest(
                    messages=[
                        UserMessage(
                            content=[
                                FilePart(
                                    data="https://example.com/file.txt", media_type="text/plain"
                                )
                            ]
                        )
                    ]
                ),
                BatchRequest(
                    messages=[UserMessage(content=[FilePart(data="eA==", media_type="text/plain")])]
                ),
            ],
            limits=BatchLimits(max_request_bytes=100, max_upload_bytes=200),
        )

    assert resolved_sources == ["eA=="]
    assert adapter.built == []
    await client.aclose()


@pytest.mark.asyncio
async def test_client_shares_decoded_media_budget_across_local_and_remote_phases(
    monkeypatch,
) -> None:
    adapter = FakeAdapter()
    monkeypatch.setattr(client_module, "_get_adapter", lambda provider, client: adapter)

    class Resolver:
        async def resolve(self, source, *, media_type=None, max_bytes):
            return ResolvedMedia(b"123456", "text/plain")

    client = Batchwork(media_resolver=Resolver())
    with pytest.raises(BatchworkError, match="aggregate decoded media"):
        await client.batch(
            model="together/model",
            requests=[
                BatchRequest(
                    messages=[UserMessage(content=[FilePart(data="eA==", media_type="text/plain")])]
                ),
                BatchRequest(
                    messages=[
                        UserMessage(
                            content=[
                                FilePart(
                                    data="https://example.com/file.txt", media_type="text/plain"
                                )
                            ]
                        )
                    ]
                ),
            ],
            limits=BatchLimits(max_request_bytes=100, max_upload_bytes=10),
        )

    assert adapter.built == []
    await client.aclose()


@pytest.mark.asyncio
async def test_client_preserves_xai_responses_file_urls_by_default(monkeypatch) -> None:
    adapter = FakeAdapter()
    monkeypatch.setattr(client_module, "_get_adapter", lambda provider, client: adapter)

    class Resolver:
        async def resolve(self, source, *, media_type=None, max_bytes):
            raise AssertionError(f"unexpected media download: {source}, {media_type}, {max_bytes}")

    client = Batchwork(media_resolver=Resolver())
    await client.batch(
        model="xai/grok-4",
        requests=[
            BatchRequest(
                messages=[
                    UserMessage(
                        content=[
                            FilePart(
                                data={"type": "url", "url": "https://example.com/file.pdf"},
                                media_type="application/pdf",
                            )
                        ]
                    )
                ]
            )
        ],
    )
    assert "https://example.com/file.pdf" in str(adapter.built[0].body)
    await client.aclose()


@pytest.mark.asyncio
async def test_client_downloads_unsupported_provider_media_url(monkeypatch) -> None:
    adapter = FakeAdapter()
    monkeypatch.setattr(client_module, "_get_adapter", lambda provider, client: adapter)

    class Resolver:
        async def resolve(self, source, *, media_type=None, max_bytes):
            assert str(source) == "https://example.com/image.png"
            assert max_bytes == 20 * 1024 * 1024
            return ResolvedMedia(b"image bytes", "image/png")

    client = Batchwork(media_resolver=Resolver())
    await client.batch(
        model="google/gemini-2.5-pro",
        requests=[
            BatchRequest(
                messages=[UserMessage(content=[ImagePart(image="https://example.com/image.png")])]
            )
        ],
    )
    assert "aW1hZ2UgYnl0ZXM=" in str(adapter.built[0].body)
    await client.aclose()


@pytest.mark.asyncio
async def test_request_limit_fails_before_media_resolution(monkeypatch) -> None:
    adapter = FakeAdapter()
    monkeypatch.setattr(client_module, "_get_adapter", lambda provider, client: adapter)

    class Resolver:
        def __init__(self) -> None:
            self.calls = 0

        async def resolve(self, source, *, media_type=None, max_bytes):
            self.calls += 1
            return ResolvedMedia(b"image bytes", "image/png")

    resolver = Resolver()
    client = Batchwork(media_resolver=resolver)
    requests = [
        BatchRequest(
            messages=[UserMessage(content=[ImagePart(image="https://example.com/image.png")])]
        ),
        BatchRequest(
            messages=[UserMessage(content=[ImagePart(image="https://example.com/image.png")])]
        ),
    ]

    with pytest.raises(BatchworkError, match="exceeds the 1 request limit"):
        await client.batch(
            model="google/gemini-2.5-pro",
            requests=requests,
            limits=BatchLimits(max_requests=1),
        )

    assert resolver.calls == 0
    assert adapter.built == []
    await client.aclose()


@pytest.mark.asyncio
async def test_get_batch_results_delegates_without_redundant_retrieve(monkeypatch) -> None:
    adapter = FakeAdapter()
    monkeypatch.setattr(client_module, "_get_adapter", lambda provider, client: adapter)
    client = Batchwork()
    results = [
        item
        async for item in client.get_batch_results(
            BatchRef(id="batch_1", provider=BatchProvider.OPENAI, api_key="secret")
        )
    ]
    assert [result.text for result in results] == ["ok"]
    assert adapter.result_calls == 1
    assert adapter.retrieve_calls == 0
    await client.aclose()
