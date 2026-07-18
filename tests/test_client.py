from collections.abc import AsyncIterator, Mapping, Sequence

import httpx
import pytest

import batchwork.client as client_module
from batchwork import Batchwork
from batchwork.body import BuiltRequest
from batchwork.errors import BatchClosedError, BatchworkError
from batchwork.media import ResolvedMedia
from batchwork.types import (
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
