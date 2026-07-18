from __future__ import annotations

import httpx
import pytest

from batchwork.errors import BatchworkError
from batchwork.providers import get_adapter
from batchwork.types import BatchProvider, ProviderCredentials


@pytest.mark.parametrize(
    ("base_url", "results_url"),
    [
        (
            "https://api.anthropic.com",
            "https://API.ANTHROPIC.COM/results/batch_1",
        ),
        (
            "https://api.anthropic.com",
            "https://api.anthropic.com:443/results/batch_1",
        ),
        (
            "https://API.ANTHROPIC.COM:443",
            "https://api.anthropic.com/results/batch_1",
        ),
    ],
)
@pytest.mark.asyncio
async def test_anthropic_accepts_equivalent_results_origins(
    base_url: str, results_url: str
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/v1/messages/batches/batch_1":
            return httpx.Response(
                200,
                json={
                    "id": "batch_1",
                    "processing_status": "ended",
                    "results_url": results_url,
                    "request_counts": {"succeeded": 1},
                },
            )
        if request.url.path == "/results/batch_1":
            return httpx.Response(
                200,
                text=(
                    '{"custom_id":"request_1","result":{"type":"succeeded",'
                    '"message":{"content":[{"type":"text","text":"done"}]}}}\n'
                ),
            )
        raise AssertionError(f"unexpected request: {request.method} {request.url}")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        adapter = get_adapter(BatchProvider.ANTHROPIC, http_client=client)
        results = [
            result
            async for result in adapter.results(
                "batch_1",
                ProviderCredentials(api_key="secret", base_url=base_url),
            )
        ]

    assert [result.text for result in results] == ["done"]
    assert [request.url.path for request in requests] == [
        "/v1/messages/batches/batch_1",
        "/results/batch_1",
    ]


@pytest.mark.parametrize(
    "results_url",
    [
        "http://api.anthropic.com/results/batch_1",
        "https://other.anthropic.com/results/batch_1",
        "https://api.anthropic.com:444/results/batch_1",
        "https://user@api.anthropic.com/results/batch_1",
        "https://@api.anthropic.com/results/batch_1",
    ],
)
@pytest.mark.asyncio
async def test_anthropic_rejects_distinct_or_credentialed_results_origins(
    results_url: str,
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/v1/messages/batches/batch_1":
            return httpx.Response(
                200,
                json={
                    "id": "batch_1",
                    "processing_status": "ended",
                    "results_url": results_url,
                    "request_counts": {},
                },
            )
        raise AssertionError("invalid-origin request must not be made")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        adapter = get_adapter(BatchProvider.ANTHROPIC, http_client=client)
        with pytest.raises(BatchworkError, match="must match"):
            _ = [
                result
                async for result in adapter.results(
                    "batch_1", ProviderCredentials(api_key="secret")
                )
            ]

    assert [request.url.path for request in requests] == ["/v1/messages/batches/batch_1"]
