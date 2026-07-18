from __future__ import annotations

import asyncio

import httpx
import pytest

from batchwork.providers import get_adapter
from batchwork.types import BatchProvider, ProviderCredentials


class ControlledResultStream(httpx.AsyncByteStream):
    def __init__(self) -> None:
        self.release_second = asyncio.Event()
        self.closed = asyncio.Event()

    async def __aiter__(self):
        yield (
            b'{"custom_id":"first","result":{"type":"succeeded","message":'
            b'{"content":[{"type":"text","text":"one"}]}}}\n'
        )
        await self.release_second.wait()
        yield (
            b'{"custom_id":"second","result":{"type":"succeeded","message":'
            b'{"content":[{"type":"text","text":"two"}]}}}\n'
        )

    async def aclose(self) -> None:
        self.closed.set()


@pytest.mark.asyncio
async def test_anthropic_results_stream_first_line_and_close_on_consumer_stop() -> None:
    stream = ControlledResultStream()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/messages/batches/batch_1":
            return httpx.Response(
                200,
                json={
                    "id": "batch_1",
                    "processing_status": "ended",
                    "results_url": "https://api.anthropic.com/results/batch_1",
                    "request_counts": {"succeeded": 2},
                },
            )
        if request.url.path == "/results/batch_1":
            return httpx.Response(200, stream=stream)
        raise AssertionError(f"unexpected request: {request.method} {request.url}")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        adapter = get_adapter(BatchProvider.ANTHROPIC, http_client=client)
        results = adapter.results("batch_1", ProviderCredentials(api_key="secret"))

        first = await asyncio.wait_for(anext(results), timeout=1)
        assert first.custom_id == "first"
        assert first.text == "one"
        assert not stream.release_second.is_set()

        await results.aclose()
        await asyncio.wait_for(stream.closed.wait(), timeout=1)
