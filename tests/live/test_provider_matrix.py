from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

import pytest

from batchwork import (
    BatchEmbeddingRequest,
    BatchImageRequest,
    BatchRequest,
    BatchResultStatus,
    BatchStatus,
    Batchwork,
)

Modality = Literal["text", "embeddings", "images"]


@dataclass(frozen=True, slots=True)
class LiveCase:
    provider: str
    modality: Modality

    @property
    def model_env(self) -> str:
        return f"BATCHWORK_LIVE_{self.provider.upper()}_{self.modality.upper()}_MODEL"


CASES = [
    *(
        LiveCase(provider, "text")
        for provider in ("anthropic", "google", "groq", "mistral", "openai", "together", "xai")
    ),
    *(LiveCase(provider, "embeddings") for provider in ("google", "mistral", "openai")),
    *(LiveCase(provider, "images") for provider in ("google", "openai", "xai")),
]

pytestmark = pytest.mark.live


@pytest.mark.parametrize("case", CASES, ids=lambda case: f"{case.provider}-{case.modality}")
async def test_live_provider_modality(case: LiveCase) -> None:
    if os.getenv("BATCHWORK_RUN_LIVE") != "1":
        pytest.skip("set BATCHWORK_RUN_LIVE=1 to enable provider acceptance tests")
    model_id = os.getenv(case.model_env)
    if not model_id:
        pytest.skip(f"set {case.model_env}")
    model = model_id if "/" in model_id else f"{case.provider}/{model_id}"

    async with Batchwork() as client:
        if case.modality == "text":
            job = await client.batch(
                model=model,
                requests=[BatchRequest(custom_id="live-text", prompt="Reply with one short word.")],
            )
        elif case.modality == "embeddings":
            job = await client.batch_embeddings(
                model=model,
                requests=[BatchEmbeddingRequest(custom_id="live-embedding", value="hello")],
            )
        else:
            job = await client.batch_images(
                model=model,
                requests=[
                    BatchImageRequest(custom_id="live-image", prompt="A solid red square on white.")
                ],
            )

        snapshot = await job.wait(
            poll_interval=float(os.getenv("BATCHWORK_LIVE_POLL_SECONDS", "15")),
            timeout=float(os.getenv("BATCHWORK_LIVE_TIMEOUT_SECONDS", "1800")),
        )
        assert snapshot.status is BatchStatus.COMPLETED, snapshot.raw
        results = await job.collect()
        assert len(results) == 1
        assert results[0].custom_id.startswith("live-")
        assert results[0].status is BatchResultStatus.SUCCEEDED, results[0].response
        if case.modality == "text":
            assert results[0].text
        elif case.modality == "embeddings":
            assert results[0].embedding
        else:
            assert results[0].images
