"""Minimal batchwork-ai package demo for OpenAI Build Week judges.

Submits a tiny 4-request batch to OpenAI's Batch API, polls until it
completes, and prints the normalized results. Token cost is a fraction
of a cent. Provider processing usually finishes in a few minutes but is
allowed up to 24 hours; interrupting this script never cancels the
remote job.
"""

from __future__ import annotations

import asyncio

from batchwork import BatchDefaults, BatchRequest, BatchSnapshot, Batchwork

PROMPTS = {
    "capital": "Reply with one word: the capital of France.",
    "gold": "Reply with one word: the chemical symbol for gold.",
    "why-batch": "Reply with one short sentence: why do batch APIs cost less?",
    "why-unified": "Reply with one short sentence: what does a unified batch client save you?",
}


async def report(snapshot: BatchSnapshot) -> None:
    counts = snapshot.request_counts
    print(
        f"status={snapshot.status.value} completed={counts.completed}/{counts.total}",
        flush=True,
    )


async def main() -> None:
    requests = [
        BatchRequest(custom_id=custom_id, prompt=prompt)
        for custom_id, prompt in PROMPTS.items()
    ]
    defaults = BatchDefaults(
        max_output_tokens=64,
        provider_options={"openai": {"reasoningEffort": "none"}},
    )

    async with Batchwork() as client:
        job = await client.batch(
            model="openai/gpt-5.4-nano",
            requests=requests,
            defaults=defaults,
        )
        print(f"submitted provider_job={job.id}", flush=True)

        await job.wait(poll_interval=15, timeout=1800, on_poll=report)

        for result in await job.collect():
            print(f"{result.custom_id}: {result.text}")


if __name__ == "__main__":
    asyncio.run(main())
