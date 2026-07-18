# Batchwork

Unified async batch API for OpenAI, Anthropic, Google Gemini, Groq, Mistral, Together AI, and xAI. Access provider-native batch pricing—often up to 50% lower than standard synchronous requests—through one typed interface while Batchwork handles provider-specific serialization, submission, polling, and result normalization. Pricing and eligibility remain provider- and model-specific. Built for Python by Ajanraj.

- One typed API for provider-native batch jobs
- Text, embedding, and image workloads
- Normalized jobs, results, usage, and errors
- Messages, tools, structured content, and remote media
- Polling, persistent stores, and signed webhooks
- No provider SDK or JavaScript runtime dependencies

📖 **Full documentation: [batchwork.ajanraj.com](https://batchwork.ajanraj.com)**

## Installation

Batchwork requires Python 3.11 or newer.

With uv:

```bash
uv add batchwork-ai
```

With pip:

```bash
pip install batchwork-ai
```

Then configure a provider credential:

```bash
export OPENAI_API_KEY="..."
```

See [Configuration](https://batchwork.ajanraj.com/docs/configuration) for every provider credential, endpoint override, model format, and batch limit.

## Quickstart

```python
import asyncio

from batchwork import BatchRequest, Batchwork


async def main() -> None:
    async with Batchwork() as client:
        job = await client.batch(
            model="openai/gpt-5.6-sol",
            requests=[BatchRequest(custom_id="hello", prompt="Say hello")],
        )
        await job.wait(timeout=3600)

        for result in await job.collect():
            print(result.custom_id, result.text)


asyncio.run(main())
```

Submitting returns a `BatchJob` immediately. Provider processing is asynchronous and may take minutes or, depending on the provider and workload, up to 24 hours.

Models use `provider/model` form. Results are correlated with requests through `custom_id`; provider output order is not guaranteed.

## Providers

Output workloads supported by Batchwork:

| Provider      | Text | Embeddings | Image generation |
| ------------- | ---- | ---------- | ---------------- |
| OpenAI        | Yes  | Yes        | Yes              |
| Anthropic     | Yes  | No         | No               |
| Google Gemini | Yes  | Yes        | Yes              |
| Groq          | Yes  | No         | No               |
| Mistral       | Yes  | Yes        | No               |
| Together AI   | Yes  | No         | No               |
| xAI           | Yes  | No         | Yes              |

Image, PDF, text-file, and audio inputs for text requests vary separately from output modalities. Provider APIs may also impose model-specific limits. See the [provider overview](https://batchwork.ajanraj.com/docs/providers) for input support, submission transport, credentials, and restrictions.

## Optional Redis store

Install the Upstash Redis integration for persistent polling state:

```bash
uv add "batchwork-ai[redis]"
# or
pip install "batchwork-ai[redis]"
```

The base package does not import or require `upstash-redis`.

## Documentation

- [Installation](https://batchwork.ajanraj.com/docs/installation)
- [Configuration](https://batchwork.ajanraj.com/docs/configuration)
- [Jobs](https://batchwork.ajanraj.com/docs/guides/jobs)
- [Results](https://batchwork.ajanraj.com/docs/guides/results)
- [Text, embeddings, and images](https://batchwork.ajanraj.com/docs/modalities/text)
- [Provider overview](https://batchwork.ajanraj.com/docs/providers)
- [Polling and webhooks](https://batchwork.ajanraj.com/docs/guides/server)
- [Stores](https://batchwork.ajanraj.com/docs/guides/stores)
- [Security](https://batchwork.ajanraj.com/docs/guides/security)
- [Examples](https://batchwork.ajanraj.com/docs/examples)
- [Public API](https://batchwork.ajanraj.com/docs/api)
- [FAQ](https://batchwork.ajanraj.com/docs/faq)

## License

[MIT](https://opensource.org/licenses/MIT) © [Ajanraj](https://github.com/ajanraj)

## Acknowledgements

Inspired by [Hayden Bleasel's Batchwork](https://github.com/haydenbleasel/batchwork).
