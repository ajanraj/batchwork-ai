"""Provider adapter registry."""

from __future__ import annotations

import httpx

from batchwork.errors import UnsupportedProviderError
from batchwork.types import BatchProvider

from .adapter import BatchAdapter
from .anthropic import AnthropicAdapter
from .google import GoogleAdapter
from .mistral import MistralAdapter
from .openai_compatible import groq_adapter, openai_adapter, together_adapter
from .xai import XAIAdapter


def get_adapter(
    provider: BatchProvider | str,
    *,
    http_client: httpx.AsyncClient | None = None,
) -> BatchAdapter:
    """Create a provider adapter bound to the caller-owned HTTP client."""

    try:
        resolved = provider if isinstance(provider, BatchProvider) else BatchProvider(provider)
    except ValueError as error:
        raise UnsupportedProviderError(str(provider)) from error
    if resolved is BatchProvider.OPENAI:
        return openai_adapter(http_client)
    if resolved is BatchProvider.ANTHROPIC:
        return AnthropicAdapter(http_client)
    if resolved is BatchProvider.GOOGLE:
        return GoogleAdapter(http_client)
    if resolved is BatchProvider.GROQ:
        return groq_adapter(http_client)
    if resolved is BatchProvider.MISTRAL:
        return MistralAdapter(http_client)
    if resolved is BatchProvider.TOGETHER:
        return together_adapter(http_client)
    if resolved is BatchProvider.XAI:
        return XAIAdapter(http_client)
    raise UnsupportedProviderError(resolved.value)


__all__ = ["BatchAdapter", "get_adapter"]
