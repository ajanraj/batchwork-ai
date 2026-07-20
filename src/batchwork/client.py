"""Async package client and provider routing."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass
from urllib.parse import urlsplit

import httpx

from ._limits import MAX_AGGREGATE_MEDIA_BYTES, MAX_DECODED_MEDIA_BYTES
from .body import (
    build_embedding_bodies,
    build_image_bodies,
    build_text_bodies,
    validate_request_count,
)
from .errors import BatchClosedError, BatchworkError, _LimitExceededError
from .job import BatchJob
from .media import DefaultMediaResolver, MediaResolver, MediaSource, ResolvedMedia
from .providers.adapter import BatchAdapter
from .types import (
    AssistantMessage,
    BatchDefaults,
    BatchEmbeddingDefaults,
    BatchEmbeddingRequest,
    BatchImageDefaults,
    BatchImageRequest,
    BatchLimits,
    BatchProvider,
    BatchRef,
    BatchRequest,
    BatchResult,
    ContentToolOutput,
    FilePart,
    ImagePart,
    ModelKind,
    ModelSpec,
    ProviderCredentials,
    ProviderFileReference,
    ReasoningFilePart,
    TaggedFileDataData,
    TaggedFileDataReference,
    TaggedFileDataText,
    TaggedFileDataUrl,
    ToolMessage,
    ToolOutputFilePart,
    ToolResultPart,
    UserMessage,
    coerce_credentials,
    provider_from_ref,
    resolve_model,
)

_API_KEY_ENV = {
    BatchProvider.ANTHROPIC: "ANTHROPIC_API_KEY",
    BatchProvider.GOOGLE: "GOOGLE_GENERATIVE_AI_API_KEY",
    BatchProvider.GROQ: "GROQ_API_KEY",
    BatchProvider.MISTRAL: "MISTRAL_API_KEY",
    BatchProvider.OPENAI: "OPENAI_API_KEY",
    BatchProvider.TOGETHER: "TOGETHER_API_KEY",
    BatchProvider.XAI: "XAI_API_KEY",
}


class _RemoteMediaDeferred(Exception):
    pass


@dataclass(slots=True)
class _MediaBudget:
    maximum: int
    consumed: int = 0

    def add(self, size: int) -> None:
        self.consumed += size
        if self.consumed > self.maximum:
            raise _LimitExceededError(
                f"batchwork: aggregate decoded media exceeds the {self.maximum} byte limit"
            )


_BASE_URL_ENV = {
    provider: f"{name.removesuffix('_API_KEY')}_BASE_URL" for provider, name in _API_KEY_ENV.items()
}
_GOOGLE_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


def _get_adapter(provider: BatchProvider, client: httpx.AsyncClient) -> BatchAdapter:
    from .providers import get_adapter

    return get_adapter(provider, http_client=client)


class Batchwork:
    """Unified async client for provider-native batch APIs."""

    def __init__(
        self,
        *,
        credentials: Mapping[BatchProvider | str, ProviderCredentials | Mapping[str, object]]
        | None = None,
        http_client: httpx.AsyncClient | None = None,
        media_resolver: MediaResolver | None = None,
        timeout: httpx.Timeout | float | None = None,
    ) -> None:
        self._credentials = {
            BatchProvider(key): coerce_credentials(value)
            for key, value in (credentials or {}).items()
        }
        self._owns_http_client = http_client is None
        self._http_client = http_client or httpx.AsyncClient(
            timeout=(
                timeout
                if timeout is not None
                else httpx.Timeout(connect=10.0, pool=10.0, read=300.0, write=300.0)
            ),
            follow_redirects=False,
        )
        self._media_resolver = media_resolver or DefaultMediaResolver()
        self._closed = False

    @property
    def media_resolver(self) -> MediaResolver:
        return self._media_resolver

    @property
    def closed(self) -> bool:
        return self._closed

    async def __aenter__(self) -> Batchwork:
        self._ensure_open()
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()

    def _ensure_open(self) -> None:
        if self._closed:
            raise BatchClosedError("batchwork: client is closed")

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._owns_http_client:
            await self._http_client.aclose()

    def _resolve_credentials(
        self,
        provider: BatchProvider,
        credentials: ProviderCredentials | Mapping[str, object] | None,
        *,
        api_key: str | None,
        base_url: str | None,
        headers: Mapping[str, str] | None,
    ) -> ProviderCredentials:
        supplied = ProviderCredentials() if credentials is None else coerce_credentials(credentials)
        configured = self._credentials.get(provider, ProviderCredentials())
        environment_key = os.getenv(_API_KEY_ENV[provider])
        if provider is BatchProvider.GOOGLE and environment_key is None:
            environment_key = os.getenv("GEMINI_API_KEY")
        merged_headers = {**configured.headers, **supplied.headers, **dict(headers or {})}
        return ProviderCredentials(
            api_key=api_key or supplied.api_key or configured.api_key or environment_key,
            base_url=(
                base_url
                or supplied.base_url
                or configured.base_url
                or os.getenv(_BASE_URL_ENV[provider])
            ),
            headers=merged_headers,
        )

    def _adapter_and_credentials(
        self,
        provider: BatchProvider,
        credentials: ProviderCredentials | Mapping[str, object] | None,
        *,
        api_key: str | None,
        base_url: str | None,
        headers: Mapping[str, str] | None,
    ) -> tuple[BatchAdapter, ProviderCredentials]:
        self._ensure_open()
        resolved = self._resolve_credentials(
            provider,
            credentials,
            api_key=api_key,
            base_url=base_url,
            headers=headers,
        )
        return _get_adapter(provider, self._http_client), resolved

    async def batch(
        self,
        *,
        model: str | ModelSpec,
        requests: Sequence[BatchRequest],
        defaults: BatchDefaults | None = None,
        metadata: Mapping[str, str] | None = None,
        limits: BatchLimits | None = None,
        credentials: ProviderCredentials | Mapping[str, object] | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> BatchJob:
        return await self._submit_text_batch(
            model=model,
            requests=requests,
            defaults=defaults,
            metadata=metadata,
            limits=limits,
            credentials=credentials,
            api_key=api_key,
            base_url=base_url,
            headers=headers,
            validate_upload=None,
        )

    async def _submit_text_batch(
        self,
        *,
        model: str | ModelSpec,
        requests: Sequence[BatchRequest],
        defaults: BatchDefaults | None = None,
        metadata: Mapping[str, str] | None = None,
        limits: BatchLimits | None = None,
        credentials: ProviderCredentials | Mapping[str, object] | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        headers: Mapping[str, str] | None = None,
        validate_upload: Callable[[int], None] | None,
    ) -> BatchJob:
        spec = resolve_model(model)
        self._validate_requests(requests)
        limits = limits or BatchLimits()
        validate_request_count(requests, limits)
        adapter, resolved_credentials = self._adapter_and_credentials(
            spec.provider, credentials, api_key=api_key, base_url=base_url, headers=headers
        )
        media_budget = _MediaBudget(min(limits.max_upload_bytes, MAX_AGGREGATE_MEDIA_BYTES))
        locally_prepared = await self._resolve_request_media(
            spec,
            requests,
            limits,
            resolved_credentials.base_url,
            remote=False,
            budget=media_budget,
        )
        build_text_bodies(
            spec.provider,
            spec.model_id,
            locally_prepared,
            defaults,
            limits,
            kind=spec.kind,
        )
        prepared = await self._resolve_request_media(
            spec,
            locally_prepared,
            limits,
            resolved_credentials.base_url,
            remote=True,
            budget=media_budget,
        )
        built = build_text_bodies(
            spec.provider, spec.model_id, prepared, defaults, limits, kind=spec.kind
        )
        if validate_upload is None:
            snapshot = await adapter.submit(
                built=built,
                credentials=resolved_credentials,
                endpoint=built[0].endpoint,
                limits=limits,
                metadata=metadata,
                model_id=spec.model_id,
            )
        else:
            snapshot = await adapter.submit(
                built=built,
                credentials=resolved_credentials,
                endpoint=built[0].endpoint,
                limits=limits,
                metadata=metadata,
                model_id=spec.model_id,
                validate_upload=validate_upload,
            )
        return BatchJob(adapter, resolved_credentials, snapshot, ensure_open=self._ensure_open)

    async def batch_embeddings(
        self,
        *,
        model: str | ModelSpec,
        requests: Sequence[BatchEmbeddingRequest],
        defaults: BatchEmbeddingDefaults | None = None,
        metadata: Mapping[str, str] | None = None,
        limits: BatchLimits | None = None,
        credentials: ProviderCredentials | Mapping[str, object] | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> BatchJob:
        return await self._submit_embedding_batch(
            model=model,
            requests=requests,
            defaults=defaults,
            metadata=metadata,
            limits=limits,
            credentials=credentials,
            api_key=api_key,
            base_url=base_url,
            headers=headers,
            validate_upload=None,
        )

    async def _submit_embedding_batch(
        self,
        *,
        model: str | ModelSpec,
        requests: Sequence[BatchEmbeddingRequest],
        defaults: BatchEmbeddingDefaults | None = None,
        metadata: Mapping[str, str] | None = None,
        limits: BatchLimits | None = None,
        credentials: ProviderCredentials | Mapping[str, object] | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        headers: Mapping[str, str] | None = None,
        validate_upload: Callable[[int], None] | None,
    ) -> BatchJob:
        spec = resolve_model(model)
        self._validate_requests(requests)
        adapter, resolved_credentials = self._adapter_and_credentials(
            spec.provider, credentials, api_key=api_key, base_url=base_url, headers=headers
        )
        limits = limits or BatchLimits()
        built = build_embedding_bodies(
            spec.provider, spec.model_id, requests, limits, defaults=defaults
        )
        if validate_upload is None:
            snapshot = await adapter.submit(
                built=built,
                credentials=resolved_credentials,
                endpoint=built[0].endpoint,
                limits=limits,
                metadata=metadata,
                model_id=spec.model_id,
            )
        else:
            snapshot = await adapter.submit(
                built=built,
                credentials=resolved_credentials,
                endpoint=built[0].endpoint,
                limits=limits,
                metadata=metadata,
                model_id=spec.model_id,
                validate_upload=validate_upload,
            )
        return BatchJob(adapter, resolved_credentials, snapshot, ensure_open=self._ensure_open)

    async def batch_images(
        self,
        *,
        model: str | ModelSpec,
        requests: Sequence[BatchImageRequest],
        defaults: BatchImageDefaults | None = None,
        metadata: Mapping[str, str] | None = None,
        limits: BatchLimits | None = None,
        credentials: ProviderCredentials | Mapping[str, object] | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> BatchJob:
        return await self._submit_image_batch(
            model=model,
            requests=requests,
            defaults=defaults,
            metadata=metadata,
            limits=limits,
            credentials=credentials,
            api_key=api_key,
            base_url=base_url,
            headers=headers,
            validate_upload=None,
        )

    async def _submit_image_batch(
        self,
        *,
        model: str | ModelSpec,
        requests: Sequence[BatchImageRequest],
        defaults: BatchImageDefaults | None = None,
        metadata: Mapping[str, str] | None = None,
        limits: BatchLimits | None = None,
        credentials: ProviderCredentials | Mapping[str, object] | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        headers: Mapping[str, str] | None = None,
        validate_upload: Callable[[int], None] | None,
    ) -> BatchJob:
        spec = resolve_model(model)
        self._validate_requests(requests)
        adapter, resolved_credentials = self._adapter_and_credentials(
            spec.provider, credentials, api_key=api_key, base_url=base_url, headers=headers
        )
        limits = limits or BatchLimits()
        built = build_image_bodies(
            spec.provider,
            spec.model_id,
            requests,
            defaults,
            limits,
            strict=True,
        )
        if validate_upload is None:
            snapshot = await adapter.submit(
                built=built,
                credentials=resolved_credentials,
                endpoint=built[0].endpoint,
                limits=limits,
                metadata=metadata,
                model_id=spec.model_id,
            )
        else:
            snapshot = await adapter.submit(
                built=built,
                credentials=resolved_credentials,
                endpoint=built[0].endpoint,
                limits=limits,
                metadata=metadata,
                model_id=spec.model_id,
                validate_upload=validate_upload,
            )
        return BatchJob(adapter, resolved_credentials, snapshot, ensure_open=self._ensure_open)

    async def get_batch(self, ref: BatchRef) -> BatchJob:
        provider = provider_from_ref(ref)
        adapter, credentials = self._adapter_and_credentials(
            provider,
            ProviderCredentials(api_key=ref.api_key, base_url=ref.base_url, headers=ref.headers),
            api_key=None,
            base_url=None,
            headers=None,
        )
        snapshot = await adapter.retrieve(ref.id, credentials)
        return BatchJob(adapter, credentials, snapshot, ensure_open=self._ensure_open)

    async def get_batch_results(self, ref: BatchRef) -> AsyncIterator[BatchResult]:
        self._ensure_open()
        provider = provider_from_ref(ref)
        adapter, credentials = self._adapter_and_credentials(
            provider,
            ProviderCredentials(api_key=ref.api_key, base_url=ref.base_url, headers=ref.headers),
            api_key=None,
            base_url=None,
            headers=None,
        )
        async for result in adapter.results(ref.id, credentials):
            self._ensure_open()
            yield result

    async def cancel_batch(self, ref: BatchRef) -> None:
        provider = provider_from_ref(ref)
        adapter, credentials = self._adapter_and_credentials(
            provider,
            ProviderCredentials(api_key=ref.api_key, base_url=ref.base_url, headers=ref.headers),
            api_key=None,
            base_url=None,
            headers=None,
        )
        await adapter.cancel(ref.id, credentials)

    @staticmethod
    def _validate_requests(requests: Sequence[object]) -> None:
        if not requests:
            raise BatchworkError("batchwork: requests must not be empty")

    async def _resolve_request_media(
        self,
        spec: ModelSpec,
        requests: Sequence[BatchRequest],
        limits: BatchLimits,
        base_url: str | None,
        *,
        remote: bool = True,
        budget: _MediaBudget | None = None,
    ) -> list[BatchRequest]:
        prepared: list[BatchRequest] = []
        selected_budget = budget or _MediaBudget(limits.max_upload_bytes)

        async def resolve(source: MediaSource, media_type: str | None) -> ResolvedMedia:
            if not remote and urlsplit(str(source)).scheme.lower() == "https":
                raise _RemoteMediaDeferred
            resolved = await self._media_resolver.resolve(
                source,
                media_type=media_type,
                max_bytes=min(limits.max_request_bytes, MAX_DECODED_MEDIA_BYTES),
            )
            selected_budget.add(len(resolved.data))
            return resolved

        for request in requests:
            if request.messages is None:
                prepared.append(request)
                continue
            messages = []
            for message in request.messages:
                if isinstance(message, ToolMessage):
                    content = []
                    for part in message.content:
                        if not isinstance(part, ToolResultPart) or not isinstance(
                            part.output, ContentToolOutput
                        ):
                            content.append(part)
                            continue
                        output_parts = []
                        for output_part in part.output.value:
                            if not isinstance(output_part, ToolOutputFilePart):
                                output_parts.append(output_part)
                                continue
                            source = output_part.data
                            if isinstance(source, TaggedFileDataUrl):
                                source_value: object = source.url
                            elif isinstance(source, TaggedFileDataReference):
                                output_parts.append(output_part)
                                continue
                            elif (
                                isinstance(source, TaggedFileDataText)
                                and spec.provider is not BatchProvider.TOGETHER
                            ):
                                output_parts.append(output_part)
                                continue
                            elif remote and isinstance(source, TaggedFileDataData):
                                output_parts.append(output_part)
                                continue
                            else:
                                source_value = source
                            if self._may_pass_media_url(
                                spec,
                                source_value,
                                output_part.media_type,
                                base_url,
                            ):
                                output_parts.append(output_part)
                                continue
                            try:
                                resolved = await resolve(source_value, output_part.media_type)
                            except _RemoteMediaDeferred:
                                output_parts.append(output_part)
                                continue
                            output_parts.append(
                                output_part.model_copy(
                                    update={
                                        "data": TaggedFileDataData(data=resolved.data),
                                        "media_type": resolved.media_type,
                                    }
                                )
                            )
                        output = part.output.model_copy(update={"value": output_parts})
                        content.append(part.model_copy(update={"output": output}))
                    messages.append(message.model_copy(update={"content": content}))
                    continue
                if not isinstance(message, (UserMessage, AssistantMessage)) or isinstance(
                    message.content, str
                ):
                    messages.append(message)
                    continue
                parts: list[object] = []
                for part in message.content:
                    if isinstance(part, ImagePart):
                        source = part.image
                        if remote and isinstance(source, bytes):
                            parts.append(part)
                            continue
                        if self._may_pass_media_url(
                            spec,
                            source,
                            part.media_type or "image/*",
                            base_url,
                        ):
                            parts.append(part)
                            continue
                        try:
                            resolved = await resolve(source, part.media_type)
                        except _RemoteMediaDeferred:
                            parts.append(part)
                            continue
                        parts.append(
                            part.model_copy(
                                update={"image": resolved.data, "media_type": resolved.media_type}
                            )
                        )
                    elif isinstance(part, (FilePart, ReasoningFilePart)):
                        source = part.data
                        if remote and isinstance(source, bytes):
                            parts.append(part)
                            continue
                        if isinstance(source, TaggedFileDataUrl):
                            source = source.url
                        elif isinstance(source, TaggedFileDataReference):
                            parts.append(part)
                            continue
                        elif (
                            isinstance(source, TaggedFileDataText)
                            and spec.provider is not BatchProvider.TOGETHER
                        ):
                            parts.append(part)
                            continue
                        elif isinstance(source, Mapping):
                            source_kind = source.get("type")
                            if source_kind == "url":
                                source = TaggedFileDataUrl.model_validate(source).url
                            elif source_kind == "data":
                                source = TaggedFileDataData.model_validate(source)
                            elif source_kind in {"reference", "provider-file-id"}:
                                parts.append(part)
                                continue
                            elif source_kind == "text":
                                if spec.provider is not BatchProvider.TOGETHER:
                                    parts.append(part)
                                    continue
                                source = TaggedFileDataText.model_validate(source)
                        if self._may_pass_media_url(spec, source, part.media_type, base_url):
                            parts.append(part)
                            continue
                        try:
                            resolved = await resolve(source, part.media_type)
                        except _RemoteMediaDeferred:
                            parts.append(part)
                            continue
                        parts.append(
                            part.model_copy(
                                update={"data": resolved.data, "media_type": resolved.media_type}
                            )
                        )
                    else:
                        parts.append(part)
                messages.append(message.model_copy(update={"content": parts}))
            prepared.append(request.model_copy(update={"messages": messages}))
        return prepared

    @staticmethod
    def _may_pass_media_url(
        spec: ModelSpec,
        source: object,
        media_type: str,
        base_url: str | None,
    ) -> bool:
        if isinstance(source, ProviderFileReference):
            return True
        if isinstance(source, Mapping) and isinstance(source.get(spec.provider.value), str):
            return True
        url = str(source)
        if not url.startswith(("http://", "https://")):
            return False
        top_level = media_type.split("/", 1)[0]
        if spec.provider == BatchProvider.OPENAI:
            return top_level == "image" or (
                spec.kind is ModelKind.RESPONSES and media_type == "application/pdf"
            )
        if spec.provider == BatchProvider.ANTHROPIC:
            return top_level == "image" or media_type == "application/pdf"
        if spec.provider == BatchProvider.GROQ:
            return top_level == "image"
        if spec.provider == BatchProvider.MISTRAL:
            return url.startswith("https://") and media_type == "application/pdf"
        if spec.provider == BatchProvider.XAI:
            return top_level in {"image", "text"} or media_type == "application/pdf"
        if spec.provider == BatchProvider.GOOGLE:
            if not url.startswith("https://"):
                return False
            provider_base = (base_url or _GOOGLE_BASE_URL).rstrip("/")
            return url.startswith(f"{provider_base}/files/") or url.startswith(
                (
                    "https://youtube.com/watch?v=",
                    "https://www.youtube.com/watch?v=",
                    "https://youtu.be/",
                )
            )
        return False


__all__ = ["Batchwork"]
