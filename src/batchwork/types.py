"""Typed public request, content, result, and configuration models."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, JsonValue, model_validator

from .errors import UnsupportedProviderError

JsonScalar: TypeAlias = str | int | float | bool | None
ProviderOptions: TypeAlias = dict[str, dict[str, JsonValue]]


def _to_camel(value: str) -> str:
    head, *tail = value.split("_")
    return head + "".join(part.capitalize() for part in tail)


class BatchworkModel(BaseModel):
    """Strict immutable value object accepting snake_case and camelCase."""

    model_config = ConfigDict(
        alias_generator=_to_camel,
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        validate_default=True,
    )


class BatchProvider(StrEnum):
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    GROQ = "groq"
    MISTRAL = "mistral"
    OPENAI = "openai"
    TOGETHER = "together"
    XAI = "xai"


class ModelKind(StrEnum):
    CHAT = "chat"
    RESPONSES = "responses"
    COMPLETION = "completion"


class ModelSpec(BatchworkModel):
    provider: BatchProvider
    model_id: str = Field(min_length=1)
    kind: ModelKind = ModelKind.CHAT


class ProviderCredentials(BatchworkModel):
    api_key: str | None = Field(default=None, repr=False)
    base_url: str | None = None
    headers: dict[str, str] = Field(default_factory=dict, repr=False)


class BatchLimits(BatchworkModel):
    max_requests: int = Field(default=50_000, ge=1)
    max_request_bytes: int = Field(default=20 * 1024 * 1024, ge=1)
    max_upload_bytes: int = Field(default=200 * 1024 * 1024, ge=1)


class ProviderFileReference(BatchworkModel):
    type: Literal["provider-file-id"] = "provider-file-id"
    value: str = Field(min_length=1)


class TaggedFileDataData(BatchworkModel):
    type: Literal["data"] = "data"
    data: bytes | str


class TaggedFileDataUrl(BatchworkModel):
    type: Literal["url"] = "url"
    url: HttpUrl


class TaggedFileDataReference(BatchworkModel):
    type: Literal["reference"] = "reference"
    reference: dict[str, str]


class TaggedFileDataText(BatchworkModel):
    type: Literal["text"] = "text"
    text: str


TaggedFileData = Annotated[
    TaggedFileDataData | TaggedFileDataUrl | TaggedFileDataReference | TaggedFileDataText,
    Field(discriminator="type"),
]
TaggedReasoningFileData = Annotated[
    TaggedFileDataData | TaggedFileDataUrl,
    Field(discriminator="type"),
]
MediaSource: TypeAlias = (
    bytes | str | HttpUrl | dict[str, str] | ProviderFileReference | TaggedFileData
)
ReasoningMediaSource: TypeAlias = bytes | str | HttpUrl | TaggedReasoningFileData


class TextPart(BatchworkModel):
    type: Literal["text"] = "text"
    text: str
    provider_options: ProviderOptions | None = None


class ImagePart(BatchworkModel):
    type: Literal["image"] = "image"
    image: bytes | str | HttpUrl | dict[str, str] | ProviderFileReference
    media_type: str | None = None
    provider_options: ProviderOptions | None = None


class FilePart(BatchworkModel):
    type: Literal["file"] = "file"
    data: MediaSource
    media_type: str
    filename: str | None = None
    provider_options: ProviderOptions | None = None


class ReasoningPart(BatchworkModel):
    type: Literal["reasoning"] = "reasoning"
    text: str
    provider_options: ProviderOptions | None = None


class ReasoningFilePart(BatchworkModel):
    type: Literal["reasoning-file"] = "reasoning-file"
    data: ReasoningMediaSource
    media_type: str
    filename: str | None = None
    provider_options: ProviderOptions | None = None


class ToolCallPart(BatchworkModel):
    type: Literal["tool-call"] = "tool-call"
    tool_call_id: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    input: JsonValue
    provider_executed: bool | None = None
    provider_options: ProviderOptions | None = None


class TextToolOutput(BatchworkModel):
    type: Literal["text"] = "text"
    value: str
    provider_options: ProviderOptions | None = None


class JsonToolOutput(BatchworkModel):
    type: Literal["json"] = "json"
    value: JsonValue
    provider_options: ProviderOptions | None = None


class ExecutionDeniedToolOutput(BatchworkModel):
    type: Literal["execution-denied"] = "execution-denied"
    reason: str | None = None
    provider_options: ProviderOptions | None = None


class ErrorTextToolOutput(BatchworkModel):
    type: Literal["error-text"] = "error-text"
    value: str
    provider_options: ProviderOptions | None = None


class ErrorJsonToolOutput(BatchworkModel):
    type: Literal["error-json"] = "error-json"
    value: JsonValue
    provider_options: ProviderOptions | None = None


class ToolOutputTextPart(BatchworkModel):
    type: Literal["text"] = "text"
    text: str
    provider_options: ProviderOptions | None = None


class ToolOutputFilePart(BatchworkModel):
    type: Literal["file"] = "file"
    data: TaggedFileData
    media_type: str
    filename: str | None = None
    provider_options: ProviderOptions | None = None


class DeprecatedToolOutputFileDataPart(BatchworkModel):
    type: Literal["file-data"] = "file-data"
    data: str
    media_type: str
    filename: str | None = None
    provider_options: ProviderOptions | None = None


class DeprecatedToolOutputFileUrlPart(BatchworkModel):
    type: Literal["file-url"] = "file-url"
    url: str
    media_type: str | None = None
    provider_options: ProviderOptions | None = None


class DeprecatedToolOutputFileIdPart(BatchworkModel):
    type: Literal["file-id"] = "file-id"
    file_id: str | dict[str, str]
    provider_options: ProviderOptions | None = None


class DeprecatedToolOutputFileReferencePart(BatchworkModel):
    type: Literal["file-reference"] = "file-reference"
    provider_reference: dict[str, str]
    provider_options: ProviderOptions | None = None


class DeprecatedToolOutputImageDataPart(BatchworkModel):
    type: Literal["image-data"] = "image-data"
    data: str
    media_type: str
    provider_options: ProviderOptions | None = None


class DeprecatedToolOutputImageUrlPart(BatchworkModel):
    type: Literal["image-url"] = "image-url"
    url: str
    provider_options: ProviderOptions | None = None


class DeprecatedToolOutputImageFileIdPart(BatchworkModel):
    type: Literal["image-file-id"] = "image-file-id"
    file_id: str | dict[str, str]
    provider_options: ProviderOptions | None = None


class DeprecatedToolOutputImageFileReferencePart(BatchworkModel):
    type: Literal["image-file-reference"] = "image-file-reference"
    provider_reference: dict[str, str]
    provider_options: ProviderOptions | None = None


class ToolOutputCustomPart(BatchworkModel):
    type: Literal["custom"] = "custom"
    provider_options: ProviderOptions | None = None


ToolOutputContentPart = Annotated[
    ToolOutputTextPart
    | ToolOutputFilePart
    | DeprecatedToolOutputFileDataPart
    | DeprecatedToolOutputFileUrlPart
    | DeprecatedToolOutputFileIdPart
    | DeprecatedToolOutputFileReferencePart
    | DeprecatedToolOutputImageDataPart
    | DeprecatedToolOutputImageUrlPart
    | DeprecatedToolOutputImageFileIdPart
    | DeprecatedToolOutputImageFileReferencePart
    | ToolOutputCustomPart,
    Field(discriminator="type"),
]


class ContentToolOutput(BatchworkModel):
    type: Literal["content"] = "content"
    value: list[ToolOutputContentPart]
    provider_options: ProviderOptions | None = None


ToolOutput = Annotated[
    TextToolOutput
    | JsonToolOutput
    | ExecutionDeniedToolOutput
    | ErrorTextToolOutput
    | ErrorJsonToolOutput
    | ContentToolOutput,
    Field(discriminator="type"),
]


class ToolResultPart(BatchworkModel):
    type: Literal["tool-result"] = "tool-result"
    tool_call_id: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    output: ToolOutput
    provider_options: ProviderOptions | None = None


class ToolApprovalRequestPart(BatchworkModel):
    type: Literal["tool-approval-request"] = "tool-approval-request"
    approval_id: str = Field(min_length=1)
    tool_call_id: str = Field(min_length=1)
    is_automatic: bool | None = None
    signature: str | None = None


class ToolApprovalResponsePart(BatchworkModel):
    type: Literal["tool-approval-response"] = "tool-approval-response"
    approval_id: str = Field(min_length=1)
    approved: bool
    reason: str | None = None
    provider_executed: bool | None = None


class CustomPart(BatchworkModel):
    type: Literal["custom"] = "custom"
    kind: str = Field(pattern=r"^[^.]+\..+$")
    provider_options: ProviderOptions | None = None


UserContentPart = Annotated[TextPart | ImagePart | FilePart, Field(discriminator="type")]
AssistantContentPart = Annotated[
    TextPart
    | FilePart
    | ReasoningPart
    | ReasoningFilePart
    | ToolCallPart
    | ToolResultPart
    | ToolApprovalRequestPart
    | CustomPart,
    Field(discriminator="type"),
]
ToolContentPart = Annotated[ToolResultPart | ToolApprovalResponsePart, Field(discriminator="type")]


class SystemMessage(BatchworkModel):
    role: Literal["system"] = "system"
    content: str
    provider_options: ProviderOptions | None = None


class UserMessage(BatchworkModel):
    role: Literal["user"] = "user"
    content: str | list[UserContentPart]
    provider_options: ProviderOptions | None = None


class AssistantMessage(BatchworkModel):
    role: Literal["assistant"] = "assistant"
    content: str | list[AssistantContentPart]
    provider_options: ProviderOptions | None = None


class ToolMessage(BatchworkModel):
    role: Literal["tool"] = "tool"
    content: list[ToolContentPart]
    provider_options: ProviderOptions | None = None


ModelMessage = Annotated[
    SystemMessage | UserMessage | AssistantMessage | ToolMessage, Field(discriminator="role")
]


class FunctionTool(BatchworkModel):
    type: Literal["function"] = "function"
    name: str = Field(min_length=1)
    description: str | None = None
    input_schema: dict[str, JsonValue]
    input_examples: list[dict[str, JsonValue]] | None = None
    strict: bool | None = None
    provider_options: ProviderOptions | None = None


class ProviderDefinedTool(BatchworkModel):
    type: Literal["provider-defined"] = "provider-defined"
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    args: dict[str, JsonValue] = Field(default_factory=dict)


Tool = Annotated[FunctionTool | ProviderDefinedTool, Field(discriminator="type")]


class NamedToolChoice(BatchworkModel):
    type: Literal["tool"] = "tool"
    tool_name: str = Field(min_length=1)


class ProviderToolChoice(BatchworkModel):
    type: Literal["provider-defined"] = "provider-defined"
    tool_name: str = Field(min_length=1)


ToolChoice: TypeAlias = Literal["auto", "none", "required"] | NamedToolChoice | ProviderToolChoice


class BatchRequestSettings(BatchworkModel):
    frequency_penalty: float | None = None
    max_output_tokens: int | None = Field(default=None, ge=1)
    presence_penalty: float | None = None
    provider_options: ProviderOptions | None = None
    seed: int | None = None
    stop_sequences: list[str] | None = None
    system: str | None = None
    temperature: float | None = None
    tool_choice: ToolChoice | None = None
    tools: list[Tool] | None = None
    top_k: int | None = Field(default=None, ge=1)
    top_p: float | None = None


class BatchRequest(BatchRequestSettings):
    custom_id: str | None = None
    messages: list[ModelMessage] | None = None
    prompt: str | None = None

    @model_validator(mode="after")
    def _one_input(self) -> BatchRequest:
        if (self.prompt is None) == (self.messages is None):
            raise ValueError("exactly one of prompt or messages is required")
        if self.messages == []:
            raise ValueError("messages must not be empty")
        return self


BatchDefaults = BatchRequestSettings


class BatchEmbeddingRequest(BatchworkModel):
    value: str
    custom_id: str | None = None
    provider_options: ProviderOptions | None = None


class BatchImageRequest(BatchworkModel):
    prompt: str = Field(min_length=1)
    aspect_ratio: str | None = Field(default=None, pattern=r"^\d+:\d+$")
    custom_id: str | None = None
    n: int | None = Field(default=None, ge=1)
    provider_options: ProviderOptions | None = None
    seed: int | None = None
    size: str | None = Field(default=None, pattern=r"^\d+x\d+$")


class BatchImageDefaults(BatchworkModel):
    aspect_ratio: str | None = Field(default=None, pattern=r"^\d+:\d+$")
    n: int = Field(default=1, ge=1)
    provider_options: ProviderOptions | None = None
    seed: int | None = None
    size: str | None = Field(default=None, pattern=r"^\d+x\d+$")


class BatchStatus(StrEnum):
    VALIDATING = "validating"
    IN_PROGRESS = "in_progress"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"


TERMINAL_STATUSES = frozenset(
    {BatchStatus.COMPLETED, BatchStatus.FAILED, BatchStatus.EXPIRED, BatchStatus.CANCELLED}
)


def is_terminal_status(status: BatchStatus | str) -> bool:
    return BatchStatus(status) in TERMINAL_STATUSES


class BatchRequestCounts(BatchworkModel):
    total: int = Field(ge=0)
    completed: int = Field(ge=0)
    failed: int = Field(ge=0)
    canceled: int | None = Field(default=None, ge=0)
    expired: int | None = Field(default=None, ge=0)
    processing: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _within_total(self) -> BatchRequestCounts:
        known = self.completed + self.failed + (self.canceled or 0) + (self.expired or 0)
        if known > self.total:
            raise ValueError("finished request counts cannot exceed total")
        return self


class BatchResultStatus(StrEnum):
    SUCCEEDED = "succeeded"
    ERRORED = "errored"
    EXPIRED = "expired"
    CANCELED = "canceled"


class BatchUsage(BatchworkModel):
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)


class BatchResultError(BatchworkModel):
    message: str
    code: str | int | None = None
    type: str | None = None


class BatchImage(BatchworkModel):
    data: str | None = None
    media_type: str | None = None
    url: str | None = None

    @model_validator(mode="after")
    def _has_location(self) -> BatchImage:
        if self.data is None and self.url is None:
            raise ValueError("at least one of data or url is required")
        return self


class BatchResult(BatchworkModel):
    custom_id: str
    status: BatchResultStatus
    embedding: list[float] | None = None
    error: BatchResultError | None = None
    images: list[BatchImage] | None = None
    response: JsonValue | None = None
    text: str | None = None
    usage: BatchUsage | None = None

    @model_validator(mode="after")
    def _error_matches_status(self) -> BatchResult:
        if self.status is BatchResultStatus.ERRORED and self.error is None:
            raise ValueError("errored result requires error")
        if self.status is not BatchResultStatus.ERRORED and self.error is not None:
            raise ValueError("only errored results may contain error")
        return self


class BatchSnapshot(BatchworkModel):
    id: str = Field(min_length=1)
    provider: BatchProvider
    status: BatchStatus
    request_counts: BatchRequestCounts
    raw: JsonValue = Field(default_factory=dict)
    created_at: datetime | None = None
    completed_at: datetime | None = None
    expires_at: datetime | None = None

    @model_validator(mode="after")
    def _aware_datetimes(self) -> BatchSnapshot:
        for field in ("created_at", "completed_at", "expires_at"):
            value = getattr(self, field)
            if value is not None and value.tzinfo is None:
                raise ValueError(f"{field} must be timezone-aware")
        return self


class BatchRef(BatchworkModel):
    id: str = Field(min_length=1)
    provider: BatchProvider | None = None
    model: str | ModelSpec | None = None
    api_key: str | None = Field(default=None, repr=False)
    base_url: str | None = None
    headers: dict[str, str] = Field(default_factory=dict, repr=False)

    @model_validator(mode="after")
    def _has_provider(self) -> BatchRef:
        if self.provider is None and self.model is None:
            raise ValueError("provide provider or model to identify the batch")
        return self


def resolve_model(model: str | ModelSpec) -> ModelSpec:
    if isinstance(model, ModelSpec):
        return model
    provider_name, separator, model_id = model.partition("/")
    if not separator or not model_id:
        raise ValueError('model must use the "provider/model" form')
    aliases = {"gemini": "google", "togetherai": "together"}
    provider_name = aliases.get(provider_name, provider_name)
    try:
        provider = BatchProvider(provider_name)
    except ValueError as error:
        raise UnsupportedProviderError(provider_name) from error
    kind = ModelKind.RESPONSES if provider is BatchProvider.XAI else ModelKind.CHAT
    return ModelSpec(provider=provider, model_id=model_id, kind=kind)


def provider_from_ref(ref: BatchRef) -> BatchProvider:
    if ref.provider is not None:
        return ref.provider
    if ref.model is None:  # guarded by validation; keeps type narrowing explicit
        raise ValueError("batch reference has no provider or model")
    return resolve_model(ref.model).provider


def utc_datetime(timestamp: float | int | None) -> datetime | None:
    return None if timestamp is None else datetime.fromtimestamp(timestamp, tz=UTC)


def coerce_credentials(value: ProviderCredentials | Mapping[str, object]) -> ProviderCredentials:
    if isinstance(value, ProviderCredentials):
        return value
    return ProviderCredentials.model_validate(value)


__all__ = [
    "TERMINAL_STATUSES",
    "AssistantContentPart",
    "AssistantMessage",
    "BatchDefaults",
    "BatchEmbeddingRequest",
    "BatchImage",
    "BatchImageDefaults",
    "BatchImageRequest",
    "BatchLimits",
    "BatchProvider",
    "BatchRef",
    "BatchRequest",
    "BatchRequestCounts",
    "BatchRequestSettings",
    "BatchResult",
    "BatchResultError",
    "BatchResultStatus",
    "BatchSnapshot",
    "BatchStatus",
    "BatchUsage",
    "ContentToolOutput",
    "CustomPart",
    "DeprecatedToolOutputFileDataPart",
    "DeprecatedToolOutputFileIdPart",
    "DeprecatedToolOutputFileReferencePart",
    "DeprecatedToolOutputFileUrlPart",
    "DeprecatedToolOutputImageDataPart",
    "DeprecatedToolOutputImageFileIdPart",
    "DeprecatedToolOutputImageFileReferencePart",
    "DeprecatedToolOutputImageUrlPart",
    "ErrorJsonToolOutput",
    "ErrorTextToolOutput",
    "ExecutionDeniedToolOutput",
    "FilePart",
    "FunctionTool",
    "ImagePart",
    "JsonScalar",
    "JsonToolOutput",
    "JsonValue",
    "MediaSource",
    "ModelKind",
    "ModelMessage",
    "ModelSpec",
    "NamedToolChoice",
    "ProviderCredentials",
    "ProviderDefinedTool",
    "ProviderFileReference",
    "ProviderOptions",
    "ProviderToolChoice",
    "ReasoningFilePart",
    "ReasoningMediaSource",
    "ReasoningPart",
    "SystemMessage",
    "TaggedFileData",
    "TaggedFileDataData",
    "TaggedFileDataReference",
    "TaggedFileDataText",
    "TaggedFileDataUrl",
    "TaggedReasoningFileData",
    "TextPart",
    "TextToolOutput",
    "Tool",
    "ToolApprovalRequestPart",
    "ToolApprovalResponsePart",
    "ToolCallPart",
    "ToolChoice",
    "ToolContentPart",
    "ToolMessage",
    "ToolOutput",
    "ToolOutputContentPart",
    "ToolOutputCustomPart",
    "ToolOutputFilePart",
    "ToolOutputTextPart",
    "ToolResultPart",
    "UserContentPart",
    "UserMessage",
    "coerce_credentials",
    "is_terminal_status",
    "provider_from_ref",
    "resolve_model",
    "utc_datetime",
]
