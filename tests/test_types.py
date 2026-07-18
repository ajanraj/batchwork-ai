from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

import batchwork
from batchwork.types import (
    AssistantMessage,
    BatchImage,
    BatchProvider,
    BatchRef,
    BatchRequest,
    BatchRequestCounts,
    BatchResult,
    BatchResultError,
    BatchResultStatus,
    BatchSnapshot,
    BatchStatus,
    ImagePart,
    ModelKind,
    TextPart,
    ToolMessage,
    UserMessage,
    provider_from_ref,
    resolve_model,
)


def test_public_exports_are_explicit_and_resolvable() -> None:
    assert len(batchwork.__all__) == len(set(batchwork.__all__))
    assert all(hasattr(batchwork, name) for name in batchwork.__all__)


def test_request_requires_exactly_one_input() -> None:
    assert BatchRequest(prompt="hello").prompt == "hello"
    with pytest.raises(ValidationError, match="exactly one"):
        BatchRequest()
    with pytest.raises(ValidationError, match="exactly one"):
        BatchRequest(prompt="hello", messages=[UserMessage(content="world")])


def test_content_union_validates_unknown_discriminator() -> None:
    message = UserMessage(content=[TextPart(text="hello"), ImagePart(image="aGVsbG8=")])
    assert len(message.content) == 2
    with pytest.raises(ValidationError, match="union_tag_invalid"):
        UserMessage.model_validate({"content": [{"type": "unknown"}], "role": "user"})


def test_ai_sdk_seven_message_extensions_validate() -> None:
    assistant = AssistantMessage.model_validate(
        {
            "content": [
                {
                    "type": "tool-approval-request",
                    "approvalId": "approval_1",
                    "toolCallId": "call_1",
                    "isAutomatic": True,
                    "signature": "signed",
                },
                {
                    "type": "custom",
                    "kind": "openai.compaction",
                    "providerOptions": {
                        "openai": {"itemId": "cmp_1", "encryptedContent": "ciphertext"}
                    },
                },
            ],
            "role": "assistant",
        }
    )
    tool = ToolMessage.model_validate(
        {
            "content": [
                {
                    "type": "tool-result",
                    "toolCallId": "call_1",
                    "toolName": "read",
                    "output": {
                        "type": "content",
                        "providerOptions": {"anthropic": {"cacheControl": {"type": "ephemeral"}}},
                        "value": [
                            {
                                "type": "file",
                                "data": {"type": "text", "text": "file contents"},
                                "mediaType": "text/plain",
                            }
                        ],
                    },
                }
            ],
            "role": "tool",
        }
    )

    assert assistant.model_dump()["content"][0]["tool_call_id"] == "call_1"
    assert assistant.model_dump()["content"][1]["kind"] == "openai.compaction"
    assert tool.model_dump()["content"][0]["output"]["type"] == "content"


def test_result_error_and_image_invariants() -> None:
    error = BatchResultError(message="bad request")
    result = BatchResult(custom_id="a", status=BatchResultStatus.ERRORED, error=error)
    assert result.error == error
    assert BatchImage(data="aGVsbG8=", media_type="text/plain").url is None
    assert BatchImage(data="aGVsbG8=").media_type is None
    with pytest.raises(ValidationError, match="requires error"):
        BatchResult(custom_id="a", status=BatchResultStatus.ERRORED)
    both = BatchImage(data="x", media_type="image/png", url="https://example.com/a.png")
    assert both.data == "x" and both.url is not None
    with pytest.raises(ValidationError, match="at least one"):
        BatchImage()


def test_snapshot_rejects_naive_datetime() -> None:
    counts = BatchRequestCounts(total=1, completed=0, failed=0)
    with pytest.raises(ValidationError, match="timezone-aware"):
        BatchSnapshot(
            id="batch_1",
            provider=BatchProvider.OPENAI,
            status=BatchStatus.IN_PROGRESS,
            request_counts=counts,
            created_at=datetime(2026, 1, 1),
        )
    snapshot = BatchSnapshot(
        id="batch_1",
        provider=BatchProvider.OPENAI,
        status=BatchStatus.IN_PROGRESS,
        request_counts=counts,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    assert snapshot.created_at is not None


def test_ref_provider_precedes_model() -> None:
    ref = BatchRef(id="batch_1", provider="xai", model="openai/gpt-5")
    assert provider_from_ref(ref) is BatchProvider.XAI
    assert resolve_model("gemini/gemini-2.5-pro").provider is BatchProvider.GOOGLE
    assert resolve_model("xai/grok-4").kind is ModelKind.RESPONSES
