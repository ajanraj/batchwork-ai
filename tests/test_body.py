from __future__ import annotations

import json
from pathlib import Path

import pytest

from batchwork.body import build_embedding_bodies, build_image_bodies, build_text_bodies
from batchwork.errors import BatchworkError, UnsupportedProviderError
from batchwork.types import (
    BatchDefaults,
    BatchEmbeddingRequest,
    BatchImageDefaults,
    BatchImageRequest,
    BatchLimits,
    BatchProvider,
    BatchRequest,
    FilePart,
    FunctionTool,
    ImagePart,
    ModelKind,
    ProviderDefinedTool,
    UserMessage,
)


def test_openai_text_serializes_multimodal_tools_and_defaults() -> None:
    built = build_text_bodies(
        BatchProvider.OPENAI,
        "gpt-4.1-mini",
        [
            BatchRequest(
                custom_id="a",
                messages=[
                    UserMessage(
                        content=[
                            {"type": "text", "text": "describe"},
                            ImagePart(image=b"png", media_type="image/png"),
                        ]
                    )
                ],
                tools=[
                    FunctionTool(
                        name="weather",
                        description="Get weather",
                        input_schema={"type": "object", "properties": {}},
                    )
                ],
            )
        ],
        {"temperature": 0.2},
    )

    assert built[0].endpoint == "/v1/chat/completions"
    assert built[0].body["temperature"] == 0.2
    assert built[0].body["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "weather",
                "description": "Get weather",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    messages = built[0].body["messages"]
    assert isinstance(messages, list)
    message = messages[0]
    assert isinstance(message, dict)
    content = message["content"]
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


@pytest.mark.parametrize("provider", list(BatchProvider))
def test_all_providers_build_text(provider: BatchProvider) -> None:
    built = build_text_bodies(provider, "model", [BatchRequest(prompt="hello")])
    assert built[0].custom_id == "request-0"
    assert built[0].body


def test_openai_model_kinds_select_endpoint_and_wire_shape() -> None:
    responses = build_text_bodies(
        BatchProvider.OPENAI,
        "gpt-4.1",
        [
            BatchRequest(
                prompt="search",
                tools=[
                    ProviderDefinedTool(id="openai.web_search_preview", name="web_search", args={})
                ],
            )
        ],
        kind=ModelKind.RESPONSES,
    )[0]
    completion = build_text_bodies(
        BatchProvider.OPENAI,
        "gpt-3.5-turbo-instruct",
        [
            BatchRequest(
                prompt="complete",
                provider_options={"openai": {"echo": True, "logprobs": True, "suffix": " done"}},
            )
        ],
        kind=ModelKind.COMPLETION,
    )[0]

    assert responses.endpoint == "/v1/responses"
    assert "input" in responses.body and "messages" not in responses.body
    assert responses.body["tools"] == [{"type": "web_search_preview"}]
    assert completion.endpoint == "/v1/completions"
    assert completion.body["prompt"] == "complete"
    assert completion.body["echo"] is True
    assert completion.body["logprobs"] == 0
    assert completion.body["suffix"] == " done"


def test_function_tool_extensions_match_provider_wire_contracts() -> None:
    tool = FunctionTool(
        name="weather",
        input_schema={"type": "object", "properties": {}},
        input_examples=[{"input": {"city": "London"}}],
        strict=True,
        provider_options={
            "anthropic": {
                "cacheControl": {"type": "ephemeral"},
                "deferLoading": True,
                "eagerInputStreaming": True,
                "allowedCallers": ["direct"],
            }
        },
    )
    anthropic = build_text_bodies(
        BatchProvider.ANTHROPIC,
        "claude-sonnet-4-6",
        [BatchRequest(prompt="hello", tools=[tool])],
    )[0]
    openai_chat = build_text_bodies(
        BatchProvider.OPENAI,
        "gpt-4.1",
        [BatchRequest(prompt="hello", tools=[tool])],
    )[0]
    openai_responses = build_text_bodies(
        BatchProvider.OPENAI,
        "gpt-5.5",
        [BatchRequest(prompt="hello", tools=[tool])],
        kind=ModelKind.RESPONSES,
    )[0]
    google = build_text_bodies(
        BatchProvider.GOOGLE,
        "gemini-3-pro",
        [BatchRequest(prompt="hello", tools=[tool])],
    )[0]

    assert anthropic.body["tools"] == [
        {
            "name": "weather",
            "input_schema": {"type": "object", "properties": {}},
            "strict": True,
            "cache_control": {"type": "ephemeral"},
            "eager_input_streaming": True,
            "defer_loading": True,
            "allowed_callers": ["direct"],
            "input_examples": [{"city": "London"}],
        }
    ]
    assert openai_chat.body["tools"][0]["function"]["strict"] is True
    assert "_input_examples" not in openai_chat.body["tools"][0]
    assert openai_responses.body["tools"][0]["strict"] is True
    assert google.body["tools"][0]["functionDeclarations"][0]["description"] == ""


def test_provider_defined_tools_use_provider_native_shapes() -> None:
    google = build_text_bodies(
        BatchProvider.GOOGLE,
        "gemini-3-pro",
        [
            BatchRequest(
                prompt="hello",
                tools=[
                    ProviderDefinedTool(
                        id="google.google_search",
                        name="search",
                        args={"timeRangeFilter": {"startTime": "2026-01-01"}},
                    )
                ],
            )
        ],
    )[0]
    xai = build_text_bodies(
        BatchProvider.XAI,
        "grok-4",
        [
            BatchRequest(
                prompt="hello",
                tools=[
                    ProviderDefinedTool(
                        id="xai.x_search",
                        name="search_x",
                        args={"allowedXHandles": ["xai"], "fromDate": "2026-01-01"},
                    )
                ],
            )
        ],
    )[0]
    openai = build_text_bodies(
        BatchProvider.OPENAI,
        "gpt-5.5",
        [
            BatchRequest(
                prompt="hello",
                tools=[
                    ProviderDefinedTool(
                        id="openai.file_search",
                        name="files",
                        args={
                            "vectorStoreIds": ["vs_1"],
                            "ranking": {"ranker": "auto", "scoreThreshold": 0.5},
                        },
                    )
                ],
            )
        ],
        kind=ModelKind.RESPONSES,
    )[0]

    assert google.body["tools"] == [
        {"googleSearch": {"timeRangeFilter": {"startTime": "2026-01-01"}}}
    ]
    assert xai.body["tools"] == [
        {
            "type": "x_search",
            "allowed_x_handles": ["xai"],
            "from_date": "2026-01-01",
        }
    ]
    assert openai.body["tools"] == [
        {
            "type": "file_search",
            "vector_store_ids": ["vs_1"],
            "ranking_options": {"ranker": "auto", "score_threshold": 0.5},
        }
    ]


def test_provider_tool_choices_and_anthropic_options_match_sdk() -> None:
    function_tools = [
        FunctionTool(name=name, input_schema={"type": "object"}) for name in ("first", "second")
    ]
    mistral = build_text_bodies(
        BatchProvider.MISTRAL,
        "mistral-large",
        [
            BatchRequest(
                prompt="hello",
                tools=function_tools,
                tool_choice={"type": "tool", "toolName": "second"},
            )
        ],
    )[0]
    anthropic = build_text_bodies(
        BatchProvider.ANTHROPIC,
        "claude-sonnet-4-6",
        [
            BatchRequest(
                prompt="hello",
                tools=function_tools,
                tool_choice="none",
                provider_options={
                    "anthropic": {
                        "disableParallelToolUse": True,
                        "mcpServers": [
                            {
                                "type": "url",
                                "name": "docs",
                                "url": "https://mcp.example.com",
                                "authorizationToken": "secret",
                                "toolConfiguration": {
                                    "enabled": True,
                                    "allowedTools": ["search"],
                                },
                            }
                        ],
                        "container": {"id": "container_1"},
                        "contextManagement": {
                            "edits": [
                                {
                                    "type": "compact_20260112",
                                    "pauseAfterCompaction": True,
                                }
                            ]
                        },
                    }
                },
            )
        ],
    )[0]

    assert mistral.body["tool_choice"] == "any"
    assert [tool["function"]["name"] for tool in mistral.body["tools"]] == ["second"]
    assert "tools" not in anthropic.body and "tool_choice" not in anthropic.body
    assert anthropic.body["container"] == "container_1"
    assert anthropic.body["mcp_servers"][0]["authorization_token"] == "secret"
    assert anthropic.body["context_management"] == {
        "edits": [{"type": "compact_20260112", "pause_after_compaction": True}]
    }


def test_openai_responses_allowed_tools_override_request_choice() -> None:
    built = build_text_bodies(
        BatchProvider.OPENAI,
        "gpt-5.5",
        [
            BatchRequest(
                prompt="hello",
                tools=[
                    FunctionTool(name="weather", input_schema={"type": "object"}),
                    ProviderDefinedTool(id="openai.web_search", name="search", args={}),
                ],
                tool_choice="none",
                provider_options={
                    "openai": {
                        "allowedTools": {
                            "toolNames": ["weather", "search"],
                            "mode": "required",
                        }
                    }
                },
            )
        ],
        kind=ModelKind.RESPONSES,
    )[0]

    assert built.body["tool_choice"] == {
        "type": "allowed_tools",
        "mode": "required",
        "tools": [
            {"type": "function", "name": "weather"},
            {"type": "function", "name": "web_search"},
        ],
    }


def test_openai_responses_ignores_allowed_tools_without_definitions() -> None:
    built = build_text_bodies(
        BatchProvider.OPENAI,
        "gpt-5.5",
        [
            BatchRequest(
                prompt="hello",
                provider_options={
                    "openai": {"allowedTools": {"toolNames": ["weather"], "mode": "required"}}
                },
            )
        ],
        kind=ModelKind.RESPONSES,
    )[0]

    assert "tool_choice" not in built.body


def test_openai_responses_serializes_compaction_approvals_and_content_output() -> None:
    built = build_text_bodies(
        BatchProvider.OPENAI,
        "gpt-5.5",
        [
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "custom",
                                "kind": "openai.compaction",
                                "providerOptions": {
                                    "openai": {
                                        "itemId": "cmp_1",
                                        "encryptedContent": "ciphertext",
                                    }
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "content": [
                            {
                                "type": "tool-approval-response",
                                "approvalId": "approval_1",
                                "approved": True,
                                "providerExecuted": True,
                            },
                            {
                                "type": "tool-result",
                                "toolCallId": "call_1",
                                "toolName": "read",
                                "output": {
                                    "type": "content",
                                    "value": [
                                        {"type": "text", "text": "result"},
                                        {
                                            "type": "file",
                                            "data": {
                                                "type": "url",
                                                "url": "https://example.com/result.pdf",
                                            },
                                            "mediaType": "application/pdf",
                                        },
                                    ],
                                },
                            },
                        ],
                    },
                ],
                "providerOptions": {"openai": {"store": False}},
            }
        ],
        kind=ModelKind.RESPONSES,
    )[0]

    assert built.body["input"] == [
        {"type": "compaction", "id": "cmp_1", "encrypted_content": "ciphertext"},
        {
            "type": "mcp_approval_response",
            "approval_request_id": "approval_1",
            "approve": True,
        },
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": [
                {"type": "input_text", "text": "result"},
                {"type": "input_file", "file_url": "https://example.com/result.pdf"},
            ],
        },
    ]


def test_openai_responses_normalizes_legacy_outputs_and_approval_denials() -> None:
    built = build_text_bodies(
        BatchProvider.OPENAI,
        "gpt-5.5",
        [
            {
                "messages": [
                    {
                        "role": "tool",
                        "content": [
                            {
                                "type": "tool-approval-response",
                                "approvalId": "local_approval",
                                "approved": False,
                            },
                            {
                                "type": "tool-result",
                                "toolCallId": "denied_call",
                                "toolName": "read",
                                "output": {
                                    "type": "execution-denied",
                                    "providerOptions": {"openai": {"approvalId": "local_approval"}},
                                },
                            },
                            {
                                "type": "tool-result",
                                "toolCallId": "file_call",
                                "toolName": "read",
                                "output": {
                                    "type": "content",
                                    "value": [
                                        {
                                            "type": "file-url",
                                            "url": "https://example.com/result.pdf",
                                        },
                                        {
                                            "type": "image-data",
                                            "data": "aW1hZ2U=",
                                            "mediaType": "image/png",
                                        },
                                    ],
                                },
                            },
                        ],
                    }
                ]
            }
        ],
        kind=ModelKind.RESPONSES,
    )[0]

    assert built.body["input"] == [
        {
            "type": "function_call_output",
            "call_id": "file_call",
            "output": [
                {"type": "input_file", "file_url": "https://example.com/result.pdf"},
                {
                    "type": "input_image",
                    "image_url": "data:image/png;base64,aW1hZ2U=",
                },
            ],
        }
    ]


def test_anthropic_serializes_tagged_files_and_structured_tool_outputs() -> None:
    built = build_text_bodies(
        BatchProvider.ANTHROPIC,
        "claude-sonnet-4-6",
        [
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "file",
                                "data": {
                                    "type": "reference",
                                    "reference": {"anthropic": "file_1"},
                                },
                                "mediaType": "application/pdf",
                            },
                            {
                                "type": "file",
                                "data": {"type": "text", "text": "contents"},
                                "mediaType": "text/plain",
                                "filename": "notes.txt",
                                "providerOptions": {
                                    "anthropic": {
                                        "citations": {"enabled": True},
                                        "context": "background",
                                    }
                                },
                            },
                        ],
                    },
                    {
                        "role": "tool",
                        "content": [
                            {
                                "type": "tool-result",
                                "toolCallId": "call_1",
                                "toolName": "read",
                                "output": {
                                    "type": "content",
                                    "providerOptions": {
                                        "anthropic": {"cacheControl": {"type": "ephemeral"}}
                                    },
                                    "value": [
                                        {"type": "text", "text": "result"},
                                        {
                                            "type": "file",
                                            "data": {
                                                "type": "url",
                                                "url": "https://example.com/image.png",
                                            },
                                            "mediaType": "image/png",
                                        },
                                        {
                                            "type": "custom",
                                            "providerOptions": {
                                                "anthropic": {
                                                    "type": "tool-reference",
                                                    "toolName": "lookup",
                                                }
                                            },
                                        },
                                    ],
                                },
                            },
                            {
                                "type": "tool-result",
                                "toolCallId": "call_2",
                                "toolName": "read",
                                "output": {"type": "error-json", "value": {"error": "bad"}},
                            },
                        ],
                    },
                ]
            }
        ],
    )[0]

    assert len(built.body["messages"]) == 1
    assert built.body["messages"][0]["content"] == [
        {"type": "document", "source": {"type": "file", "file_id": "file_1"}},
        {
            "type": "document",
            "source": {"type": "text", "media_type": "text/plain", "data": "contents"},
            "title": "notes.txt",
            "context": "background",
            "citations": {"enabled": True},
        },
        {
            "type": "tool_result",
            "tool_use_id": "call_1",
            "content": [
                {"type": "text", "text": "result"},
                {
                    "type": "image",
                    "source": {"type": "url", "url": "https://example.com/image.png"},
                },
                {"type": "tool_reference", "tool_name": "lookup"},
            ],
            "cache_control": {"type": "ephemeral"},
        },
        {
            "type": "tool_result",
            "tool_use_id": "call_2",
            "content": '{"error":"bad"}',
            "is_error": True,
        },
    ]


def test_google_preserves_json_and_multimodal_tool_outputs() -> None:
    built = build_text_bodies(
        BatchProvider.GOOGLE,
        "gemini-3-pro",
        [
            {
                "messages": [
                    {
                        "role": "tool",
                        "content": [
                            {
                                "type": "tool-result",
                                "toolCallId": "json_call",
                                "toolName": "lookup",
                                "output": {"type": "json", "value": {"ok": True}},
                            },
                            {
                                "type": "tool-result",
                                "toolCallId": "file_call",
                                "toolName": "render",
                                "output": {
                                    "type": "content",
                                    "value": [
                                        {"type": "text", "text": "done"},
                                        {
                                            "type": "file",
                                            "data": {"type": "data", "data": "aW1hZ2U="},
                                            "mediaType": "image/png",
                                        },
                                    ],
                                },
                            },
                        ],
                    }
                ]
            }
        ],
    )[0]

    assert built.body["contents"][0]["parts"] == [
        {
            "functionResponse": {
                "id": "json_call",
                "name": "lookup",
                "response": {"name": "lookup", "content": {"ok": True}},
            }
        },
        {
            "functionResponse": {
                "id": "file_call",
                "name": "render",
                "response": {"name": "render", "content": "done"},
                "parts": [{"inlineData": {"mimeType": "image/png", "data": "aW1hZ2U="}}],
            }
        },
    ]


def test_tagged_file_sources_match_openai_wire_shapes() -> None:
    chat = build_text_bodies(
        BatchProvider.OPENAI,
        "gpt-4.1",
        [
            BatchRequest(
                messages=[
                    UserMessage(
                        content=[
                            FilePart(
                                data={"type": "data", "data": "cGRm"},
                                media_type="application/pdf",
                            )
                        ]
                    )
                ]
            )
        ],
    )[0]
    responses = build_text_bodies(
        BatchProvider.OPENAI,
        "gpt-5.5",
        [
            BatchRequest(
                messages=[
                    UserMessage(
                        content=[
                            FilePart(
                                data={
                                    "type": "url",
                                    "url": "https://example.com/file.pdf",
                                },
                                media_type="application/pdf",
                            ),
                            FilePart(
                                data={
                                    "type": "reference",
                                    "reference": {"openai": "file_1"},
                                },
                                media_type="application/pdf",
                            ),
                        ]
                    )
                ]
            )
        ],
        kind=ModelKind.RESPONSES,
    )[0]

    assert chat.body["messages"][0]["content"] == [
        {
            "type": "file",
            "file": {
                "file_data": "data:application/pdf;base64,cGRm",
                "filename": "part-0.pdf",
            },
        }
    ]
    assert responses.body["input"][0]["content"] == [
        {"type": "input_file", "file_url": "https://example.com/file.pdf"},
        {"type": "input_file", "file_id": "file_1"},
    ]


def test_tagged_data_that_looks_like_url_stays_inline() -> None:
    built = build_text_bodies(
        BatchProvider.OPENAI,
        "gpt-5.5",
        [
            BatchRequest(
                messages=[
                    UserMessage(
                        content=[
                            FilePart(
                                data={"type": "data", "data": "https://example.com/file.pdf"},
                                media_type="application/pdf",
                            )
                        ]
                    )
                ]
            )
        ],
        kind=ModelKind.RESPONSES,
    )[0]

    assert built.body["input"][0]["content"][0]["file_data"].startswith(
        "data:application/pdf;base64,https://"
    )


@pytest.mark.parametrize(
    "provider", [BatchProvider.OPENAI, BatchProvider.GOOGLE, BatchProvider.MISTRAL]
)
def test_embedding_matrix(provider: BatchProvider) -> None:
    built = build_embedding_bodies(provider, "embed", [BatchEmbeddingRequest(value="hello")])
    assert built[0].endpoint.endswith(("/embeddings", ":embedContent"))


@pytest.mark.parametrize(
    "provider", [BatchProvider.OPENAI, BatchProvider.GOOGLE, BatchProvider.XAI]
)
def test_image_matrix(provider: BatchProvider) -> None:
    built = build_image_bodies(provider, "image", [BatchImageRequest(prompt="cat")])
    assert built[0].body


def test_image_defaults_and_provider_specific_wire_shapes() -> None:
    openai = build_image_bodies(
        BatchProvider.OPENAI,
        "gpt-image-2",
        [
            BatchImageRequest(
                prompt="cat",
                n=2,
                size="1536x1024",
                provider_options={
                    "openai": {
                        "background": "transparent",
                        "moderation": "low",
                        "outputCompression": 80,
                        "outputFormat": "webp",
                        "quality": "high",
                        "style": "natural",
                        "user": "user-1",
                    }
                },
            )
        ],
    )[0]
    assert openai.endpoint == "/v1/images/generations"
    assert openai.body == {
        "model": "gpt-image-2",
        "prompt": "cat",
        "n": 2,
        "size": "1536x1024",
        "background": "transparent",
        "moderation": "low",
        "output_compression": 80,
        "output_format": "webp",
        "quality": "high",
        "style": "natural",
        "user": "user-1",
    }
    dalle = build_image_bodies(
        BatchProvider.OPENAI,
        "dall-e-3",
        [BatchImageRequest(prompt="cat")],
    )[0]
    assert dalle.body["response_format"] == "b64_json"

    inherited, overridden = build_image_bodies(
        BatchProvider.XAI,
        "grok-imagine",
        [BatchImageRequest(prompt="one"), BatchImageRequest(prompt="two", n=2)],
        BatchImageDefaults(n=4, aspect_ratio="16:9", seed=7, size="1024x1024"),
    )
    assert inherited.body == {
        "model": "grok-imagine",
        "prompt": "one",
        "n": 4,
        "response_format": "b64_json",
        "aspect_ratio": "16:9",
    }
    assert overridden.body["n"] == 2

    google = build_image_bodies(
        BatchProvider.GOOGLE,
        "gemini-image",
        [
            BatchImageRequest(
                prompt="cat",
                aspect_ratio="4:3",
                seed=8,
                provider_options={
                    "google": {"googleSearch": {"timeRangeFilter": {"startTime": "2026-01-01"}}}
                },
            )
        ],
    )[0]
    assert google.body["generationConfig"] == {
        "responseModalities": ["IMAGE"],
        "imageConfig": {"aspectRatio": "4:3"},
        "seed": 8,
    }
    assert google.body["tools"] == [
        {"googleSearch": {"timeRangeFilter": {"startTime": "2026-01-01"}}}
    ]
    with pytest.raises(BatchworkError, match="exactly one image"):
        build_image_bodies(
            BatchProvider.GOOGLE,
            "gemini-image",
            [BatchImageRequest(prompt="cats", n=2)],
        )


def test_unsupported_modalities_raise_specific_error() -> None:
    with pytest.raises(UnsupportedProviderError) as image_error:
        build_image_bodies(
            BatchProvider.ANTHROPIC,
            "claude",
            [BatchImageRequest(prompt="cat")],
        )
    assert image_error.value.provider == "anthropic"
    with pytest.raises(UnsupportedProviderError) as embedding_error:
        build_embedding_bodies(
            BatchProvider.ANTHROPIC,
            "claude",
            [BatchEmbeddingRequest(value="hello")],
        )
    assert embedding_error.value.provider == "anthropic"


def test_provider_specific_text_settings_and_tool_outputs() -> None:
    request = {
        "frequencyPenalty": 0.2,
        "maxOutputTokens": 64,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "hello"}]},
            {
                "role": "assistant",
                "content": [
                    {"type": "reasoning", "text": "thought"},
                    {"type": "text", "text": "answer"},
                    {
                        "type": "tool-call",
                        "toolCallId": "call-1",
                        "toolName": "lookup",
                        "input": {"q": 1},
                    },
                ],
            },
            {
                "role": "tool",
                "content": [
                    {
                        "type": "tool-result",
                        "toolCallId": "call-1",
                        "toolName": "lookup",
                        "output": {"type": "text", "value": "plain"},
                    }
                ],
            },
        ],
        "presencePenalty": 0.3,
        "seed": 42,
        "stopSequences": ["stop"],
        "temperature": 0.5,
        "topK": 10,
        "topP": 0.9,
    }
    bodies = {
        provider: build_text_bodies(provider, "model", [request])[0] for provider in BatchProvider
    }
    expected = json.loads(
        (Path(__file__).parent / "fixtures" / "provider-text-bodies.json").read_text()
    )
    assert {
        provider.value: {"body": built.body, "endpoint": built.endpoint}
        for provider, built in bodies.items()
    } == expected
    assert bodies[BatchProvider.GROQ].body["max_tokens"] == 64
    assert bodies[BatchProvider.GROQ].body["messages"][1]["reasoning"] == "thought"
    assert bodies[BatchProvider.MISTRAL].body["random_seed"] == 42
    assert "frequency_penalty" not in bodies[BatchProvider.MISTRAL].body
    assert bodies[BatchProvider.MISTRAL].body["messages"][2]["name"] == "lookup"
    assert bodies[BatchProvider.TOGETHER].body["messages"][1]["reasoning_content"] == "thought"
    assert bodies[BatchProvider.OPENAI].body["messages"][2]["content"] == "plain"
    assert bodies[BatchProvider.XAI].endpoint == "/v1/responses"
    assert bodies[BatchProvider.XAI].body["input"][-1]["output"] == "plain"


def test_openai_reasoning_model_uses_reasoning_wire_contract() -> None:
    built = build_text_bodies(
        BatchProvider.OPENAI,
        "gpt-5.5",
        [BatchRequest(prompt="hello", max_output_tokens=32, temperature=0.4)],
    )[0]
    assert built.body["max_completion_tokens"] == 32
    assert "max_tokens" not in built.body
    assert "temperature" not in built.body

    responses = build_text_bodies(
        BatchProvider.OPENAI,
        "gpt-5.5",
        [
            {
                "prompt": "hello",
                "system": "system setting",
                "temperature": 0.4,
                "maxOutputTokens": 32,
                "providerOptions": {
                    "openai": {
                        "reasoningEffort": "none",
                        "textVerbosity": "low",
                        "store": False,
                        "logprobs": 3,
                        "instructions": "provider instruction",
                    }
                },
            }
        ],
        kind=ModelKind.RESPONSES,
    )[0]
    assert responses.endpoint == "/v1/responses"
    assert responses.body == {
        "model": "gpt-5.5",
        "input": [
            {"role": "developer", "content": "system setting"},
            {"role": "user", "content": [{"type": "input_text", "text": "hello"}]},
        ],
        "temperature": 0.4,
        "max_output_tokens": 32,
        "text": {"verbosity": "low"},
        "store": False,
        "instructions": "provider instruction",
        "include": ["message.output_text.logprobs", "reasoning.encrypted_content"],
        "top_logprobs": 3,
        "reasoning": {"effort": "none"},
    }


@pytest.mark.parametrize("kind", [ModelKind.CHAT, ModelKind.RESPONSES])
def test_openai_force_reasoning_overrides_model_detection(kind: ModelKind) -> None:
    forced = build_text_bodies(
        BatchProvider.OPENAI,
        "gpt-4.1",
        [
            {
                "prompt": "hello",
                "system": "rules",
                "temperature": 0.4,
                "topP": 0.8,
                "maxOutputTokens": 32,
                "providerOptions": {"openai": {"forceReasoning": True, "reasoningEffort": "low"}},
            }
        ],
        kind=kind,
    )[0]
    non_reasoning = build_text_bodies(
        BatchProvider.OPENAI,
        "gpt-5",
        [
            {
                "prompt": "hello",
                "system": "rules",
                "temperature": 0.4,
                "topP": 0.8,
                "providerOptions": {"openai": {"forceReasoning": False, "reasoningEffort": "low"}},
            }
        ],
        kind=kind,
    )[0]

    entries_key = "messages" if kind is ModelKind.CHAT else "input"
    assert "temperature" not in forced.body and "top_p" not in forced.body
    assert forced.body[entries_key][0]["role"] == "developer"
    assert non_reasoning.body["temperature"] == 0.4
    assert non_reasoning.body["top_p"] == 0.8
    assert non_reasoning.body[entries_key][0]["role"] == "system"
    if kind is ModelKind.CHAT:
        assert forced.body["max_completion_tokens"] == 32
    else:
        assert forced.body["max_output_tokens"] == 32
        assert forced.body["reasoning"] == {"effort": "low", "summary": "detailed"}
        assert "reasoning" not in non_reasoning.body


@pytest.mark.parametrize("kind", [ModelKind.CHAT, ModelKind.RESPONSES])
@pytest.mark.parametrize(
    ("model_id", "tier", "supported"),
    [
        ("gpt-4.1", "flex", False),
        ("o3", "flex", True),
        ("gpt-5-nano", "priority", False),
        ("gpt-4.1", "priority", True),
    ],
)
def test_openai_service_tier_capabilities_match_across_endpoints(
    kind: ModelKind, model_id: str, tier: str, supported: bool
) -> None:
    built = build_text_bodies(
        BatchProvider.OPENAI,
        model_id,
        [
            {
                "prompt": "hello",
                "providerOptions": {"openai": {"serviceTier": tier}},
            }
        ],
        kind=kind,
    )[0]

    assert (built.body.get("service_tier") == tier) is supported


@pytest.mark.parametrize("kind", [ModelKind.CHAT, ModelKind.RESPONSES])
@pytest.mark.parametrize(
    ("mode", "expected_role"),
    [("system", "system"), ("developer", "developer"), ("remove", None)],
)
def test_openai_system_message_modes_match_across_endpoints(
    kind: ModelKind, mode: str, expected_role: str | None
) -> None:
    built = build_text_bodies(
        BatchProvider.OPENAI,
        "gpt-4.1",
        [
            {
                "prompt": "hello",
                "system": "rules",
                "providerOptions": {"openai": {"systemMessageMode": mode}},
            }
        ],
        kind=kind,
    )[0]

    entries_key = "messages" if kind is ModelKind.CHAT else "input"
    roles = [entry["role"] for entry in built.body[entries_key]]
    assert roles == ([expected_role, "user"] if expected_role is not None else ["user"])


def test_provider_options_map_to_provider_wire_fields() -> None:
    mistral = build_text_bodies(
        BatchProvider.MISTRAL,
        "model",
        [{"prompt": "hello", "providerOptions": {"mistral": {"safePrompt": True}}}],
    )[0]
    google = build_text_bodies(
        BatchProvider.GOOGLE,
        "model",
        [
            {
                "prompt": "hello",
                "providerOptions": {"google": {"responseModalities": ["TEXT"]}},
            }
        ],
    )[0]
    openai = build_text_bodies(
        BatchProvider.OPENAI,
        "model",
        [
            {
                "prompt": "hello",
                "providerOptions": {"openai": {"reasoningEffort": "low"}},
            }
        ],
    )[0]
    together = build_text_bodies(
        BatchProvider.TOGETHER,
        "model",
        [
            {
                "prompt": "hello",
                "providerOptions": {"together": {"custom_wire": "value", "strictJsonSchema": True}},
            }
        ],
    )[0]
    xai = build_text_bodies(
        BatchProvider.XAI,
        "model",
        [
            {
                "prompt": "hello",
                "providerOptions": {"xai": {"store": False}},
            }
        ],
    )[0]
    assert mistral.body["safe_prompt"] is True
    assert google.body["generationConfig"] == {"responseModalities": ["TEXT"]}
    assert openai.body["reasoning_effort"] == "low"
    assert together.body["custom_wire"] == "value"
    assert together.body["strictJsonSchema"] is True
    assert xai.body["store"] is False
    assert xai.body["include"] == ["reasoning.encrypted_content"]


def test_text_provider_options_shallow_merge_selected_branch() -> None:
    built = build_text_bodies(
        BatchProvider.GOOGLE,
        "gemini-test",
        [
            {
                "prompt": "hello",
                "providerOptions": {"google": {"thinkingConfig": {"includeThoughts": True}}},
            }
        ],
        BatchDefaults(
            provider_options={
                "google": {
                    "thinkingConfig": {"thinkingBudget": 1024},
                    "serviceTier": "flex",
                }
            }
        ),
        strict=True,
    )[0]

    assert built.body["generationConfig"]["thinkingConfig"] == {"includeThoughts": True}
    assert built.body["serviceTier"] == "flex"


@pytest.mark.parametrize(
    "provider",
    [provider for provider in BatchProvider if provider is not BatchProvider.TOGETHER],
)
def test_strict_text_provider_options_reject_unknown_keys(provider: BatchProvider) -> None:
    with pytest.raises(BatchworkError, match='provider option "unknownOption" is unsupported'):
        build_text_bodies(
            provider,
            "model",
            [
                {
                    "prompt": "hello",
                    "providerOptions": {provider.value: {"unknownOption": True}},
                }
            ],
            strict=True,
        )


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        ("reasoningEffort", [], 'provider option "reasoningEffort" must be a string'),
        ("systemMessageMode", "silent", 'provider option "systemMessageMode" must be one of'),
        ("truncation", "middle", 'provider option "truncation" must be one of'),
    ],
)
def test_strict_openai_provider_options_reject_invalid_values(
    option: str, value: object, message: str
) -> None:
    with pytest.raises(BatchworkError, match=message):
        build_text_bodies(
            BatchProvider.OPENAI,
            "gpt-4.1",
            [{"prompt": "hello", "providerOptions": {"openai": {option: value}}}],
            strict=True,
            kind=ModelKind.RESPONSES,
        )


def test_strict_openai_responses_accepts_conversation_object() -> None:
    built = build_text_bodies(
        BatchProvider.OPENAI,
        "gpt-4.1",
        [
            {
                "prompt": "hello",
                "providerOptions": {"openai": {"conversation": {"id": "conv_1"}}},
            }
        ],
        strict=True,
        kind=ModelKind.RESPONSES,
    )[0]

    assert built.body["conversation"] == {"id": "conv_1"}


@pytest.mark.parametrize("setting", ["temperature", "top_p", "frequency_penalty"])
def test_strict_openai_reasoning_rejects_dropped_settings(setting: str) -> None:
    with pytest.raises(BatchworkError, match=f"canonical {setting} conflicts"):
        build_text_bodies(
            BatchProvider.OPENAI,
            "gpt-5",
            [{"prompt": "hello", setting: 0.2}],
            strict=True,
        )


def test_together_passthrough_rejects_reserved_wire_fields() -> None:
    with pytest.raises(BatchworkError, match='provider option "model" is reserved'):
        build_text_bodies(
            BatchProvider.TOGETHER,
            "model",
            [{"prompt": "hello", "providerOptions": {"together": {"model": "other"}}}],
            strict=True,
        )


def test_strict_text_preflight_rejects_semantic_collision_and_unsupported_setting() -> None:
    with pytest.raises(BatchworkError, match=r"max_output_tokens.*maxCompletionTokens"):
        build_text_bodies(
            BatchProvider.OPENAI,
            "model",
            [
                {
                    "prompt": "hello",
                    "maxOutputTokens": 32,
                    "providerOptions": {"openai": {"maxCompletionTokens": 64}},
                }
            ],
            strict=True,
        )


def test_xai_top_logprobs_accepts_documented_range() -> None:
    built = build_text_bodies(
        BatchProvider.XAI,
        "model",
        [{"prompt": "hello", "providerOptions": {"xai": {"topLogprobs": 0}}}],
        kind=ModelKind.RESPONSES,
        strict=True,
    )[0]
    assert built.body["top_logprobs"] == 0

    with pytest.raises(BatchworkError, match="integer between 0 and 8"):
        build_text_bodies(
            BatchProvider.XAI,
            "model",
            [{"prompt": "hello", "providerOptions": {"xai": {"topLogprobs": 9}}}],
            kind=ModelKind.RESPONSES,
            strict=True,
        )

    with pytest.raises(BatchworkError, match='canonical setting "frequency_penalty"'):
        build_text_bodies(
            BatchProvider.XAI,
            "model",
            [{"prompt": "hello", "frequencyPenalty": 0.2}],
            kind=ModelKind.RESPONSES,
            strict=True,
        )


def test_embedding_provider_options_and_wire_shapes() -> None:
    openai = build_embedding_bodies(
        BatchProvider.OPENAI,
        "embed",
        [
            BatchEmbeddingRequest(
                value="hello",
                provider_options={"openai": {"dimensions": 256, "user": "u"}},
            )
        ],
    )[0]
    google = build_embedding_bodies(
        BatchProvider.GOOGLE,
        "embed",
        [
            BatchEmbeddingRequest(
                value="hello",
                provider_options={
                    "google": {
                        "outputDimensionality": 128,
                        "taskType": "RETRIEVAL_QUERY",
                        "title": "Knowledge base entry",
                        "content": [[{"text": "context"}]],
                    }
                },
            )
        ],
    )[0]
    mistral = build_embedding_bodies(
        BatchProvider.MISTRAL,
        "embed",
        [BatchEmbeddingRequest(value="hello", provider_options={"mistral": {"ignored": 1}})],
    )[0]
    assert openai.body == {
        "model": "embed",
        "input": ["hello"],
        "encoding_format": "float",
        "dimensions": 256,
        "user": "u",
    }
    assert google.body == {
        "model": "models/embed",
        "content": {"parts": [{"text": "hello"}, {"text": "context"}]},
        "outputDimensionality": 128,
        "taskType": "RETRIEVAL_QUERY",
        "title": "Knowledge base entry",
    }
    assert mistral.body == {
        "model": "embed",
        "input": ["hello"],
        "encoding_format": "float",
    }


def test_embedding_defaults_apply_only_when_record_value_is_absent() -> None:
    inherited, overridden = build_embedding_bodies(
        BatchProvider.OPENAI,
        "embed",
        [
            BatchEmbeddingRequest(value="one"),
            BatchEmbeddingRequest(
                value="two",
                dimensions=64,
                provider_options={"openai": {"user": "record"}},
            ),
        ],
        defaults={
            "dimensions": 128,
            "provider_options": {"openai": {"user": "default"}},
        },
    )

    assert inherited.body["dimensions"] == 128
    assert inherited.body["user"] == "default"
    assert overridden.body["dimensions"] == 64
    assert overridden.body["user"] == "record"


def test_embedding_strict_validation_rejects_collisions_and_unsupported_settings() -> None:
    with pytest.raises(BatchworkError, match="conflicts"):
        build_embedding_bodies(
            BatchProvider.OPENAI,
            "embed",
            [
                BatchEmbeddingRequest(
                    value="hello",
                    dimensions=128,
                    provider_options={"openai": {"dimensions": 64}},
                )
            ],
            strict=True,
        )
    with pytest.raises(BatchworkError, match="does not support"):
        build_embedding_bodies(
            BatchProvider.MISTRAL,
            "embed",
            [BatchEmbeddingRequest(value="hello", dimensions=128)],
            strict=True,
        )
    with pytest.raises(BatchworkError, match="not supported"):
        build_embedding_bodies(
            BatchProvider.MISTRAL,
            "embed",
            [
                BatchEmbeddingRequest(
                    value="hello",
                    provider_options={"mistral": {"unknown": True}},
                )
            ],
            strict=True,
        )


@pytest.mark.parametrize(
    ("provider", "options", "message"),
    (
        (BatchProvider.OPENAI, {"dimensions": "wide"}, "positive integer"),
        (BatchProvider.OPENAI, {"user": 42}, "must be a string"),
        (BatchProvider.GOOGLE, {"outputDimensionality": None}, "positive integer"),
        (BatchProvider.GOOGLE, {"taskType": "UNKNOWN"}, "task type"),
        (BatchProvider.GOOGLE, {"title": 42}, "must be a string"),
    ),
)
def test_embedding_strict_validation_rejects_invalid_provider_option_values(
    provider: BatchProvider,
    options: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(BatchworkError, match=message):
        build_embedding_bodies(
            provider,
            "embed",
            [
                BatchEmbeddingRequest(
                    value="hello",
                    provider_options={provider.value: options},
                )
            ],
            strict=True,
        )


def test_embedding_dimension_collision_uses_key_presence() -> None:
    with pytest.raises(BatchworkError, match="conflicts"):
        build_embedding_bodies(
            BatchProvider.GOOGLE,
            "embed",
            [
                BatchEmbeddingRequest(
                    value="hello",
                    dimensions=128,
                    provider_options={"google": {"outputDimensionality": None}},
                )
            ],
            strict=True,
        )


def test_duplicate_ids_and_limits_fail_before_serialization() -> None:
    with pytest.raises(BatchworkError, match="duplicate customId"):
        build_text_bodies(
            BatchProvider.OPENAI,
            "model",
            [
                BatchRequest(custom_id="same", prompt="a"),
                BatchRequest(custom_id="same", prompt="b"),
            ],
        )
    with pytest.raises(BatchworkError, match="request limit"):
        build_text_bodies(
            BatchProvider.OPENAI,
            "model",
            [BatchRequest(prompt="a"), BatchRequest(prompt="b")],
            limits=BatchLimits(max_requests=1),
        )


def test_default_request_limit_is_fifty_thousand_but_can_be_overridden() -> None:
    requests = [{"prompt": "hello"}] * 50_001
    with pytest.raises(BatchworkError, match="50000 request limit"):
        build_text_bodies(BatchProvider.OPENAI, "model", requests)
    assert BatchLimits(max_requests=50_001).max_requests == 50_001
