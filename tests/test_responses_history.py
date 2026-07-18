from batchwork.body import build_text_bodies
from batchwork.types import BatchProvider, ModelKind


def test_openai_responses_reuses_stored_history_items() -> None:
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
                                "type": "text",
                                "text": "answer",
                                "providerOptions": {"openai": {"itemId": "msg_1"}},
                            },
                            {
                                "type": "reasoning",
                                "text": "summary",
                                "providerOptions": {"openai": {"itemId": "rs_1"}},
                            },
                            {
                                "type": "tool-call",
                                "toolCallId": "call_1",
                                "toolName": "search",
                                "input": {"query": "news"},
                                "providerExecuted": True,
                                "providerOptions": {"openai": {"itemId": "ws_1"}},
                            },
                        ],
                    }
                ],
                "tools": [
                    {
                        "type": "provider-defined",
                        "id": "openai.web_search",
                        "name": "search",
                        "args": {},
                    }
                ],
            }
        ],
        kind=ModelKind.RESPONSES,
    )[0]

    assert built.body["input"] == [
        {"type": "item_reference", "id": "msg_1"},
        {"type": "item_reference", "id": "rs_1"},
        {"type": "item_reference", "id": "ws_1"},
    ]


def test_openai_responses_replays_encrypted_reasoning_without_storage() -> None:
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
                                "type": "reasoning",
                                "text": "summary",
                                "providerOptions": {
                                    "openai": {"reasoningEncryptedContent": "ciphertext"}
                                },
                            }
                        ],
                    }
                ],
                "providerOptions": {"openai": {"store": False}},
            }
        ],
        kind=ModelKind.RESPONSES,
    )[0]

    assert built.body["input"] == [
        {
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": "summary"}],
            "encrypted_content": "ciphertext",
        }
    ]


def test_openai_provider_tool_choice_uses_native_type() -> None:
    built = build_text_bodies(
        BatchProvider.OPENAI,
        "gpt-5.5",
        [
            {
                "prompt": "search",
                "tools": [
                    {
                        "type": "provider-defined",
                        "id": "openai.web_search",
                        "name": "search",
                        "args": {},
                    }
                ],
                "toolChoice": {"type": "provider-defined", "toolName": "search"},
            }
        ],
        kind=ModelKind.RESPONSES,
    )[0]

    assert built.body["tool_choice"] == {"type": "web_search"}


def test_xai_preserves_system_file_references_and_schema_contract() -> None:
    built = build_text_bodies(
        BatchProvider.XAI,
        "grok-4",
        [
            {
                "prompt": "hello",
                "system": "rules",
                "tools": [
                    {
                        "type": "function",
                        "name": "lookup",
                        "inputSchema": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "query": {
                                    "type": "object",
                                    "additionalProperties": False,
                                }
                            },
                        },
                    }
                ],
            },
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "file",
                                "data": {
                                    "type": "reference",
                                    "reference": {"xai": "file-1"},
                                },
                                "mediaType": "application/pdf",
                            }
                        ],
                    }
                ]
            },
        ],
        kind=ModelKind.RESPONSES,
    )

    assert built[0].body["input"][:2] == [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": [{"type": "input_text", "text": "hello"}]},
    ]
    schema = built[0].body["tools"][0]["parameters"]
    assert "additionalProperties" not in schema
    assert "additionalProperties" not in schema["properties"]["query"]
    assert built[1].body["input"] == [
        {
            "role": "user",
            "content": [{"type": "input_file", "file_id": "file-1"}],
        }
    ]
