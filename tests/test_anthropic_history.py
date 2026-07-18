from __future__ import annotations

from batchwork._anthropic_serialization import anthropic_messages, anthropic_prompt


def test_system_settings_and_messages_preserve_api_block_position() -> None:
    system, messages = anthropic_prompt(
        {
            "system": "request system",
            "messages": [
                {
                    "role": "system",
                    "content": "cached system",
                    "providerOptions": {"anthropic": {"cacheControl": {"type": "ephemeral"}}},
                },
                {"role": "user", "content": "question"},
                {"role": "system", "content": "mid-conversation system"},
                {"role": "assistant", "content": "answer   "},
            ],
        }
    )

    assert system == [
        {"type": "text", "text": "request system"},
        {
            "type": "text",
            "text": "cached system",
            "cache_control": {"type": "ephemeral"},
        },
    ]
    assert messages == [
        {"role": "user", "content": [{"type": "text", "text": "question"}]},
        {
            "role": "system",
            "content": [{"type": "text", "text": "mid-conversation system"}],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "answer"}]},
    ]


def test_reasoning_history_uses_signed_and_redacted_blocks() -> None:
    messages = anthropic_messages(
        {
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool-call",
                            "toolCallId": "before",
                            "toolName": "lookup",
                            "input": {},
                        },
                        {
                            "type": "reasoning",
                            "text": "private chain",
                            "providerOptions": {
                                "anthropic": {
                                    "signature": "signed",
                                    "cacheControl": {"type": "ephemeral"},
                                }
                            },
                        },
                        {
                            "type": "tool-call",
                            "toolCallId": "after",
                            "toolName": "lookup",
                            "input": {},
                        },
                        {"type": "text", "text": "visible"},
                        {
                            "type": "reasoning",
                            "text": "not sent",
                            "providerOptions": {"anthropic": {"redactedData": "encrypted"}},
                        },
                        {"type": "reasoning", "text": "missing metadata"},
                    ],
                }
            ]
        }
    )

    assert messages == [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "before",
                    "name": "lookup",
                    "input": {},
                },
                {"type": "thinking", "thinking": "private chain", "signature": "signed"},
                {"type": "text", "text": "visible"},
                {
                    "type": "tool_use",
                    "id": "after",
                    "name": "lookup",
                    "input": {},
                },
                {"type": "redacted_thinking", "data": "encrypted"},
            ],
        }
    ]


def test_user_files_and_tool_results_preserve_cache_boundaries() -> None:
    messages = anthropic_messages(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "inspect"},
                        {
                            "type": "image",
                            "image": b"png",
                            "mediaType": "image/png",
                        },
                        {
                            "type": "file",
                            "data": {"type": "url", "url": "https://example.com/report.pdf"},
                            "mediaType": "application/pdf",
                            "filename": "report.pdf",
                            "providerOptions": {
                                "anthropic": {
                                    "citations": {"enabled": True},
                                    "context": "quarterly report",
                                }
                            },
                        },
                    ],
                    "providerOptions": {"anthropic": {"cacheControl": {"type": "ephemeral"}}},
                },
                {
                    "role": "tool",
                    "content": [
                        {
                            "type": "tool-result",
                            "toolCallId": "call-1",
                            "toolName": "lookup",
                            "output": {
                                "type": "content",
                                "value": [
                                    {"type": "text", "text": "result"},
                                    {
                                        "type": "file",
                                        "data": {
                                            "type": "url",
                                            "url": "https://example.com/result.png",
                                        },
                                        "mediaType": "image/png",
                                        "providerOptions": {
                                            "anthropic": {"cacheControl": {"type": "ephemeral"}}
                                        },
                                    },
                                ],
                            },
                        }
                    ],
                },
            ]
        }
    )

    assert messages == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "inspect"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "cG5n",
                    },
                },
                {
                    "type": "document",
                    "source": {
                        "type": "url",
                        "url": "https://example.com/report.pdf",
                    },
                    "title": "report.pdf",
                    "context": "quarterly report",
                    "citations": {"enabled": True},
                    "cache_control": {"type": "ephemeral"},
                },
                {
                    "type": "tool_result",
                    "tool_use_id": "call-1",
                    "content": [
                        {"type": "text", "text": "result"},
                        {
                            "type": "image",
                            "source": {
                                "type": "url",
                                "url": "https://example.com/result.png",
                            },
                        },
                    ],
                    "cache_control": {"type": "ephemeral"},
                },
            ],
        }
    ]


def test_provider_executed_tool_calls_use_public_history_shapes() -> None:
    messages = anthropic_messages(
        {
            "tools": [
                {
                    "type": "provider-defined",
                    "id": "anthropic.web_search_20260209",
                    "name": "search_docs",
                    "args": {},
                },
                {
                    "type": "provider-defined",
                    "id": "anthropic.code_execution_20260120",
                    "name": "run_code",
                    "args": {},
                },
                {
                    "type": "provider-defined",
                    "id": "anthropic.advisor_20260301",
                    "name": "advise",
                    "args": {},
                },
            ],
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool-call",
                            "toolCallId": "search-1",
                            "toolName": "search_docs",
                            "input": {"query": "batch API"},
                            "providerExecuted": True,
                        },
                        {
                            "type": "tool-call",
                            "toolCallId": "code-1",
                            "toolName": "run_code",
                            "input": {"type": "programmatic-tool-call", "code": "print(1)"},
                            "providerExecuted": True,
                        },
                        {
                            "type": "tool-call",
                            "toolCallId": "advisor-1",
                            "toolName": "advise",
                            "input": {"ignored": True},
                            "providerExecuted": True,
                        },
                        {
                            "type": "tool-call",
                            "toolCallId": "mcp-1",
                            "toolName": "remote_search",
                            "input": {"query": "docs"},
                            "providerExecuted": True,
                            "providerOptions": {
                                "anthropic": {
                                    "type": "mcp-tool-use",
                                    "serverName": "docs",
                                    "cacheControl": {"type": "ephemeral"},
                                }
                            },
                        },
                    ],
                }
            ],
        }
    )

    assert messages[0]["content"] == [
        {
            "type": "server_tool_use",
            "id": "search-1",
            "name": "web_search",
            "input": {"query": "batch API"},
        },
        {
            "type": "server_tool_use",
            "id": "code-1",
            "name": "code_execution",
            "input": {"code": "print(1)"},
        },
        {
            "type": "server_tool_use",
            "id": "advisor-1",
            "name": "advisor",
            "input": {},
        },
        {
            "type": "mcp_tool_use",
            "id": "mcp-1",
            "name": "remote_search",
            "input": {"query": "docs"},
            "server_name": "docs",
            "cache_control": {"type": "ephemeral"},
        },
    ]


def test_regular_tool_calls_map_caller_and_message_cache_control() -> None:
    messages = anthropic_messages(
        {
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool-call",
                            "toolCallId": "call-1",
                            "toolName": "lookup",
                            "input": {"query": "weather"},
                            "providerOptions": {
                                "anthropic": {
                                    "caller": {
                                        "type": "code_execution_20260120",
                                        "toolId": "code-1",
                                    }
                                }
                            },
                        }
                    ],
                    "providerOptions": {"anthropic": {"cacheControl": {"type": "ephemeral"}}},
                }
            ]
        }
    )

    assert messages == [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "call-1",
                    "name": "lookup",
                    "input": {"query": "weather"},
                    "caller": {"type": "code_execution_20260120", "tool_id": "code-1"},
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        }
    ]
