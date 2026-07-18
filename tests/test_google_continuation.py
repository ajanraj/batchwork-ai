from batchwork._google_schema import google_openapi_schema
from batchwork._google_serialization import (
    SKIP_THOUGHT_SIGNATURE_VALIDATOR,
    google_messages,
)


def test_google_system_messages_preserve_order_and_gemma_prefixes_them() -> None:
    item = {
        "system": "Pinned",
        "messages": [
            {"role": "system", "content": "First"},
            {"role": "system", "content": "Second"},
            {"role": "user", "content": "Hello"},
        ],
    }

    contents, system_instruction = google_messages(item, "gemini-2.5-pro")

    assert system_instruction == {
        "parts": [{"text": "Pinned"}, {"text": "First"}, {"text": "Second"}]
    }
    assert contents == [{"role": "user", "parts": [{"text": "Hello"}]}]

    gemma_contents, gemma_system = google_messages(item, "gemma-3-27b-it")

    assert gemma_system is None
    assert gemma_contents == [
        {
            "role": "user",
            "parts": [{"text": "Pinned\n\nFirst\n\nSecond\n\n"}, {"text": "Hello"}],
        }
    ]


def test_google_preserves_thought_signatures_on_continuation_parts() -> None:
    contents, system_instruction = google_messages(
        {
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "reasoning",
                            "text": "thinking",
                            "providerOptions": {
                                "google": {"thoughtSignature": "reasoning-signature"}
                            },
                        },
                        {
                            "type": "text",
                            "text": "answer",
                            "provider_options": {"google": {"thoughtSignature": "text-signature"}},
                        },
                        {
                            "type": "tool-call",
                            "toolCallId": "call_1",
                            "toolName": "weather",
                            "input": {"city": "London"},
                            "providerOptions": {"google": {"thoughtSignature": "tool-signature"}},
                        },
                        {
                            "type": "tool-call",
                            "toolCallId": "call_2",
                            "toolName": "clock",
                            "input": {},
                        },
                    ],
                }
            ]
        },
        "gemini-3-pro",
    )

    assert system_instruction is None
    assert contents == [
        {
            "role": "model",
            "parts": [
                {
                    "text": "thinking",
                    "thought": True,
                    "thoughtSignature": "reasoning-signature",
                },
                {"text": "answer", "thoughtSignature": "text-signature"},
                {
                    "functionCall": {
                        "id": "call_1",
                        "name": "weather",
                        "args": {"city": "London"},
                    },
                    "thoughtSignature": "tool-signature",
                },
                {
                    "functionCall": {"id": "call_2", "name": "clock", "args": {}},
                    "thoughtSignature": SKIP_THOUGHT_SIGNATURE_VALIDATOR,
                },
            ],
        }
    ]


def test_google_openapi_schema_converts_nullable_const_and_strips_keywords() -> None:
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "unit": {"const": "celsius", "default": "celsius"},
            "note": {
                "anyOf": [
                    {"type": "string", "minLength": 1, "pattern": "^[a-z]+$"},
                    {"type": "null"},
                ]
            },
            "choice": {"type": ["string", "integer", "null"]},
            "metadata": {
                "type": "object",
                "description": "Free-form metadata",
                "additionalProperties": True,
            },
        },
        "required": ["unit"],
    }

    converted = google_openapi_schema(schema)

    assert converted == {
        "type": "object",
        "properties": {
            "unit": {"enum": ["celsius"]},
            "note": {"nullable": True, "type": "string", "minLength": 1},
            "choice": {
                "anyOf": [{"type": "string"}, {"type": "integer"}],
                "nullable": True,
            },
            "metadata": {"description": "Free-form metadata", "type": "object"},
        },
        "required": ["unit"],
    }


def test_google_openapi_schema_omits_empty_root_but_preserves_nested_object() -> None:
    assert google_openapi_schema({"type": "object", "properties": {}}) is None
    assert google_openapi_schema({"type": "object", "additionalProperties": {}}) == {
        "type": "object"
    }
    assert google_openapi_schema(
        {
            "type": "object",
            "additionalProperties": True,
            "properties": {"empty": {"type": "object", "properties": {}}},
        }
    ) == {"type": "object", "properties": {"empty": {"type": "object"}}}
