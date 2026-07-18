"""Private Google message and tool-result serialization helpers."""

from __future__ import annotations

import json
import mimetypes
from collections.abc import Mapping, Sequence

from ._serialization import _messages, _provider_options, _source, _wire
from ._typing import is_string_mapping
from .types import BatchProvider

SKIP_THOUGHT_SIGNATURE_VALIDATOR = "skip_thought_signature_validator"


def _thought_signature(part: Mapping[str, object]) -> str | None:
    value = _provider_options(part, BatchProvider.GOOGLE).get("thoughtSignature")
    return str(value) if value is not None else None


def _with_signature(
    converted: dict[str, object], part: Mapping[str, object], *, fallback: str | None = None
) -> dict[str, object]:
    signature = _thought_signature(part)
    if signature is None:
        signature = fallback
    return {**converted, **({"thoughtSignature": signature} if signature is not None else {})}


def _tool_file(part: Mapping[str, object]) -> tuple[str, bool, str] | None:
    kind = part.get("type")
    media_type = part.get("media_type", part.get("mediaType"))
    raw_data = part.get("data")
    if kind == "file-data":
        raw_data = {"type": "data", "data": raw_data}
    elif kind == "file-url":
        raw_data = {"type": "url", "url": part.get("url")}
        url = part.get("url")
        media_type = media_type or (mimetypes.guess_type(url)[0] if isinstance(url, str) else None)
    elif kind == "image-data":
        raw_data = {"type": "data", "data": raw_data}
    elif kind == "image-url":
        raw_data = {"type": "url", "url": part.get("url")}
        media_type = "image"
    elif kind != "file":
        return None
    if not isinstance(media_type, str):
        media_type = "application/octet-stream"
    source, inline = _source(raw_data, BatchProvider.GOOGLE)
    return (source, inline, media_type) if source is not None else None


def _tool_content(
    name: object, call_id: object, value: object, *, supports_parts: bool
) -> list[object]:
    if (
        not isinstance(name, str)
        or not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
    ):
        return []
    if not supports_parts:
        result: list[object] = []
        for item in value:
            if is_string_mapping(item) and item.get("type") == "text":
                result.append(
                    {
                        "functionResponse": {
                            "id": call_id,
                            "name": name,
                            "response": {"name": name, "content": item.get("text", "")},
                        }
                    }
                )
                continue
            file = _tool_file(item) if is_string_mapping(item) else None
            if file is not None and file[1] and file[2].split("/", 1)[0] == "image":
                result.extend(
                    [
                        {"inlineData": {"mimeType": file[2], "data": file[0]}},
                        {
                            "text": (
                                "Tool executed successfully and returned this image as a response"
                            )
                        },
                    ]
                )
            elif item is not None:
                result.append({"text": json.dumps(item, separators=(",", ":"))})
        return result
    text: list[str] = []
    response_parts: list[object] = []
    for item in value:
        if is_string_mapping(item) and item.get("type") == "text":
            text.append(str(item.get("text", "")))
            continue
        file = _tool_file(item) if is_string_mapping(item) else None
        if file is not None and file[1]:
            response_parts.append({"inlineData": {"mimeType": file[2], "data": file[0]}})
        elif item is not None:
            text.append(json.dumps(item, separators=(",", ":")))
    function_response: dict[str, object] = {
        "id": call_id,
        "name": name,
        "response": {
            "name": name,
            "content": "\n".join(text) if text else "Tool executed successfully.",
        },
    }
    if response_parts:
        function_response["parts"] = response_parts
    return [{"functionResponse": function_response}]


def _part(
    part: object,
    *,
    assistant: bool,
    gemini_3: bool,
    supports_response_parts: bool,
) -> object | list[object]:
    if isinstance(part, str):
        return {"text": part}
    if not is_string_mapping(part):
        return part
    kind = part.get("type")
    if kind == "text":
        converted = {"text": part.get("text", "")}
        return _with_signature(converted, part) if assistant else converted
    if kind == "reasoning":
        return _with_signature({"text": part.get("text", ""), "thought": True}, part)
    if kind in {"image", "file", "reasoning-file"}:
        raw_source = part.get("image") if kind == "image" else part.get("data")
        source, inline = _source(raw_source, BatchProvider.GOOGLE)
        media_type = part.get("media_type", part.get("mediaType", "application/octet-stream"))
        if source is not None and inline:
            converted = {"inlineData": {"mimeType": media_type, "data": source}}
            if kind == "reasoning-file":
                converted["thought"] = True
            return _with_signature(converted, part) if assistant else converted
        if source is not None:
            converted = {"fileData": {"mimeType": media_type, "fileUri": source}}
            if kind == "reasoning-file":
                converted["thought"] = True
            return _with_signature(converted, part) if assistant else converted
    if kind == "tool-call":
        converted = {
            "functionCall": {
                "id": part.get("tool_call_id", part.get("toolCallId")),
                "name": part.get("tool_name", part.get("toolName")),
                "args": part.get("input", {}),
            }
        }
        fallback = SKIP_THOUGHT_SIGNATURE_VALIDATOR if gemini_3 else None
        return _with_signature(converted, part, fallback=fallback)
    if kind == "tool-result":
        name = part.get("tool_name", part.get("toolName"))
        output = part.get("output", part.get("content", {}))
        if is_string_mapping(output) and output.get("type") == "content":
            return _tool_content(
                name,
                part.get("tool_call_id", part.get("toolCallId")),
                output.get("value"),
                supports_parts=supports_response_parts,
            )
        if is_string_mapping(output):
            content = (
                output.get("reason") or "Tool call execution denied."
                if output.get("type") == "execution-denied"
                else output.get("value")
            )
        else:
            content = output
        return {
            "functionResponse": {
                "id": part.get("tool_call_id", part.get("toolCallId")),
                "name": name,
                "response": {"name": name, "content": content},
            }
        }
    native = {key: value for key, value in part.items() if key != "provider_options"}
    return _wire(part, BatchProvider.GOOGLE, native)


def google_messages(
    item: Mapping[str, object], model_id: str
) -> tuple[list[dict[str, object]], dict[str, object] | None]:
    contents: list[dict[str, object]] = []
    system_parts: list[dict[str, str]] = []
    system = item.get("system")
    if isinstance(system, str):
        system_parts.append({"text": system})
    normalized_model_id = model_id.lower()
    gemini_3 = "gemini-3" in normalized_model_id
    is_gemma = normalized_model_id.startswith("gemma-")
    supports_response_parts = gemini_3
    for message in _messages(item):
        role = message.get("role")
        if role == "system":
            content = message.get("content")
            if isinstance(content, str):
                system_parts.append({"text": content})
            continue
        content = message.get("content")
        source_parts = (
            content
            if isinstance(content, Sequence) and not isinstance(content, (str, bytes, bytearray))
            else [content]
        )
        parts: list[object] = []
        for part in source_parts:
            converted = _part(
                part,
                assistant=role == "assistant",
                gemini_3=gemini_3,
                supports_response_parts=supports_response_parts,
            )
            parts.extend(converted if isinstance(converted, list) else [converted])
        contents.append(
            _wire(
                message,
                BatchProvider.GOOGLE,
                {"role": "model" if role == "assistant" else "user", "parts": parts},
            )
        )
    if is_gemma:
        if system_parts and contents and contents[0].get("role") == "user":
            system_text = "\n\n".join(part["text"] for part in system_parts)
            raw_parts = contents[0].get("parts")
            if isinstance(raw_parts, list):
                prefixed_parts: list[object] = [{"text": f"{system_text}\n\n"}]
                prefixed_parts.extend(raw_parts)
                contents[0]["parts"] = prefixed_parts
        return contents, None
    system_instruction: dict[str, object] | None = {"parts": system_parts} if system_parts else None
    return contents, system_instruction


__all__ = ["SKIP_THOUGHT_SIGNATURE_VALIDATOR", "google_messages"]
