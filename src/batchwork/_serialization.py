"""Private provider message, media, and tool serialization helpers."""

from __future__ import annotations

import base64
import json
import mimetypes
from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

from ._compatible_media import compatible_file_content
from ._google_schema import google_openapi_schema
from ._typing import is_string_mapping
from .errors import BatchworkError
from .types import BatchProvider, ModelKind


@runtime_checkable
class _Dumpable(Protocol):
    def model_dump(
        self, *, by_alias: bool = ..., exclude_none: bool = ...
    ) -> dict[str, object]: ...


Request = Mapping[str, object] | _Dumpable


def _provider_options(item: Mapping[str, object], provider: BatchProvider) -> dict[str, object]:
    raw = item.get("provider_options", item.get("providerOptions"))
    if not is_string_mapping(raw):
        return {}
    selected = raw.get(provider.value)
    return {key: value for key, value in selected.items()} if is_string_mapping(selected) else {}


def _mapped_options(options: Mapping[str, object], mapping: Mapping[str, str]) -> dict[str, object]:
    return {
        target: options[source]
        for source, target in mapping.items()
        if source in options and options[source] is not None
    }


def _wire(
    source: Mapping[str, object],
    provider: BatchProvider,
    native: Mapping[str, object],
) -> dict[str, object]:
    """Merge provider options while preserving typed wire invariants."""

    return {**_provider_options(source, provider), **native}


def _messages(item: Mapping[str, object]) -> list[dict[str, object]]:
    raw = item.get("messages")
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        result: list[dict[str, object]] = []
        for message in raw:
            if is_string_mapping(message):
                result.append({key: value for key, value in message.items()})
            elif isinstance(message, _Dumpable):
                result.append(message.model_dump(by_alias=False, exclude_none=True))
        return result
    prompt = item.get("prompt")
    return [{"role": "user", "content": prompt}] if isinstance(prompt, str) else []


def _source(value: object, provider: BatchProvider | None = None) -> tuple[str | None, bool]:
    """Return a JSON-safe media value and whether it contains base64 bytes."""

    if isinstance(value, bytes):
        return base64.b64encode(value).decode(), True
    if isinstance(value, str):
        return value, not value.startswith(("http://", "https://", "file-"))
    if is_string_mapping(value):
        kind = value.get("type")
        if kind == "data":
            data = value.get("data")
            if isinstance(data, bytes):
                return base64.b64encode(data).decode(), True
            if isinstance(data, str):
                return data, True
            return _source(data, provider)
        if kind == "url":
            url = value.get("url")
            return (str(url), False) if url is not None else (None, False)
        text = value.get("text")
        if kind == "text" and isinstance(text, str):
            return base64.b64encode(text.encode()).decode(), True
        if kind == "reference":
            reference = value.get("reference")
            if is_string_mapping(reference) and provider is not None:
                provider_reference = reference.get(provider.value)
                if isinstance(provider_reference, str):
                    return provider_reference, False
                raise BatchworkError(
                    f"batchwork: file reference has no {provider.value} provider value."
                )
        reference = value.get("value")
        if kind == "provider-file-id" and isinstance(reference, str):
            return reference, False
        if provider is not None and isinstance(value.get(provider.value), str):
            return str(value[provider.value]), False
    if value is not None:
        rendered = str(value)
        if rendered.startswith(("http://", "https://")):
            return rendered, False
    return None, False


def _tool_output(value: object, *, content_as_text: bool = False) -> str:
    if not is_string_mapping(value):
        return value if isinstance(value, str) else json.dumps(value, separators=(",", ":"))
    kind = value.get("type")
    if kind in {"text", "error-text"}:
        output = value.get("value")
        return output if isinstance(output, str) else ""
    if kind in {"json", "error-json"}:
        return json.dumps(value.get("value"), separators=(",", ":"))
    if kind == "execution-denied":
        reason = value.get("reason")
        return reason if isinstance(reason, str) else "Tool call execution denied."
    if kind == "content":
        content = value.get("value")
        if content_as_text and isinstance(content, Sequence):
            return "".join(
                str(part.get("text", ""))
                for part in content
                if is_string_mapping(part) and part.get("type") == "text"
            )
        return json.dumps(content if content is not None else [], separators=(",", ":"))
    return json.dumps(dict(value), separators=(",", ":"))


def _openai_response_output(value: object, provider: BatchProvider) -> object:
    if not is_string_mapping(value) or value.get("type") != "content":
        return _tool_output(value)
    content = value.get("value")
    if not isinstance(content, Sequence) or isinstance(content, (str, bytes, bytearray)):
        return []
    result: list[dict[str, object]] = []
    for part in content:
        if not is_string_mapping(part):
            continue
        part_kind = part.get("type")
        if part_kind == "text" and isinstance(part.get("text"), str):
            result.append({"type": "input_text", "text": part["text"]})
            continue
        normalized = dict(part)
        if part_kind == "file-data":
            normalized.update({"type": "file", "data": {"type": "data", "data": part.get("data")}})
        elif part_kind == "file-url":
            url = part.get("url")
            inferred_type = (
                mimetypes.guess_type(url)[0] if isinstance(url, str) else None
            ) or "application/octet-stream"
            normalized.update(
                {
                    "type": "file",
                    "data": {"type": "url", "url": url},
                    "media_type": part.get("media_type", part.get("mediaType", inferred_type)),
                }
            )
        elif part_kind == "file-id":
            file_id = part.get("file_id", part.get("fileId"))
            reference = {provider.value: file_id} if isinstance(file_id, str) else file_id
            normalized.update(
                {
                    "type": "file",
                    "data": {"type": "reference", "reference": reference},
                    "media_type": "application",
                }
            )
        elif part_kind == "file-reference":
            normalized.update(
                {
                    "type": "file",
                    "data": {
                        "type": "reference",
                        "reference": part.get("provider_reference", part.get("providerReference")),
                    },
                    "media_type": "application",
                }
            )
        elif part_kind == "image-data":
            normalized.update({"type": "file", "data": {"type": "data", "data": part.get("data")}})
        elif part_kind == "image-url":
            normalized.update(
                {
                    "type": "file",
                    "data": {"type": "url", "url": part.get("url")},
                    "media_type": "image",
                }
            )
        elif part_kind == "image-file-id":
            file_id = part.get("file_id", part.get("fileId"))
            reference = {provider.value: file_id} if isinstance(file_id, str) else file_id
            normalized.update(
                {
                    "type": "file",
                    "data": {"type": "reference", "reference": reference},
                    "media_type": "image",
                }
            )
        elif part_kind == "image-file-reference":
            normalized.update(
                {
                    "type": "file",
                    "data": {
                        "type": "reference",
                        "reference": part.get("provider_reference", part.get("providerReference")),
                    },
                    "media_type": "image",
                }
            )
        if normalized.get("type") != "file":
            continue
        data = normalized.get("data")
        if not is_string_mapping(data):
            continue
        media_type = normalized.get("media_type", normalized.get("mediaType"))
        if not isinstance(media_type, str):
            continue
        top_level = media_type.split("/", 1)[0]
        detail = _provider_options(normalized, provider).get("imageDetail")
        if data.get("type") == "data":
            source, _ = _source(data, provider)
            if source is None:
                continue
            url = f"data:{media_type};base64,{source}"
            if top_level == "image":
                result.append(
                    {
                        "type": "input_image",
                        "image_url": url,
                        **({"detail": detail} if detail is not None else {}),
                    }
                )
            else:
                result.append(
                    {
                        "type": "input_file",
                        "filename": normalized.get("filename", "data"),
                        "file_data": url,
                    }
                )
        elif data.get("type") == "url":
            url = str(data.get("url"))
            if top_level == "image":
                result.append(
                    {
                        "type": "input_image",
                        "image_url": url,
                        **({"detail": detail} if detail is not None else {}),
                    }
                )
            else:
                result.append({"type": "input_file", "file_url": url})
    return result


def _openai_content(
    content: object, provider: BatchProvider
) -> tuple[object, list[dict[str, object]]]:
    if isinstance(content, str):
        return content, []
    if not isinstance(content, Sequence) or isinstance(content, (bytes, bytearray)):
        return content, []
    parts: list[object] = []
    tool_calls: list[dict[str, object]] = []
    for index, part in enumerate(content):
        if not is_string_mapping(part):
            parts.append(part)
            continue
        kind = part.get("type")
        if kind == "text":
            parts.append(_wire(part, provider, {"type": "text", "text": part.get("text", "")}))
        elif kind == "image":
            source, inline = _source(part.get("image", part.get("data")), provider)
            media_type = part.get("media_type", part.get("mediaType", "image/png"))
            if source is not None:
                url = f"data:{media_type};base64,{source}" if inline else source
                parts.append(
                    _wire(part, provider, {"type": "image_url", "image_url": {"url": url}})
                )
        elif kind in {"file", "reasoning-file"}:
            if provider in {
                BatchProvider.GROQ,
                BatchProvider.MISTRAL,
                BatchProvider.TOGETHER,
            }:
                parts.append(compatible_file_content(part, provider))
                continue
            raw_data = part.get("data")
            source, inline = _source(raw_data, provider)
            if source is not None:
                media_type = part.get("media_type", part.get("mediaType"))
                top_level = media_type.split("/", 1)[0] if isinstance(media_type, str) else None
                if provider == BatchProvider.OPENAI and top_level == "image":
                    url = f"data:{media_type};base64,{source}" if inline else source
                    detail = _provider_options(part, provider).get("imageDetail")
                    parts.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": url,
                                **({"detail": detail} if detail is not None else {}),
                            },
                        }
                    )
                    continue
                if provider == BatchProvider.OPENAI and top_level == "audio" and inline:
                    audio_format = (
                        "wav"
                        if media_type == "audio/wav"
                        else "mp3"
                        if media_type in {"audio/mp3", "audio/mpeg"}
                        else None
                    )
                    if audio_format is not None:
                        parts.append(
                            {
                                "type": "input_audio",
                                "input_audio": {"data": source, "format": audio_format},
                            }
                        )
                    continue
                file_value: dict[str, object] = {}
                if inline:
                    file_value["file_data"] = (
                        f"data:{media_type};base64,{source}"
                        if provider == BatchProvider.OPENAI and isinstance(media_type, str)
                        else source
                    )
                else:
                    file_value["file_id"] = source
                if part.get("filename") is not None:
                    file_value["filename"] = part["filename"]
                elif (
                    provider == BatchProvider.OPENAI and inline and media_type == "application/pdf"
                ):
                    file_value["filename"] = f"part-{index}.pdf"
                parts.append(_wire(part, provider, {"type": "file", "file": file_value}))
        elif kind == "tool-call":
            tool_calls.append(
                _wire(
                    part,
                    provider,
                    {
                        "id": part.get("tool_call_id", part.get("toolCallId")),
                        "type": "function",
                        "function": {
                            "name": part.get("tool_name", part.get("toolName")),
                            "arguments": json.dumps(part.get("input"), separators=(",", ":")),
                        },
                    },
                )
            )
        elif kind in {"reasoning", "custom"}:
            # Provider-specific representations belong in provider_options.
            continue
        else:
            parts.append(dict(part))
    return parts or None, tool_calls


def _openai_messages(
    item: Mapping[str, object],
    provider: BatchProvider,
    *,
    include_system_setting: bool = True,
) -> list[dict[str, object]]:
    messages: list[dict[str, object]] = []
    system = item.get("system")
    if include_system_setting and isinstance(system, str):
        messages.insert(0, {"role": "system", "content": system})
    source_messages = _messages(item)
    for index, message in enumerate(source_messages):
        role = message.get("role")
        raw_content = message.get("content")
        if role == "tool" and isinstance(raw_content, Sequence):
            for result in raw_content:
                if is_string_mapping(result) and result.get("type") == "tool-result":
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": result.get("tool_call_id", result.get("toolCallId")),
                            "content": _tool_output(result.get("output")),
                        }
                        | (
                            {"name": result.get("tool_name", result.get("toolName"))}
                            if provider == BatchProvider.MISTRAL
                            else {}
                        )
                    )
            continue
        if (
            role == "assistant"
            and isinstance(raw_content, Sequence)
            and not isinstance(raw_content, (str, bytes, bytearray))
        ):
            text_parts: list[str] = []
            reasoning_parts: list[str] = []
            tool_calls: list[dict[str, object]] = []
            for part in raw_content:
                if not is_string_mapping(part):
                    continue
                kind = part.get("type")
                part_text = part.get("text")
                if kind == "text" and isinstance(part_text, str):
                    text_parts.append(part_text)
                elif kind == "reasoning" and isinstance(part_text, str):
                    reasoning_parts.append(part_text)
                    if provider == BatchProvider.MISTRAL:
                        text_parts.append(part_text)
                elif kind == "tool-call":
                    tool_calls.append(
                        {
                            "id": part.get("tool_call_id", part.get("toolCallId")),
                            "type": "function",
                            "function": {
                                "name": part.get("tool_name", part.get("toolName")),
                                "arguments": json.dumps(part.get("input"), separators=(",", ":")),
                            },
                        }
                    )
            normalized: dict[str, object] = {
                "role": "assistant",
                "content": "".join(text_parts),
            }
            if provider == BatchProvider.GROQ and reasoning_parts:
                normalized["reasoning"] = "".join(reasoning_parts)
            elif provider == BatchProvider.TOGETHER and reasoning_parts:
                normalized["reasoning_content"] = "".join(reasoning_parts)
            if tool_calls:
                normalized["tool_calls"] = tool_calls
            if provider == BatchProvider.MISTRAL and index == len(source_messages) - 1:
                normalized["prefix"] = True
            messages.append(normalized)
            continue
        content, tool_calls = _openai_content(raw_content, provider)
        if (
            role == "user"
            and provider != BatchProvider.MISTRAL
            and isinstance(content, list)
            and len(content) == 1
            and is_string_mapping(content[0])
            and content[0].get("type") == "text"
            and isinstance(content[0].get("text"), str)
        ):
            content = content[0]["text"]
        normalized: dict[str, object] = {
            "role": role if isinstance(role, str) else "user",
            "content": content,
        }
        if tool_calls:
            normalized["tool_calls"] = tool_calls
        messages.append(_wire(message, provider, normalized))
    return messages


def _xai_responses_input(item: Mapping[str, object]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    system = item.get("system")
    if isinstance(system, str):
        result.append({"role": "system", "content": system})
    for message in _messages(item):
        role = message.get("role")
        raw_content = message.get("content")
        if role == "system" and isinstance(raw_content, str):
            result.append({"role": "system", "content": raw_content})
            continue
        if role == "user":
            content = [raw_content] if isinstance(raw_content, str) else raw_content
            parts: list[dict[str, object]] = []
            if isinstance(content, Sequence):
                for part in content:
                    if isinstance(part, str):
                        parts.append({"type": "input_text", "text": part})
                    elif is_string_mapping(part) and part.get("type") == "text":
                        parts.append({"type": "input_text", "text": part.get("text", "")})
                    elif is_string_mapping(part) and part.get("type") == "image":
                        source, inline = _source(
                            part.get("image", part.get("data")), BatchProvider.XAI
                        )
                        if source is not None:
                            media_type = part.get("media_type", "image/png")
                            parts.append(
                                {
                                    "type": "input_image",
                                    "image_url": f"data:{media_type};base64,{source}"
                                    if inline
                                    else source,
                                }
                            )
                    elif is_string_mapping(part) and part.get("type") in {
                        "file",
                        "reasoning-file",
                    }:
                        source, inline = _source(part.get("data"), BatchProvider.XAI)
                        if source is not None and not inline:
                            data_source = part.get("data")
                            target = (
                                "file_id"
                                if is_string_mapping(data_source)
                                and data_source.get("type") in {"provider-file-id", "reference"}
                                else "file_url"
                            )
                            parts.append({"type": "input_file", target: source})
            result.append({"role": "user", "content": parts})
            continue
        if role == "assistant":
            content = [raw_content] if isinstance(raw_content, str) else raw_content
            if not isinstance(content, Sequence):
                continue
            for part in content:
                if isinstance(part, str):
                    result.append({"role": "assistant", "content": part})
                    continue
                if not is_string_mapping(part):
                    continue
                options = _provider_options(part, BatchProvider.XAI)
                item_id = options.get("itemId")
                kind = part.get("type")
                if kind == "text" and isinstance(part.get("text"), str):
                    entry: dict[str, object] = {
                        "role": "assistant",
                        "content": part["text"],
                    }
                    if isinstance(item_id, str):
                        entry["id"] = item_id
                    result.append(entry)
                elif kind == "tool-call" and not part.get(
                    "provider_executed", part.get("providerExecuted", False)
                ):
                    call_id = part.get("tool_call_id", part.get("toolCallId"))
                    result.append(
                        {
                            "type": "function_call",
                            "id": item_id if isinstance(item_id, str) else call_id,
                            "call_id": call_id,
                            "name": part.get("tool_name", part.get("toolName")),
                            "arguments": json.dumps(part.get("input"), separators=(",", ":")),
                            "status": "completed",
                        }
                    )
                elif kind == "reasoning":
                    encrypted = options.get("reasoningEncryptedContent")
                    if isinstance(item_id, str) or isinstance(encrypted, str):
                        reasoning: dict[str, object] = {
                            "type": "reasoning",
                            "id": item_id if isinstance(item_id, str) else "",
                            "summary": [],
                            "status": "completed",
                        }
                        text = part.get("text")
                        if isinstance(text, str) and text:
                            reasoning["summary"] = [{"type": "summary_text", "text": text}]
                        if isinstance(encrypted, str):
                            reasoning["encrypted_content"] = encrypted
                        result.append(reasoning)
            continue
        if role == "tool" and isinstance(raw_content, Sequence):
            for part in raw_content:
                if not is_string_mapping(part) or part.get("type") != "tool-result":
                    continue
                result.append(
                    {
                        "type": "function_call_output",
                        "call_id": part.get("tool_call_id", part.get("toolCallId")),
                        "output": _tool_output(part.get("output"), content_as_text=True),
                    }
                )
    return result


def _openai_responses_input(
    item: Mapping[str, object], provider: BatchProvider
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    options = _provider_options(item, provider)
    store = options.get("store", True) is not False
    has_conversation = options.get("conversation") is not None
    raw_tools = item.get("tools")
    custom_tool_names = (
        {
            definition.get("name")
            for definition in raw_tools
            if is_string_mapping(definition) and definition.get("id") == "openai.custom"
        }
        if isinstance(raw_tools, list)
        else set()
    )
    provider_tool_names = (
        {
            definition.get("name")
            for definition in raw_tools
            if is_string_mapping(definition) and definition.get("type") == "provider-defined"
        }
        if isinstance(raw_tools, list)
        else set()
    )
    referenced_reasoning_ids: set[str] = set()
    has_previous_response = options.get("previousResponseId") is not None
    system = item.get("system")
    if isinstance(system, str):
        result.append({"role": "system", "content": system})
    for message in _messages(item):
        role = message.get("role")
        raw_content = message.get("content")
        if role == "system" and isinstance(raw_content, str):
            result.append({"role": "system", "content": raw_content})
            continue
        content = [raw_content] if isinstance(raw_content, str) else raw_content
        if not isinstance(content, Sequence):
            continue
        message_parts: list[dict[str, object]] = []
        for index, part in enumerate(content):
            if isinstance(part, str):
                message_parts.append(
                    {
                        "type": "output_text" if role == "assistant" else "input_text",
                        "text": part,
                    }
                )
                continue
            if not is_string_mapping(part):
                continue
            kind = part.get("type")
            if kind == "text":
                if role == "assistant":
                    part_options = _provider_options(part, provider)
                    item_id = part_options.get("itemId")
                    if isinstance(item_id, str) and has_conversation:
                        continue
                    if isinstance(item_id, str) and store:
                        result.append({"type": "item_reference", "id": item_id})
                        continue
                    assistant_item: dict[str, object] = {
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": part.get("text", "")}],
                    }
                    if isinstance(item_id, str):
                        assistant_item["id"] = item_id
                    phase = part_options.get("phase")
                    if phase in {"commentary", "final_answer"}:
                        assistant_item["phase"] = phase
                    result.append(assistant_item)
                    continue
                message_parts.append(
                    _wire(
                        part,
                        provider,
                        {
                            "type": "output_text" if role == "assistant" else "input_text",
                            "text": part.get("text", ""),
                        },
                    )
                )
            elif kind == "image":
                source, inline = _source(part.get("image", part.get("data")), provider)
                if source is not None:
                    media_type = part.get("media_type", "image/png")
                    url = f"data:{media_type};base64,{source}" if inline else source
                    message_parts.append(
                        _wire(
                            part,
                            provider,
                            {"type": "input_image", "image_url": url},
                        )
                    )
            elif kind in {"file", "reasoning-file"}:
                raw_data = part.get("data")
                if is_string_mapping(raw_data) and raw_data.get("type") == "text":
                    raise BatchworkError("batchwork: OpenAI does not support text file parts.")
                source, inline = _source(raw_data, provider)
                if source is not None:
                    media_type = part.get("media_type", part.get("mediaType"))
                    top_level = media_type.split("/", 1)[0] if isinstance(media_type, str) else None
                    reference = (
                        is_string_mapping(raw_data)
                        and raw_data.get("type")
                        in {
                            "reference",
                            "provider-file-id",
                        }
                    ) or (
                        is_string_mapping(raw_data)
                        and isinstance(raw_data.get(provider.value), str)
                    )
                    url_source = (
                        is_string_mapping(raw_data) and raw_data.get("type") == "url"
                    ) or str(raw_data).startswith(("http://", "https://"))
                    input_part = dict[str, object]()
                    if top_level == "image":
                        input_part["type"] = "input_image"
                        target = "file_id" if reference else "image_url"
                        input_part[target] = (
                            f"data:{media_type};base64,{source}" if inline else source
                        )
                    else:
                        input_part["type"] = "input_file"
                        if reference:
                            input_part["file_id"] = source
                        elif url_source:
                            input_part["file_url"] = source
                        else:
                            input_part["file_data"] = (
                                f"data:{media_type};base64,{source}"
                                if isinstance(media_type, str)
                                else source
                            )
                            default_filename = (
                                f"part-{index}.pdf"
                                if media_type == "application/pdf"
                                else f"part-{index}"
                            )
                            filename = part.get("filename")
                            input_part["filename"] = (
                                filename if isinstance(filename, str) else default_filename
                            )
                    message_parts.append(_wire(part, provider, input_part))
            elif kind == "tool-call":
                part_options = _provider_options(part, provider)
                item_id = part_options.get("itemId")
                provider_executed = part.get(
                    "provider_executed", part.get("providerExecuted", False)
                )
                tool_name = part.get("tool_name", part.get("toolName"))
                if isinstance(item_id, str) and has_conversation:
                    continue
                if provider_executed:
                    if store and isinstance(item_id, str):
                        result.append({"type": "item_reference", "id": item_id})
                    continue
                if store and isinstance(item_id, str) and tool_name in provider_tool_names:
                    result.append({"type": "item_reference", "id": item_id})
                    continue
                call: dict[str, object] = {
                    "type": "function_call",
                    "call_id": part.get("tool_call_id", part.get("toolCallId")),
                    "name": tool_name,
                    "arguments": json.dumps(part.get("input"), separators=(",", ":")),
                }
                namespace = part_options.get("namespace")
                if isinstance(namespace, str):
                    call["namespace"] = namespace
                result.append(call)
            elif kind == "tool-result":
                output = part.get("output")
                if is_string_mapping(output) and output.get("type") == "execution-denied":
                    output_options = _provider_options(output, provider)
                    if role == "assistant" or isinstance(output_options.get("approvalId"), str):
                        continue
                tool_name = part.get("tool_name", part.get("toolName"))
                if role == "assistant":
                    if has_conversation:
                        continue
                    item_id = _provider_options(part, provider).get("itemId")
                    if store:
                        result.append(
                            {
                                "type": "item_reference",
                                "id": item_id
                                if isinstance(item_id, str)
                                else part.get("tool_call_id", part.get("toolCallId")),
                            }
                        )
                    continue
                result.append(
                    _wire(
                        part,
                        provider,
                        {
                            "type": (
                                "custom_tool_call_output"
                                if tool_name in custom_tool_names
                                else "function_call_output"
                            ),
                            "call_id": part.get("tool_call_id", part.get("toolCallId")),
                            "output": _openai_response_output(output, provider),
                        },
                    )
                )
            elif kind == "reasoning" and role == "assistant":
                part_options = _provider_options(part, provider)
                item_id = part_options.get("itemId")
                encrypted = part_options.get("reasoningEncryptedContent")
                if isinstance(item_id, str) and (has_conversation or has_previous_response):
                    continue
                if isinstance(item_id, str) and store:
                    if item_id not in referenced_reasoning_ids:
                        result.append({"type": "item_reference", "id": item_id})
                        referenced_reasoning_ids.add(item_id)
                    continue
                if isinstance(item_id, str) or isinstance(encrypted, str):
                    summary = []
                    text = part.get("text")
                    if isinstance(text, str) and text:
                        summary.append({"type": "summary_text", "text": text})
                    reasoning: dict[str, object] = {
                        "type": "reasoning",
                        "summary": summary,
                    }
                    if isinstance(item_id, str):
                        reasoning["id"] = item_id
                    if isinstance(encrypted, str):
                        reasoning["encrypted_content"] = encrypted
                    result.append(reasoning)
            elif kind == "tool-approval-response":
                provider_executed = part.get("provider_executed", part.get("providerExecuted"))
                if provider_executed is not True:
                    continue
                approval_id = part.get("approval_id", part.get("approvalId"))
                if isinstance(approval_id, str):
                    if store:
                        result.append({"type": "item_reference", "id": approval_id})
                    result.append(
                        {
                            "type": "mcp_approval_response",
                            "approval_request_id": approval_id,
                            "approve": part.get("approved", False),
                        }
                    )
            elif kind == "custom" and part.get("kind") == "openai.compaction":
                part_options = _provider_options(part, provider)
                item_id = part_options.get("itemId")
                if not isinstance(item_id, str) or has_conversation:
                    continue
                if store:
                    result.append({"type": "item_reference", "id": item_id})
                    continue
                encrypted = part_options.get("encryptedContent")
                result.append(
                    {
                        "type": "compaction",
                        "id": item_id,
                        **({"encrypted_content": encrypted} if isinstance(encrypted, str) else {}),
                    }
                )
        if message_parts:
            result.append(
                _wire(
                    message,
                    provider,
                    {
                        "role": role if isinstance(role, str) else "user",
                        "content": message_parts,
                    },
                )
            )
    return result


def _anthropic_cache_control(part: Mapping[str, object]) -> object | None:
    return _provider_options(part, BatchProvider.ANTHROPIC).get("cacheControl")


def _anthropic_file_block(
    part: Mapping[str, object], *, tool_output: bool = False
) -> dict[str, object] | None:
    raw_data = part.get("data")
    media_type = part.get("media_type", part.get("mediaType"))
    if not isinstance(media_type, str):
        media_type = "application/octet-stream"
    top_level = media_type.split("/", 1)[0]
    tagged_kind = raw_data.get("type") if is_string_mapping(raw_data) else None
    source, inline = _source(raw_data, BatchProvider.ANTHROPIC)
    if source is None:
        return None

    if tagged_kind == "reference" or tagged_kind == "provider-file-id":
        if tool_output:
            return None
        block_type = "image" if top_level == "image" else "document"
        return {
            "type": block_type,
            "source": {"type": "file", "file_id": source},
        }

    if tagged_kind == "text":
        if tool_output:
            return None
        text = raw_data.get("text") if is_string_mapping(raw_data) else None
        if not isinstance(text, str):
            return None
        return {
            "type": "document",
            "source": {"type": "text", "media_type": "text/plain", "data": text},
        }

    if not inline:
        return {
            "type": "image" if top_level == "image" else "document",
            "source": {"type": "url", "url": source},
        }

    if top_level == "image":
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": source},
        }
    if media_type == "application/pdf":
        return {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": "application/pdf",
                "data": source,
            },
        }
    if media_type == "text/plain" and not tool_output:
        try:
            text = base64.b64decode(source, validate=True).decode()
        except (ValueError, UnicodeDecodeError) as error:
            raise BatchworkError(
                "batchwork: Anthropic text file data must be valid base64-encoded UTF-8."
            ) from error
        return {
            "type": "document",
            "source": {"type": "text", "media_type": "text/plain", "data": text},
        }
    if tool_output:
        return None
    raise BatchworkError(f"batchwork: Anthropic does not support media type {media_type!r}.")


def _anthropic_tool_output(value: object) -> tuple[object, bool]:
    if not is_string_mapping(value):
        return _tool_output(value), False
    kind = value.get("type")
    if kind == "content":
        raw_content = value.get("value")
        if not isinstance(raw_content, Sequence) or isinstance(
            raw_content, (str, bytes, bytearray)
        ):
            return [], False
        content: list[object] = []
        for part in raw_content:
            if not is_string_mapping(part):
                continue
            part_kind = part.get("type")
            if part_kind == "text":
                content.append({"type": "text", "text": part.get("text", "")})
                continue
            if part_kind == "file":
                block = _anthropic_file_block(part, tool_output=True)
                if block is not None:
                    content.append(block)
                continue
            if part_kind == "custom":
                options = _provider_options(part, BatchProvider.ANTHROPIC)
                tool_name = options.get("toolName")
                if options.get("type") == "tool-reference" and isinstance(tool_name, str):
                    content.append({"type": "tool_reference", "tool_name": tool_name})
        return content, False
    return _tool_output(value), kind in {"error-text", "error-json"}


def _anthropic_content(content: object) -> object:
    if isinstance(content, str):
        return content
    if not isinstance(content, Sequence) or isinstance(content, (bytes, bytearray)):
        return content
    blocks: list[object] = []
    for part in content:
        if not is_string_mapping(part):
            blocks.append(part)
            continue
        kind = part.get("type")
        if kind == "image":
            data, inline = _source(part.get("image", part.get("data")), BatchProvider.ANTHROPIC)
            media_type = part.get("media_type", part.get("mediaType", "image/png"))
            if data is not None and inline:
                blocks.append(
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": media_type, "data": data},
                    }
                )
                continue
            if data is not None:
                blocks.append({"type": "image", "source": {"type": "url", "url": data}})
                continue
        if kind in {"file", "reasoning-file"}:
            block = _anthropic_file_block(part)
            if block is not None:
                options = _provider_options(part, BatchProvider.ANTHROPIC)
                filename = part.get("filename")
                if block["type"] == "document":
                    title = options.get("title", filename)
                    if title is not None:
                        block["title"] = title
                    if options.get("context") is not None:
                        block["context"] = options["context"]
                    citations = options.get("citations")
                    if is_string_mapping(citations) and citations.get("enabled") is True:
                        block["citations"] = {"enabled": True}
                cache_control = _anthropic_cache_control(part)
                if cache_control is not None:
                    block["cache_control"] = cache_control
                blocks.append(block)
                continue
        if kind == "tool-call":
            blocks.append(
                {
                    "type": "tool_use",
                    "id": part.get("tool_call_id", part.get("toolCallId")),
                    "name": part.get("tool_name", part.get("toolName")),
                    "input": part.get("input", {}),
                }
            )
            continue
        if kind == "tool-result":
            output = part.get("output", part.get("content", ""))
            serialized, is_error = _anthropic_tool_output(output)
            result: dict[str, object] = {
                "type": "tool_result",
                "tool_use_id": part.get("tool_call_id", part.get("toolCallId")),
                "content": serialized,
            }
            if is_error:
                result["is_error"] = True
            cache_control = _anthropic_cache_control(part)
            if cache_control is None and is_string_mapping(output):
                cache_control = _anthropic_cache_control(output)
            if cache_control is not None:
                result["cache_control"] = cache_control
            blocks.append(result)
            continue
        if kind in {"reasoning", "custom"}:
            continue
        native = {key: value for key, value in part.items() if key != "provider_options"}
        blocks.append(_wire(part, BatchProvider.ANTHROPIC, native))
    return blocks


def _anthropic_messages(item: Mapping[str, object]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for message in _messages(item):
        role = message.get("role")
        if role == "system":
            continue
        normalized = "assistant" if role == "assistant" else "user"
        result.append(
            _wire(
                message,
                BatchProvider.ANTHROPIC,
                {"role": normalized, "content": _anthropic_content(message.get("content"))},
            )
        )
    return result


def _defined_tool(
    provider: BatchProvider,
    identifier: str,
    name: object,
    args: Mapping[str, object],
    model_id: str,
    kind: ModelKind,
) -> dict[str, object] | None:
    if provider == BatchProvider.ANTHROPIC:
        tool_type = identifier.removeprefix("anthropic.")
        fixed_names = {
            "code_execution_20250522": "code_execution",
            "code_execution_20250825": "code_execution",
            "code_execution_20260120": "code_execution",
            "computer_20241022": "computer",
            "computer_20250124": "computer",
            "computer_20251124": "computer",
            "text_editor_20241022": "str_replace_editor",
            "text_editor_20250124": "str_replace_editor",
            "text_editor_20250429": "str_replace_based_edit_tool",
            "text_editor_20250728": "str_replace_based_edit_tool",
            "bash_20241022": "bash",
            "bash_20250124": "bash",
            "memory_20250818": "memory",
            "web_fetch_20250910": "web_fetch",
            "web_fetch_20260209": "web_fetch",
            "web_search_20250305": "web_search",
            "web_search_20260209": "web_search",
            "tool_search_regex_20251119": "tool_search_tool_regex",
            "tool_search_bm25_20251119": "tool_search_tool_bm25",
            "advisor_20260301": "advisor",
        }
        if tool_type not in fixed_names:
            return None
        fixed_name = fixed_names[tool_type]
        if tool_type == "tool_search_regex_20251119":
            tool_type = "tool_search_tool_regex_20251119"
        elif tool_type == "tool_search_bm25_20251119":
            tool_type = "tool_search_tool_bm25_20251119"
        wire: dict[str, object] = {"type": tool_type, "name": fixed_name}
        mappings = {
            "displayWidthPx": "display_width_px",
            "displayHeightPx": "display_height_px",
            "displayNumber": "display_number",
            "enableZoom": "enable_zoom",
            "maxCharacters": "max_characters",
            "maxUses": "max_uses",
            "allowedDomains": "allowed_domains",
            "blockedDomains": "blocked_domains",
            "userLocation": "user_location",
            "citations": "citations",
            "maxContentTokens": "max_content_tokens",
            "model": "model",
            "caching": "caching",
        }
        wire.update(_mapped_options(args, mappings))
        return wire
    if provider == BatchProvider.GOOGLE:
        latest = model_id in {
            "gemini-flash-latest",
            "gemini-flash-lite-latest",
            "gemini-pro-latest",
        }
        modern = any(value in model_id for value in ("gemini-2", "gemini-3", "nano-banana"))
        modern = modern or latest
        if not modern:
            return None
        if identifier == "google.google_search":
            return {"googleSearch": dict(args)}
        if identifier == "google.enterprise_web_search":
            return {"enterpriseWebSearch": {}}
        if identifier == "google.url_context":
            return {"urlContext": {}}
        if identifier == "google.code_execution":
            return {"codeExecution": {}}
        if identifier == "google.file_search" and any(
            value in model_id for value in ("gemini-2.5", "gemini-3")
        ):
            return {"fileSearch": dict(args)}
        if identifier == "google.vertex_rag_store":
            return {
                "retrieval": {
                    "vertex_rag_store": {
                        "rag_resources": {"rag_corpus": args.get("ragCorpus")},
                        **(
                            {"similarity_top_k": args["topK"]}
                            if args.get("topK") is not None
                            else {}
                        ),
                    }
                }
            }
        if identifier == "google.google_maps":
            return {"googleMaps": {}}
        return None
    if provider == BatchProvider.GROQ:
        if identifier == "groq.browser_search" and model_id in {
            "openai/gpt-oss-20b",
            "openai/gpt-oss-120b",
        }:
            return {"type": "browser_search"}
        return None
    if provider == BatchProvider.OPENAI:
        if kind is not ModelKind.RESPONSES or not identifier.startswith("openai."):
            return None
        tool_type = identifier.removeprefix("openai.")
        if tool_type in {"local_shell", "apply_patch"}:
            return {"type": tool_type}
        if tool_type == "file_search":
            ranking = args.get("ranking")
            return {
                "type": tool_type,
                "vector_store_ids": args.get("vectorStoreIds"),
                **({"max_num_results": args["maxNumResults"]} if "maxNumResults" in args else {}),
                **(
                    {
                        "ranking_options": {
                            "ranker": ranking.get("ranker"),
                            "score_threshold": ranking.get("scoreThreshold"),
                        }
                    }
                    if is_string_mapping(ranking)
                    else {}
                ),
                **({"filters": args["filters"]} if "filters" in args else {}),
            }
        if tool_type in {"web_search_preview", "web_search"}:
            wire = {
                "type": tool_type,
                **_mapped_options(
                    args,
                    {
                        "searchContextSize": "search_context_size",
                        "userLocation": "user_location",
                        "externalWebAccess": "external_web_access",
                    },
                ),
            }
            filters = args.get("filters")
            if tool_type == "web_search" and is_string_mapping(filters):
                wire["filters"] = {"allowed_domains": filters.get("allowedDomains")}
            return wire
        if tool_type == "code_interpreter":
            container = args.get("container")
            if container is None:
                container = {"type": "auto"}
            elif is_string_mapping(container):
                container = {"type": "auto", "file_ids": container.get("fileIds")}
            return {"type": tool_type, "container": container}
        if tool_type == "custom":
            return {
                "type": tool_type,
                "name": name,
                **_mapped_options(args, {"description": "description", "format": "format"}),
            }
        if tool_type == "tool_search":
            return {
                "type": tool_type,
                **_mapped_options(
                    args,
                    {
                        "execution": "execution",
                        "description": "description",
                        "parameters": "parameters",
                    },
                ),
            }
        if tool_type == "shell":
            environment = args.get("environment")
            mapped_environment: object = environment
            if is_string_mapping(environment):
                environment_type = environment.get("type")
                if environment_type == "containerReference":
                    mapped_environment = {
                        "type": "container_reference",
                        "container_id": environment.get("containerId"),
                    }
                elif environment_type == "containerAuto":
                    mapped_environment = {
                        "type": "container_auto",
                        **_mapped_options(
                            environment,
                            {"fileIds": "file_ids", "memoryLimit": "memory_limit"},
                        ),
                    }
            return {
                "type": tool_type,
                **({"environment": mapped_environment} if mapped_environment is not None else {}),
            }
        if tool_type == "image_generation":
            wire = {
                "type": tool_type,
                **_mapped_options(
                    args,
                    {
                        "background": "background",
                        "inputFidelity": "input_fidelity",
                        "model": "model",
                        "moderation": "moderation",
                        "partialImages": "partial_images",
                        "quality": "quality",
                        "outputCompression": "output_compression",
                        "outputFormat": "output_format",
                        "size": "size",
                    },
                ),
            }
            mask = args.get("inputImageMask")
            if is_string_mapping(mask):
                wire["input_image_mask"] = {
                    **_mapped_options(mask, {"fileId": "file_id", "imageUrl": "image_url"})
                }
            return wire
        if tool_type == "mcp":
            allowed = args.get("allowedTools")
            mapped_allowed: object = allowed
            if is_string_mapping(allowed):
                mapped_allowed = {
                    **_mapped_options(allowed, {"readOnly": "read_only", "toolNames": "tool_names"})
                }
            require_approval = args.get("requireApproval")
            if is_string_mapping(require_approval):
                never = require_approval.get("never")
                require_approval = (
                    {"never": {"tool_names": never.get("toolNames")}}
                    if is_string_mapping(never)
                    else None
                )
            return {
                "type": tool_type,
                "require_approval": require_approval or "never",
                **({"allowed_tools": mapped_allowed} if mapped_allowed is not None else {}),
                **_mapped_options(
                    args,
                    {
                        "authorization": "authorization",
                        "connectorId": "connector_id",
                        "headers": "headers",
                        "serverDescription": "server_description",
                        "serverLabel": "server_label",
                        "serverUrl": "server_url",
                    },
                ),
            }
        return None
    if provider == BatchProvider.XAI:
        mappings = {
            "xai.web_search": (
                "web_search",
                {
                    "allowedDomains": "allowed_domains",
                    "excludedDomains": "excluded_domains",
                    "enableImageSearch": "enable_image_search",
                    "enableImageUnderstanding": "enable_image_understanding",
                },
            ),
            "xai.x_search": (
                "x_search",
                {
                    "allowedXHandles": "allowed_x_handles",
                    "excludedXHandles": "excluded_x_handles",
                    "fromDate": "from_date",
                    "toDate": "to_date",
                    "enableImageUnderstanding": "enable_image_understanding",
                    "enableVideoUnderstanding": "enable_video_understanding",
                },
            ),
            "xai.file_search": (
                "file_search",
                {"vectorStoreIds": "vector_store_ids", "maxNumResults": "max_num_results"},
            ),
            "xai.mcp": (
                "mcp",
                {
                    "serverUrl": "server_url",
                    "serverLabel": "server_label",
                    "serverDescription": "server_description",
                    "allowedTools": "allowed_tools",
                    "headers": "headers",
                    "authorization": "authorization",
                },
            ),
        }
        fixed = {
            "xai.code_execution": "code_interpreter",
            "xai.view_image": "view_image",
            "xai.view_x_video": "view_x_video",
        }
        if identifier in fixed:
            return {"type": fixed[identifier]}
        mapping = mappings.get(identifier)
        if mapping is not None:
            tool_type, fields = mapping
            return {"type": tool_type, **_mapped_options(args, fields)}
    return None


def _tools(
    item: Mapping[str, object],
    provider: BatchProvider,
    model_id: str,
    kind: ModelKind,
) -> list[dict[str, object]] | None:
    raw = item.get("tools")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        return None
    tools: list[dict[str, object]] = []
    for definition in raw:
        if not is_string_mapping(definition):
            continue
        if definition.get("type") == "provider-defined":
            identifier = definition.get("id")
            name = definition.get("name")
            args = definition.get("args")
            options = {key: value for key, value in args.items()} if is_string_mapping(args) else {}
            if isinstance(identifier, str):
                native = _defined_tool(provider, identifier, name, options, model_id, kind)
                if native is not None:
                    tools.append(native)
            continue
        name = definition.get("name")
        if not isinstance(name, str):
            continue
        description = definition.get("description")
        schema = definition.get(
            "input_schema", definition.get("inputSchema", definition.get("parameters", {}))
        )
        if provider == BatchProvider.GOOGLE:
            schema = google_openapi_schema(schema)
        strict = definition.get("strict")
        function: dict[str, object] = {
            "name": name,
            "parameters": schema,
            **(
                {"description": "" if description is None else description}
                if provider == BatchProvider.GOOGLE
                else {"description": description}
                if description is not None
                else {}
            ),
            **({"strict": strict} if isinstance(strict, bool) else {}),
        }
        tools.append(
            {
                "type": "function",
                "function": function,
                **(
                    {"_input_examples": input_examples}
                    if (
                        input_examples := definition.get(
                            "input_examples", definition.get("inputExamples")
                        )
                    )
                    is not None
                    else {}
                ),
                **(
                    {"_provider_options": provider_options}
                    if (
                        provider_options := definition.get(
                            "provider_options", definition.get("providerOptions")
                        )
                    )
                    is not None
                    else {}
                ),
            }
        )
    return tools or None


def _response_function_tool(
    function: Mapping[str, object],
    tool: Mapping[str, object],
    provider: BatchProvider,
) -> dict[str, object]:
    raw_options = tool.get("_provider_options")
    options = (
        _provider_options({"provider_options": raw_options}, provider)
        if is_string_mapping(raw_options)
        else {}
    )
    parameters = function.get("parameters", {})
    if provider == BatchProvider.XAI:
        parameters = _without_additional_properties_false(parameters)
    return {
        "type": "function",
        "name": function.get("name"),
        "description": function.get("description"),
        "parameters": parameters,
        **({"strict": function["strict"]} if isinstance(function.get("strict"), bool) else {}),
        **(
            {"defer_loading": options["deferLoading"]}
            if provider == BatchProvider.OPENAI and options.get("deferLoading") is not None
            else {}
        ),
    }


def _without_additional_properties_false(value: object) -> object:
    if is_string_mapping(value):
        return {
            key: _without_additional_properties_false(item)
            for key, item in value.items()
            if not (key == "additionalProperties" and item is False)
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_without_additional_properties_false(item) for item in value]
    return value


def _chat_tool(tool: Mapping[str, object]) -> dict[str, object]:
    return {key: value for key, value in tool.items() if not key.startswith("_")}
