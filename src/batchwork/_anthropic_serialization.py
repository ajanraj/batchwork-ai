"""Anthropic-native prompt history serialization."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ._serialization import (
    _anthropic_file_block,
    _anthropic_tool_output,
    _messages,
    _provider_options,
    _source,
)
from ._typing import is_string_mapping
from .types import BatchProvider

_PROVIDER_TOOL_NAMES = {
    "anthropic.code_execution_20250522": "code_execution",
    "anthropic.code_execution_20250825": "code_execution",
    "anthropic.code_execution_20260120": "code_execution",
    "anthropic.computer_20241022": "computer",
    "anthropic.computer_20250124": "computer",
    "anthropic.computer_20251124": "computer",
    "anthropic.text_editor_20241022": "str_replace_editor",
    "anthropic.text_editor_20250124": "str_replace_editor",
    "anthropic.text_editor_20250429": "str_replace_based_edit_tool",
    "anthropic.text_editor_20250728": "str_replace_based_edit_tool",
    "anthropic.bash_20241022": "bash",
    "anthropic.bash_20250124": "bash",
    "anthropic.memory_20250818": "memory",
    "anthropic.web_fetch_20250910": "web_fetch",
    "anthropic.web_fetch_20260209": "web_fetch",
    "anthropic.web_search_20250305": "web_search",
    "anthropic.web_search_20260209": "web_search",
    "anthropic.tool_search_regex_20251119": "tool_search_tool_regex",
    "anthropic.tool_search_bm25_20251119": "tool_search_tool_bm25",
    "anthropic.advisor_20260301": "advisor",
}


def _value(item: Mapping[str, object], snake: str, camel: str) -> object | None:
    return item[snake] if snake in item else item.get(camel)


def _parts(content: object) -> list[object]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, Sequence) and not isinstance(content, (str, bytes, bytearray)):
        return list(content)
    return []


def _cache_control(item: Mapping[str, object]) -> object | None:
    return _provider_options(item, BatchProvider.ANTHROPIC).get("cacheControl")


def _part_cache_control(
    part: Mapping[str, object], message: Mapping[str, object], *, last: bool
) -> object | None:
    cache_control = _cache_control(part)
    return _cache_control(message) if cache_control is None and last else cache_control


def _with_cache(block: dict[str, object], cache_control: object | None) -> dict[str, object]:
    if cache_control is not None:
        block["cache_control"] = cache_control
    return block


def _system_content(message: Mapping[str, object]) -> list[dict[str, object]]:
    content = message.get("content")
    if not isinstance(content, str):
        return []
    return [_with_cache({"type": "text", "text": content}, _cache_control(message))]


def _file_block(
    part: Mapping[str, object], cache_control: object | None, *, tool_output: bool = False
) -> dict[str, object] | None:
    block = _anthropic_file_block(part, tool_output=tool_output)
    if block is None:
        return None
    if block.get("type") == "document" and not tool_output:
        options = _provider_options(part, BatchProvider.ANTHROPIC)
        filename = part.get("filename")
        title = options.get("title", filename)
        if title is not None:
            block["title"] = title
        context = options.get("context")
        if context is not None:
            block["context"] = context
        citations = options.get("citations")
        if is_string_mapping(citations) and citations.get("enabled") is True:
            block["citations"] = {"enabled": True}
    return _with_cache(block, cache_control)


def _user_part(
    part: object, message: Mapping[str, object], *, last: bool
) -> dict[str, object] | None:
    if not is_string_mapping(part):
        return None
    cache_control = _part_cache_control(part, message, last=last)
    kind = part.get("type")
    if kind == "text":
        return _with_cache({"type": "text", "text": part.get("text", "")}, cache_control)
    if kind == "image":
        data, inline = _source(part.get("image", part.get("data")), BatchProvider.ANTHROPIC)
        if data is None:
            return None
        media_type = _value(part, "media_type", "mediaType")
        source: dict[str, object]
        if inline:
            source = {
                "type": "base64",
                "media_type": media_type if isinstance(media_type, str) else "image/png",
                "data": data,
            }
        else:
            source = {"type": "url", "url": data}
        return _with_cache({"type": "image", "source": source}, cache_control)
    if kind in {"file", "reasoning-file"}:
        return _file_block(part, cache_control)
    return None


def _output_cache_control(output: Mapping[str, object]) -> object | None:
    cache_control = _cache_control(output)
    if cache_control is not None or output.get("type") != "content":
        return cache_control
    content = output.get("value")
    if not isinstance(content, Sequence) or isinstance(content, (str, bytes, bytearray)):
        return None
    for part in content:
        if is_string_mapping(part) and (cache_control := _cache_control(part)) is not None:
            return cache_control
    return None


def _tool_result(
    part: Mapping[str, object], message: Mapping[str, object], *, last: bool
) -> dict[str, object]:
    output = part.get("output", part.get("content", ""))
    serialized, is_error = _anthropic_tool_output(output)
    result: dict[str, object] = {
        "type": "tool_result",
        "tool_use_id": _value(part, "tool_call_id", "toolCallId"),
        "content": serialized,
    }
    if is_error:
        result["is_error"] = True
    cache_control = _cache_control(part)
    if cache_control is None and is_string_mapping(output):
        cache_control = _output_cache_control(output)
    if cache_control is None and last:
        cache_control = _cache_control(message)
    return _with_cache(result, cache_control)


def _user_block(messages: Sequence[Mapping[str, object]]) -> dict[str, object]:
    content: list[object] = []
    for message in messages:
        parts = _parts(message.get("content"))
        for index, part in enumerate(parts):
            last = index == len(parts) - 1
            if message.get("role") == "tool":
                if not is_string_mapping(part) or part.get("type") != "tool-result":
                    continue
                content.append(_tool_result(part, message, last=last))
                continue
            converted = _user_part(part, message, last=last)
            if converted is not None:
                content.append(converted)
    return {"role": "user", "content": content}


def _provider_tool_name_mapping(item: Mapping[str, object]) -> dict[str, str]:
    tools = item.get("tools")
    if not isinstance(tools, Sequence) or isinstance(tools, (str, bytes, bytearray)):
        return {}
    result: dict[str, str] = {}
    for tool in tools:
        if not is_string_mapping(tool) or tool.get("type") not in {
            "provider",
            "provider-defined",
        }:
            continue
        identifier = tool.get("id")
        name = tool.get("name")
        if isinstance(identifier, str) and isinstance(name, str):
            provider_name = _PROVIDER_TOOL_NAMES.get(identifier)
            if provider_name is not None:
                result[name] = provider_name
    return result


def _caller(options: Mapping[str, object]) -> dict[str, object] | None:
    caller = options.get("caller")
    if not is_string_mapping(caller):
        return None
    caller_type = caller.get("type")
    if caller_type == "direct":
        return {"type": "direct"}
    tool_id = _value(caller, "tool_id", "toolId")
    if caller_type in {"code_execution_20250825", "code_execution_20260120"} and isinstance(
        tool_id, str
    ):
        return {"type": caller_type, "tool_id": tool_id}
    return None


def _provider_tool_use(
    part: Mapping[str, object], provider_name: str, cache_control: object | None
) -> dict[str, object] | None:
    options = _provider_options(part, BatchProvider.ANTHROPIC)
    call_id = _value(part, "tool_call_id", "toolCallId")
    tool_name = _value(part, "tool_name", "toolName")
    tool_input = part.get("input", {})
    if options.get("type") == "mcp-tool-use":
        server_name = _value(options, "server_name", "serverName")
        if not isinstance(server_name, str):
            return None
        return _with_cache(
            {
                "type": "mcp_tool_use",
                "id": call_id,
                "name": tool_name,
                "input": tool_input,
                "server_name": server_name,
            },
            cache_control,
        )
    if provider_name == "code_execution" and is_string_mapping(tool_input):
        input_type = tool_input.get("type")
        if input_type in {"bash_code_execution", "text_editor_code_execution"}:
            return _with_cache(
                {
                    "type": "server_tool_use",
                    "id": call_id,
                    "name": input_type,
                    "input": dict(tool_input),
                },
                cache_control,
            )
        if input_type == "programmatic-tool-call":
            return _with_cache(
                {
                    "type": "server_tool_use",
                    "id": call_id,
                    "name": "code_execution",
                    "input": {key: value for key, value in tool_input.items() if key != "type"},
                },
                cache_control,
            )
    if provider_name not in {
        "code_execution",
        "web_fetch",
        "web_search",
        "tool_search_tool_regex",
        "tool_search_tool_bm25",
        "advisor",
    }:
        return None
    return _with_cache(
        {
            "type": "server_tool_use",
            "id": call_id,
            "name": provider_name,
            "input": {} if provider_name == "advisor" else tool_input,
        },
        cache_control,
    )


def _tool_use(
    part: Mapping[str, object],
    message: Mapping[str, object],
    tool_names: Mapping[str, str],
    *,
    last: bool,
) -> dict[str, object] | None:
    cache_control = _part_cache_control(part, message, last=last)
    tool_name = _value(part, "tool_name", "toolName")
    provider_name = tool_names.get(tool_name, tool_name) if isinstance(tool_name, str) else ""
    provider_executed = _value(part, "provider_executed", "providerExecuted")
    if provider_executed is True:
        return _provider_tool_use(part, provider_name, cache_control)
    result: dict[str, object] = {
        "type": "tool_use",
        "id": _value(part, "tool_call_id", "toolCallId"),
        "name": tool_name,
        "input": part.get("input", {}),
    }
    caller = _caller(_provider_options(part, BatchProvider.ANTHROPIC))
    if caller is not None:
        result["caller"] = caller
    return _with_cache(result, cache_control)


def _reasoning(part: Mapping[str, object]) -> dict[str, object] | None:
    options = _provider_options(part, BatchProvider.ANTHROPIC)
    signature = options.get("signature")
    if isinstance(signature, str):
        return {"type": "thinking", "thinking": part.get("text", ""), "signature": signature}
    redacted_data = _value(options, "redacted_data", "redactedData")
    if isinstance(redacted_data, str):
        return {"type": "redacted_thinking", "data": redacted_data}
    return None


def _assistant_part(
    part: object,
    message: Mapping[str, object],
    tool_names: Mapping[str, str],
    *,
    last: bool,
    trim: bool,
) -> dict[str, object] | None:
    if not is_string_mapping(part):
        return None
    kind = part.get("type")
    if kind == "reasoning":
        return _reasoning(part)
    if kind == "tool-call":
        return _tool_use(part, message, tool_names, last=last)
    if kind == "tool-result":
        return _tool_result(part, message, last=last)
    cache_control = _part_cache_control(part, message, last=last)
    if kind == "text":
        text = part.get("text", "")
        if trim and isinstance(text, str):
            text = text.strip()
        options = _provider_options(part, BatchProvider.ANTHROPIC)
        block_type = "compaction" if options.get("type") == "compaction" else "text"
        key = "content" if block_type == "compaction" else "text"
        return _with_cache({"type": block_type, key: text}, cache_control)
    if kind in {"file", "reasoning-file"}:
        return _file_block(part, cache_control)
    return None


def _move_tool_uses_to_end(content: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    segment: list[dict[str, object]] = []

    def flush() -> None:
        result.extend(part for part in segment if part.get("type") != "tool_use")
        result.extend(part for part in segment if part.get("type") == "tool_use")
        segment.clear()

    for part in content:
        if part.get("type") in {"thinking", "redacted_thinking"}:
            flush()
            result.append(part)
        else:
            segment.append(part)
    flush()
    return result


def _assistant_block(
    messages: Sequence[Mapping[str, object]],
    tool_names: Mapping[str, str],
    *,
    final: bool,
) -> dict[str, object]:
    content: list[dict[str, object]] = []
    for message_index, message in enumerate(messages):
        parts = _parts(message.get("content"))
        for part_index, part in enumerate(parts):
            last = part_index == len(parts) - 1
            converted = _assistant_part(
                part,
                message,
                tool_names,
                last=last,
                trim=final and message_index == len(messages) - 1 and last,
            )
            if converted is not None:
                content.append(converted)
    return {"role": "assistant", "content": _move_tool_uses_to_end(content)}


def _grouped_messages(item: Mapping[str, object]) -> list[tuple[str, list[dict[str, object]]]]:
    groups: list[tuple[str, list[dict[str, object]]]] = []
    for message in _messages(item):
        role = message.get("role")
        block_type = "user" if role in {"user", "tool"} else str(role)
        if groups and groups[-1][0] == block_type:
            groups[-1][1].append(message)
        else:
            groups.append((block_type, [message]))
    return groups


def anthropic_prompt(
    item: Mapping[str, object],
) -> tuple[list[dict[str, object]] | None, list[dict[str, object]]]:
    """Return Anthropic API ``system`` blocks and conversation messages."""

    top_level_system = item.get("system")
    system: list[dict[str, object]] | None = None
    if isinstance(top_level_system, str):
        system = [{"type": "text", "text": top_level_system}]
    messages: list[dict[str, object]] = []
    groups = _grouped_messages(item)
    tool_names = _provider_tool_name_mapping(item)
    system_seen = system is not None
    for index, (block_type, block_messages) in enumerate(groups):
        if block_type == "system":
            content = [part for message in block_messages for part in _system_content(message)]
            if not system_seen or (not messages and index == 0):
                system = [*(system or []), *content]
            else:
                messages.append({"role": "system", "content": content})
            system_seen = True
        elif block_type == "assistant":
            messages.append(
                _assistant_block(block_messages, tool_names, final=index == len(groups) - 1)
            )
        elif block_type == "user":
            messages.append(_user_block(block_messages))
    return system, messages


def anthropic_messages(item: Mapping[str, object]) -> list[dict[str, object]]:
    """Return only Anthropic conversation messages for body serializers."""

    return anthropic_prompt(item)[1]


__all__ = ["anthropic_messages", "anthropic_prompt"]
