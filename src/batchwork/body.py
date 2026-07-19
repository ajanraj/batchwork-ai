"""Provider-native request serialization for batch items.

Request bodies are built directly from typed Python inputs while keeping
provider-owned options at the boundary.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ._anthropic_serialization import anthropic_prompt
from ._google_serialization import google_messages
from ._serialization import (
    Request,
    _chat_tool,
    _Dumpable,
    _mapped_options,
    _openai_messages,
    _openai_responses_input,
    _provider_options,
    _response_function_tool,
    _tools,
    _xai_responses_input,
)
from ._typing import is_string_mapping
from .errors import BatchworkError, UnsupportedProviderError
from .types import BatchLimits, BatchProvider, ModelKind


@dataclass(frozen=True, slots=True)
class BuiltRequest:
    """One serialized provider request correlated with its batch result."""

    body: dict[str, object]
    custom_id: str
    endpoint: str


_TEXT_PROVIDERS = frozenset(BatchProvider)
_EMBEDDING_PROVIDERS = frozenset(
    {BatchProvider.OPENAI, BatchProvider.GOOGLE, BatchProvider.MISTRAL}
)
_IMAGE_PROVIDERS = frozenset({BatchProvider.OPENAI, BatchProvider.GOOGLE, BatchProvider.XAI})


def _anthropic_capabilities(model_id: str) -> tuple[int, bool, bool, bool]:
    if any(
        name in model_id
        for name in ("claude-opus-4-8", "claude-opus-4-7", "claude-fable-5", "claude-sonnet-5")
    ):
        return 128_000, True, True, True
    if any(name in model_id for name in ("claude-sonnet-4-6", "claude-opus-4-6")):
        return 128_000, True, False, True
    if any(
        name in model_id for name in ("claude-sonnet-4-5", "claude-opus-4-5", "claude-haiku-4-5")
    ):
        return 64_000, True, False, True
    if "claude-opus-4-1" in model_id:
        return 32_000, True, False, True
    if "claude-sonnet-4-" in model_id:
        return 64_000, True, False, False
    if "claude-opus-4-" in model_id:
        return 32_000, True, False, False
    if "claude-3-haiku" in model_id:
        return 4_096, True, False, False
    return 4_096, False, False, False


def _mapping(value: Request | None) -> dict[str, object]:
    if value is None:
        return {}
    if is_string_mapping(value):
        return {key: item for key, item in value.items()}
    if isinstance(value, _Dumpable):
        return value.model_dump(by_alias=False, exclude_none=True)
    raise TypeError("batchwork: request must be a string-keyed mapping or Pydantic model")


def _limits_value(limits: BatchLimits | None, name: str, default: int) -> int:
    if limits is None:
        return default
    value = getattr(limits, name, default)
    return default if value is None else value


def validate_request_count(requests: Sequence[object], limits: BatchLimits | None) -> None:
    maximum = _limits_value(limits, "max_requests", 50_000)
    if len(requests) > maximum:
        raise BatchworkError(
            f"batchwork: requests length {len(requests)} exceeds the {maximum} request limit."
        )


def _validate_size(custom_id: str, body: Mapping[str, object], limits: BatchLimits | None) -> None:
    maximum = _limits_value(limits, "max_request_bytes", 20 * 1024 * 1024)
    size = len(json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode())
    if size > maximum:
        raise BatchworkError(
            f'batchwork: request "{custom_id}" is {size} bytes, exceeding the {maximum} byte limit.'
        )


def _assigned(requests: Sequence[Request]) -> list[tuple[str, dict[str, object]]]:
    seen: set[str] = set()
    assigned: list[tuple[str, dict[str, object]]] = []
    for index, request in enumerate(requests):
        item = _mapping(request)
        value = item.pop("custom_id", item.pop("customId", None))
        custom_id = value if isinstance(value, str) else f"request-{index}"
        if custom_id in seen:
            raise BatchworkError(
                f'batchwork: duplicate customId "{custom_id}". '
                "customId values must be unique within a batch."
            )
        seen.add(custom_id)
        assigned.append((custom_id, item))
    return assigned


def _chat_body_options(provider: BatchProvider, options: Mapping[str, object]) -> dict[str, object]:
    if provider == BatchProvider.GROQ:
        return _mapped_options(
            options,
            {
                "user": "user",
                "parallelToolCalls": "parallel_tool_calls",
                "reasoningFormat": "reasoning_format",
                "reasoningEffort": "reasoning_effort",
                "serviceTier": "service_tier",
            },
        )
    if provider == BatchProvider.MISTRAL:
        return _mapped_options(
            options,
            {
                "safePrompt": "safe_prompt",
                "reasoningEffort": "reasoning_effort",
                "documentImageLimit": "document_image_limit",
                "documentPageLimit": "document_page_limit",
            },
        )
    if provider == BatchProvider.OPENAI:
        return _mapped_options(
            options,
            {
                "logitBias": "logit_bias",
                "user": "user",
                "parallelToolCalls": "parallel_tool_calls",
                "maxCompletionTokens": "max_completion_tokens",
                "store": "store",
                "metadata": "metadata",
                "prediction": "prediction",
                "reasoningEffort": "reasoning_effort",
                "serviceTier": "service_tier",
                "promptCacheKey": "prompt_cache_key",
                "promptCacheRetention": "prompt_cache_retention",
                "safetyIdentifier": "safety_identifier",
                "textVerbosity": "verbosity",
            },
        )
    if provider == BatchProvider.TOGETHER:
        recognized = {"user", "reasoningEffort", "textVerbosity", "strictJsonSchema"}
        result = {key: value for key, value in options.items() if key not in recognized}
        result.update(
            _mapped_options(
                options,
                {
                    "user": "user",
                    "reasoningEffort": "reasoning_effort",
                    "textVerbosity": "verbosity",
                },
            )
        )
        return result
    return {}


def _openai_is_reasoning_model(model_id: str) -> bool:
    return model_id.startswith(("o1", "o3", "o4-mini")) or (
        model_id.startswith("gpt-5") and not model_id.startswith("gpt-5-chat")
    )


def _openai_reasoning_enabled(model_id: str, options: Mapping[str, object]) -> bool:
    forced = options.get("forceReasoning")
    return forced if isinstance(forced, bool) else _openai_is_reasoning_model(model_id)


def _openai_supports_non_reasoning(model_id: str) -> bool:
    return model_id.startswith(("gpt-5.1", "gpt-5.2", "gpt-5.3", "gpt-5.4", "gpt-5.5"))


def _openai_supports_service_tier(model_id: str, tier: object) -> bool:
    if tier == "flex":
        return model_id.startswith(("o3", "o4-mini")) or (
            model_id.startswith("gpt-5") and not model_id.startswith("gpt-5-chat")
        )
    if tier == "priority":
        return model_id.startswith(("gpt-4", "o3", "o4-mini")) or (
            model_id.startswith("gpt-5")
            and not model_id.startswith(("gpt-5-nano", "gpt-5-chat", "gpt-5.4-nano"))
        )
    return True


def _filter_openai_service_tier(
    body: dict[str, object], model_id: str, options: Mapping[str, object]
) -> None:
    if not _openai_supports_service_tier(model_id, options.get("serviceTier")):
        body.pop("service_tier", None)


def _openai_system_message_mode(options: Mapping[str, object], reasoning_enabled: bool) -> str:
    mode = options.get("systemMessageMode")
    if isinstance(mode, str) and mode in {"system", "developer", "remove"}:
        return mode
    return "developer" if reasoning_enabled else "system"


def _rewrite_openai_system_entries(entries: Sequence[object], mode: str) -> list[object]:
    rewritten: list[object] = []
    for entry in entries:
        if is_string_mapping(entry) and entry.get("role") == "system":
            if mode == "remove":
                continue
            rewritten.append({**entry, "role": mode})
        else:
            rewritten.append(entry)
    return rewritten


def _openai_responses_options(options: Mapping[str, object], model_id: str) -> dict[str, object]:
    body = _mapped_options(
        options,
        {
            "conversation": "conversation",
            "maxToolCalls": "max_tool_calls",
            "metadata": "metadata",
            "parallelToolCalls": "parallel_tool_calls",
            "previousResponseId": "previous_response_id",
            "store": "store",
            "user": "user",
            "instructions": "instructions",
            "serviceTier": "service_tier",
            "promptCacheKey": "prompt_cache_key",
            "promptCacheRetention": "prompt_cache_retention",
            "safetyIdentifier": "safety_identifier",
            "truncation": "truncation",
        },
    )
    include = options.get("include")
    normalized_include = list(include) if isinstance(include, list) else []
    logprobs = options.get("logprobs")
    top_logprobs: int | float | None = None
    if logprobs is True:
        top_logprobs = 20
    elif isinstance(logprobs, (int, float)) and not isinstance(logprobs, bool):
        top_logprobs = logprobs
    if top_logprobs:
        body["top_logprobs"] = top_logprobs
        if "message.output_text.logprobs" not in normalized_include:
            normalized_include.append("message.output_text.logprobs")

    reasoning_model = _openai_reasoning_enabled(model_id, options)
    effort = options.get("reasoningEffort")
    summary = options.get("reasoningSummary")
    if summary is None and effort is not None and effort != "none":
        summary = "detailed"
    if reasoning_model and (effort is not None or summary is not None):
        body["reasoning"] = {
            **({"effort": effort} if effort is not None else {}),
            **({"summary": summary} if summary is not None else {}),
        }
    if options.get("store") is False and reasoning_model:
        if "reasoning.encrypted_content" not in normalized_include:
            normalized_include.append("reasoning.encrypted_content")
    if normalized_include:
        body["include"] = normalized_include
    verbosity = options.get("textVerbosity")
    if verbosity is not None:
        body["text"] = {"verbosity": verbosity}
    context = options.get("contextManagement")
    if isinstance(context, list):
        normalized_context: list[dict[str, object]] = []
        for item in context:
            if not is_string_mapping(item):
                continue
            normalized_context.append(
                {
                    "type": item.get("type"),
                    "compact_threshold": item.get("compactThreshold"),
                }
            )
        if normalized_context:
            body["context_management"] = normalized_context
    return body


def _openai_completion_options(options: Mapping[str, object]) -> dict[str, object]:
    body = _mapped_options(
        options,
        {
            "echo": "echo",
            "logitBias": "logit_bias",
            "suffix": "suffix",
            "user": "user",
        },
    )
    logprobs = options.get("logprobs")
    if logprobs is True:
        body["logprobs"] = 0
    elif isinstance(logprobs, (int, float)) and not isinstance(logprobs, bool):
        body["logprobs"] = logprobs
    return body


def _value(item: Mapping[str, object], snake: str, camel: str | None = None) -> object | None:
    if snake in item:
        return item[snake]
    return item.get(camel) if camel else None


def _text_body(
    provider: BatchProvider,
    model_id: str,
    item: Mapping[str, object],
    kind: ModelKind,
) -> tuple[dict[str, object], str]:
    options = _provider_options(item, provider)
    if provider == BatchProvider.ANTHROPIC:
        max_tokens, known_model, rejects_sampling, supports_strict = _anthropic_capabilities(
            model_id
        )
        requested_tokens = _value(item, "max_output_tokens", "maxOutputTokens")
        resolved_tokens = requested_tokens if isinstance(requested_tokens, int) else max_tokens
        system, messages = anthropic_prompt(item)
        body: dict[str, object] = {
            "model": model_id,
            "messages": messages,
            "max_tokens": resolved_tokens,
        }
        if system is not None:
            body["system"] = system
        for source, target in {
            "temperature": "temperature",
            "top_k": "top_k",
            "top_p": "top_p",
            "stop_sequences": "stop_sequences",
        }.items():
            camel = "".join([source.split("_")[0], *[p.title() for p in source.split("_")[1:]]])
            value = _value(item, source, camel)
            if value is not None:
                body[target] = value
        body.update(
            _mapped_options(
                options,
                {
                    "speed": "speed",
                    "inferenceGeo": "inference_geo",
                    "fallbacks": "fallbacks",
                    "cacheControl": "cache_control",
                },
            )
        )
        metadata = options.get("metadata")
        if is_string_mapping(metadata) and isinstance(metadata.get("userId"), str):
            body["metadata"] = {"user_id": metadata["userId"]}
        mcp_servers = options.get("mcpServers")
        if isinstance(mcp_servers, list) and mcp_servers:
            body["mcp_servers"] = [
                {
                    "type": server.get("type"),
                    "name": server.get("name"),
                    "url": server.get("url"),
                    **(
                        {"authorization_token": server["authorizationToken"]}
                        if server.get("authorizationToken") is not None
                        else {}
                    ),
                    **(
                        {
                            "tool_configuration": {
                                **(
                                    {"allowed_tools": configuration["allowedTools"]}
                                    if configuration.get("allowedTools") is not None
                                    else {}
                                ),
                                **(
                                    {"enabled": configuration["enabled"]}
                                    if configuration.get("enabled") is not None
                                    else {}
                                ),
                            }
                        }
                        if is_string_mapping(configuration := server.get("toolConfiguration"))
                        else {}
                    ),
                }
                for server in mcp_servers
                if is_string_mapping(server)
            ]
        container = options.get("container")
        if is_string_mapping(container):
            skills = container.get("skills")
            if isinstance(skills, list) and skills:
                normalized_skills: list[dict[str, object]] = []
                for skill in skills:
                    if not is_string_mapping(skill):
                        continue
                    skill_id = skill.get("skillId")
                    if skill.get("type") == "custom":
                        reference = skill.get("providerReference")
                        skill_id = (
                            reference.get("anthropic") if is_string_mapping(reference) else None
                        )
                    normalized_skills.append(
                        {
                            "type": skill.get("type"),
                            "skill_id": skill_id,
                            **(
                                {"version": skill["version"]}
                                if skill.get("version") is not None
                                else {}
                            ),
                        }
                    )
                body["container"] = {
                    "id": container.get("id"),
                    "skills": normalized_skills,
                }
            elif container.get("id") is not None:
                body["container"] = container["id"]
        context_management = options.get("contextManagement")
        if is_string_mapping(context_management):
            edits = context_management.get("edits")
            normalized_edits: list[dict[str, object]] = []
            if isinstance(edits, list):
                edit_fields = {
                    "trigger": "trigger",
                    "keep": "keep",
                    "clearAtLeast": "clear_at_least",
                    "clearToolInputs": "clear_tool_inputs",
                    "excludeTools": "exclude_tools",
                    "pauseAfterCompaction": "pause_after_compaction",
                    "instructions": "instructions",
                }
                known_edits = {
                    "clear_tool_uses_20250919",
                    "clear_thinking_20251015",
                    "compact_20260112",
                }
                for edit in edits:
                    if is_string_mapping(edit) and edit.get("type") in known_edits:
                        normalized_edits.append(
                            {
                                "type": edit["type"],
                                **_mapped_options(edit, edit_fields),
                            }
                        )
            body["context_management"] = {"edits": normalized_edits}
        output_config: dict[str, object] = {}
        if options.get("effort") is not None:
            output_config["effort"] = options["effort"]
        task_budget = options.get("taskBudget")
        if is_string_mapping(task_budget):
            output_config["task_budget"] = {
                "type": task_budget.get("type"),
                "total": task_budget.get("total"),
                **(
                    {"remaining": task_budget["remaining"]}
                    if task_budget.get("remaining") is not None
                    else {}
                ),
            }
        if output_config:
            body["output_config"] = output_config
        thinking = options.get("thinking")
        if is_string_mapping(thinking):
            normalized_thinking: dict[str, object] = {"type": thinking.get("type")}
            budget = thinking.get("budgetTokens")
            if thinking.get("type") == "enabled" and not isinstance(budget, (int, float)):
                budget = 1_024
            if budget is not None:
                normalized_thinking["budget_tokens"] = budget
            if thinking.get("display") is not None:
                normalized_thinking["display"] = thinking["display"]
            body["thinking"] = normalized_thinking
            if thinking.get("type") in {"enabled", "adaptive"}:
                body.pop("temperature", None)
                body.pop("top_k", None)
                body.pop("top_p", None)
                if isinstance(budget, (int, float)) and not isinstance(budget, bool):
                    body["max_tokens"] = resolved_tokens + int(budget)
        if rejects_sampling:
            body.pop("temperature", None)
            body.pop("top_k", None)
            body.pop("top_p", None)
        elif known_model and "temperature" in body and "top_p" in body:
            body.pop("top_p", None)
        if known_model and isinstance(body.get("max_tokens"), int):
            body["max_tokens"] = min(body["max_tokens"], max_tokens)
        tools = _tools(item, provider, model_id, kind)
        if tools:
            anthropic_tools: list[object] = []
            for tool in tools:
                function = tool.get("function")
                if is_string_mapping(function):
                    raw_tool_options = tool.get("_provider_options")
                    tool_options = (
                        _provider_options({"provider_options": raw_tool_options}, provider)
                        if is_string_mapping(raw_tool_options)
                        else {}
                    )
                    input_examples = tool.get("_input_examples")
                    anthropic_tools.append(
                        {
                            "name": function.get("name"),
                            "input_schema": function.get("parameters", {}),
                            **(
                                {"description": function["description"]}
                                if "description" in function
                                else {}
                            ),
                            **(
                                {"strict": function["strict"]}
                                if supports_strict and isinstance(function.get("strict"), bool)
                                else {}
                            ),
                            **(
                                {"cache_control": tool_options["cacheControl"]}
                                if tool_options.get("cacheControl") is not None
                                else {}
                            ),
                            **(
                                {"eager_input_streaming": True}
                                if tool_options.get(
                                    "eagerInputStreaming", options.get("toolStreaming", True)
                                )
                                is True
                                else {}
                            ),
                            **(
                                {"defer_loading": tool_options["deferLoading"]}
                                if tool_options.get("deferLoading") is not None
                                else {}
                            ),
                            **(
                                {"allowed_callers": tool_options["allowedCallers"]}
                                if tool_options.get("allowedCallers") is not None
                                else {}
                            ),
                            **(
                                {
                                    "input_examples": [
                                        example["input"]
                                        for example in input_examples
                                        if is_string_mapping(example) and "input" in example
                                    ]
                                }
                                if isinstance(input_examples, list)
                                else {}
                            ),
                        }
                    )
                else:
                    anthropic_tools.append(tool)
            body["tools"] = anthropic_tools
        tool_choice = _value(item, "tool_choice", "toolChoice")
        disable_parallel = options.get("disableParallelToolUse")
        if tool_choice == "none":
            body.pop("tools", None)
        elif tool_choice == "required":
            body["tool_choice"] = {
                "type": "any",
                **(
                    {"disable_parallel_tool_use": disable_parallel}
                    if isinstance(disable_parallel, bool)
                    else {}
                ),
            }
        elif tool_choice == "auto":
            body["tool_choice"] = {
                "type": "auto",
                **(
                    {"disable_parallel_tool_use": disable_parallel}
                    if isinstance(disable_parallel, bool)
                    else {}
                ),
            }
        elif is_string_mapping(tool_choice):
            name = tool_choice.get("tool_name", tool_choice.get("toolName"))
            if isinstance(name, str):
                body["tool_choice"] = {
                    "type": "tool",
                    "name": name,
                    **(
                        {"disable_parallel_tool_use": disable_parallel}
                        if isinstance(disable_parallel, bool)
                        else {}
                    ),
                }
        elif isinstance(disable_parallel, bool):
            body["tool_choice"] = {
                "type": "auto",
                "disable_parallel_tool_use": disable_parallel,
            }
        endpoint = "/v1/messages"
    elif provider == BatchProvider.GOOGLE:
        contents, system = google_messages(item, model_id)
        generation_config: dict[str, object] = {}
        settings = {
            "temperature": "temperature",
            "top_p": "topP",
            "top_k": "topK",
            "max_output_tokens": "maxOutputTokens",
            "stop_sequences": "stopSequences",
            "seed": "seed",
            "presence_penalty": "presencePenalty",
            "frequency_penalty": "frequencyPenalty",
        }
        for source, target in settings.items():
            value = _value(
                item,
                source,
                "".join([source.split("_")[0], *[p.title() for p in source.split("_")[1:]]]),
            )
            if value is not None:
                generation_config[target] = value
        for key in (
            "responseModalities",
            "thinkingConfig",
            "mediaResolution",
            "imageConfig",
            "audioTimestamp",
        ):
            if options.get(key) is not None:
                generation_config[key] = options[key]
        body = {"contents": contents}
        if generation_config:
            body["generationConfig"] = generation_config
        if system is not None:
            body["systemInstruction"] = system
        for key in ("safetySettings", "cachedContent", "labels", "serviceTier"):
            if options.get(key) is not None:
                body[key] = options[key]
        mixed_tools = False
        tools = _tools(item, provider, model_id, kind)
        if tools:
            declarations: list[object] = []
            native_tools: list[object] = []
            for tool in tools:
                function = tool.get("function")
                if is_string_mapping(function):
                    declaration: dict[str, object] = {
                        "name": function.get("name"),
                        "description": function.get("description", ""),
                    }
                    if function.get("parameters") is not None:
                        declaration["parameters"] = function["parameters"]
                    declarations.append(declaration)
                else:
                    native_tools.append(tool)
            mixed_tools = bool(declarations and native_tools)
            if mixed_tools and "gemini-3" not in model_id:
                declarations = []
            if declarations:
                native_tools.insert(0, {"functionDeclarations": declarations})
            if native_tools:
                body["tools"] = native_tools
        tool_choice = _value(item, "tool_choice", "toolChoice")
        if tool_choice is not None:
            mode = (
                "ANY" if tool_choice == "required" else "NONE" if tool_choice == "none" else "AUTO"
            )
            tool_config: dict[str, object] = {"mode": mode}
            if is_string_mapping(tool_choice):
                name = tool_choice.get("tool_name", tool_choice.get("toolName"))
                if isinstance(name, str):
                    tool_config.update({"mode": "ANY", "allowedFunctionNames": [name]})
            body["toolConfig"] = {
                "functionCallingConfig": tool_config,
                **(
                    {"includeServerSideToolInvocations": True}
                    if mixed_tools and "gemini-3" in model_id
                    else {}
                ),
            }
        elif mixed_tools and "gemini-3" in model_id:
            body["toolConfig"] = {
                "functionCallingConfig": {"mode": "VALIDATED"},
                "includeServerSideToolInvocations": True,
            }
        retrieval_config = options.get("retrievalConfig")
        if retrieval_config is not None:
            raw_tool_config = body.get("toolConfig")
            body["toolConfig"] = {
                **(dict(raw_tool_config) if is_string_mapping(raw_tool_config) else {}),
                "retrievalConfig": retrieval_config,
            }
        endpoint = f"/v1beta/models/{model_id}:generateContent"
    elif provider == BatchProvider.XAI:
        body = {
            "model": model_id,
            "input": _xai_responses_input(item),
        }
        for source, target in {
            "max_output_tokens": "max_output_tokens",
            "temperature": "temperature",
            "top_p": "top_p",
            "seed": "seed",
        }.items():
            camel = "".join([source.split("_")[0], *[p.title() for p in source.split("_")[1:]]])
            value = _value(item, source, camel)
            if value is not None:
                body[target] = value
        if options.get("logprobs") is True or options.get("topLogprobs") is not None:
            body["logprobs"] = True
        if options.get("topLogprobs") is not None:
            body["top_logprobs"] = options["topLogprobs"]
        reasoning: dict[str, object] = {}
        if options.get("reasoningEffort") is not None:
            reasoning["effort"] = options["reasoningEffort"]
        if options.get("reasoningSummary") is not None:
            reasoning["summary"] = options["reasoningSummary"]
        if reasoning:
            body["reasoning"] = reasoning
        include = options.get("include")
        if options.get("store") is False:
            body["store"] = False
            base_include = list(include) if isinstance(include, list) else []
            body["include"] = [*base_include, "reasoning.encrypted_content"]
        elif isinstance(include, list):
            body["include"] = include
        if options.get("previousResponseId") is not None:
            body["previous_response_id"] = options["previousResponseId"]
        tools = _tools(item, provider, model_id, kind)
        if tools:
            response_tools: list[object] = []
            for tool in tools:
                function = tool.get("function")
                if is_string_mapping(function):
                    response_tools.append(_response_function_tool(function, tool, provider))
                else:
                    response_tools.append(tool)
            body["tools"] = response_tools
        tool_choice = _value(item, "tool_choice", "toolChoice")
        if is_string_mapping(tool_choice):
            name = tool_choice.get("tool_name", tool_choice.get("toolName"))
            if isinstance(name, str):
                body["tool_choice"] = {"type": "function", "name": name}
        elif tool_choice is not None:
            body["tool_choice"] = tool_choice
        endpoint = "/v1/responses"
    else:
        body = (
            _openai_responses_options(options, model_id)
            if provider == BatchProvider.OPENAI and kind is ModelKind.RESPONSES
            else _openai_completion_options(options)
            if provider == BatchProvider.OPENAI and kind is ModelKind.COMPLETION
            else _chat_body_options(provider, options)
        )
        if provider == BatchProvider.OPENAI and kind is ModelKind.CHAT:
            logprobs = options.get("logprobs")
            if logprobs is True or (
                isinstance(logprobs, (int, float)) and not isinstance(logprobs, bool)
            ):
                body["logprobs"] = True
            if isinstance(logprobs, (int, float)) and not isinstance(logprobs, bool):
                body["top_logprobs"] = logprobs
        if provider == BatchProvider.OPENAI and kind is ModelKind.RESPONSES:
            body.update(
                {
                    "model": model_id,
                    "input": _openai_responses_input(item, provider),
                }
            )
        elif provider == BatchProvider.OPENAI and kind is ModelKind.COMPLETION:
            prompt = item.get("prompt")
            if not isinstance(prompt, str):
                raise BatchworkError("batchwork: OpenAI completions require a string prompt.")
            body.update({"model": model_id, "prompt": prompt})
        else:
            body.update({"model": model_id, "messages": _openai_messages(item, provider)})
        settings = {
            "temperature": "temperature",
            "top_p": "top_p",
            "max_output_tokens": "max_tokens",
            "frequency_penalty": "frequency_penalty",
            "presence_penalty": "presence_penalty",
            "seed": "seed",
            "stop_sequences": "stop",
        }
        for source, target in settings.items():
            camel = "".join([source.split("_")[0], *[p.title() for p in source.split("_")[1:]]])
            value = _value(item, source, camel)
            if value is not None:
                if kind is ModelKind.RESPONSES and target == "max_tokens":
                    body["max_output_tokens"] = value
                elif kind is ModelKind.RESPONSES and target in {
                    "frequency_penalty",
                    "presence_penalty",
                    "stop",
                }:
                    continue
                elif provider == BatchProvider.MISTRAL and target in {
                    "frequency_penalty",
                    "presence_penalty",
                }:
                    continue
                elif provider == BatchProvider.MISTRAL and target == "seed":
                    body["random_seed"] = value
                else:
                    body[target] = value
        if provider == BatchProvider.OPENAI and kind in {ModelKind.CHAT, ModelKind.RESPONSES}:
            reasoning_enabled = _openai_reasoning_enabled(model_id, options)
            supports_non_reasoning = _openai_supports_non_reasoning(model_id)
            if kind is ModelKind.CHAT and reasoning_enabled:
                max_tokens = body.pop("max_tokens", None)
                if max_tokens is not None and "max_completion_tokens" not in body:
                    body["max_completion_tokens"] = max_tokens
                body.pop("frequency_penalty", None)
                body.pop("presence_penalty", None)
                if not (supports_non_reasoning and options.get("reasoningEffort") == "none"):
                    body.pop("temperature", None)
                    body.pop("top_p", None)
            elif (
                not (
                    reasoning_enabled
                    and options.get("reasoningEffort") == "none"
                    and supports_non_reasoning
                )
                and reasoning_enabled
            ):
                body.pop("temperature", None)
                body.pop("top_p", None)
            _filter_openai_service_tier(body, model_id, options)
            entries_key = "messages" if kind is ModelKind.CHAT else "input"
            entries = body.get(entries_key)
            if isinstance(entries, list):
                mode = _openai_system_message_mode(options, reasoning_enabled)
                body[entries_key] = _rewrite_openai_system_entries(entries, mode)
        tools = _tools(item, provider, model_id, kind)
        if tools:
            if kind is ModelKind.RESPONSES:
                response_tools: list[object] = []
                for tool in tools:
                    function = tool.get("function")
                    if is_string_mapping(function):
                        response_tools.append(_response_function_tool(function, tool, provider))
                    else:
                        response_tools.append(tool)
                body["tools"] = response_tools
            elif kind is not ModelKind.COMPLETION:
                body["tools"] = [_chat_tool(tool) for tool in tools]
                if (
                    provider == BatchProvider.MISTRAL
                    and options.get("parallelToolCalls") is not None
                ):
                    body["parallel_tool_calls"] = options["parallelToolCalls"]
        allowed_tools = (
            options.get("allowedTools")
            if provider == BatchProvider.OPENAI and kind is ModelKind.RESPONSES
            else None
        )
        allowed_tool_names = (
            allowed_tools.get("toolNames") if is_string_mapping(allowed_tools) else None
        )
        if tools and is_string_mapping(allowed_tools) and isinstance(allowed_tool_names, list):
            raw_definitions = item.get("tools")
            definitions = (
                raw_definitions
                if isinstance(raw_definitions, Sequence)
                and not isinstance(raw_definitions, (str, bytes, bytearray))
                else []
            )
            names: list[dict[str, str]] = []
            for allowed_name in allowed_tool_names:
                if not isinstance(allowed_name, str):
                    continue
                resolved_name = allowed_name
                for definition in definitions:
                    identifier = definition.get("id") if is_string_mapping(definition) else None
                    if (
                        is_string_mapping(definition)
                        and definition.get("type") == "provider-defined"
                        and definition.get("name") == allowed_name
                        and isinstance(identifier, str)
                        and identifier != "openai.custom"
                    ):
                        resolved_name = identifier.removeprefix("openai.")
                        break
                names.append({"type": "function", "name": resolved_name})
            body["tool_choice"] = {
                "type": "allowed_tools",
                "mode": allowed_tools.get("mode", "auto"),
                "tools": names,
            }
        tool_choice = _value(item, "tool_choice", "toolChoice")
        if allowed_tools is None and tool_choice is not None and kind is not ModelKind.COMPLETION:
            if is_string_mapping(tool_choice):
                name = tool_choice.get("tool_name", tool_choice.get("toolName"))
                if isinstance(name, str) and provider == BatchProvider.MISTRAL:
                    raw_tools = body.get("tools")
                    if isinstance(raw_tools, list):
                        selected_tools: list[object] = []
                        for tool in raw_tools:
                            function = tool.get("function") if is_string_mapping(tool) else None
                            if is_string_mapping(function) and function.get("name") == name:
                                selected_tools.append(tool)
                        body["tools"] = selected_tools
                    body["tool_choice"] = "any"
                elif isinstance(name, str) and kind is ModelKind.RESPONSES:
                    definitions = item.get("tools")
                    resolved_name = name
                    if isinstance(definitions, Sequence) and not isinstance(
                        definitions, (str, bytes, bytearray)
                    ):
                        for definition in definitions:
                            identifier = (
                                definition.get("id")
                                if is_string_mapping(definition)
                                and definition.get("type") == "provider-defined"
                                and definition.get("name") == name
                                else None
                            )
                            if isinstance(identifier, str):
                                resolved_name = identifier.removeprefix("openai.")
                                break
                    raw_tools = body.get("tools")
                    custom = isinstance(raw_tools, list) and any(
                        is_string_mapping(tool)
                        and tool.get("type") == "custom"
                        and tool.get("name") == resolved_name
                        for tool in raw_tools
                    )
                    provider_choice_types = {
                        "apply_patch",
                        "code_interpreter",
                        "file_search",
                        "image_generation",
                        "mcp",
                        "web_search",
                        "web_search_preview",
                    }
                    body["tool_choice"] = (
                        {"type": resolved_name}
                        if resolved_name in provider_choice_types
                        else {
                            "type": "custom" if custom else "function",
                            "name": resolved_name,
                        }
                    )
                else:
                    body["tool_choice"] = (
                        {"type": "function", "function": {"name": name}}
                        if isinstance(name, str)
                        else tool_choice
                    )
            elif provider == BatchProvider.MISTRAL and tool_choice == "required":
                body["tool_choice"] = "any"
            else:
                body["tool_choice"] = tool_choice
        if provider == BatchProvider.OPENAI and kind is ModelKind.RESPONSES:
            endpoint = "/v1/responses"
        elif provider == BatchProvider.OPENAI and kind is ModelKind.COMPLETION:
            endpoint = "/v1/completions"
        else:
            endpoint = (
                "/openai/v1/chat/completions"
                if provider == BatchProvider.GROQ
                else "/v1/chat/completions"
            )
    body.pop("stream", None)
    return body, endpoint


def build_text_bodies(
    provider: BatchProvider,
    model_id: str,
    requests: Sequence[Request],
    defaults: Request | None = None,
    limits: BatchLimits | None = None,
    *,
    kind: ModelKind = ModelKind.CHAT,
) -> list[BuiltRequest]:
    """Serialize text requests using request-over-default precedence."""

    if provider not in _TEXT_PROVIDERS:
        raise BatchworkError(
            f"batchwork: provider {provider.value} does not offer batch text generation."
        )
    validate_request_count(requests, limits)
    base = _mapping(defaults)
    built: list[BuiltRequest] = []
    for custom_id, request in _assigned(requests):
        merged = {**base, **request}
        body, endpoint = _text_body(provider, model_id, merged, kind)
        _validate_size(custom_id, body, limits)
        built.append(BuiltRequest(body=body, custom_id=custom_id, endpoint=endpoint))
    return built


def build_embedding_bodies(
    provider: BatchProvider,
    model_id: str,
    requests: Sequence[Request],
    limits: BatchLimits | None = None,
) -> list[BuiltRequest]:
    if provider not in _EMBEDDING_PROVIDERS:
        raise UnsupportedProviderError(
            provider.value,
            f'batchwork: provider "{provider.value}" does not offer batch embeddings. '
            "Embeddings are supported for: openai, mistral, google.",
        )
    validate_request_count(requests, limits)
    built: list[BuiltRequest] = []
    for custom_id, item in _assigned(requests):
        options = _provider_options(item, provider)
        value = item.get("value")
        if not isinstance(value, str):
            raise BatchworkError("batchwork: embedding request value must be a string.")
        if provider == BatchProvider.GOOGLE:
            parts: list[object] = [{"text": value}]
            multimodal = options.get("content")
            if multimodal is not None:
                if not isinstance(multimodal, list) or len(multimodal) != 1:
                    raise BatchworkError(
                        "batchwork: Google embedding content must contain one entry per value."
                    )
                extra = multimodal[0]
                if extra is not None:
                    if not isinstance(extra, list):
                        raise BatchworkError(
                            "batchwork: Google embedding content entries must be lists."
                        )
                    parts.extend(extra)
            body = {"model": f"models/{model_id}", "content": {"parts": parts}}
            for key in ("outputDimensionality", "taskType", "title"):
                if options.get(key) is not None:
                    body[key] = options[key]
            endpoint = f"/v1beta/models/{model_id}:embedContent"
        else:
            body = {"model": model_id, "input": [value], "encoding_format": "float"}
            if provider == BatchProvider.OPENAI:
                for key in ("dimensions", "user"):
                    if options.get(key) is not None:
                        body[key] = options[key]
            endpoint = "/v1/embeddings"
        _validate_size(custom_id, body, limits)
        built.append(BuiltRequest(body=body, custom_id=custom_id, endpoint=endpoint))
    return built


def build_image_bodies(
    provider: BatchProvider,
    model_id: str,
    requests: Sequence[Request],
    defaults: Request | None = None,
    limits: BatchLimits | None = None,
) -> list[BuiltRequest]:
    if provider not in _IMAGE_PROVIDERS:
        raise UnsupportedProviderError(
            provider.value,
            f'batchwork: provider "{provider.value}" does not offer batch image generation. '
            "Image batches are supported for: openai, google, xai.",
        )
    validate_request_count(requests, limits)
    base = _mapping(defaults)
    built: list[BuiltRequest] = []
    for custom_id, request in _assigned(requests):
        item = {**base, **request}
        prompt = item.get("prompt")
        if not isinstance(prompt, str):
            raise BatchworkError("batchwork: image request prompt must be a string.")
        options = _provider_options(item, provider)
        if provider == BatchProvider.GOOGLE:
            n = item.get("n", 1)
            if n not in (None, 1):
                raise BatchworkError(
                    "batchwork: Google batch image generation supports exactly one image "
                    "per request."
                )
            body: dict[str, object] = {}
            body["contents"] = [{"role": "user", "parts": [{"text": prompt}]}]
            generation: dict[str, object] = {"responseModalities": ["IMAGE"]}
            for key in ("responseModalities", "thinkingConfig", "mediaResolution"):
                if options.get(key) is not None:
                    generation[key] = options[key]
            image_config = options.get("imageConfig")
            normalized_image_config = dict(image_config) if is_string_mapping(image_config) else {}
            aspect = _value(item, "aspect_ratio", "aspectRatio")
            if aspect is not None:
                normalized_image_config["aspectRatio"] = aspect
            if normalized_image_config:
                generation["imageConfig"] = normalized_image_config
            seed = item.get("seed")
            if seed is not None:
                generation["seed"] = seed
            body["generationConfig"] = generation
            google_search = options.get("googleSearch")
            if is_string_mapping(google_search):
                body["tools"] = [{"googleSearch": dict(google_search)}]
            endpoint = f"/v1beta/models/{model_id}:generateContent"
        elif provider == BatchProvider.OPENAI:
            body = {
                "model": model_id,
                "prompt": prompt,
                "n": item.get("n", 1),
            }
            size = item.get("size")
            if size is not None:
                body["size"] = size
            for source, target in (
                ("quality", "quality"),
                ("style", "style"),
                ("background", "background"),
                ("moderation", "moderation"),
                ("outputFormat", "output_format"),
                ("outputCompression", "output_compression"),
                ("user", "user"),
            ):
                if options.get(source) is not None:
                    body[target] = options[source]
            if not model_id.startswith(
                (
                    "chatgpt-image-",
                    "gpt-image-1-mini",
                    "gpt-image-1.5",
                    "gpt-image-1",
                    "gpt-image-2",
                )
            ):
                body["response_format"] = "b64_json"
            endpoint = "/v1/images/generations"
        else:
            body = {}
            body.update(
                {
                    "model": model_id,
                    "prompt": prompt,
                    "n": item.get("n", 1),
                    "response_format": "b64_json",
                }
            )
            aspect = _value(item, "aspect_ratio", "aspectRatio")
            if aspect is not None:
                body["aspect_ratio"] = aspect
            for key in (
                "output_format",
                "sync_mode",
                "resolution",
                "quality",
                "user",
            ):
                if options.get(key) is not None:
                    body[key] = options[key]
            if aspect is None and options.get("aspect_ratio") is not None:
                body["aspect_ratio"] = options["aspect_ratio"]
            endpoint = "/v1/images/generations"
        _validate_size(custom_id, body, limits)
        built.append(BuiltRequest(body=body, custom_id=custom_id, endpoint=endpoint))
    return built


# Compatibility alias retained for existing callers.
build_request_bodies = build_text_bodies
