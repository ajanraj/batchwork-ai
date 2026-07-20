"""Provider text option merging and strict preflight validation."""

from __future__ import annotations

from collections.abc import Mapping

from ._serialization import _provider_options
from ._typing import is_string_mapping
from .errors import _OptionConflictError, _ProviderOptionError, _UnsupportedSettingError
from .types import BatchProvider, ModelKind

_TEXT_PROVIDER_OPTIONS = {
    BatchProvider.ANTHROPIC: frozenset(
        {
            "cacheControl",
            "container",
            "contextManagement",
            "disableParallelToolUse",
            "effort",
            "fallbacks",
            "inferenceGeo",
            "mcpServers",
            "metadata",
            "speed",
            "taskBudget",
            "thinking",
            "toolStreaming",
        }
    ),
    BatchProvider.GOOGLE: frozenset(
        {
            "audioTimestamp",
            "cachedContent",
            "imageConfig",
            "labels",
            "mediaResolution",
            "responseModalities",
            "retrievalConfig",
            "safetySettings",
            "serviceTier",
            "thinkingConfig",
        }
    ),
    BatchProvider.GROQ: frozenset(
        {"parallelToolCalls", "reasoningEffort", "reasoningFormat", "serviceTier", "user"}
    ),
    BatchProvider.MISTRAL: frozenset(
        {
            "documentImageLimit",
            "documentPageLimit",
            "parallelToolCalls",
            "reasoningEffort",
            "safePrompt",
        }
    ),
    BatchProvider.XAI: frozenset(
        {
            "include",
            "logprobs",
            "previousResponseId",
            "reasoningEffort",
            "reasoningSummary",
            "store",
            "topLogprobs",
        }
    ),
}

_OPENAI_COMMON_OPTIONS = frozenset(
    {
        "forceReasoning",
        "logprobs",
        "promptCacheKey",
        "promptCacheRetention",
        "safetyIdentifier",
        "serviceTier",
        "systemMessageMode",
        "user",
    }
)
_OPENAI_ENDPOINT_OPTIONS = {
    ModelKind.CHAT: _OPENAI_COMMON_OPTIONS
    | {
        "logitBias",
        "maxCompletionTokens",
        "metadata",
        "parallelToolCalls",
        "prediction",
        "reasoningEffort",
        "store",
        "textVerbosity",
    },
    ModelKind.RESPONSES: _OPENAI_COMMON_OPTIONS
    | {
        "allowedTools",
        "contextManagement",
        "conversation",
        "include",
        "instructions",
        "maxToolCalls",
        "metadata",
        "parallelToolCalls",
        "previousResponseId",
        "reasoningEffort",
        "reasoningSummary",
        "store",
        "textVerbosity",
        "truncation",
    },
    ModelKind.COMPLETION: frozenset({"echo", "logitBias", "logprobs", "suffix", "user"}),
}
_TOGETHER_RESERVED_OPTIONS = frozenset(
    {"custom_id", "customId", "input", "messages", "model", "prompt", "stream"}
)
_ENDPOINT_KINDS = {
    "chat-completions": ModelKind.CHAT,
    "responses": ModelKind.RESPONSES,
    "completions": ModelKind.COMPLETION,
}
_TEXT_ENDPOINTS = tuple(_ENDPOINT_KINDS)
_ENDPOINT_NAMES = {kind: endpoint for endpoint, kind in _ENDPOINT_KINDS.items()}
_PROVIDER_ENDPOINTS = {
    BatchProvider.OPENAI: frozenset(_ENDPOINT_KINDS),
    BatchProvider.XAI: frozenset({"responses"}),
    BatchProvider.GROQ: frozenset({"chat-completions"}),
    BatchProvider.MISTRAL: frozenset({"chat-completions"}),
    BatchProvider.TOGETHER: frozenset({"chat-completions"}),
    BatchProvider.ANTHROPIC: frozenset(),
    BatchProvider.GOOGLE: frozenset(),
}


def _canonical_value(item: Mapping[str, object], snake: str) -> object | None:
    camel = "".join([snake.split("_")[0], *[part.title() for part in snake.split("_")[1:]]])
    return item.get(snake, item.get(camel))


def _merge_text_request(
    base: Mapping[str, object], request: Mapping[str, object], provider: BatchProvider
) -> dict[str, object]:
    merged = {**base, **{key: value for key, value in request.items() if value is not None}}
    options = {
        **_provider_options(base, provider),
        **_provider_options(request, provider),
    }
    if options:
        merged.pop("providerOptions", None)
        merged["provider_options"] = {provider.value: options}
    return merged


def _text_endpoint_kind(provider: BatchProvider, endpoint: str) -> ModelKind:
    if endpoint not in _PROVIDER_ENDPOINTS[provider]:
        raise _UnsupportedSettingError(
            f'Endpoint "{endpoint}" is unsupported for provider {provider.value}; '
            "omit --endpoint to use its native text API."
        )
    return _ENDPOINT_KINDS[endpoint]


def _openai_reasoning_enabled(model_id: str, options: Mapping[str, object]) -> bool:
    forced = options.get("forceReasoning")
    if isinstance(forced, bool):
        return forced
    return model_id.startswith(("o1", "o3", "o4-mini")) or (
        model_id.startswith("gpt-5") and not model_id.startswith("gpt-5-chat")
    )


def _openai_supports_non_reasoning(model_id: str) -> bool:
    return model_id.startswith(("gpt-5.1", "gpt-5.2", "gpt-5.3", "gpt-5.4", "gpt-5.5"))


def _provider_option_failure(provider: BatchProvider, kind: ModelKind, key: str) -> None:
    endpoint = _ENDPOINT_NAMES[kind]
    raise _ProviderOptionError(
        f'batchwork: {provider.value} {endpoint} provider option "{key}" is unsupported; '
        f"see the {provider.value} provider documentation for exact option keys."
    )


def _validate_text_preflight(
    provider: BatchProvider,
    model_id: str,
    item: Mapping[str, object],
    options: Mapping[str, object],
    kind: ModelKind,
) -> None:
    if provider is BatchProvider.OPENAI:
        allowed = _OPENAI_ENDPOINT_OPTIONS[kind]
    elif provider is BatchProvider.TOGETHER:
        reserved = sorted(_TOGETHER_RESERVED_OPTIONS.intersection(options))
        if reserved:
            raise _ProviderOptionError(
                f'batchwork: together provider option "{reserved[0]}" is reserved by the '
                "canonical batch request contract."
            )
        allowed = None
    else:
        allowed = _TEXT_PROVIDER_OPTIONS[provider]
    if allowed is not None:
        unknown = sorted(set(options) - allowed)
        if unknown:
            _provider_option_failure(provider, kind, unknown[0])

    def require_shape(key: str, valid: bool, expected: str) -> None:
        if key in options and options[key] is not None and not valid:
            raise _ProviderOptionError(
                f'batchwork: {provider.value} provider option "{key}" must be {expected}.'
            )

    object_keys = {
        "allowedTools",
        "cacheControl",
        "container",
        "imageConfig",
        "labels",
        "logitBias",
        "metadata",
        "prediction",
        "retrievalConfig",
        "taskBudget",
        "thinking",
        "thinkingConfig",
    }
    for key in object_keys.intersection(options):
        require_shape(key, is_string_mapping(options[key]), "a JSON object")
    list_keys = {"fallbacks", "include", "mcpServers", "responseModalities", "safetySettings"}
    for key in list_keys.intersection(options):
        require_shape(key, isinstance(options[key], list), "a JSON array")
    boolean_keys = {
        "audioTimestamp",
        "disableParallelToolUse",
        "forceReasoning",
        "parallelToolCalls",
        "safePrompt",
        "store",
        "toolStreaming",
    }
    for key in boolean_keys.intersection(options):
        require_shape(key, isinstance(options[key], bool), "a boolean")
    string_keys = {
        "cachedContent",
        "effort",
        "inferenceGeo",
        "previousResponseId",
        "promptCacheKey",
        "promptCacheRetention",
        "reasoningEffort",
        "reasoningFormat",
        "reasoningSummary",
        "safetyIdentifier",
        "serviceTier",
        "speed",
        "suffix",
        "systemMessageMode",
        "textVerbosity",
        "truncation",
        "user",
    }
    for key in string_keys.intersection(options):
        require_shape(key, isinstance(options[key], str), "a string")
    if "conversation" in options:
        require_shape(
            "conversation",
            isinstance(options["conversation"], str) or is_string_mapping(options["conversation"]),
            "a string or JSON object",
        )
    if "logprobs" in options:
        logprobs = options["logprobs"]
        require_shape(
            "logprobs",
            isinstance(logprobs, bool)
            or (isinstance(logprobs, int) and not isinstance(logprobs, bool) and logprobs >= 0),
            "a boolean or non-negative integer",
        )
    enum_values = {
        "promptCacheRetention": {"24h", "in-memory"},
        "reasoningEffort": {"default", "high", "low", "medium", "minimal", "none", "xhigh"},
        "reasoningSummary": {"auto", "concise", "detailed"},
        "systemMessageMode": {"developer", "remove", "system"},
        "textVerbosity": {"high", "low", "medium"},
        "truncation": {"auto", "disabled"},
    }
    for key, valid_values in enum_values.items():
        value = options.get(key)
        if value is not None and isinstance(value, str) and value not in valid_values:
            choices = ", ".join(sorted(valid_values))
            raise _ProviderOptionError(
                f'batchwork: {provider.value} provider option "{key}" must be one of: {choices}.'
            )
    positive_integer_keys = {
        "documentImageLimit",
        "documentPageLimit",
        "maxCompletionTokens",
        "maxToolCalls",
    }
    for key in positive_integer_keys.intersection(options):
        value = options[key]
        require_shape(
            key,
            isinstance(value, int) and not isinstance(value, bool) and value >= 1,
            "a positive integer",
        )
    if "topLogprobs" in options:
        top_logprobs = options["topLogprobs"]
        require_shape(
            "topLogprobs",
            isinstance(top_logprobs, int)
            and not isinstance(top_logprobs, bool)
            and 0 <= top_logprobs <= 8,
            "an integer between 0 and 8",
        )
    if "contextManagement" in options:
        context = options["contextManagement"]
        valid_context = (
            isinstance(context, list)
            if provider is BatchProvider.OPENAI
            else is_string_mapping(context)
        )
        require_shape(
            "contextManagement",
            valid_context,
            "a JSON array" if provider is BatchProvider.OPENAI else "a JSON object",
        )

    max_tokens = _canonical_value(item, "max_output_tokens")
    if (
        provider is BatchProvider.OPENAI
        and max_tokens is not None
        and "maxCompletionTokens" in options
    ):
        raise _OptionConflictError(
            "batchwork: canonical max_output_tokens conflicts with OpenAI provider option "
            "maxCompletionTokens. Remove one source of the token limit."
        )
    if (
        provider is BatchProvider.OPENAI
        and kind is ModelKind.RESPONSES
        and _canonical_value(item, "system") is not None
        and "instructions" in options
    ):
        raise _OptionConflictError(
            "batchwork: canonical system conflicts with OpenAI provider option instructions."
        )
    if (
        provider is BatchProvider.OPENAI
        and kind is ModelKind.RESPONSES
        and _canonical_value(item, "tool_choice") is not None
        and "allowedTools" in options
    ):
        raise _OptionConflictError(
            "batchwork: canonical tool_choice conflicts with OpenAI provider option allowedTools."
        )
    thinking = options.get("thinking")
    if (
        provider is BatchProvider.ANTHROPIC
        and is_string_mapping(thinking)
        and thinking.get("type") in {"enabled", "adaptive"}
    ):
        sampling = [
            field
            for field in ("temperature", "top_k", "top_p")
            if _canonical_value(item, field) is not None
        ]
        if sampling:
            raise _OptionConflictError(
                f"batchwork: canonical {sampling[0]} conflicts with Anthropic provider option "
                "thinking because thinking disables sampling controls."
            )

    if provider is BatchProvider.OPENAI and _openai_reasoning_enabled(model_id, options):
        conflicts = {"frequency_penalty", "presence_penalty"} if kind is ModelKind.CHAT else set()
        if not (
            _openai_supports_non_reasoning(model_id) and options.get("reasoningEffort") == "none"
        ):
            conflicts.update({"temperature", "top_p"})
        for field in sorted(conflicts):
            if _canonical_value(item, field) is not None:
                raise _OptionConflictError(
                    f"batchwork: canonical {field} conflicts with OpenAI reasoning mode; "
                    "remove the sampling setting or disable reasoning for a compatible model."
                )

    unsupported: set[str] = set()
    if provider is BatchProvider.ANTHROPIC:
        unsupported = {"frequency_penalty", "presence_penalty", "seed"}
    elif provider in {BatchProvider.GROQ, BatchProvider.TOGETHER}:
        unsupported = {"top_k"}
    elif provider is BatchProvider.MISTRAL:
        unsupported = {"frequency_penalty", "presence_penalty", "top_k"}
    elif provider is BatchProvider.XAI:
        unsupported = {"frequency_penalty", "presence_penalty", "stop_sequences", "top_k"}
    elif provider is BatchProvider.OPENAI:
        unsupported = {"top_k"}
        if kind is ModelKind.RESPONSES:
            unsupported.update({"frequency_penalty", "presence_penalty", "stop_sequences"})
        elif kind is ModelKind.COMPLETION:
            unsupported.add("tool_choice")
    for field in sorted(unsupported):
        if _canonical_value(item, field) is not None:
            raise _UnsupportedSettingError(
                f'batchwork: canonical setting "{field}" is unsupported for '
                f"{provider.value}/{kind.value}; remove it or choose a compatible endpoint."
            )

    if provider is BatchProvider.TOGETHER:
        collisions = {
            "frequency_penalty": "frequency_penalty",
            "max_tokens": "max_output_tokens",
            "presence_penalty": "presence_penalty",
            "seed": "seed",
            "stop": "stop_sequences",
            "temperature": "temperature",
            "tool_choice": "tool_choice",
            "top_p": "top_p",
        }
        for option, canonical in collisions.items():
            if option in options and _canonical_value(item, canonical) is not None:
                raise _OptionConflictError(
                    f"batchwork: canonical {canonical} conflicts with Together provider option "
                    f"{option}."
                )
