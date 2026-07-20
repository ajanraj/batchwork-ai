"""Adapter-owned image-generation capability validation."""

from __future__ import annotations

from collections.abc import Mapping

from batchwork._typing import is_string_mapping
from batchwork.errors import (
    _LimitExceededError,
    _OptionConflictError,
    _ProviderOptionError,
    _UnsupportedSettingError,
)
from batchwork.types import BatchProvider

_GOOGLE_ASPECT_RATIOS = frozenset(
    {"1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"}
)
_XAI_ASPECT_RATIOS = frozenset({"1:1", "2:3", "3:2", "3:4", "4:3", "9:16", "16:9"})
_OPENAI_GPT_SIZES = frozenset({"1024x1024", "1024x1536", "1536x1024"})
_OPENAI_DALLE_3_SIZES = frozenset({"1024x1024", "1024x1792", "1792x1024"})
_OPENAI_DALLE_2_SIZES = frozenset({"256x256", "512x512", "1024x1024"})


def _value(item: Mapping[str, object], snake: str) -> object | None:
    if snake in item:
        return item[snake]
    head, *tail = snake.split("_")
    return item.get(head + "".join(part.title() for part in tail))


def _allowed_options(provider: BatchProvider) -> frozenset[str]:
    return {
        BatchProvider.OPENAI: frozenset(
            {
                "background",
                "moderation",
                "outputCompression",
                "outputFormat",
                "quality",
                "style",
                "user",
            }
        ),
        BatchProvider.GOOGLE: frozenset(
            {
                "googleSearch",
                "imageConfig",
                "mediaResolution",
                "responseModalities",
                "thinkingConfig",
            }
        ),
        BatchProvider.XAI: frozenset(
            {"aspect_ratio", "output_format", "quality", "resolution", "sync_mode", "user"}
        ),
    }[provider]


def _validate_count(provider: BatchProvider, model_id: str, item: Mapping[str, object]) -> None:
    n = item.get("n", 1)
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise _UnsupportedSettingError('batchwork: canonical setting "n" must be positive.')
    maximum = 1 if provider is BatchProvider.GOOGLE or model_id.startswith("dall-e-3") else 10
    if n > maximum:
        raise _LimitExceededError(
            f"batchwork: provider {provider.value} supports at most {maximum} generated "
            f"image{'s' if maximum != 1 else ''} per request."
        )


def _validate_canonical_settings(
    provider: BatchProvider,
    model_id: str,
    item: Mapping[str, object],
    options: Mapping[str, object],
) -> None:
    unsupported = {
        BatchProvider.OPENAI: ("aspect_ratio", "seed"),
        BatchProvider.GOOGLE: ("size",),
        BatchProvider.XAI: ("seed", "size"),
    }[provider]
    for setting in unsupported:
        if _value(item, setting) is not None:
            raise _UnsupportedSettingError(
                f'batchwork: provider "{provider.value}" does not support canonical '
                f'setting "{setting}" for image generation.'
            )

    aspect = _value(item, "aspect_ratio")
    if aspect is not None and provider is not BatchProvider.OPENAI:
        allowed = _GOOGLE_ASPECT_RATIOS if provider is BatchProvider.GOOGLE else _XAI_ASPECT_RATIOS
        if aspect not in allowed:
            raise _UnsupportedSettingError(
                f'batchwork: provider "{provider.value}" does not support image '
                f'aspect_ratio "{aspect}".'
            )
    size = item.get("size")
    if provider is BatchProvider.OPENAI and size is not None:
        allowed_sizes = (
            _OPENAI_DALLE_3_SIZES
            if model_id.startswith("dall-e-3")
            else _OPENAI_DALLE_2_SIZES
            if model_id.startswith("dall-e-2")
            else _OPENAI_GPT_SIZES
        )
        if size not in allowed_sizes:
            raise _UnsupportedSettingError(
                f'batchwork: OpenAI model "{model_id}" does not support image size "{size}".'
            )

    image_config = options.get("imageConfig")
    if aspect is not None and (
        (provider is BatchProvider.XAI and "aspect_ratio" in options)
        or (
            provider is BatchProvider.GOOGLE
            and is_string_mapping(image_config)
            and "aspectRatio" in image_config
        )
    ):
        key = "aspect_ratio" if provider is BatchProvider.XAI else "imageConfig.aspectRatio"
        raise _OptionConflictError(
            f'batchwork: canonical setting "aspect_ratio" conflicts with provider option "{key}".'
        )


def _validate_openai_options(model_id: str, options: Mapping[str, object]) -> None:
    if model_id.startswith("dall-e-3"):
        model_options = {"quality", "style", "user"}
        quality_values = {"hd", "standard"}
    elif model_id.startswith("dall-e-2"):
        model_options = {"user"}
        quality_values = set()
    else:
        model_options = {
            "background",
            "moderation",
            "outputCompression",
            "outputFormat",
            "quality",
            "user",
        }
        quality_values = {"auto", "high", "low", "medium"}
    unsupported = sorted(set(options) - model_options)
    if unsupported:
        raise _ProviderOptionError(
            f'batchwork: provider option "{unsupported[0]}" is not supported by '
            f'OpenAI image model "{model_id}".'
        )
    if "outputCompression" in options:
        compression = options["outputCompression"]
        if (
            not isinstance(compression, int)
            or isinstance(compression, bool)
            or not 0 <= compression <= 100
        ):
            raise _ProviderOptionError(
                'batchwork: provider option "outputCompression" must be an integer '
                "between 0 and 100."
            )
    enum_options = {
        "background": {"auto", "opaque", "transparent"},
        "moderation": {"auto", "low"},
        "outputFormat": {"jpeg", "png", "webp"},
        "quality": quality_values,
        "style": {"natural", "vivid"},
    }
    for key, allowed_values in enum_options.items():
        value = options.get(key)
        if value is not None and value not in allowed_values:
            allowed = ", ".join(sorted(allowed_values))
            raise _ProviderOptionError(
                f'batchwork: provider option "{key}" must be one of: {allowed}.'
            )
    output_format = options.get("outputFormat", "png")
    if "outputCompression" in options and output_format not in {"jpeg", "webp"}:
        raise _ProviderOptionError(
            'batchwork: provider option "outputCompression" requires '
            '"outputFormat" to be "jpeg" or "webp".'
        )
    if options.get("background") == "transparent" and output_format == "jpeg":
        raise _ProviderOptionError(
            "batchwork: transparent background is incompatible with JPEG output."
        )


def _validate_google_options(options: Mapping[str, object]) -> None:
    for key in ("googleSearch", "imageConfig", "thinkingConfig"):
        if key in options and not is_string_mapping(options[key]):
            raise _ProviderOptionError(f'batchwork: provider option "{key}" must be an object.')
    image_config = options.get("imageConfig")
    if is_string_mapping(image_config) and "aspectRatio" in image_config:
        if image_config["aspectRatio"] not in _GOOGLE_ASPECT_RATIOS:
            raise _ProviderOptionError(
                'batchwork: provider option "imageConfig.aspectRatio" is not supported by Google.'
            )
    if is_string_mapping(image_config) and "imageSize" in image_config:
        if image_config["imageSize"] not in {"1K", "2K", "4K"}:
            raise _ProviderOptionError(
                'batchwork: provider option "imageConfig.imageSize" must be one of: 1K, 2K, 4K.'
            )
    if "mediaResolution" in options and options["mediaResolution"] not in {
        "MEDIA_RESOLUTION_LOW",
        "MEDIA_RESOLUTION_MEDIUM",
        "MEDIA_RESOLUTION_HIGH",
        "MEDIA_RESOLUTION_ULTRA_HIGH",
    }:
        raise _ProviderOptionError(
            'batchwork: provider option "mediaResolution" is not a supported Google value.'
        )
    if "responseModalities" in options:
        modalities = options["responseModalities"]
        if (
            not isinstance(modalities, list)
            or "IMAGE" not in modalities
            or any(value not in {"IMAGE", "TEXT"} for value in modalities)
        ):
            raise _ProviderOptionError(
                'batchwork: provider option "responseModalities" must contain only '
                '"IMAGE" and optional "TEXT".'
            )


def _validate_xai_options(options: Mapping[str, object]) -> None:
    if "sync_mode" in options and not isinstance(options["sync_mode"], bool):
        raise _ProviderOptionError('batchwork: provider option "sync_mode" must be a boolean.')
    if "aspect_ratio" in options and options["aspect_ratio"] not in _XAI_ASPECT_RATIOS:
        raise _ProviderOptionError(
            'batchwork: provider option "aspect_ratio" is not supported by xAI.'
        )
    enum_options = {
        "output_format": {"jpeg", "jpg", "png", "webp"},
        "quality": {"high", "low", "medium"},
        "resolution": {"1k", "2k"},
    }
    for key, allowed_values in enum_options.items():
        value = options.get(key)
        if value is not None and value not in allowed_values:
            allowed = ", ".join(sorted(allowed_values))
            raise _ProviderOptionError(
                f'batchwork: provider option "{key}" must be one of: {allowed}.'
            )


def validate_image_preflight(
    provider: BatchProvider,
    model_id: str,
    item: Mapping[str, object],
    options: Mapping[str, object],
) -> None:
    """Validate the selected adapter's image capability matrix."""
    unknown = sorted(set(options) - _allowed_options(provider))
    if unknown:
        raise _ProviderOptionError(
            f'batchwork: provider option "{unknown[0]}" is not supported for '
            f"{provider.value} image generation."
        )
    _validate_count(provider, model_id, item)
    _validate_canonical_settings(provider, model_id, item, options)

    string_options = {
        BatchProvider.OPENAI: {
            "background",
            "moderation",
            "outputFormat",
            "quality",
            "style",
            "user",
        },
        BatchProvider.GOOGLE: {"mediaResolution"},
        BatchProvider.XAI: {"aspect_ratio", "output_format", "quality", "resolution", "user"},
    }[provider]
    for key in string_options:
        if key in options and not isinstance(options[key], str):
            raise _ProviderOptionError(f'batchwork: provider option "{key}" must be a string.')

    if provider is BatchProvider.OPENAI:
        _validate_openai_options(model_id, options)
    elif provider is BatchProvider.GOOGLE:
        _validate_google_options(options)
    else:
        _validate_xai_options(options)
