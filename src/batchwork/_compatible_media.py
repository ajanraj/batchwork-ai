"""Media conversion for chat-completion-compatible providers."""

from __future__ import annotations

import base64
import binascii
from collections.abc import Mapping

from ._typing import is_string_mapping
from .errors import BatchworkError
from .types import BatchProvider


def _media_type(part: Mapping[str, object]) -> str:
    value = part.get("media_type", part.get("mediaType"))
    if isinstance(value, str) and value:
        return value
    raise BatchworkError("batchwork: file part requires a media type.")


def _file_source(part: Mapping[str, object], provider: BatchProvider) -> tuple[str, str]:
    data = part.get("data")
    if isinstance(data, bytes):
        return "data", base64.b64encode(data).decode()
    if isinstance(data, str):
        return ("url", data) if data.startswith(("http://", "https://")) else ("data", data)
    if not is_string_mapping(data):
        raise BatchworkError(f"batchwork: {provider.value} file part data is invalid.")

    kind = data.get("type")
    if kind == "url":
        url = data.get("url")
        if url is not None:
            return "url", str(url)
    elif kind == "data":
        value = data.get("data")
        if isinstance(value, bytes):
            return "data", base64.b64encode(value).decode()
        if isinstance(value, str):
            return "data", value
    elif kind in {"reference", "text", "provider-file-id"}:
        raise BatchworkError(
            f"batchwork: {provider.value} does not support file data type {kind!r}."
        )
    raise BatchworkError(f"batchwork: {provider.value} file part data is invalid.")


def _data_url(media_type: str, data: str) -> str:
    return f"data:{media_type};base64,{data}"


def _groq_file_content(part: Mapping[str, object]) -> dict[str, object]:
    media_type = _media_type(part)
    if media_type.split("/", 1)[0] != "image":
        raise BatchworkError("batchwork: Groq supports only image file parts.")
    kind, source = _file_source(part, BatchProvider.GROQ)
    return {
        "type": "image_url",
        "image_url": {"url": source if kind == "url" else _data_url(media_type, source)},
    }


def _mistral_file_content(part: Mapping[str, object]) -> dict[str, object]:
    media_type = _media_type(part)
    kind, source = _file_source(part, BatchProvider.MISTRAL)
    value = source if kind == "url" else _data_url(media_type, source)
    if media_type.split("/", 1)[0] == "image":
        return {"type": "image_url", "image_url": value}
    if media_type == "application/pdf":
        return {"type": "document_url", "document_url": value}
    raise BatchworkError("batchwork: Mistral supports only image and PDF file parts.")


def _decode_text(data: str) -> str:
    try:
        return base64.b64decode(data, validate=True).decode()
    except (binascii.Error, UnicodeDecodeError, ValueError) as error:
        raise BatchworkError(
            "batchwork: Together text file data must be valid base64-encoded UTF-8."
        ) from error


def _together_file_content(part: Mapping[str, object]) -> dict[str, object]:
    media_type = _media_type(part)
    top_level = media_type.split("/", 1)[0]
    kind, source = _file_source(part, BatchProvider.TOGETHER)

    if top_level == "image":
        return {
            "type": "image_url",
            "image_url": {"url": source if kind == "url" else _data_url(media_type, source)},
        }
    if top_level == "audio":
        if kind == "url":
            raise BatchworkError("batchwork: Together does not support audio file URLs.")
        audio_format = (
            "wav"
            if media_type == "audio/wav"
            else "mp3"
            if media_type in {"audio/mp3", "audio/mpeg"}
            else None
        )
        if audio_format is None:
            raise BatchworkError(
                f"batchwork: Together does not support audio media type {media_type!r}."
            )
        return {
            "type": "input_audio",
            "input_audio": {"data": source, "format": audio_format},
        }
    if top_level == "application":
        if kind == "url":
            raise BatchworkError("batchwork: Together does not support PDF file URLs.")
        if media_type != "application/pdf":
            raise BatchworkError(
                f"batchwork: Together does not support file media type {media_type!r}."
            )
        filename = part.get("filename")
        return {
            "type": "file",
            "file": {
                "filename": filename if isinstance(filename, str) else "document.pdf",
                "file_data": _data_url("application/pdf", source),
            },
        }
    if top_level == "text":
        return {"type": "text", "text": source if kind == "url" else _decode_text(source)}
    raise BatchworkError(f"batchwork: Together does not support file media type {media_type!r}.")


def compatible_file_content(
    part: Mapping[str, object], provider: BatchProvider
) -> dict[str, object]:
    """Convert a file part to the provider's native chat content shape."""

    if provider is BatchProvider.GROQ:
        return _groq_file_content(part)
    if provider is BatchProvider.MISTRAL:
        return _mistral_file_content(part)
    if provider is BatchProvider.TOGETHER:
        return _together_file_content(part)
    raise BatchworkError(
        f"batchwork: provider {provider.value!r} does not use compatible media conversion."
    )


__all__ = ["compatible_file_content"]
