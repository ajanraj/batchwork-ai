import base64

import pytest

from batchwork._compatible_media import compatible_file_content
from batchwork.client import Batchwork
from batchwork.errors import BatchworkError
from batchwork.media import ResolvedMedia
from batchwork.types import (
    BatchLimits,
    BatchProvider,
    BatchRequest,
    ContentToolOutput,
    FilePart,
    ModelSpec,
    TaggedFileDataData,
    TaggedFileDataReference,
    TaggedFileDataText,
    TaggedFileDataUrl,
    ToolMessage,
    ToolOutputFilePart,
    ToolResultPart,
    UserMessage,
)


def _file(data: object, media_type: str, filename: str | None = None) -> dict[str, object]:
    part: dict[str, object] = {"type": "file", "data": data, "media_type": media_type}
    if filename is not None:
        part["filename"] = filename
    return part


def test_groq_converts_only_image_file_content() -> None:
    assert compatible_file_content(
        _file({"type": "url", "url": "https://example.com/image.png"}, "image/png"),
        BatchProvider.GROQ,
    ) == {
        "type": "image_url",
        "image_url": {"url": "https://example.com/image.png"},
    }
    assert compatible_file_content(
        _file({"type": "data", "data": b"image"}, "image/png"), BatchProvider.GROQ
    ) == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,aW1hZ2U="},
    }
    with pytest.raises(BatchworkError, match="only image"):
        compatible_file_content(
            _file({"type": "data", "data": b"pdf"}, "application/pdf"),
            BatchProvider.GROQ,
        )


@pytest.mark.parametrize(
    ("data", "media_type", "expected"),
    [
        (
            {"type": "url", "url": "https://example.com/image.png"},
            "image/png",
            {"type": "image_url", "image_url": "https://example.com/image.png"},
        ),
        (
            {"type": "data", "data": b"image"},
            "image/png",
            {"type": "image_url", "image_url": "data:image/png;base64,aW1hZ2U="},
        ),
        (
            {"type": "url", "url": "https://example.com/document.pdf"},
            "application/pdf",
            {
                "type": "document_url",
                "document_url": "https://example.com/document.pdf",
            },
        ),
        (
            {"type": "data", "data": b"pdf"},
            "application/pdf",
            {"type": "document_url", "document_url": "data:application/pdf;base64,cGRm"},
        ),
    ],
)
def test_mistral_converts_image_and_pdf_file_content(
    data: object, media_type: str, expected: dict[str, object]
) -> None:
    assert compatible_file_content(_file(data, media_type), BatchProvider.MISTRAL) == expected


def test_mistral_rejects_unsupported_file_content() -> None:
    with pytest.raises(BatchworkError, match="only image and PDF"):
        compatible_file_content(
            _file({"type": "data", "data": b"hello"}, "text/plain"),
            BatchProvider.MISTRAL,
        )


@pytest.mark.parametrize(
    ("data", "media_type", "expected"),
    [
        (
            {"type": "url", "url": "https://example.com/image.png"},
            "image/png",
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/image.png"},
            },
        ),
        (
            {"type": "data", "data": b"pdf"},
            "application/pdf",
            {
                "type": "file",
                "file": {
                    "filename": "document.pdf",
                    "file_data": "data:application/pdf;base64,cGRm",
                },
            },
        ),
        (
            {"type": "data", "data": base64.b64encode(b"hello").decode()},
            "text/plain",
            {"type": "text", "text": "hello"},
        ),
        (
            {"type": "data", "data": b"audio"},
            "audio/mpeg",
            {"type": "input_audio", "input_audio": {"data": "YXVkaW8=", "format": "mp3"}},
        ),
    ],
)
def test_together_converts_openai_compatible_file_content(
    data: object, media_type: str, expected: dict[str, object]
) -> None:
    assert compatible_file_content(_file(data, media_type), BatchProvider.TOGETHER) == expected


@pytest.mark.parametrize(
    ("data", "media_type", "error"),
    [
        (
            {"type": "url", "url": "https://example.com/document.pdf"},
            "application/pdf",
            "PDF file URLs",
        ),
        (
            {"type": "url", "url": "https://example.com/audio.mp3"},
            "audio/mpeg",
            "audio file URLs",
        ),
        ({"type": "data", "data": b"audio"}, "audio/ogg", "audio media type"),
    ],
)
def test_together_rejects_unsupported_file_content(
    data: object, media_type: str, error: str
) -> None:
    with pytest.raises(BatchworkError, match=error):
        compatible_file_content(_file(data, media_type), BatchProvider.TOGETHER)


@pytest.mark.asyncio
async def test_client_resolves_together_tagged_text_file_data() -> None:
    class Resolver:
        async def resolve(
            self, source: object, *, media_type: str | None = None, max_bytes: int
        ) -> ResolvedMedia:
            assert source == TaggedFileDataText(text="hello")
            assert media_type == "text/plain"
            assert max_bytes == 20 * 1024 * 1024
            return ResolvedMedia(b"hello", "text/plain")

    client = Batchwork(media_resolver=Resolver())
    request = BatchRequest(
        messages=[
            UserMessage(
                content=[FilePart(data=TaggedFileDataText(text="hello"), media_type="text/plain")]
            )
        ]
    )

    prepared = await client._resolve_request_media(
        ModelSpec(provider=BatchProvider.TOGETHER, model_id="moonshotai/Kimi-K2-Instruct"),
        [request],
        BatchLimits(),
        None,
    )

    message = prepared[0].messages[0]
    assert isinstance(message, UserMessage)
    assert not isinstance(message.content, str)
    part = message.content[0]
    assert isinstance(part, FilePart)
    assert part.data == b"hello"
    assert compatible_file_content(part.model_dump(exclude_none=True), BatchProvider.TOGETHER) == {
        "type": "text",
        "text": "hello",
    }
    await client.aclose()


@pytest.mark.asyncio
async def test_client_preserves_together_tagged_file_reference() -> None:
    class Resolver:
        async def resolve(
            self, source: object, *, media_type: str | None, max_bytes: int
        ) -> ResolvedMedia:
            raise AssertionError(f"unexpected media resolution: {source}")

    reference = TaggedFileDataReference(reference={"together": "file-1"})
    client = Batchwork(media_resolver=Resolver())
    request = BatchRequest(
        messages=[UserMessage(content=[FilePart(data=reference, media_type="application/pdf")])]
    )

    prepared = await client._resolve_request_media(
        ModelSpec(provider=BatchProvider.TOGETHER, model_id="moonshotai/Kimi-K2-Instruct"),
        [request],
        BatchLimits(),
        None,
    )

    message = prepared[0].messages[0]
    assert isinstance(message, UserMessage)
    assert not isinstance(message.content, str)
    part = message.content[0]
    assert isinstance(part, FilePart)
    assert part.data == reference
    await client.aclose()


@pytest.mark.asyncio
async def test_client_resolves_together_tagged_text_tool_output() -> None:
    class Resolver:
        async def resolve(
            self, source: object, *, media_type: str | None = None, max_bytes: int
        ) -> ResolvedMedia:
            assert source == TaggedFileDataText(text="hello")
            assert media_type == "text/plain"
            assert max_bytes == 20 * 1024 * 1024
            return ResolvedMedia(b"hello", "text/plain")

    client = Batchwork(media_resolver=Resolver())
    request = BatchRequest(
        messages=[
            ToolMessage(
                content=[
                    ToolResultPart(
                        tool_call_id="call_1",
                        tool_name="read",
                        output=ContentToolOutput(
                            value=[
                                ToolOutputFilePart(
                                    data=TaggedFileDataText(text="hello"),
                                    media_type="text/plain",
                                )
                            ]
                        ),
                    )
                ]
            )
        ]
    )

    prepared = await client._resolve_request_media(
        ModelSpec(provider=BatchProvider.TOGETHER, model_id="moonshotai/Kimi-K2-Instruct"),
        [request],
        BatchLimits(),
        None,
    )

    message = prepared[0].messages[0]
    assert isinstance(message, ToolMessage)
    result = message.content[0]
    assert isinstance(result, ToolResultPart)
    assert isinstance(result.output, ContentToolOutput)
    part = result.output.value[0]
    assert isinstance(part, ToolOutputFilePart)
    assert isinstance(part.data, TaggedFileDataData)
    assert part.data.data == b"hello"
    await client.aclose()


@pytest.mark.asyncio
async def test_client_resolves_nested_tool_output_url_without_losing_tag() -> None:
    class Resolver:
        async def resolve(
            self, source: object, *, media_type: str | None = None, max_bytes: int
        ) -> ResolvedMedia:
            assert str(source) == "https://example.com/image.png"
            assert media_type == "image/png"
            assert max_bytes == 20 * 1024 * 1024
            return ResolvedMedia(b"image", "image/png")

    client = Batchwork(media_resolver=Resolver())
    request = BatchRequest(
        messages=[
            ToolMessage(
                content=[
                    ToolResultPart(
                        tool_call_id="call_1",
                        tool_name="render",
                        output=ContentToolOutput(
                            value=[
                                ToolOutputFilePart(
                                    data=TaggedFileDataUrl(url="https://example.com/image.png"),
                                    media_type="image/png",
                                )
                            ]
                        ),
                    )
                ]
            )
        ]
    )

    prepared = await client._resolve_request_media(
        ModelSpec(provider=BatchProvider.GOOGLE, model_id="gemini-2.5-pro"),
        [request],
        BatchLimits(),
        None,
    )
    message = prepared[0].messages[0]
    assert isinstance(message, ToolMessage)
    result = message.content[0]
    assert isinstance(result, ToolResultPart)
    assert isinstance(result.output, ContentToolOutput)
    part = result.output.value[0]
    assert isinstance(part, ToolOutputFilePart)
    assert isinstance(part.data, TaggedFileDataData)
    assert part.data.type == "data"
    assert part.data.data == b"image"
    await client.aclose()
