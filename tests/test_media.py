import asyncio
import os
import socket
import threading
from pathlib import Path

import httpcore
import httpx
import pytest

import batchwork.media as media_module
from batchwork.errors import MediaResolutionError
from batchwork.media import DefaultMediaResolver, _validate_address
from batchwork.types import (
    TaggedFileDataData,
    TaggedFileDataReference,
    TaggedFileDataText,
)


@pytest.mark.asyncio
async def test_inline_data_url_and_size_limit() -> None:
    resolver = DefaultMediaResolver()
    resolved = await resolver.resolve("data:text/plain;base64,aGVsbG8=", max_bytes=5)
    assert resolved.data == b"hello"
    assert resolved.media_type == "text/plain"
    with pytest.raises(MediaResolutionError, match="exceeding"):
        await resolver.resolve(b"too large", media_type="text/plain", max_bytes=2)


@pytest.mark.asyncio
async def test_tagged_file_data_is_resolved_without_stringifying_models() -> None:
    resolver = DefaultMediaResolver()
    encoded = await resolver.resolve(
        TaggedFileDataData(data="aGVsbG8="), media_type="text/plain", max_bytes=5
    )
    text = await resolver.resolve(
        TaggedFileDataText(text="hello"), media_type="text/plain", max_bytes=5
    )
    assert encoded.data == b"hello"
    assert text.data == b"hello"
    with pytest.raises(MediaResolutionError, match="cannot be downloaded"):
        await resolver.resolve(
            TaggedFileDataReference(reference={"openai": "file_1"}),
            media_type="application/pdf",
            max_bytes=5,
        )


@pytest.mark.asyncio
async def test_remote_download_is_bounded_and_does_not_auto_redirect(monkeypatch) -> None:
    requested: list[tuple[str, str, str | None]] = []

    async def resolve_public_addresses(host: str, port: int) -> tuple[str, ...]:
        assert (host, port) == ("example.com", 443)
        return ("93.184.216.34",)

    monkeypatch.setattr(media_module, "_resolve_public_addresses", resolve_public_addresses)

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(
            (str(request.url), request.headers["host"], request.extensions.get("sni_hostname"))
        )
        if request.url.path == "/start":
            return httpx.Response(302, headers={"location": "/image"})
        return httpx.Response(200, content=b"hello", headers={"content-type": "text/plain"})

    resolver = DefaultMediaResolver(transport=httpx.MockTransport(handler))
    resolved = await resolver.resolve("https://example.com/start", max_bytes=10)
    assert resolved.data == b"hello"
    assert requested == [
        ("https://93.184.216.34/start", "example.com", "example.com"),
        ("https://93.184.216.34/image", "example.com", "example.com"),
    ]


@pytest.mark.parametrize(
    "headers",
    [
        [("content-length", "bogus")],
        [("content-length", "5"), ("content-length", "5")],
    ],
)
@pytest.mark.asyncio
async def test_remote_download_treats_invalid_content_length_as_unknown(
    monkeypatch: pytest.MonkeyPatch, headers: list[tuple[str, str]]
) -> None:
    async def resolve_public_addresses(_host: str, _port: int) -> tuple[str, ...]:
        return ("93.184.216.34",)

    monkeypatch.setattr(media_module, "_resolve_public_addresses", resolve_public_addresses)

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"12345", headers=headers)

    resolved = await DefaultMediaResolver(transport=httpx.MockTransport(handler)).resolve(
        "https://example.com/file.txt", media_type="text/plain", max_bytes=5
    )

    assert resolved.data == b"12345"


@pytest.mark.parametrize("declared_length", ["bogus", "1", "-1"])
@pytest.mark.asyncio
async def test_remote_download_stream_limit_overrides_untrusted_content_length(
    monkeypatch: pytest.MonkeyPatch, declared_length: str
) -> None:
    async def resolve_public_addresses(_host: str, _port: int) -> tuple[str, ...]:
        return ("93.184.216.34",)

    monkeypatch.setattr(media_module, "_resolve_public_addresses", resolve_public_addresses)

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"123456",
            headers={"content-length": declared_length},
        )

    resolver = DefaultMediaResolver(transport=httpx.MockTransport(handler))
    with pytest.raises(MediaResolutionError, match="media exceeds the 5 byte limit") as captured:
        await resolver.resolve("https://example.com/file.txt", media_type="text/plain", max_bytes=5)

    assert declared_length not in str(captured.value)


@pytest.mark.asyncio
async def test_remote_download_rejects_declared_oversize_before_reading_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def resolve_public_addresses(_host: str, _port: int) -> tuple[str, ...]:
        return ("93.184.216.34",)

    monkeypatch.setattr(media_module, "_resolve_public_addresses", resolve_public_addresses)

    class UnreadStream(httpx.AsyncByteStream):
        def __init__(self) -> None:
            self.was_read = False

        async def __aiter__(self):
            self.was_read = True
            yield b"123456"

    stream = UnreadStream()

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=stream, headers={"content-length": "6"})

    resolver = DefaultMediaResolver(transport=httpx.MockTransport(handler))
    with pytest.raises(MediaResolutionError, match="media exceeds the 5 byte limit"):
        await resolver.resolve("https://example.com/file.txt", media_type="text/plain", max_bytes=5)

    assert stream.was_read is False


@pytest.mark.asyncio
async def test_pinned_backend_tries_each_validated_address(monkeypatch) -> None:
    async def resolve_public_addresses(host: str, port: int) -> tuple[str, ...]:
        assert (host, port) == ("example.com", 443)
        return ("93.184.216.34", "93.184.216.35")

    attempts: list[tuple[str, int, float | None, str | None, object]] = []
    stream = httpcore.AsyncNetworkStream()

    async def connect_tcp(
        host: str,
        port: int,
        **options: object,
    ) -> httpcore.AsyncNetworkStream:
        timeout = options.get("timeout")
        local_address = options.get("local_address")
        attempts.append(
            (
                host,
                port,
                timeout if isinstance(timeout, float) else None,
                local_address if isinstance(local_address, str) else None,
                options.get("socket_options"),
            )
        )
        if host == "93.184.216.34":
            raise httpcore.ConnectTimeout("first address timed out")
        return stream

    monkeypatch.setattr(media_module, "_resolve_public_addresses", resolve_public_addresses)
    backend = media_module._PinnedNetworkBackend()
    monkeypatch.setattr(backend._backend, "connect_tcp", connect_tcp)

    connected = await backend.connect_tcp(
        "example.com",
        443,
        timeout=5,
        local_address="0.0.0.0",
        socket_options=((socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1),),
    )

    assert connected is stream
    assert [attempt[:2] for attempt in attempts] == [
        ("93.184.216.34", 443),
        ("93.184.216.35", 443),
    ]
    assert [attempt[2] for attempt in attempts] == [5.0, 5.0]
    assert all(attempt[3] == "0.0.0.0" for attempt in attempts)
    assert all(attempt[4] == [(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)] for attempt in attempts)


@pytest.mark.asyncio
async def test_pinned_backend_rejects_all_addresses_when_one_is_private(monkeypatch) -> None:
    loop = asyncio.get_running_loop()

    async def getaddrinfo(
        host: str,
        port: int,
        *,
        family: int,
        type: int,
    ) -> list[tuple[int, int, int, str, tuple[str, int]]]:
        assert (host, port, family, type) == (
            "example.com",
            443,
            socket.AF_UNSPEC,
            socket.SOCK_STREAM,
        )
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443)),
        ]

    attempted: list[str] = []

    async def connect_tcp(host: str, **options: object) -> httpcore.AsyncNetworkStream:
        del options
        attempted.append(host)
        return httpcore.AsyncNetworkStream()

    monkeypatch.setattr(loop, "getaddrinfo", getaddrinfo)
    backend = media_module._PinnedNetworkBackend()
    monkeypatch.setattr(backend._backend, "connect_tcp", connect_tcp)

    with pytest.raises(MediaResolutionError, match="not globally routable"):
        await backend.connect_tcp("example.com", 443)

    assert attempted == []


@pytest.mark.asyncio
async def test_media_dns_rejects_invalid_address(monkeypatch) -> None:
    loop = asyncio.get_running_loop()

    async def getaddrinfo(
        _host: str,
        _port: int,
        *,
        family: int,
        type: int,
    ) -> list[tuple[int, int, int, str, tuple[str, int]]]:
        assert (family, type) == (socket.AF_UNSPEC, socket.SOCK_STREAM)
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("invalid-address", 443))]

    monkeypatch.setattr(loop, "getaddrinfo", getaddrinfo)

    with pytest.raises(MediaResolutionError, match="DNS returned invalid address"):
        await media_module._resolve_public_addresses("example.com", 443)


@pytest.mark.asyncio
async def test_injected_transport_remains_open_across_resolutions(monkeypatch) -> None:
    async def resolve_public_addresses(host: str, port: int) -> tuple[str, ...]:
        assert (host, port) == ("example.com", 443)
        return ("93.184.216.34",)

    monkeypatch.setattr(media_module, "_resolve_public_addresses", resolve_public_addresses)

    class ReusableTransport(httpx.AsyncBaseTransport):
        def __init__(self) -> None:
            self.closed = False
            self.requests = 0

        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            if self.closed:
                raise RuntimeError("transport is closed")
            self.requests += 1
            return httpx.Response(
                200,
                content=request.url.path.encode(),
                headers={"content-type": "text/plain"},
            )

        async def aclose(self) -> None:
            self.closed = True

    transport = ReusableTransport()
    resolver = DefaultMediaResolver(transport=transport)

    first = await resolver.resolve("https://example.com/first", max_bytes=20)
    second = await resolver.resolve("https://example.com/second", max_bytes=20)

    assert first.data == b"/first"
    assert second.data == b"/second"
    assert transport.requests == 2
    assert transport.closed is False
    await transport.aclose()


@pytest.mark.asyncio
async def test_injected_transport_tries_each_validated_address(monkeypatch) -> None:
    async def resolve_public_addresses(host: str, port: int) -> tuple[str, ...]:
        assert (host, port) == ("example.com", 443)
        return ("93.184.216.34", "93.184.216.35")

    monkeypatch.setattr(media_module, "_resolve_public_addresses", resolve_public_addresses)
    attempts: list[tuple[str, str, object]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(
            (request.url.host, request.headers["host"], request.extensions.get("sni_hostname"))
        )
        if request.url.host == "93.184.216.34":
            raise httpx.ConnectTimeout("first address timed out", request=request)
        return httpx.Response(200, content=b"hello", headers={"content-type": "text/plain"})

    resolver = DefaultMediaResolver(transport=httpx.MockTransport(handler))
    resolved = await resolver.resolve("https://example.com/file.txt", max_bytes=10)

    assert resolved.data == b"hello"
    assert attempts == [
        ("93.184.216.34", "example.com", "example.com"),
        ("93.184.216.35", "example.com", "example.com"),
    ]


@pytest.mark.asyncio
async def test_injected_transport_rejects_private_target_before_request() -> None:
    requested = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requested
        requested = True
        return httpx.Response(200, content=b"secret", headers={"content-type": "text/plain"})

    resolver = DefaultMediaResolver(transport=httpx.MockTransport(handler))
    with pytest.raises(MediaResolutionError, match="not globally routable"):
        await resolver.resolve("https://127.0.0.1/latest/meta-data", max_bytes=100)
    assert requested is False


@pytest.mark.parametrize(
    "url",
    [
        "https://user@example.com/image.png",
        "https://:secret@example.com/image.png",
    ],
)
@pytest.mark.asyncio
async def test_remote_url_rejects_credentials_before_request(url: str) -> None:
    requested = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requested
        requested = True
        return httpx.Response(200, content=b"image", headers={"content-type": "image/png"})

    resolver = DefaultMediaResolver(transport=httpx.MockTransport(handler))
    with pytest.raises(MediaResolutionError, match="must not include credentials"):
        await resolver.resolve(url, max_bytes=100)
    assert requested is False


@pytest.mark.asyncio
async def test_remote_redirect_rejects_credentials_before_following_request(monkeypatch) -> None:
    requested: list[str] = []

    async def resolve_public_addresses(_host: str, _port: int) -> tuple[str, ...]:
        return ("93.184.216.34",)

    monkeypatch.setattr(media_module, "_resolve_public_addresses", resolve_public_addresses)

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(request.url.path)
        return httpx.Response(
            302,
            headers={"location": "https://user:secret@example.com/image.png"},
        )

    resolver = DefaultMediaResolver(transport=httpx.MockTransport(handler))
    with pytest.raises(MediaResolutionError, match="must not include credentials"):
        await resolver.resolve("https://example.com/start", max_bytes=100)
    assert requested == ["/start"]


def test_private_and_reserved_addresses_are_rejected() -> None:
    for address in ("127.0.0.1", "10.0.0.1", "::1", "169.254.169.254"):
        with pytest.raises(MediaResolutionError, match="not globally routable"):
            _validate_address(address)


@pytest.mark.parametrize(
    ("data", "declared", "expected"),
    [
        (b"\x89PNG\r\n\x1a\nrest", "image", "image/png"),
        (b"\x89PNG\r\n\x1a\nrest", "image/*", "image/png"),
        (b"%PDF-rest", "application", "application/pdf"),
        (b"%PDF-rest", "application/*", "application/pdf"),
    ],
)
@pytest.mark.asyncio
async def test_resolves_partial_media_type_from_inline_bytes(
    data: bytes, declared: str, expected: str
) -> None:
    resolved = await DefaultMediaResolver().resolve(data, media_type=declared, max_bytes=100)

    assert resolved.media_type == expected


@pytest.mark.asyncio
async def test_full_media_type_still_rejects_detected_subtype_mismatch() -> None:
    resolver = DefaultMediaResolver()
    with pytest.raises(MediaResolutionError, match="does not match"):
        await resolver.resolve(b"\x89PNG\r\n\x1a\nrest", media_type="image/jpeg", max_bytes=100)


@pytest.mark.asyncio
async def test_rejects_insecure_remote_url() -> None:
    with pytest.raises(MediaResolutionError, match="must use HTTPS"):
        await DefaultMediaResolver().resolve("http://example.com/a.png", max_bytes=100)


@pytest.mark.asyncio
async def test_raw_media_strings_classify_base64_before_local_paths(tmp_path: Path) -> None:
    (tmp_path / "dGVzdA==").write_bytes(b"file contents")
    resolver = DefaultMediaResolver(base_path=tmp_path)

    encoded = await resolver.resolve("dGVzdA==", media_type="text/plain", max_bytes=20)
    explicit_path = await resolver.resolve("./dGVzdA==", media_type="text/plain", max_bytes=20)

    assert encoded.data == b"test"
    assert explicit_path.data == b"file contents"


@pytest.mark.asyncio
async def test_local_media_paths_resolve_from_configured_base_and_stay_literal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media_dir = tmp_path / "source"
    media_dir.mkdir()
    relative = media_dir / "image.png"
    relative.write_bytes(b"\x89PNG\r\n\x1a\npayload")
    (media_dir / "linked.png").symlink_to(relative)
    absolute = tmp_path / "absolute.pdf"
    absolute.write_bytes(b"%PDF-payload")
    monkeypatch.setenv("MEDIA_FILE", str(relative))
    resolver = DefaultMediaResolver(base_path=media_dir)

    resolved_relative = await resolver.resolve("image.png", max_bytes=100)
    resolved_symlink = await resolver.resolve("linked.png", max_bytes=100)
    resolved_absolute = await resolver.resolve(str(absolute), max_bytes=100)

    assert resolved_relative == media_module.ResolvedMedia(relative.read_bytes(), "image/png")
    assert resolved_symlink == resolved_relative
    assert resolved_absolute == media_module.ResolvedMedia(absolute.read_bytes(), "application/pdf")
    for literal in ("~/image.png", "$MEDIA_FILE", "*.png"):
        with pytest.raises(MediaResolutionError, match="local media path"):
            await resolver.resolve(literal, max_bytes=100)


@pytest.mark.asyncio
async def test_local_media_requires_regular_bounded_type_checked_file(tmp_path: Path) -> None:
    resolver = DefaultMediaResolver(base_path=tmp_path)
    (tmp_path / "large.txt").write_bytes(b"1234")
    (tmp_path / "wrong.jpg").write_bytes(b"\x89PNG\r\n\x1a\npayload")
    (tmp_path / "directory").mkdir()

    with pytest.raises(MediaResolutionError, match="exceeding the 3 byte limit"):
        await resolver.resolve("large.txt", media_type="text/plain", max_bytes=3)
    with pytest.raises(MediaResolutionError, match="does not match"):
        await resolver.resolve("wrong.jpg", media_type="image/jpeg", max_bytes=100)
    with pytest.raises(MediaResolutionError, match="regular file"):
        await resolver.resolve("directory", media_type="text/plain", max_bytes=100)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="named pipes require POSIX")
@pytest.mark.asyncio
async def test_local_media_rejects_fifo_without_blocking(tmp_path: Path) -> None:
    fifo = tmp_path / "media.pipe"
    os.mkfifo(fifo)
    resolver = DefaultMediaResolver(base_path=tmp_path)

    with pytest.raises(MediaResolutionError, match="regular file"):
        await asyncio.wait_for(
            resolver.resolve("media.pipe", media_type="text/plain", max_bytes=100),
            timeout=1,
        )


@pytest.mark.asyncio
async def test_relative_media_base_is_frozen_when_resolver_is_created(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "document.pdf").write_bytes(b"%PDF-frozen-base")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(tmp_path)
    resolver = DefaultMediaResolver(base_path=Path("source"))

    monkeypatch.chdir(elsewhere)
    resolved = await resolver.resolve("document.pdf", max_bytes=100)

    assert resolved.data == b"%PDF-frozen-base"


@pytest.mark.asyncio
async def test_local_media_read_runs_off_event_loop_thread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "document.pdf").write_bytes(b"%PDF-threaded")
    resolver = DefaultMediaResolver(base_path=tmp_path)
    event_loop_thread = threading.get_ident()
    read_threads: list[int] = []
    original_read = resolver._read_local

    def tracked_read(
        value: str, *, media_type: str | None, max_bytes: int
    ) -> media_module.ResolvedMedia:
        read_threads.append(threading.get_ident())
        return original_read(value, media_type=media_type, max_bytes=max_bytes)

    monkeypatch.setattr(resolver, "_read_local", tracked_read)
    await resolver.resolve("document.pdf", max_bytes=100)

    assert read_threads
    assert read_threads[0] != event_loop_thread
