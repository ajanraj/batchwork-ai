"""Secure media resolution for providers that require inline data."""

from __future__ import annotations

import asyncio
import base64
import binascii
import mimetypes
import ssl
from collections.abc import AsyncIterable, AsyncIterator, Iterable
from dataclasses import dataclass
from typing import Protocol, TypeAlias
from urllib.parse import unquote_to_bytes, urljoin, urlsplit

import httpcore
import httpx
from httpcore._backends.anyio import AnyIOBackend

from ._network import (
    AddressResolutionFailureReason,
    ResolvedAddresses,
    resolve_public_addresses,
    validate_public_address,
)
from .errors import MediaResolutionError
from .types import (
    MediaSource,
    ProviderFileReference,
    TaggedFileDataData,
    TaggedFileDataReference,
    TaggedFileDataText,
    TaggedFileDataUrl,
)

_MAX_REDIRECTS = 5
SocketOption: TypeAlias = (
    tuple[int, int, int] | tuple[int, int, bytes | bytearray] | tuple[int, int, None, int]
)


def _socket_options(value: object) -> list[SocketOption] | None:
    if value is None:
        return None
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError("batchwork: socket_options must be an iterable of socket option tuples")
    result: list[SocketOption] = []
    for option in value:
        if not isinstance(option, tuple):
            raise TypeError("batchwork: each socket option must be a tuple")
        if len(option) == 3:
            level, name, payload = option
            if not isinstance(level, int) or not isinstance(name, int):
                raise TypeError("batchwork: socket option identifiers must be integers")
            if isinstance(payload, int):
                result.append((level, name, payload))
            elif isinstance(payload, (bytes, bytearray)):
                result.append((level, name, payload))
            else:
                raise TypeError("batchwork: socket option payload has an unsupported type")
        elif len(option) == 4:
            level, name, payload, length = option
            if (
                not isinstance(level, int)
                or not isinstance(name, int)
                or payload is not None
                or not isinstance(length, int)
            ):
                raise TypeError("batchwork: socket option tuple has invalid values")
            result.append((level, name, None, length))
        else:
            raise TypeError("batchwork: socket option tuple must contain three or four values")
    return result


def _timeout(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    raise TypeError("batchwork: network timeout must be numeric or None")


@dataclass(frozen=True, slots=True)
class ResolvedMedia:
    data: bytes
    media_type: str


class MediaResolver(Protocol):
    async def resolve(
        self,
        source: MediaSource,
        *,
        media_type: str | None = None,
        max_bytes: int,
    ) -> ResolvedMedia: ...


def _validate_address(address: str) -> None:
    failure = validate_public_address(address)
    if failure is None:
        return
    if failure.reason is AddressResolutionFailureReason.INVALID:
        raise MediaResolutionError(
            f'batchwork: DNS returned invalid address "{address}"'
        ) from failure.cause
    if failure.reason is AddressResolutionFailureReason.NON_GLOBAL:
        raise MediaResolutionError(
            f'batchwork: refused media address "{address}" because it is not globally routable'
        )
    raise RuntimeError("batchwork: unexpected address validation failure")


async def _resolve_public_addresses(host: str, port: int) -> tuple[str, ...]:
    result = await resolve_public_addresses(host, port)
    if isinstance(result, ResolvedAddresses):
        return result.addresses
    if result.reason is AddressResolutionFailureReason.LOOKUP:
        raise MediaResolutionError(
            f'batchwork: failed to resolve media host "{host}"'
        ) from result.cause
    if result.reason is AddressResolutionFailureReason.EMPTY:
        raise MediaResolutionError(f'batchwork: media host "{host}" resolved to no addresses')
    address = str(result.address)
    _validate_address(address)
    raise RuntimeError("batchwork: unexpected media address resolution failure")


class _PinnedNetworkBackend(httpcore.AsyncNetworkBackend):
    """Resolve, validate, and pin every new TCP connection."""

    def __init__(self) -> None:
        self._backend = AnyIOBackend()

    async def connect_tcp(
        self,
        host: str,
        port: int,
        *args: object,
        local_address: str | None = None,
        socket_options: Iterable[SocketOption] | None = None,
        **options: object,
    ) -> httpcore.AsyncNetworkStream:
        if len(args) > 3:
            raise TypeError("batchwork: connect_tcp received too many positional arguments")
        timeout_seconds = _timeout(args[0] if args else options.get("timeout"))
        raw_local_address = args[1] if len(args) > 1 else local_address
        if raw_local_address is not None and not isinstance(raw_local_address, str):
            raise TypeError("batchwork: local_address must be a string or None")
        resolved_socket_options = _socket_options(args[2] if len(args) > 2 else socket_options)
        addresses = await _resolve_public_addresses(host, port)
        last_error: httpcore.ConnectError | httpcore.ConnectTimeout | None = None
        for address in addresses:
            try:
                return await self._backend.connect_tcp(
                    address,
                    timeout=timeout_seconds,
                    port=port,
                    local_address=raw_local_address,
                    socket_options=resolved_socket_options,
                )
            except (httpcore.ConnectError, httpcore.ConnectTimeout) as error:
                last_error = error
        message = f'batchwork: failed to connect to media host "{host}" at any validated address'
        if isinstance(last_error, httpcore.ConnectTimeout):
            raise httpcore.ConnectTimeout(message) from last_error
        if isinstance(last_error, httpcore.ConnectError):
            raise httpcore.ConnectError(message) from last_error
        raise MediaResolutionError(message)

    async def connect_unix_socket(
        self,
        path: str,
        *args: object,
        socket_options: Iterable[SocketOption] | None = None,
        **options: object,
    ) -> httpcore.AsyncNetworkStream:
        del path, args, socket_options, options
        raise MediaResolutionError("batchwork: Unix sockets are not valid media sources")

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)


class _CoreStream(httpx.AsyncByteStream):
    def __init__(self, stream: AsyncIterable[object]) -> None:
        self._stream = stream

    async def __aiter__(self) -> AsyncIterator[bytes]:
        async for chunk in self._stream:
            if not isinstance(chunk, bytes):
                raise MediaResolutionError("batchwork: media transport returned a non-byte chunk")
            yield chunk

    async def aclose(self) -> None:
        close = getattr(self._stream, "aclose", None)
        if close is not None:
            await close()


class _PinnedTransport(httpx.AsyncBaseTransport):
    def __init__(self) -> None:
        self._pool = httpcore.AsyncConnectionPool(
            ssl_context=ssl.create_default_context(),
            network_backend=_PinnedNetworkBackend(),
            max_keepalive_connections=0,
            retries=0,
        )

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        response = await self._pool.handle_async_request(
            httpcore.Request(
                method=request.method,
                url=httpcore.URL(
                    scheme=request.url.raw_scheme,
                    host=request.url.raw_host,
                    port=request.url.port,
                    target=request.url.raw_path,
                ),
                headers=request.headers.raw,
                content=request.stream,
                extensions=request.extensions,
            )
        )
        stream = response.stream
        if not isinstance(stream, AsyncIterable):
            raise MediaResolutionError("batchwork: media transport returned an invalid stream")
        return httpx.Response(
            status_code=response.status,
            headers=response.headers,
            stream=_CoreStream(stream),
            extensions=response.extensions,
        )

    async def aclose(self) -> None:
        await self._pool.aclose()


class _PinnedBorrowedTransport(httpx.AsyncBaseTransport):
    """Pin validated targets while preserving caller ownership."""

    def __init__(self, transport: httpx.AsyncBaseTransport) -> None:
        self._transport = transport

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        host = request.url.host
        if host is None:
            raise MediaResolutionError("batchwork: media URL must include a host")
        authority = request.url.netloc.decode("ascii")
        addresses = await _resolve_public_addresses(host, request.url.port or 443)
        last_error: httpx.ConnectError | httpx.ConnectTimeout | None = None
        for address in addresses:
            pinned = httpx.Request(
                request.method,
                request.url.copy_with(host=address),
                headers=request.headers,
                stream=request.stream,
                extensions={**request.extensions, "sni_hostname": host},
            )
            pinned.headers["host"] = authority
            try:
                return await self._transport.handle_async_request(pinned)
            except (httpx.ConnectError, httpx.ConnectTimeout) as error:
                last_error = error
        message = f'batchwork: failed to connect to media host "{host}" at any validated address'
        if isinstance(last_error, httpx.ConnectTimeout):
            raise httpx.ConnectTimeout(message, request=request) from last_error
        if isinstance(last_error, httpx.ConnectError):
            raise httpx.ConnectError(message, request=request) from last_error
        raise MediaResolutionError(message)


def _sniff_media_type(data: bytes) -> str | None:
    signatures = (
        (b"\x89PNG\r\n\x1a\n", "image/png"),
        (b"\xff\xd8\xff", "image/jpeg"),
        (b"GIF87a", "image/gif"),
        (b"GIF89a", "image/gif"),
        (b"RIFF", "image/webp"),
        (b"%PDF-", "application/pdf"),
    )
    for signature, detected in signatures:
        if data.startswith(signature):
            if detected == "image/webp" and data[8:12] != b"WEBP":
                continue
            return detected
    return None


def _validate_size(data: bytes, max_bytes: int) -> None:
    if len(data) > max_bytes:
        raise MediaResolutionError(
            f"batchwork: media is {len(data)} bytes, exceeding the {max_bytes} byte limit"
        )


def _decode_inline(value: str) -> tuple[bytes, str | None]:
    if value.startswith("data:"):
        header, separator, payload = value[5:].partition(",")
        if not separator:
            raise MediaResolutionError("batchwork: malformed data URL")
        segments = header.split(";")
        declared = segments[0] or None
        try:
            data = (
                base64.b64decode(payload, validate=True)
                if "base64" in segments
                else unquote_to_bytes(payload)
            )
        except (binascii.Error, ValueError) as error:
            raise MediaResolutionError("batchwork: malformed data URL payload") from error
        return data, declared
    try:
        return base64.b64decode(value, validate=True), None
    except (binascii.Error, ValueError) as error:
        raise MediaResolutionError(
            "batchwork: string media must be an HTTPS URL, data URL, or base64 payload"
        ) from error


def _validated_type(data: bytes, declared: str | None, fallback: str | None = None) -> str:
    declared = declared.split(";", 1)[0].strip().lower() if declared else None
    detected = _sniff_media_type(data)
    declared_top_level = declared.split("/", 1)[0] if declared is not None else None
    declared_is_partial = declared is not None and ("/" not in declared or declared.endswith("/*"))
    types_match = detected == declared or (
        declared_is_partial
        and detected is not None
        and detected.startswith(f"{declared_top_level}/")
    )
    if detected is not None and declared is not None and not types_match:
        raise MediaResolutionError(
            f'batchwork: media type "{declared}" does not match detected type "{detected}"'
        )
    selected = detected or declared or fallback
    if selected is None or "/" not in selected:
        raise MediaResolutionError("batchwork: media type is unknown; provide media_type")
    return selected


class DefaultMediaResolver:
    """Resolve inline data or HTTPS media with SSRF and rebinding protection."""

    def __init__(
        self,
        *,
        timeout: httpx.Timeout | float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._timeout = timeout
        self._transport = transport

    async def resolve(
        self,
        source: MediaSource,
        *,
        media_type: str | None = None,
        max_bytes: int,
    ) -> ResolvedMedia:
        if isinstance(source, (ProviderFileReference, TaggedFileDataReference)):
            raise MediaResolutionError("batchwork: provider file references cannot be downloaded")
        if isinstance(source, TaggedFileDataData):
            raw_data = source.data
            if isinstance(raw_data, bytes):
                _validate_size(raw_data, max_bytes)
                return ResolvedMedia(raw_data, _validated_type(raw_data, media_type))
            data, declared = _decode_inline(raw_data)
            _validate_size(data, max_bytes)
            return ResolvedMedia(data, _validated_type(data, media_type or declared))
        elif isinstance(source, TaggedFileDataUrl):
            source = source.url
        elif isinstance(source, TaggedFileDataText):
            source = source.text.encode()
        if isinstance(source, bytes):
            _validate_size(source, max_bytes)
            return ResolvedMedia(source, _validated_type(source, media_type))

        value = str(source)
        parsed = urlsplit(value)
        if parsed.scheme and parsed.scheme != "https":
            if parsed.scheme != "data":
                raise MediaResolutionError("batchwork: remote media URLs must use HTTPS")
        if parsed.scheme != "https":
            data, declared = _decode_inline(value)
            _validate_size(data, max_bytes)
            return ResolvedMedia(data, _validated_type(data, media_type or declared))
        if parsed.username is not None or parsed.password is not None:
            raise MediaResolutionError("batchwork: remote media URLs must not include credentials")
        return await self._download(value, media_type=media_type, max_bytes=max_bytes)

    async def _download(self, url: str, *, media_type: str | None, max_bytes: int) -> ResolvedMedia:
        transport = (
            _PinnedTransport()
            if self._transport is None
            else _PinnedBorrowedTransport(self._transport)
        )
        async with httpx.AsyncClient(
            transport=transport,
            timeout=self._timeout,
            follow_redirects=False,
            headers={"accept-encoding": "identity"},
        ) as client:
            current = url
            for redirects in range(_MAX_REDIRECTS + 1):
                try:
                    async with client.stream("GET", current) as response:
                        if response.status_code in {301, 302, 303, 307, 308}:
                            location = response.headers.get("location")
                            if location is None:
                                raise MediaResolutionError(
                                    "batchwork: media redirect omitted Location"
                                )
                            if redirects == _MAX_REDIRECTS:
                                raise MediaResolutionError("batchwork: too many media redirects")
                            current = urljoin(current, location)
                            redirected = urlsplit(current)
                            if redirected.scheme != "https":
                                raise MediaResolutionError(
                                    "batchwork: media redirects must remain on HTTPS"
                                )
                            if redirected.username is not None or redirected.password is not None:
                                raise MediaResolutionError(
                                    "batchwork: remote media URLs must not include credentials"
                                )
                            continue
                        try:
                            response.raise_for_status()
                        except httpx.HTTPStatusError as error:
                            raise MediaResolutionError(
                                f"batchwork: media download returned HTTP {response.status_code}"
                            ) from error
                        length = response.headers.get("content-length")
                        if length is not None and int(length) > max_bytes:
                            raise MediaResolutionError(
                                f"batchwork: media exceeds the {max_bytes} byte limit"
                            )
                        chunks: list[bytes] = []
                        size = 0
                        async for chunk in response.aiter_bytes():
                            size += len(chunk)
                            if size > max_bytes:
                                raise MediaResolutionError(
                                    f"batchwork: media exceeds the {max_bytes} byte limit"
                                )
                            chunks.append(chunk)
                        data = b"".join(chunks)
                        fallback, _ = mimetypes.guess_type(current)
                        declared = media_type or response.headers.get("content-type") or fallback
                        return ResolvedMedia(data, _validated_type(data, declared))
                except httpx.HTTPError as error:
                    raise MediaResolutionError("batchwork: media download failed") from error
        raise AssertionError("unreachable")
