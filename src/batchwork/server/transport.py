"""SSRF-resistant webhook URL validation and delivery."""

from __future__ import annotations

import asyncio
import contextlib
import ssl
import string
from collections.abc import Awaitable, Callable, Mapping
from typing import Protocol
from urllib.parse import SplitResult, quote, urlsplit

from batchwork._network import (
    AddressResolutionFailureReason,
    ResolvedAddresses,
)
from batchwork._network import (
    resolve_public_addresses as resolve_addresses,
)
from batchwork.errors import BatchworkError

WebhookUrlValidator = Callable[[SplitResult], None | Awaitable[None]]

_HEADER_TOKEN = frozenset(
    "!#$%&'*+-.^_`|~0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
)
_RESERVED_HEADERS = frozenset({"connection", "content-length", "host"})


def _quote_request_target(value: str, *, safe: str) -> str:
    """Quote text while retaining valid escapes from the raw URL."""

    parts: list[str] = []
    start = 0
    index = 0
    while index < len(value):
        if (
            value[index] == "%"
            and index + 2 < len(value)
            and value[index + 1] in string.hexdigits
            and value[index + 2] in string.hexdigits
        ):
            parts.append(quote(value[start:index], safe=safe))
            parts.append(value[index : index + 3])
            index += 3
            start = index
        else:
            index += 1
    parts.append(quote(value[start:], safe=safe))
    return "".join(parts)


class WebhookTransport(Protocol):
    """Injectable outbound webhook transport."""

    async def post(self, url: str, body: bytes, headers: Mapping[str, str]) -> int: ...


def parse_webhook_url(raw_url: str) -> SplitResult:
    """Parse and apply policy checks that do not require DNS."""

    if any(ord(character) < 32 for character in raw_url):
        raise BatchworkError("batchwork: webhook_url must be a valid URL.")
    try:
        parsed = urlsplit(raw_url)
        port = parsed.port
    except ValueError as error:
        raise BatchworkError("batchwork: webhook_url must be a valid URL.") from error
    if parsed.scheme != "https":
        raise BatchworkError("batchwork: webhook_url must use https.")
    if not parsed.hostname:
        raise BatchworkError("batchwork: webhook_url must include a hostname.")
    if parsed.username or parsed.password:
        raise BatchworkError("batchwork: webhook_url must not include credentials.")
    if port is not None and not 1 <= port <= 65535:
        raise BatchworkError("batchwork: webhook_url has an invalid port.")
    return parsed


def _validated_headers(headers: Mapping[str, str]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for name, value in headers.items():
        if not name or any(character not in _HEADER_TOKEN for character in name):
            raise BatchworkError("batchwork: webhook header name is invalid.")
        lowered = name.lower()
        if lowered in _RESERVED_HEADERS:
            raise BatchworkError(f"batchwork: webhook header {name!r} is reserved.")
        if any(ord(character) < 32 or ord(character) == 127 for character in value):
            raise BatchworkError(f"batchwork: webhook header {name!r} has an invalid value.")
        try:
            value.encode("ascii")
        except UnicodeEncodeError as error:
            raise BatchworkError(
                f"batchwork: webhook header {name!r} has an invalid value."
            ) from error
        normalized[lowered] = value
    return normalized


async def resolve_public_addresses(host: str, port: int) -> tuple[str, ...]:
    """Resolve a host, rejecting the whole target if any address is non-global."""

    result = await resolve_addresses(host, port)
    if isinstance(result, ResolvedAddresses):
        return result.addresses
    if result.reason is AddressResolutionFailureReason.LOOKUP:
        raise BatchworkError(
            f"batchwork: webhook hostname {host!r} could not be resolved."
        ) from result.cause
    if result.reason is AddressResolutionFailureReason.EMPTY:
        raise BatchworkError(f"batchwork: webhook hostname {host!r} resolved to no addresses.")
    if result.reason is AddressResolutionFailureReason.NON_GLOBAL:
        raise BatchworkError(
            "batchwork: webhook_url must not target localhost, private, or reserved networks."
        )
    message = f"batchwork: webhook hostname {host!r} returned an invalid address."
    raise BatchworkError(message) from result.cause


async def validate_webhook_url(parsed: SplitResult) -> None:
    """Default URL policy, including DNS resolution."""

    host = parsed.hostname
    if host is None:
        raise BatchworkError("batchwork: webhook_url must include a hostname.")
    await resolve_public_addresses(host, parsed.port or 443)


class PinnedWebhookTransport:
    """HTTPS/1.1 transport pinned to a validated address for each attempt."""

    def __init__(
        self,
        *,
        connect_timeout: float = 10.0,
        response_timeout: float = 300.0,
        max_header_bytes: int = 65_536,
    ) -> None:
        self._connect_timeout = connect_timeout
        self._response_timeout = response_timeout
        self._max_header_bytes = max_header_bytes
        self._ssl_context = ssl.create_default_context()

    async def post(self, url: str, body: bytes, headers: Mapping[str, str]) -> int:
        parsed = parse_webhook_url(url)
        supplied_headers = _validated_headers(headers)
        host = parsed.hostname
        if host is None:
            raise BatchworkError("batchwork: webhook_url must include a hostname.")
        port = parsed.port or 443
        addresses = await resolve_public_addresses(host, port)

        connection_error: OSError | TimeoutError | None = None
        for address in addresses:
            try:
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(
                        address,
                        port,
                        ssl=self._ssl_context,
                        server_hostname=host,
                    ),
                    timeout=self._connect_timeout,
                )
            except (OSError, TimeoutError) as error:
                connection_error = error
                continue
            break
        else:
            raise BatchworkError(
                f"batchwork: webhook delivery to {url} could not connect to any resolved address."
            ) from connection_error

        try:
            target = _quote_request_target(parsed.path or "/", safe="/:@-._~!$&'()*+,;=")
            if parsed.query:
                query = _quote_request_target(parsed.query, safe="=&;:@/?+-._~")
                target = f"{target}?{query}"
            wire_host = host.encode("idna").decode("ascii")
            if ":" in wire_host:
                wire_host = f"[{wire_host}]"
            authority = wire_host if port == 443 else f"{wire_host}:{port}"
            outbound = {
                "connection": "close",
                "content-length": str(len(body)),
                "host": authority,
                **supplied_headers,
            }
            request_head = [f"POST {target} HTTP/1.1"]
            request_head.extend(f"{key}: {value}" for key, value in outbound.items())
            wire = "\r\n".join(request_head).encode("ascii") + b"\r\n\r\n" + body
            writer.write(wire)
            await asyncio.wait_for(writer.drain(), timeout=self._response_timeout)
            response_head = await asyncio.wait_for(
                reader.readuntil(b"\r\n\r\n"), timeout=self._response_timeout
            )
        except (OSError, TimeoutError, asyncio.IncompleteReadError) as error:
            raise BatchworkError(
                f"batchwork: webhook delivery to {url} failed while reading the response."
            ) from error
        except asyncio.LimitOverrunError as error:
            raise BatchworkError(
                "batchwork: webhook response headers exceeded the safe limit."
            ) from error
        finally:
            writer.close()
            with contextlib.suppress(OSError):
                await writer.wait_closed()

        if len(response_head) > self._max_header_bytes:
            raise BatchworkError("batchwork: webhook response headers exceeded the safe limit.")
        try:
            status_line = response_head.split(b"\r\n", 1)[0].decode("ascii")
            version, raw_status, _reason = status_line.split(" ", 2)
            status = int(raw_status)
        except (UnicodeDecodeError, ValueError) as error:
            raise BatchworkError(
                "batchwork: webhook target returned an invalid HTTP response."
            ) from error
        if not version.startswith("HTTP/1.") or not 100 <= status <= 599:
            raise BatchworkError("batchwork: webhook target returned an invalid HTTP response.")
        return status


__all__ = [
    "PinnedWebhookTransport",
    "WebhookTransport",
    "WebhookUrlValidator",
    "parse_webhook_url",
    "resolve_public_addresses",
    "validate_webhook_url",
]
