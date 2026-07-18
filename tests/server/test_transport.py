from __future__ import annotations

import asyncio
import socket

import pytest

from batchwork.errors import BatchworkError
from batchwork.server import (
    PinnedWebhookTransport,
    parse_webhook_url,
    resolve_public_addresses,
    sign_webhook,
)


def test_webhook_url_requires_https_without_credentials() -> None:
    with pytest.raises(BatchworkError, match="must use https"):
        parse_webhook_url("http://example.com/hook")
    with pytest.raises(BatchworkError, match="must not include credentials"):
        parse_webhook_url("https://user:secret@example.com/hook")


@pytest.mark.parametrize(
    "address",
    ["127.0.0.1", "10.0.0.1", "169.254.1.1", "::1", "fc00::1"],
)
async def test_private_and_reserved_literals_are_rejected(address: str) -> None:
    with pytest.raises(BatchworkError, match="private, or reserved"):
        await resolve_public_addresses(address, 443)


async def test_webhook_dns_rejects_invalid_address(monkeypatch: pytest.MonkeyPatch) -> None:
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

    with pytest.raises(BatchworkError, match="returned an invalid address"):
        await resolve_public_addresses("hooks.example.test", 443)


@pytest.mark.parametrize(
    ("headers", "message"),
    [
        ({"bad\r\nname": "value"}, "name is invalid"),
        ({"x-test": "value\r\ninjected: yes"}, "invalid value"),
        ({"host": "attacker.test"}, "reserved"),
    ],
)
async def test_pinned_transport_rejects_unsafe_headers_before_network(
    headers: dict[str, str], message: str
) -> None:
    with pytest.raises(BatchworkError, match=message):
        await PinnedWebhookTransport().post("https://example.com/hook", b"{}", headers)


@pytest.mark.parametrize(
    ("url", "request_target"),
    [
        ("https://hooks.example.test/path?attempt=1", b"/path?attempt=1"),
        ("https://hooks.example.test/hooks/a%2Fb", b"/hooks/a%2Fb"),
        ("https://hooks.example.test/hooks/a%20b", b"/hooks/a%20b"),
    ],
)
async def test_pinned_transport_writes_valid_http_request(
    monkeypatch: pytest.MonkeyPatch,
    url: str,
    request_target: bytes,
) -> None:
    from batchwork.server import transport as transport_module

    class Reader:
        async def readuntil(self, separator: bytes) -> bytes:
            assert separator == b"\r\n\r\n"
            return b"HTTP/1.1 204 No Content\r\ncontent-length: 0\r\n\r\n"

    class Writer:
        def __init__(self) -> None:
            self.wire = b""

        def write(self, data: bytes) -> None:
            self.wire += data

        async def drain(self) -> None:
            return None

        def close(self) -> None:
            return None

        async def wait_closed(self) -> None:
            return None

    writer = Writer()

    async def resolve(_host: str, _port: int) -> tuple[str, ...]:
        return ("203.0.113.1",)

    async def connect(*_args: object, **_kwargs: object) -> tuple[Reader, Writer]:
        return Reader(), writer

    monkeypatch.setattr(transport_module, "resolve_public_addresses", resolve)
    monkeypatch.setattr(transport_module.asyncio, "open_connection", connect)
    status = await PinnedWebhookTransport().post(
        url,
        b"{}",
        {"content-type": "application/json", "webhook-id": "delivery-1"},
    )
    assert status == 204
    assert writer.wire.startswith(b"POST " + request_target + b" HTTP/1.1\r\n")
    assert b"host: hooks.example.test\r\n" in writer.wire
    assert b"webhook-id: delivery-1\r\n\r\n{}" in writer.wire


async def test_pinned_transport_tries_each_validated_address(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from batchwork.server import transport as transport_module

    class Reader:
        async def readuntil(self, separator: bytes) -> bytes:
            assert separator == b"\r\n\r\n"
            return b"HTTP/1.1 204 No Content\r\ncontent-length: 0\r\n\r\n"

    class Writer:
        def __init__(self) -> None:
            self.closed = False

        def write(self, _data: bytes) -> None:
            return None

        async def drain(self) -> None:
            return None

        def close(self) -> None:
            self.closed = True

        async def wait_closed(self) -> None:
            return None

    writer = Writer()
    attempts: list[tuple[str, object]] = []

    async def resolve(_host: str, _port: int) -> tuple[str, ...]:
        return ("203.0.113.1", "203.0.113.2")

    async def connect(address: str, _port: int, **kwargs: object) -> tuple[Reader, Writer]:
        attempts.append((address, kwargs.get("server_hostname")))
        if address == "203.0.113.1":
            raise OSError("first address unavailable")
        return Reader(), writer

    monkeypatch.setattr(transport_module, "resolve_public_addresses", resolve)
    monkeypatch.setattr(transport_module.asyncio, "open_connection", connect)

    status = await PinnedWebhookTransport().post("https://hooks.example.test/path", b"{}", {})

    assert status == 204
    assert attempts == [
        ("203.0.113.1", "hooks.example.test"),
        ("203.0.113.2", "hooks.example.test"),
    ]
    assert writer.closed


async def test_pinned_transport_reports_when_all_validated_addresses_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from batchwork.server import transport as transport_module

    attempts: list[str] = []

    async def resolve(_host: str, _port: int) -> tuple[str, ...]:
        return ("203.0.113.1", "203.0.113.2")

    async def connect(address: str, _port: int, **_kwargs: object) -> None:
        attempts.append(address)
        raise OSError(f"{address} unavailable")

    monkeypatch.setattr(transport_module, "resolve_public_addresses", resolve)
    monkeypatch.setattr(transport_module.asyncio, "open_connection", connect)

    with pytest.raises(BatchworkError, match="could not connect to any resolved address"):
        await PinnedWebhookTransport().post("https://hooks.example.test/path", b"{}", {})

    assert attempts == ["203.0.113.1", "203.0.113.2"]


@pytest.mark.parametrize("event_id", ["event\r\ninjected: yes", "event\nnext"])
def test_signing_rejects_unsafe_header_identity(event_id: str) -> None:
    with pytest.raises(BatchworkError, match="valid HTTP header value"):
        sign_webhook("secret", event_id, b"{}")
