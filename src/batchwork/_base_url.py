"""Provider base URL validation and normalization."""

from __future__ import annotations

from ipaddress import ip_address
from urllib.parse import urlsplit, urlunsplit


class BaseUrlError(ValueError):
    """A provider base URL violates the outbound credential policy."""


def _policy_error(label: str) -> BaseUrlError:
    return BaseUrlError(
        f"{label} must be absolute HTTPS (HTTP allowed for loopback), without "
        "userinfo, query, or fragment."
    )


def normalize_base_url(value: str, label: str = "base URL") -> str:
    """Validate a provider endpoint and remove trailing path slashes."""
    raw_scheme, separator, remainder = value.partition("://")
    authority = remainder
    for delimiter in "/?#":
        authority = authority.partition(delimiter)[0]
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
    except ValueError as error:
        raise _policy_error(label) from error
    try:
        _ = parsed.port
    except ValueError as error:
        raise BaseUrlError(f"{label} contains an invalid port.") from error

    loopback = hostname == "localhost"
    if hostname is not None:
        try:
            loopback = loopback or ip_address(hostname).is_loopback
        except ValueError:
            pass

    if (
        not separator
        or raw_scheme.lower() != parsed.scheme
        or not hostname
        or any(character.isspace() for character in authority)
        or parsed.username is not None
        or parsed.password is not None
        or "?" in value
        or "#" in value
        or (parsed.scheme != "https" and not (parsed.scheme == "http" and loopback))
    ):
        raise _policy_error(label)

    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", ""))
