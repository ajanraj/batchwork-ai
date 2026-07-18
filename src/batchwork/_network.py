"""Shared private network-address policy."""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from dataclasses import dataclass
from enum import Enum


class AddressResolutionFailureReason(Enum):
    LOOKUP = "lookup"
    EMPTY = "empty"
    INVALID = "invalid"
    NON_GLOBAL = "non_global"


@dataclass(frozen=True, slots=True)
class AddressResolutionFailure:
    reason: AddressResolutionFailureReason
    address: object | None = None
    cause: Exception | None = None


@dataclass(frozen=True, slots=True)
class ResolvedAddresses:
    addresses: tuple[str, ...]


AddressResolution = ResolvedAddresses | AddressResolutionFailure


def validate_public_address(address: object) -> AddressResolutionFailure | None:
    if not isinstance(address, str):
        return AddressResolutionFailure(AddressResolutionFailureReason.INVALID, address)
    try:
        parsed = ipaddress.ip_address(address)
    except ValueError as error:
        return AddressResolutionFailure(AddressResolutionFailureReason.INVALID, address, error)
    if not parsed.is_global:
        return AddressResolutionFailure(AddressResolutionFailureReason.NON_GLOBAL, address)
    return None


async def resolve_public_addresses(host: str, port: int) -> AddressResolution:
    """Resolve one target and reject its complete address set on any unsafe record."""

    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        try:
            records = await asyncio.get_running_loop().getaddrinfo(
                host,
                port,
                family=socket.AF_UNSPEC,
                type=socket.SOCK_STREAM,
            )
        except OSError as error:
            return AddressResolutionFailure(AddressResolutionFailureReason.LOOKUP, cause=error)
        addresses: list[str] = []
        for record in records:
            address = record[4][0]
            if not isinstance(address, str):
                return AddressResolutionFailure(AddressResolutionFailureReason.INVALID, address)
            failure = validate_public_address(address)
            if failure is not None:
                return failure
            if address not in addresses:
                addresses.append(address)
    else:
        address = str(literal)
        failure = validate_public_address(address)
        if failure is not None:
            return failure
        addresses = [address]
    if not addresses:
        return AddressResolutionFailure(AddressResolutionFailureReason.EMPTY)
    return ResolvedAddresses(tuple(addresses))
