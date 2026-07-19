"""Incremental compact JSON encoding with a strict byte ceiling."""

from __future__ import annotations

import json
from collections.abc import Mapping


class JsonSizeExceeded(ValueError):
    """Raised as soon as compact JSON crosses its byte ceiling."""

    def __init__(self, maximum: int, known_size: int | None = None) -> None:
        super().__init__()
        self.maximum = maximum
        self.known_size = known_size


def encode_bounded_json(value: Mapping[str, object], maximum: int) -> bytes:
    """Encode compact JSON, stopping at the first encoded chunk over ``maximum``."""
    chunks = iter(json.JSONEncoder(ensure_ascii=False, separators=(",", ":")).iterencode(value))
    encoded = bytearray()
    for chunk in chunks:
        part = chunk.encode()
        if len(encoded) + len(part) > maximum:
            raise JsonSizeExceeded(maximum, len(encoded) + len(part))
        encoded.extend(part)
    return bytes(encoded)
