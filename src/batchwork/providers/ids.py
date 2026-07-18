"""Untrusted provider identifier validation."""

from __future__ import annotations

import re

from batchwork.errors import BatchworkError

_SIMPLE = re.compile(r"^[A-Za-z0-9_-]+$")


def simple_provider_id(label: str, value: str) -> str:
    if not _SIMPLE.fullmatch(value):
        raise BatchworkError(f"batchwork: invalid {label}.")
    return value


def prefixed_provider_id(label: str, value: str, prefix: str) -> str:
    parts = value.split("/")
    if len(parts) != 2 or parts[0] != prefix or not _SIMPLE.fullmatch(parts[1]):
        raise BatchworkError(f"batchwork: invalid {label}.")
    return value
