"""Runtime guards that preserve JSON object key types for static analysis."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeGuard


def is_string_mapping(value: object) -> TypeGuard[Mapping[str, object]]:
    """Return whether value is a mapping with string keys."""

    return isinstance(value, Mapping) and all(isinstance(key, str) for key in value)


__all__ = ["is_string_mapping"]
