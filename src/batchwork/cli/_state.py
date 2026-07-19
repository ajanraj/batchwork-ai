"""Root command state shared by private CLI commands."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class OutputMode(StrEnum):
    HUMAN = "human"
    JSON = "json"
    JSONL = "jsonl"


@dataclass(frozen=True, slots=True)
class RootOptions:
    config: Path | None
    registry: Path | None
    profile: str | None
    output_mode: OutputMode | None
    quiet: bool
    progress: bool
    color: bool | None
