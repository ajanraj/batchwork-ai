"""Shared provider capability checks."""

from __future__ import annotations

from collections.abc import Mapping

from batchwork.errors import _UnsupportedSettingError
from batchwork.types import BatchProvider

BATCH_METADATA_PROVIDERS = frozenset(
    {
        BatchProvider.OPENAI,
        BatchProvider.GROQ,
        BatchProvider.MISTRAL,
        BatchProvider.TOGETHER,
    }
)


def supports_batch_metadata(provider: BatchProvider) -> bool:
    """Return whether the provider accepts submission-level batch metadata."""

    return provider in BATCH_METADATA_PROVIDERS


def validate_batch_metadata(provider: BatchProvider, metadata: Mapping[str, str] | None) -> None:
    """Reject unsupported batch metadata before provider-side mutation."""

    if metadata and not supports_batch_metadata(provider):
        raise _UnsupportedSettingError(
            f'batchwork: provider "{provider.value}" does not support '
            "submission-level batch metadata."
        )
