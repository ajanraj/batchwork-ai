"""Public exception hierarchy."""

from __future__ import annotations


class BatchworkError(Exception):
    """Base class for all package errors."""


class BatchStateError(BatchworkError):
    """The requested operation is invalid for the batch's current state."""


class BatchTimeoutError(BatchworkError):
    """Waiting for a batch exceeded the requested timeout."""


class BatchClosedError(BatchworkError):
    """An operation was attempted after the owning client closed."""


class UnsupportedProviderError(BatchworkError):
    """The requested provider or modality is unsupported."""

    def __init__(self, provider: str, detail: str | None = None) -> None:
        self.provider = provider
        super().__init__(
            detail
            or (
                f'batchwork: provider "{provider}" is not supported; supported providers: '
                "openai, anthropic, google, groq, mistral, together, xai"
            )
        )


class MissingDependencyError(BatchworkError):
    """An optional package required by the requested adapter is unavailable."""

    def __init__(self, package: str, extra: str) -> None:
        super().__init__(
            f'batchwork: install the optional dependency with `uv add "batchwork-ai[{extra}]"` '
            f"to use {package}."
        )


class MediaResolutionError(BatchworkError):
    """Remote or inline media could not be resolved safely."""
