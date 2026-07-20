"""Safe, terminal-aware human presentation for the CLI."""

from __future__ import annotations

import os
import re
import shutil
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, TextIO
from urllib.parse import urlsplit, urlunsplit

import click

from batchwork.types import BatchRequestCounts, BatchResult, BatchResultStatus, BatchSnapshot

from ._contract import ErrorDetail, Job, Materialization
from ._registry import RegistryJob
from ._state import OutputMode, RootOptions

if TYPE_CHECKING:
    from ._lifecycle import LifecycleResult

_PREVIEW_LIMIT = 160
_URL = re.compile(r"https?://[^\s<>'\"]+", re.IGNORECASE)
_INLINE_DATA = re.compile(r"data:[^,\s]+,[^\s]+", re.IGNORECASE)
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SECRET_FIELD = re.compile(
    r"(?i)\b(authorization|proxy-authorization|api[-_ ]?key|x-api-key|"
    r"cookie|set-cookie|secret|access[-_ ]?token)\s*[:=]\s*"
    r"(?:bearer\s+)?[^\s,;]+"
)
_BEARER = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}")
_CREDENTIAL = re.compile(r"\b(?:sk|xai|key|token)-[A-Za-z0-9_-]{12,}\b", re.IGNORECASE)
_BASE64_TOKEN = re.compile(
    r"(?<![A-Za-z0-9+/=])(?=[A-Za-z0-9+/=]{100,}(?![A-Za-z0-9+/=]))"
    r"(?=[A-Za-z0-9+/=]*[+/=])[A-Za-z0-9+/]{98,}={0,2}"
)
_SIGNED_PATH_PARTS = frozenset({"auth", "signature", "sig", "token"})


def terminal_color(root: RootOptions, mode: OutputMode, stream: TextIO) -> bool:
    """Return whether ANSI color is safe and selected for this stream."""
    if mode is not OutputMode.HUMAN or root.color is False:
        return False
    if os.environ.get("TERM", "").lower() == "dumb":
        return False
    if root.color is True:
        return True
    return stream.isatty() and "NO_COLOR" not in os.environ


def styled(value: str, *, color: bool, fg: str | None = None, bold: bool = False) -> str:
    return click.style(value, fg=fg, bold=bold) if color else value


def _selector_lines(job: Job) -> list[str]:
    selector = job.name or job.record_id or job.provider_reference
    lines = [f"  Selector      {selector}"]
    if job.record_id is not None:
        lines.append(f"  Record        {job.record_id}")
    lines.append(f"  Direct        {job.provider_reference}")
    return lines


def human_job(job: Job, *, color: bool = False) -> str:
    title = styled("Job submitted", color=color, fg="green", bold=True)
    canonical_selector = job.record_id or job.provider_reference
    lines = [title, f"Job: {canonical_selector}", *_selector_lines(job)]
    lines.extend(
        (
            f"  Provider      {job.provider.value}",
            f"  Provider job  {job.provider_job_id}",
            f"  Modality      {job.modality or 'batch'}",
        )
    )
    if job.model is not None:
        lines.append(f"  Model         {job.model}")
    lines.append(f"  Status        {job.status.value if job.status else 'unknown'}")
    if job.request_counts is not None:
        lines.append(f"  Requests      {job.request_counts.total}")
    lines.extend(("", f"Resume: batchwork status {canonical_selector}"))
    return "\n".join(lines) + "\n"


def _finished(counts: BatchRequestCounts) -> int:
    return counts.completed + counts.failed + (counts.canceled or 0) + (counts.expired or 0)


def human_snapshot(
    result: LifecycleResult,
    *,
    title: str,
    color: bool = False,
) -> str:
    snapshot = result.snapshot
    counts = snapshot.request_counts
    heading = styled(title, color=color, fg="cyan", bold=True)
    job = result.resolved.record.job if result.resolved.record is not None else None
    if job is not None:
        selector_lines = _selector_lines(job)
    else:
        selector_lines = [
            f"  Selector      {result.resolved.machine_job}",
            f"  Direct        {result.resolved.provider.value}:{result.resolved.provider_job_id}",
        ]
    return (
        "\n".join(
            (
                heading,
                *selector_lines,
                f"  Provider      {snapshot.provider.value}",
                f"  Status        {snapshot.status.value}",
                (
                    f"  Requests      {counts.total} total · {counts.completed} completed · "
                    f"{counts.failed} failed"
                ),
            )
        )
        + "\n"
    )


def _safe_url(match: re.Match[str]) -> str:
    value = match.group(0)
    trailing = ""
    while value and value[-1] in ".,;:!?)":
        trailing = value[-1] + trailing
        value = value[:-1]
    try:
        parsed = urlsplit(value)
    except ValueError:
        return "[URL omitted]" + trailing
    path_parts = {part.lower() for part in parsed.path.split("/") if part}
    if path_parts & _SIGNED_PATH_PARTS:
        safe = urlunsplit((parsed.scheme, parsed.hostname or "", "/[signed path omitted]", "", ""))
        return safe + trailing
    if any(
        len(part) >= 24
        and any(character.isalpha() for character in part)
        and any(character.isdigit() for character in part)
        for part in path_parts
    ):
        safe = urlunsplit((parsed.scheme, parsed.hostname or "", "/[opaque path omitted]", "", ""))
        return safe + trailing
    if parsed.query or parsed.fragment or parsed.username or parsed.password:
        safe = urlunsplit((parsed.scheme, parsed.hostname or "", parsed.path, "", ""))
        return f"{safe}?[parameters omitted]{trailing}"
    return value + trailing


def safe_preview(
    value: str,
    *,
    limit: int = _PREVIEW_LIMIT,
    sensitive_names: Sequence[str] = (),
    sensitive_values: Sequence[str] = (),
    redact_urls: bool = False,
) -> tuple[str, bool]:
    """Return a bounded one-line preview with common secret-bearing forms removed."""
    preview = _CONTROL.sub("", value).replace("\r", " ").replace("\n", " ")
    for secret in sorted({item for item in sensitive_values if item}, key=len, reverse=True):
        preview = preview.replace(secret, "[redacted]")
    for name in sorted({item for item in sensitive_names if item}, key=len, reverse=True):
        preview = re.sub(
            rf"(?i)\b{re.escape(name)}\s*[:=]\s*[^\s,;]+",
            f"{name}: [redacted]",
            preview,
        )
    preview = _INLINE_DATA.sub("[inline data omitted]", preview)
    preview = _SECRET_FIELD.sub(lambda match: f"{match.group(1)}: [redacted]", preview)
    preview = _BEARER.sub("Bearer [redacted]", preview)
    preview = _CREDENTIAL.sub("[credential omitted]", preview)
    preview = _BASE64_TOKEN.sub("[inline data omitted]", preview)
    preview = _URL.sub("[URL omitted]" if redact_urls else _safe_url, preview)
    preview = " ".join(preview.split())
    if len(preview) <= limit:
        return preview, False
    return preview[: max(0, limit - 1)].rstrip() + "…", True


def _result_line(
    item: BatchResult,
    materialized: dict[str, list[str]],
    *,
    sensitive_names: Sequence[str],
    sensitive_values: Sequence[str],
) -> tuple[list[str], bool]:
    custom_id, truncated = safe_preview(
        item.custom_id,
        limit=80,
        sensitive_names=sensitive_names,
        sensitive_values=sensitive_values,
        redact_urls=True,
    )
    lines = [f"{custom_id}  {item.status.value}"]
    if item.embedding is not None:
        lines[0] += f"  {len(item.embedding)} dimensions"
    elif item.images is not None:
        count = len(item.images)
        lines[0] += f"  {count} image{'s' if count != 1 else ''}"
        paths = materialized.get(item.custom_id, [])
        if paths:
            lines.extend(f"  {path}" for path in paths)
        else:
            lines.append("  Not materialized; use --output-dir DIR to save images.")
    elif item.text:
        preview, text_truncated = safe_preview(
            item.text,
            sensitive_names=sensitive_names,
            sensitive_values=sensitive_values,
            redact_urls=True,
        )
        truncated = truncated or text_truncated
        lines.append(f"  {preview}")
    if item.error is not None:
        preview, error_truncated = safe_preview(
            item.error.message,
            sensitive_names=sensitive_names,
            sensitive_values=sensitive_values,
            redact_urls=True,
        )
        truncated = truncated or error_truncated
        code = f" [{item.error.code}]" if item.error.code is not None else ""
        lines.append(f"  Error{code}: {preview}")
    return lines, truncated


def human_results(
    result: LifecycleResult,
    *,
    materialization: Materialization | None = None,
    color: bool = False,
) -> str:
    items = result.results or []
    heading = styled("Results", color=color, fg="cyan", bold=True)
    materialized: dict[str, list[str]] = {}
    if materialization is not None:
        root = Path(materialization.output_dir)
        for image in materialization.images:
            materialized.setdefault(image.custom_id, []).append(str(root / image.path))
    sensitive_names = (
        *result.resolved.route.headers,
        *result.resolved.route.registry.header_env,
    )
    sensitive_values = (
        result.resolved.route.api_key,
        *result.resolved.route.headers.values(),
    )

    lines = [heading, f"Job: {result.resolved.machine_job}", ""]
    truncated = False
    for item in items:
        rendered, item_truncated = _result_line(
            item,
            materialized,
            sensitive_names=sensitive_names,
            sensitive_values=sensitive_values,
        )
        lines.extend(rendered)
        lines.append("")
        truncated = truncated or item_truncated

    successes = sum(item.status is BatchResultStatus.SUCCEEDED for item in items)
    failures = len(items) - successes
    lines.append(f"{successes} succeeded · {failures} errored")
    if materialization is not None:
        count = len(materialization.images)
        lines.extend(
            (
                f"{count} image{'s' if count != 1 else ''} saved",
                f"Manifest: {Path(materialization.output_dir) / materialization.manifest}",
            )
        )
    if truncated:
        lines.extend(("", "Preview truncated; use --json or --jsonl for complete data."))
    return "\n".join(lines) + "\n"


def human_error(error: ErrorDetail) -> str:
    """Render one bounded, recovery-oriented human diagnostic."""
    command = error.recovery.command if error.recovery is not None else None
    recovery = f"\nRecovery: {' '.join(command)}" if command else ""
    message, truncated = safe_preview(error.message, redact_urls=True)
    guidance = "\nDiagnostic truncated; use --json for structured details." if truncated else ""
    return f"Error [{error.code}]\n  {message}{guidance}{recovery}\n"


def _table_time(value: datetime | None) -> str:
    if value is None:
        return "-"
    return value.astimezone(UTC).strftime("%Y-%m-%d %H:%MZ")


def human_table(records: Sequence[RegistryJob], *, width: int | None = None) -> str:
    if not records:
        return "No local jobs.\n"
    terminal_width = width or shutil.get_terminal_size(fallback=(100, 24)).columns
    if terminal_width < 96:
        blocks: list[str] = []
        for record in records:
            job = record.job
            blocks.append(
                "\n".join(
                    (
                        f"Selector   {job.name or job.record_id}",
                        f"Record     {job.record_id or '-'}",
                        f"Provider   {job.provider.value}",
                        f"Status     {job.status.value if job.status else '-'}",
                        f"Submitted  {_table_time(job.provider_created_at or job.registered_at)}",
                        f"Completed  {_table_time(job.completed_at or job.terminal_at)}",
                        f"Direct     {job.provider_reference}",
                    )
                )
            )
        return "\n\n".join(blocks) + "\n"

    selector_width = max(
        len("SELECTOR"),
        *(len(record.job.name or record.job.record_id or "") for record in records),
    )
    selector_width = min(selector_width, max(16, terminal_width - 120))
    lines = [
        f"{'SELECTOR':<{selector_width}}  {'RECORD':<35}  PROVIDER   STATUS       "
        f"{'SUBMITTED':<17}  {'COMPLETED':<17}  DIRECT"
    ]
    for record in records:
        job = record.job
        selector = job.name or job.record_id or job.provider_reference
        # A long selector may exceed its column, but it is deliberately never truncated.
        lines.append(
            f"{selector:<{selector_width}}  {(job.record_id or '-'):<35}  "
            f"{job.provider.value:<9}  "
            f"{(job.status.value if job.status else '-'):<11}  "
            f"{_table_time(job.provider_created_at or job.registered_at):<17}  "
            f"{_table_time(job.completed_at or job.terminal_at):<17}  "
            f"{job.provider_reference}"
        )
    return "\n".join(lines) + "\n"


class ProgressReporter:
    """Render normalized wait state on stderr without poll-log noise."""

    def __init__(self, root: RootOptions, mode: OutputMode) -> None:
        self._interactive = sys.stderr.isatty()
        self._enabled = not root.quiet and (root.progress or self._interactive)
        self._color = terminal_color(root, mode, sys.stderr)
        self._last: str | None = None
        self._open_line = False
        self._last_width = 0

    def update(self, snapshot: BatchSnapshot) -> None:
        if not self._enabled:
            return
        counts = snapshot.request_counts
        finished = _finished(counts)
        status = styled(snapshot.status.value, color=self._color, fg="cyan", bold=True)
        visible = f"Waiting  {status}  {finished}/{counts.total} finished"
        visible_plain = f"Waiting  {snapshot.status.value}  {finished}/{counts.total} finished"
        plain = f"{snapshot.status.value}:{finished}:{counts.total}"
        if plain == self._last:
            return
        if self._interactive:
            padding = " " * max(0, self._last_width - len(visible_plain))
            click.echo(f"\r{visible}{padding}", nl=False, err=True, color=self._color)
            self._open_line = True
            self._last_width = len(visible_plain)
        else:
            click.echo(visible, err=True, color=self._color)
        self._last = plain

    def close(self) -> None:
        if self._open_line:
            click.echo(err=True)
            self._open_line = False
