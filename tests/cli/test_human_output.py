from __future__ import annotations

import io

from batchwork.cli._contract import ImageManifestEntry, Job, Materialization
from batchwork.cli._human import (
    ProgressReporter,
    human_job,
    human_results,
    human_snapshot,
    human_table,
    terminal_color,
)
from batchwork.cli._lifecycle import LifecycleResult, ResolvedJob
from batchwork.cli._registry import RegistryJob, RegistryRoute
from batchwork.cli._state import OutputMode, RootOptions
from batchwork.cli._submit_text import ResolvedRoute
from batchwork.types import (
    BatchImage,
    BatchProvider,
    BatchRequestCounts,
    BatchResult,
    BatchResultError,
    BatchResultStatus,
    BatchSnapshot,
    BatchStatus,
)

_FINGERPRINT = "a" * 64
_RECORD_ID = "bw_0123456789abcdef0123456789abcdef"


def _job(*, name: str | None = "nightly") -> Job:
    return Job(
        record_id=_RECORD_ID,
        name=name,
        provider=BatchProvider.OPENAI,
        provider_job_id="batch_123",
        provider_reference="openai:batch_123",
        routing_fingerprint=_FINGERPRINT,
        modality="text",
        model="openai/gpt-test",
        status=BatchStatus.IN_PROGRESS,
        request_counts=BatchRequestCounts(total=2, completed=1, failed=0),
    )


def _result(items: list[BatchResult]) -> LifecycleResult:
    job = _job()
    route = RegistryRoute(
        fingerprint=_FINGERPRINT,
        api_key_env="OPENAI_API_KEY",
        base_url=None,
        headers={"x-custom-auth": "literal-name-only"},
        header_env={},
    )
    registry_job = RegistryJob(job=job, route=route)
    resolved = ResolvedJob(
        selector="nightly",
        provider=BatchProvider.OPENAI,
        provider_job_id="batch_123",
        route=ResolvedRoute(
            api_key="secret",
            base_url=None,
            headers={"x-custom-auth": "hunter2"},
            registry=route,
        ),
        record=registry_job,
    )
    snapshot = BatchSnapshot(
        id="batch_123",
        provider=BatchProvider.OPENAI,
        status=BatchStatus.COMPLETED,
        request_counts=BatchRequestCounts(total=len(items), completed=len(items), failed=0),
    )
    return LifecycleResult(resolved, snapshot, items)


def test_human_job_and_snapshot_keep_every_recovery_selector_copyable() -> None:
    job = _job()

    submitted = human_job(job)
    snapshot = human_snapshot(_result([]), title="Job status")

    for output in (submitted, snapshot):
        assert "nightly" in output
        assert _RECORD_ID in output
        assert "openai:batch_123" in output
    assert f"batchwork status {_RECORD_ID}" in submitted


def test_human_results_redact_lossless_payloads_and_mark_text_truncation() -> None:
    signed_url = "https://images.example/result.png?X-Amz-Signature=private"
    result = _result(
        [
            BatchResult(
                custom_id="text",
                status=BatchResultStatus.SUCCEEDED,
                text="x" * 300,
                response={"secret": "raw provider body"},
            ),
            BatchResult(
                custom_id="embedding",
                status=BatchResultStatus.SUCCEEDED,
                embedding=[0.1, 0.2, 0.3],
            ),
            BatchResult(
                custom_id="url",
                status=BatchResultStatus.SUCCEEDED,
                text="https://files.example/short-signed-path",
            ),
            BatchResult(
                custom_id="image",
                status=BatchResultStatus.SUCCEEDED,
                images=[BatchImage(data="aGVsbG8=", url=signed_url)],
            ),
            BatchResult(
                custom_id="Authorization: Bearer custom-id-secret\nsecond-line",
                status=BatchResultStatus.ERRORED,
                error=BatchResultError(
                    message=(
                        f"download failed at {signed_url}; "
                        "X-API-Key=provider-secret; Bearer provider-token; "
                        "X-Custom-Auth: hunter2"
                    )
                ),
            ),
        ]
    )

    output = human_results(result)

    assert "raw provider body" not in output
    assert "[0.1" not in output
    assert "aGVsbG8=" not in output
    assert "X-Amz-Signature" not in output
    assert "https://files.example" not in output
    assert "custom-id-secret" not in output
    assert "provider-secret" not in output
    assert "provider-token" not in output
    assert "hunter2" not in output
    assert "\x1b[" not in output
    assert "3 dimensions" in output
    assert "1 image" in output
    assert "truncated" in output
    assert "--json" in output and "--jsonl" in output
    assert f"Job: {_RECORD_ID}" in output


def test_human_results_show_materialized_image_paths_and_manifest() -> None:
    result = _result(
        [
            BatchResult(
                custom_id="image",
                status=BatchResultStatus.SUCCEEDED,
                images=[BatchImage(data="aGVsbG8=", media_type="image/png")],
            )
        ]
    )
    materialization = Materialization(
        output_dir="/tmp/images",
        images=[
            ImageManifestEntry(
                path="image--0123456789ab--1.png",
                custom_id="image",
                image_index=1,
                source_kind="data",
                media_type="image/png",
                byte_count=5,
                sha256="b" * 64,
            )
        ],
    )

    output = human_results(result, materialization=materialization)

    assert "/tmp/images/image--0123456789ab--1.png" in output
    assert "Manifest: /tmp/images/manifest.json" in output
    assert "1 image saved" in output


def test_long_custom_id_keeps_lossless_mode_guidance_with_short_text() -> None:
    result = _result(
        [
            BatchResult(
                custom_id="request-" + "x" * 120,
                status=BatchResultStatus.SUCCEEDED,
                text="short",
            )
        ]
    )

    output = human_results(result)

    assert "request-" + "x" * 120 not in output
    assert "request-" in output and "…" in output
    assert "Preview truncated; use --json or --jsonl" in output


def test_human_registry_table_never_truncates_the_only_usable_selector() -> None:
    unnamed = RegistryJob(
        job=_job(name=None),
        route=RegistryRoute(
            fingerprint=_FINGERPRINT,
            api_key_env="OPENAI_API_KEY",
            base_url=None,
            headers={},
            header_env={},
        ),
    )

    output = human_table([unnamed], width=24)

    assert _RECORD_ID in output
    assert "…" not in next(line for line in output.splitlines() if _RECORD_ID in line)

    named_output = human_table([RegistryJob(job=_job(), route=unnamed.route)], width=120)
    assert "nightly" in named_output
    assert _RECORD_ID in named_output
    assert "RECORD" in named_output


def _root(*, quiet: bool = False, progress: bool = False, color: bool | None = None) -> RootOptions:
    return RootOptions(
        config=None,
        registry=None,
        profile=None,
        output_mode=None,
        quiet=quiet,
        progress=progress,
        color=color,
    )


def test_forced_redirected_progress_uses_stderr_and_deduplicates_unchanged_state(
    monkeypatch,
) -> None:
    stderr = io.StringIO()
    monkeypatch.setattr("sys.stderr", stderr)
    snapshot = BatchSnapshot(
        id="batch_123",
        provider=BatchProvider.OPENAI,
        status=BatchStatus.IN_PROGRESS,
        request_counts=BatchRequestCounts(total=2, completed=0, failed=0),
    )
    final = snapshot.model_copy(
        update={
            "status": BatchStatus.COMPLETED,
            "request_counts": BatchRequestCounts(total=2, completed=2, failed=0),
        }
    )
    reporter = ProgressReporter(_root(progress=True), OutputMode.JSON)

    reporter.update(snapshot)
    reporter.update(snapshot)
    reporter.update(final)
    reporter.close()

    assert stderr.getvalue().splitlines() == [
        "Waiting  in_progress  0/2 finished",
        "Waiting  completed  2/2 finished",
    ]
    assert "\x1b[" not in stderr.getvalue()


def test_quiet_suppresses_even_forced_progress(monkeypatch) -> None:
    stderr = io.StringIO()
    monkeypatch.setattr("sys.stderr", stderr)
    snapshot = BatchSnapshot(
        id="batch_123",
        provider=BatchProvider.OPENAI,
        status=BatchStatus.IN_PROGRESS,
        request_counts=BatchRequestCounts(total=1, completed=0, failed=0),
    )
    reporter = ProgressReporter(_root(quiet=True, progress=True), OutputMode.HUMAN)

    reporter.update(snapshot)
    reporter.close()

    assert stderr.getvalue() == ""


def test_machine_mode_never_enables_ansi_when_color_is_forced() -> None:
    assert terminal_color(_root(color=True), OutputMode.JSON, io.StringIO()) is False


def test_forced_human_color_is_visible_but_dumb_term_disables_it(monkeypatch) -> None:
    monkeypatch.delenv("TERM", raising=False)
    assert terminal_color(_root(color=True), OutputMode.HUMAN, io.StringIO()) is True
    assert "\x1b[" in human_job(_job(), color=True)

    monkeypatch.setenv("TERM", "dumb")
    assert terminal_color(_root(color=True), OutputMode.HUMAN, io.StringIO()) is False


def test_no_color_disables_automatic_terminal_color(monkeypatch) -> None:
    class _Tty(io.StringIO):
        def isatty(self) -> bool:
            return True

    monkeypatch.delenv("TERM", raising=False)
    monkeypatch.setenv("NO_COLOR", "1")

    assert terminal_color(_root(), OutputMode.HUMAN, _Tty()) is False
