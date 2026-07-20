from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

from click.testing import CliRunner

from batchwork.cli._commands import cli
from batchwork.cli._registry import RegistryRoute, adopt_job, insert_job, prune_jobs, update_job
from batchwork.cli._submit_text import resolve_route_descriptor
from batchwork.types import BatchProvider, BatchRequestCounts, BatchSnapshot, BatchStatus

FINGERPRINT = "1" * 64
OTHER_FINGERPRINT = "2" * 64


def _snapshot(
    job_id: str,
    *,
    status: BatchStatus = BatchStatus.IN_PROGRESS,
    created_at: datetime | None = None,
) -> BatchSnapshot:
    return BatchSnapshot(
        id=job_id,
        provider=BatchProvider.OPENAI,
        status=status,
        request_counts=BatchRequestCounts(total=2, completed=0, failed=0),
        created_at=created_at,
        completed_at=created_at if status is BatchStatus.COMPLETED else None,
    )


def _route(fingerprint: str = FINGERPRINT) -> RegistryRoute:
    return RegistryRoute(
        fingerprint=fingerprint,
        api_key_env="TEST_API_KEY",
        base_url="https://gateway.example/v1",
        headers={"x-client": "batchwork"},
        header_env={"authorization": "TEST_GATEWAY_AUTH"},
    )


def _insert(
    registry: Path,
    job_id: str,
    *,
    name: str | None,
    registered_at: datetime,
    status: BatchStatus = BatchStatus.IN_PROGRESS,
    fingerprint: str = FINGERPRINT,
) -> str:
    job = insert_job(
        registry,
        name=name,
        model="openai/gpt-test",
        profile=None,
        route=_route(fingerprint),
        snapshot=_snapshot(job_id, status=status, created_at=registered_at),
        registered_at=registered_at,
    )
    assert job.record_id is not None
    if status is BatchStatus.COMPLETED:
        update_job(registry, job.record_id, _snapshot(job_id, status=status), registered_at)
    return job.record_id


def _invoke(registry: Path, *arguments: str):
    return CliRunner().invoke(
        cli,
        ["--json", "--registry", str(registry), *arguments],
        prog_name="batchwork",
    )


def test_route_fingerprint_has_stable_canonical_vector() -> None:
    route = resolve_route_descriptor(
        BatchProvider.OPENAI,
        api_key_env="TEST_API_KEY",
        base_url="https://gateway.example/v1",
        header=("X-Client=batchwork",),
        header_env=("Authorization=TEST_GATEWAY_AUTH",),
    )

    assert route.fingerprint == "5abec0694826587a6913badfe801a411b2395e186f8d53b907eb7dfef7bb1f73"


def test_record_id_shape_cannot_be_adopted_as_alias(tmp_path: Path) -> None:
    record_id = "bw_0123456789abcdef0123456789abcdef"
    result = CliRunner().invoke(
        cli,
        [
            "--json",
            "--registry",
            str(tmp_path / "registry.sqlite3"),
            "status",
            "openai:batch_123",
            "--save",
            "--name",
            record_id,
        ],
        prog_name="batchwork",
        env={"XDG_CONFIG_HOME": str(tmp_path / "config")},
    )

    assert result.exit_code == 2
    assert "cannot be a record ID" in result.stderr


def test_list_is_local_filtered_and_deterministically_sorted(tmp_path: Path) -> None:
    registry = tmp_path / "registry.sqlite3"
    now = datetime.now(UTC)
    older = _insert(registry, "batch_old", name="old", registered_at=now - timedelta(days=2))
    newer = _insert(registry, "batch_new", name="new", registered_at=now - timedelta(days=1))
    _insert(
        registry,
        "batch_complete",
        name="done",
        registered_at=now,
        status=BatchStatus.COMPLETED,
    )

    result = _invoke(
        registry,
        "list",
        "--provider",
        "openai",
        "--modality",
        "text",
        "--status",
        "in_progress",
    )

    assert result.exit_code == 0, result.stderr
    assert [job["record_id"] for job in json.loads(result.stdout)["jobs"]] == [newer, older]
    assert result.stderr == ""

    limited = _invoke(registry, "list", "--name", "new", "--limit", "1")
    assert [job["record_id"] for job in json.loads(limited.stdout)["jobs"]] == [newer]


def test_jsonl_list_has_no_silent_cap_and_empty_list_has_no_output(tmp_path: Path) -> None:
    registry = tmp_path / "registry.sqlite3"
    now = datetime.now(UTC)
    for index in range(12):
        _insert(registry, f"batch_{index}", name=f"job-{index}", registered_at=now)

    result = CliRunner().invoke(
        cli,
        ["--jsonl", "--registry", str(registry), "list"],
        prog_name="batchwork",
    )

    assert result.exit_code == 0, result.stderr
    assert len(result.stdout.splitlines()) == 12
    assert all(json.loads(line)["type"] == "job" for line in result.stdout.splitlines())

    empty = CliRunner().invoke(
        cli,
        ["--jsonl", "--registry", str(registry), "list", "--name", "missing"],
        prog_name="batchwork",
    )
    assert empty.exit_code == 0
    assert empty.stdout == ""


def test_output_dir_with_unknown_record_modality_requires_explicit_images(tmp_path: Path) -> None:
    registry = tmp_path / "registry.sqlite3"
    adopted = adopt_job(
        registry,
        name="adopted",
        profile=None,
        route=_route(),
        snapshot=_snapshot("batch_adopted", status=BatchStatus.COMPLETED),
        registered_at=datetime.now(UTC),
        modality=None,
    )
    assert adopted.job.record_id is not None
    output_dir = tmp_path / "images"

    result = CliRunner().invoke(
        cli,
        [
            "--json",
            "--registry",
            str(registry),
            "results",
            adopted.job.record_id,
            "--output-dir",
            str(output_dir),
        ],
        prog_name="batchwork",
        env={"TEST_API_KEY": "test-key", "TEST_GATEWAY_AUTH": "test-auth"},
    )

    assert result.exit_code == 2
    assert "unknown modality requires --modality images" in result.stderr
    assert not output_dir.exists()


def test_forget_removes_exactly_one_local_record(tmp_path: Path) -> None:
    registry = tmp_path / "registry.sqlite3"
    now = datetime.now(UTC)
    forgotten = _insert(registry, "batch_shared", name="forget-me", registered_at=now)
    retained = _insert(
        registry,
        "batch_shared",
        name="keep-me",
        registered_at=now,
        fingerprint=OTHER_FINGERPRINT,
    )

    result = _invoke(registry, "forget", "forget-me")

    assert result.exit_code == 0, result.stderr
    envelope = json.loads(result.stdout)
    assert envelope["operation"] == "forget"
    assert envelope["record_id"] == forgotten
    with sqlite3.connect(registry) as connection:
        assert connection.execute("SELECT record_id FROM jobs").fetchall() == [(retained,)]

    direct = _invoke(registry, "forget", "openai:batch_shared")
    assert direct.exit_code == 2
    assert "local alias or record ID" in direct.stderr


def test_prune_previews_then_deletes_only_strictly_old_terminal_records(tmp_path: Path) -> None:
    registry = tmp_path / "registry.sqlite3"
    now = datetime.now(UTC)
    old_terminal = _insert(
        registry,
        "batch_old",
        name="old",
        registered_at=now - timedelta(days=31),
        status=BatchStatus.COMPLETED,
    )
    recent_terminal = _insert(
        registry,
        "batch_recent",
        name="recent",
        registered_at=now - timedelta(days=29),
        status=BatchStatus.COMPLETED,
    )
    active = _insert(
        registry,
        "batch_active",
        name="active",
        registered_at=now - timedelta(days=40),
    )

    preview = _invoke(registry, "prune", "--older-than", "30d")
    assert preview.exit_code == 0, preview.stderr
    assert json.loads(preview.stdout)["candidate_records"] == 1
    with sqlite3.connect(registry) as connection:
        assert connection.execute("SELECT COUNT(*) FROM jobs").fetchone() == (3,)

    committed = _invoke(registry, "prune", "--older-than", "30d", "--yes")
    assert committed.exit_code == 0, committed.stderr
    assert json.loads(committed.stdout)["changed_records"] == 1
    with sqlite3.connect(registry) as connection:
        remaining = {row[0] for row in connection.execute("SELECT record_id FROM jobs")}
    assert remaining == {recent_terminal, active}
    assert old_terminal not in remaining


def test_prune_cutoff_equality_does_not_qualify(tmp_path: Path) -> None:
    registry = tmp_path / "registry.sqlite3"
    terminal_at = datetime.now(UTC) - timedelta(days=30)
    record_id = _insert(
        registry,
        "batch_boundary",
        name="boundary",
        registered_at=terminal_at,
        status=BatchStatus.COMPLETED,
    )

    assert prune_jobs(registry, terminal_at, commit=False) == 0
    assert prune_jobs(registry, terminal_at, commit=True) == 0
    with sqlite3.connect(registry) as connection:
        assert connection.execute("SELECT record_id FROM jobs").fetchone() == (record_id,)


def test_concurrent_adoption_reuses_one_exact_route_record(tmp_path: Path) -> None:
    registry = tmp_path / "registry.sqlite3"
    snapshot = _snapshot("batch_shared")
    registered_at = datetime.now(UTC)

    def adopt() -> str | None:
        return adopt_job(
            registry,
            name=None,
            profile=None,
            route=_route(),
            snapshot=snapshot,
            registered_at=registered_at,
        ).job.record_id

    with ThreadPoolExecutor(max_workers=4) as executor:
        record_ids = set(executor.map(lambda _: adopt(), range(8)))

    assert len(record_ids) == 1
    with sqlite3.connect(registry) as connection:
        assert connection.execute("SELECT COUNT(*) FROM jobs").fetchone() == (1,)


def test_adoption_alias_collision_changes_nothing(tmp_path: Path) -> None:
    registry = tmp_path / "registry.sqlite3"
    now = datetime.now(UTC)
    existing = _insert(registry, "batch_existing", name="taken", registered_at=now)

    try:
        adopt_job(
            registry,
            name="taken",
            profile=None,
            route=_route(OTHER_FINGERPRINT),
            snapshot=_snapshot("batch_new"),
            registered_at=now,
        )
    except sqlite3.IntegrityError:
        pass
    else:
        raise AssertionError("expected alias collision")

    with sqlite3.connect(registry) as connection:
        assert connection.execute("SELECT record_id, name FROM jobs").fetchall() == [
            (existing, "taken")
        ]


def test_adoption_preserves_unknown_modality(tmp_path: Path) -> None:
    registry = tmp_path / "registry.sqlite3"
    adopted = adopt_job(
        registry,
        name="adopted",
        profile=None,
        route=_route(),
        snapshot=_snapshot("batch_adopted"),
        registered_at=datetime.now(UTC),
    )

    assert adopted.job.modality is None
    text_jobs = _invoke(registry, "list", "--modality", "text")
    assert json.loads(text_jobs.stdout)["jobs"] == []


def test_stale_refresh_cannot_regress_terminal_state(tmp_path: Path) -> None:
    registry = tmp_path / "registry.sqlite3"
    now = datetime.now(UTC)
    record_id = _insert(registry, "batch_terminal", name=None, registered_at=now)
    update_job(
        registry,
        record_id,
        _snapshot("batch_terminal", status=BatchStatus.COMPLETED),
        now + timedelta(seconds=1),
    )

    update_job(
        registry,
        record_id,
        _snapshot("batch_terminal", status=BatchStatus.IN_PROGRESS),
        now + timedelta(seconds=2),
    )

    with sqlite3.connect(registry) as connection:
        assert connection.execute(
            "SELECT status, terminal_at FROM jobs WHERE record_id = ?", (record_id,)
        ).fetchone() == ("completed", (now + timedelta(seconds=1)).isoformat())


def test_new_terminal_observation_can_correct_terminal_state(tmp_path: Path) -> None:
    registry = tmp_path / "registry.sqlite3"
    now = datetime.now(UTC)
    record_id = _insert(
        registry,
        "batch_terminal",
        name=None,
        registered_at=now,
        status=BatchStatus.FAILED,
    )

    update_job(
        registry,
        record_id,
        _snapshot("batch_terminal", status=BatchStatus.COMPLETED),
        now + timedelta(seconds=1),
    )

    with sqlite3.connect(registry) as connection:
        assert connection.execute(
            "SELECT status, terminal_at FROM jobs WHERE record_id = ?", (record_id,)
        ).fetchone() == ("completed", now.isoformat())
