from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import pytest
from click.testing import CliRunner

import batchwork.cli._registry as registry_module
from batchwork.cli._commands import cli
from batchwork.cli._registry import (
    CURRENT_SCHEMA_VERSION,
    RegistryRoute,
    check_registry,
    get_job,
    insert_job,
    reset_registry,
)
from batchwork.types import BatchProvider, BatchRequestCounts, BatchSnapshot, BatchStatus

V1_SCHEMA = """
CREATE TABLE jobs (
    record_id TEXT PRIMARY KEY,
    name TEXT UNIQUE,
    provider TEXT NOT NULL,
    provider_job_id TEXT NOT NULL,
    provider_reference TEXT NOT NULL,
    routing_fingerprint TEXT NOT NULL,
    api_key_env TEXT NOT NULL,
    base_url TEXT,
    headers_json TEXT NOT NULL,
    header_env_json TEXT NOT NULL,
    modality TEXT NOT NULL,
    model TEXT,
    profile TEXT,
    status TEXT NOT NULL,
    request_total INTEGER NOT NULL,
    request_completed INTEGER NOT NULL,
    request_failed INTEGER NOT NULL,
    registered_at TEXT NOT NULL,
    provider_created_at TEXT,
    UNIQUE (provider, provider_job_id, routing_fingerprint)
)
"""


def _snapshot(job_id: str = "batch_123") -> BatchSnapshot:
    return BatchSnapshot(
        id=job_id,
        provider=BatchProvider.OPENAI,
        status=BatchStatus.IN_PROGRESS,
        request_counts=BatchRequestCounts(total=1, completed=0, failed=0),
    )


def _route() -> RegistryRoute:
    return RegistryRoute(
        fingerprint="1" * 64,
        api_key_env="TEST_API_KEY",
        base_url=None,
        headers={},
        header_env={},
    )


def _insert_v1_row(connection: sqlite3.Connection) -> None:
    connection.execute(
        """INSERT INTO jobs VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )""",
        (
            "bw_0123456789abcdef0123456789abcdef",
            "legacy",
            "openai",
            "batch_123",
            "openai:batch_123",
            "1" * 64,
            "TEST_API_KEY",
            None,
            "{}",
            "{}",
            "text",
            "openai/gpt-test",
            None,
            "in_progress",
            1,
            0,
            0,
            datetime.now(UTC).isoformat(),
            None,
        ),
    )


def test_new_registry_uses_wal_current_version_and_database_constraints(tmp_path: Path) -> None:
    registry = tmp_path / "registry.sqlite3"
    insert_job(
        registry,
        name="job",
        model="openai/gpt-test",
        profile=None,
        route=_route(),
        snapshot=_snapshot(),
        registered_at=datetime.now(UTC),
    )

    with sqlite3.connect(registry) as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone() == ("wal",)
        assert connection.execute("PRAGMA user_version").fetchone() == (CURRENT_SCHEMA_VERSION,)
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "UPDATE jobs SET routing_fingerprint = ?, request_total = -1",
                ("not-a-fingerprint",),
            )


def test_v1_migration_creates_consistent_backup_then_commits_transactionally(
    tmp_path: Path,
) -> None:
    registry = tmp_path / "registry.sqlite3"
    with sqlite3.connect(registry) as connection:
        connection.execute(V1_SCHEMA)
        _insert_v1_row(connection)
        connection.execute("PRAGMA user_version = 1")

    migrated = get_job(registry, "legacy")

    assert migrated is not None
    with sqlite3.connect(registry) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (CURRENT_SCHEMA_VERSION,)
    backups = list(tmp_path.glob("registry.sqlite3.migration-v1-*.sqlite3"))
    assert len(backups) == 1
    with sqlite3.connect(backups[0]) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (1,)
        assert connection.execute("SELECT name FROM jobs").fetchone() == ("legacy",)


def test_failed_migration_rolls_back_original_and_retains_backup(tmp_path: Path) -> None:
    registry = tmp_path / "registry.sqlite3"
    with sqlite3.connect(registry) as connection:
        connection.execute(V1_SCHEMA)
        _insert_v1_row(connection)
        connection.execute("UPDATE jobs SET request_total = -1")
        connection.execute("PRAGMA user_version = 1")

    with pytest.raises(sqlite3.IntegrityError):
        get_job(registry, "legacy")

    with sqlite3.connect(registry) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (1,)
        assert connection.execute("SELECT request_total FROM jobs").fetchone() == (-1,)
        assert (
            connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'jobs_v1'"
            ).fetchone()
            is None
        )
    assert len(list(tmp_path.glob("registry.sqlite3.migration-v1-*.sqlite3"))) == 1


def test_newer_schema_fails_closed_without_changing_any_registry_file(tmp_path: Path) -> None:
    registry = tmp_path / "registry.sqlite3"
    with sqlite3.connect(registry) as connection:
        connection.execute("CREATE TABLE marker (value TEXT)")
        connection.execute("INSERT INTO marker VALUES ('preserve-me')")
        connection.execute("PRAGMA user_version = 99")
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}

    result = CliRunner().invoke(
        cli,
        ["--json", "--registry", str(registry), "list"],
        prog_name="batchwork",
    )

    assert result.exit_code == 8
    assert json.loads(result.stderr)["error"]["code"] == "registry_schema_unsupported"
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before


def test_registry_check_reports_explicit_integrity_result(tmp_path: Path) -> None:
    registry = tmp_path / "registry.sqlite3"
    insert_job(
        registry,
        name=None,
        model="openai/gpt-test",
        profile=None,
        route=_route(),
        snapshot=_snapshot(),
        registered_at=datetime.now(UTC),
    )

    report = check_registry(registry)
    result = CliRunner().invoke(
        cli,
        ["--json", "--registry", str(registry), "registry", "check"],
        prog_name="batchwork",
    )

    assert report.ok is True
    assert report.integrity == "ok"
    assert result.exit_code == 0, result.stderr
    assert json.loads(result.stdout) == {
        "schema_version": 1,
        "type": "registry_check",
        "path": str(registry),
        "ok": True,
        "user_version": CURRENT_SCHEMA_VERSION,
        "integrity": "ok",
    }


def test_registry_check_does_not_migrate_supported_older_schema(tmp_path: Path) -> None:
    registry = tmp_path / "registry.sqlite3"
    with sqlite3.connect(registry) as connection:
        connection.execute(V1_SCHEMA)
        _insert_v1_row(connection)
        connection.execute("PRAGMA user_version = 1")

    report = check_registry(registry)

    assert (report.ok, report.user_version, report.integrity) == (True, 1, "ok")
    with sqlite3.connect(registry) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (1,)
    assert list(tmp_path.glob("registry.sqlite3.migration-v1-*.sqlite3")) == []


def test_registry_check_reports_corruption_without_replacing_file(tmp_path: Path) -> None:
    registry = tmp_path / "registry.sqlite3"
    registry.write_bytes(b"not sqlite")

    result = CliRunner().invoke(
        cli,
        ["--json", "--registry", str(registry), "registry", "check"],
        prog_name="batchwork",
    )

    assert result.exit_code == 8
    assert result.stdout == ""
    document = json.loads(result.stderr)["error"]
    assert document["code"] == "registry_unavailable"
    assert registry.read_bytes() == b"not sqlite"


def test_registry_check_reports_process_lock_failure_as_machine_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = tmp_path / "registry.sqlite3"
    registry.write_bytes(b"database")

    @contextmanager
    def fail_lock(*_: object, **__: object) -> Iterator[None]:
        raise TimeoutError("injected lock timeout")
        yield

    monkeypatch.setattr(registry_module, "_process_lock", fail_lock)

    result = CliRunner().invoke(
        cli,
        ["--json", "--registry", str(registry), "registry", "check"],
        prog_name="batchwork",
    )

    assert result.exit_code == 8
    assert result.stdout == ""
    assert json.loads(result.stderr)["error"]["code"] == "registry_unavailable"


def test_registry_reset_preserves_database_and_sidecars_as_one_recovery_set(
    tmp_path: Path,
) -> None:
    registry = tmp_path / "registry.sqlite3"
    files = {
        registry: b"database",
        Path(f"{registry}-wal"): b"wal",
        Path(f"{registry}-shm"): b"shm",
    }
    for path, content in files.items():
        path.write_bytes(content)

    result = reset_registry(registry)

    assert result.backup_path is not None
    assert {path.name: path.read_bytes() for path in result.backup_path.iterdir()} == {
        path.name: content for path, content in files.items()
    }
    with sqlite3.connect(registry) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (CURRENT_SCHEMA_VERSION,)
        assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)


def test_registry_reset_reports_machine_recovery_path(tmp_path: Path) -> None:
    registry = tmp_path / "registry.sqlite3"
    registry.write_bytes(b"damaged")

    result = CliRunner().invoke(
        cli,
        ["--json", "--registry", str(registry), "registry", "reset", "--backup"],
        prog_name="batchwork",
    )

    assert result.exit_code == 0, result.stderr
    document = json.loads(result.stdout)
    assert document["operation"] == "reset"
    assert document["user_version"] == CURRENT_SCHEMA_VERSION
    assert document["records_count_known"] is False
    assert Path(document["backup_path"], registry.name).read_bytes() == b"damaged"


def test_registry_reset_preservation_failure_restores_every_original_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = tmp_path / "registry.sqlite3"
    files = {
        registry: b"database",
        Path(f"{registry}-wal"): b"wal",
        Path(f"{registry}-shm"): b"shm",
    }
    for path, content in files.items():
        path.write_bytes(content)
    original_copy = registry_module.shutil.copy2

    def fail_on_wal(source: Path, target: Path) -> str:
        if source == Path(f"{registry}-wal"):
            raise OSError("injected preservation failure")
        return original_copy(source, target)

    monkeypatch.setattr(registry_module.shutil, "copy2", fail_on_wal)

    with pytest.raises(OSError, match="injected preservation failure"):
        reset_registry(registry)

    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == {
        path.name: content for path, content in files.items()
    }


def test_registry_reset_fails_closed_while_external_writer_is_active(tmp_path: Path) -> None:
    registry = tmp_path / "registry.sqlite3"
    insert_job(
        registry,
        name=None,
        model="openai/gpt-test",
        profile=None,
        route=_route(),
        snapshot=_snapshot(),
        registered_at=datetime.now(UTC),
    )
    writer = sqlite3.connect(registry)
    writer.execute("BEGIN IMMEDIATE")
    try:
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            reset_registry(registry)
    finally:
        writer.rollback()
        writer.close()

    with sqlite3.connect(registry) as connection:
        assert connection.execute("SELECT COUNT(*) FROM jobs").fetchone() == (1,)
        assert connection.execute("PRAGMA user_version").fetchone() == (CURRENT_SCHEMA_VERSION,)
    assert list(tmp_path.glob("registry.sqlite3.recovery-*")) == []
