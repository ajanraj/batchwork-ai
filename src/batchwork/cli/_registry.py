"""Metadata-only SQLite continuity for CLI jobs."""

from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from batchwork.types import (
    BatchProvider,
    BatchRequestCounts,
    BatchSnapshot,
    BatchStatus,
    is_terminal_status,
)

from ._contract import Job

_RECORD_ID = re.compile(r"^bw_[0-9a-f]{32}$")
_JOB_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_TERMINAL_STATUS_VALUES = tuple(
    status.value for status in BatchStatus if is_terminal_status(status)
)
_TERMINAL_STATUS_PLACEHOLDERS = ", ".join("?" for _ in _TERMINAL_STATUS_VALUES)
_PROVIDER_CONSTRAINT_VALUES = ", ".join(repr(provider.value) for provider in BatchProvider)
_STATUS_CONSTRAINT_VALUES = ", ".join(repr(status.value) for status in BatchStatus)
CURRENT_SCHEMA_VERSION = 2
BUSY_TIMEOUT_MILLISECONDS = 5_000

SCHEMA = f"""
CREATE TABLE jobs (
    record_id TEXT PRIMARY KEY
        CHECK (
            length(record_id) = 35
            AND substr(record_id, 1, 3) = 'bw_'
            AND substr(record_id, 4) NOT GLOB '*[^0-9a-f]*'
        ),
    name TEXT UNIQUE
        CHECK (
            name IS NULL
            OR (
                length(name) BETWEEN 1 AND 64
                AND substr(name, 1, 1) NOT GLOB '[^A-Za-z0-9]'
                AND name NOT GLOB '*[^A-Za-z0-9._-]*'
                AND NOT (
                    length(name) = 35
                    AND substr(name, 1, 3) = 'bw_'
                    AND substr(name, 4) NOT GLOB '*[^0-9a-f]*'
                )
            )
        ),
    provider TEXT NOT NULL CHECK (
        provider IN ({_PROVIDER_CONSTRAINT_VALUES})
    ),
    provider_job_id TEXT NOT NULL CHECK (length(provider_job_id) > 0),
    provider_reference TEXT NOT NULL
        CHECK (provider_reference = provider || ':' || provider_job_id),
    routing_fingerprint TEXT NOT NULL
        CHECK (
            length(routing_fingerprint) = 64
            AND routing_fingerprint NOT GLOB '*[^0-9a-f]*'
        ),
    api_key_env TEXT NOT NULL CHECK (length(api_key_env) > 0),
    base_url TEXT,
    headers_json TEXT NOT NULL,
    header_env_json TEXT NOT NULL,
    modality TEXT CHECK (modality IS NULL OR modality IN ('text', 'embeddings', 'images')),
    model TEXT,
    profile TEXT,
    status TEXT NOT NULL CHECK (
        status IN ({_STATUS_CONSTRAINT_VALUES})
    ),
    request_total INTEGER NOT NULL CHECK (request_total >= 0),
    request_completed INTEGER NOT NULL CHECK (request_completed >= 0),
    request_failed INTEGER NOT NULL CHECK (request_failed >= 0),
    registered_at TEXT NOT NULL,
    provider_created_at TEXT,
    completed_at TEXT,
    expires_at TEXT,
    terminal_at TEXT,
    last_refreshed_at TEXT,
    UNIQUE (provider, provider_job_id, routing_fingerprint),
    CHECK (request_completed + request_failed <= request_total)
)
"""

_V1_COLUMNS = """
record_id, name, provider, provider_job_id, provider_reference,
routing_fingerprint, api_key_env, base_url, headers_json, header_env_json,
modality, model, profile, status, request_total, request_completed,
request_failed, registered_at, provider_created_at, completed_at, expires_at,
terminal_at, last_refreshed_at
"""

_INSERT_JOB = """
INSERT INTO jobs (
    record_id, name, provider, provider_job_id, provider_reference,
    routing_fingerprint, api_key_env, base_url, headers_json, header_env_json,
    modality, model, profile, status, request_total, request_completed,
    request_failed, registered_at, provider_created_at, completed_at, expires_at,
    terminal_at, last_refreshed_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


@dataclass(frozen=True, slots=True)
class RegistryRoute:
    fingerprint: str
    api_key_env: str
    base_url: str | None
    headers: dict[str, str]
    header_env: dict[str, str]


@dataclass(frozen=True, slots=True)
class RegistryJob:
    job: Job
    route: RegistryRoute


@dataclass(frozen=True, slots=True)
class RegistryCheckResult:
    ok: bool
    user_version: int
    integrity: str


@dataclass(frozen=True, slots=True)
class RegistryResetResult:
    backup_path: Path | None
    records_count: int | None


class RegistryNameConflict(sqlite3.IntegrityError):
    pass


class RegistrySchemaError(sqlite3.DatabaseError):
    pass


class RegistryIntegrityError(sqlite3.DatabaseError):
    pass


def is_record_id(value: str) -> bool:
    return _RECORD_ID.fullmatch(value) is not None


def is_job_name(value: str) -> bool:
    return _JOB_NAME.fullmatch(value) is not None and not is_record_id(value)


def _raise_name_conflict(
    connection: sqlite3.Connection, name: str | None, error: sqlite3.IntegrityError
) -> None:
    if (
        name is not None
        and connection.execute("SELECT 1 FROM jobs WHERE name = ?", (name,)).fetchone()
    ):
        raise RegistryNameConflict(name) from error
    raise error


def _enable_wal(connection: sqlite3.Connection) -> None:
    deadline = time.monotonic() + 5
    while True:
        try:
            connection.execute("PRAGMA journal_mode = WAL")
            return
        except sqlite3.OperationalError as error:
            if error.sqlite_errorcode not in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED}:
                raise
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.01)


@contextmanager
def _process_lock(path: Path, *, exclusive: bool, create_lock_file: bool = True) -> Iterator[None]:
    deadline = time.monotonic() + BUSY_TIMEOUT_MILLISECONDS / 1_000
    if os.name == "posix":
        import fcntl

        descriptor = os.open(path.parent, os.O_RDONLY)
        operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        while True:
            try:
                fcntl.flock(descriptor, operation | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    os.close(descriptor)
                    raise TimeoutError("timed out waiting for the registry process lock") from None
                time.sleep(0.01)
        try:
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)
        return

    import msvcrt

    lock_path = Path(f"{path}.lock")
    if not create_lock_file and (not lock_path.exists() or lock_path.stat().st_size == 0):
        yield
        return
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    if os.fstat(descriptor).st_size == 0:
        os.write(descriptor, b"\0")
    os.lseek(descriptor, 0, os.SEEK_SET)
    while True:
        try:
            msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
            break
        except OSError:
            if time.monotonic() >= deadline:
                os.close(descriptor)
                raise TimeoutError("timed out waiting for the registry process lock") from None
            time.sleep(0.01)
    try:
        yield
    finally:
        os.lseek(descriptor, 0, os.SEEK_SET)
        msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
        os.close(descriptor)


def _integrity(connection: sqlite3.Connection, pragma: str) -> str:
    rows = connection.execute(f"PRAGMA {pragma}").fetchall()
    messages = [str(row[0]) for row in rows if row]
    return "ok" if messages == ["ok"] else "; ".join(messages) or "no result"


def _user_version(connection: sqlite3.Connection) -> int:
    row = connection.execute("PRAGMA user_version").fetchone()
    return int(row[0]) if row is not None else 0


def _migration_backup(path: Path, version: int) -> Path:
    backup = path.with_name(f"{path.name}.migration-v{version}-{uuid4().hex}.sqlite3")
    source = sqlite3.connect(path, timeout=BUSY_TIMEOUT_MILLISECONDS / 1_000)
    destination = sqlite3.connect(backup)
    try:
        source.backup(destination)
    except BaseException:
        source.close()
        destination.close()
        backup.unlink(missing_ok=True)
        raise
    source.close()
    destination.close()
    if os.name == "posix":
        backup.chmod(0o600)
    return backup


def _migrate_v1_to_v2(path: Path, connection: sqlite3.Connection) -> None:
    connection.execute("BEGIN IMMEDIATE")
    try:
        version = _user_version(connection)
        if version == CURRENT_SCHEMA_VERSION:
            connection.commit()
            return
        if version != 1:
            raise RegistrySchemaError(f"registry changed to schema {version} during migration")
        existing_columns = {row[1] for row in connection.execute("PRAGMA table_info(jobs)")}
        for name in ("completed_at", "expires_at", "terminal_at", "last_refreshed_at"):
            if name not in existing_columns:
                connection.execute(f"ALTER TABLE jobs ADD COLUMN {name} TEXT")
        connection.execute("ALTER TABLE jobs RENAME TO jobs_v1")
        connection.execute(SCHEMA)
        connection.execute(f"INSERT INTO jobs ({_V1_COLUMNS}) SELECT {_V1_COLUMNS} FROM jobs_v1")
        connection.execute("DROP TABLE jobs_v1")
        connection.execute(f"PRAGMA user_version = {CURRENT_SCHEMA_VERSION}")
    except BaseException:
        connection.rollback()
        raise
    connection.commit()


def _initialize(connection: sqlite3.Connection) -> None:
    connection.execute("BEGIN IMMEDIATE")
    try:
        version = _user_version(connection)
        has_jobs = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'jobs'"
        ).fetchone()
        if version == CURRENT_SCHEMA_VERSION and has_jobs is not None:
            connection.commit()
            return
        if version != 0 or has_jobs is not None:
            raise RegistrySchemaError("registry changed while it was being initialized")
        connection.execute(SCHEMA)
        connection.execute(f"PRAGMA user_version = {CURRENT_SCHEMA_VERSION}")
    except BaseException:
        connection.rollback()
        raise
    connection.commit()


@contextmanager
def _open_registry_unlocked(path: Path, *, create: bool) -> Iterator[sqlite3.Connection]:
    existed = path.exists()
    if not existed and not create:
        raise FileNotFoundError(path)
    if create:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=BUSY_TIMEOUT_MILLISECONDS / 1_000)
    try:
        connection.row_factory = sqlite3.Row
        connection.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MILLISECONDS}")
        version = _user_version(connection)
        if version > CURRENT_SCHEMA_VERSION:
            raise RegistrySchemaError(
                f"registry schema {version} is newer than supported schema {CURRENT_SCHEMA_VERSION}"
            )
        integrity = _integrity(connection, "quick_check")
        if integrity != "ok":
            raise RegistryIntegrityError(f"registry integrity check failed: {integrity}")
        has_jobs = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'jobs'"
        ).fetchone()
        if version == 0:
            if has_jobs is not None:
                raise RegistrySchemaError("unversioned registry schema is unsupported")
            _initialize(connection)
        elif version == 1:
            _migration_backup(path, version)
            _migrate_v1_to_v2(path, connection)
        elif has_jobs is None:
            raise RegistrySchemaError("registry jobs table is missing")
        _enable_wal(connection)
        if os.name == "posix":
            path.chmod(0o600)
        try:
            yield connection
        except BaseException:
            connection.rollback()
            raise
        connection.commit()
    finally:
        connection.close()


@contextmanager
def _open_registry(path: Path, *, create: bool) -> Iterator[sqlite3.Connection]:
    if create:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with _process_lock(path, exclusive=False):
        with _open_registry_unlocked(path, create=create) as connection:
            yield connection


def _job_values(
    *,
    record_id: str,
    name: str | None,
    modality: str | None,
    model: str | None,
    profile: str | None,
    route: RegistryRoute,
    snapshot: BatchSnapshot,
    registered_at: datetime,
) -> tuple[str | int | None, ...]:
    counts = snapshot.request_counts
    return (
        record_id,
        name,
        snapshot.provider.value,
        snapshot.id,
        f"{snapshot.provider.value}:{snapshot.id}",
        route.fingerprint,
        route.api_key_env,
        route.base_url,
        json.dumps(route.headers, ensure_ascii=False, separators=(",", ":"), sort_keys=True),
        json.dumps(route.header_env, ensure_ascii=False, separators=(",", ":"), sort_keys=True),
        modality,
        model,
        profile,
        snapshot.status.value,
        counts.total,
        counts.completed,
        counts.failed,
        registered_at.isoformat(),
        snapshot.created_at.isoformat() if snapshot.created_at else None,
        snapshot.completed_at.isoformat() if snapshot.completed_at else None,
        snapshot.expires_at.isoformat() if snapshot.expires_at else None,
        registered_at.isoformat() if is_terminal_status(snapshot.status) else None,
        registered_at.isoformat(),
    )


def insert_job(
    path: Path,
    *,
    name: str | None,
    model: str,
    profile: str | None,
    route: RegistryRoute,
    snapshot: BatchSnapshot,
    registered_at: datetime,
) -> Job:
    if name is not None and not is_job_name(name):
        raise ValueError("name must be shell-safe and cannot match a record ID")
    record_id = f"bw_{uuid4().hex}"
    provider_reference = f"{snapshot.provider.value}:{snapshot.id}"
    counts = snapshot.request_counts
    with _open_registry(path, create=True) as connection:
        try:
            connection.execute(
                _INSERT_JOB,
                _job_values(
                    record_id=record_id,
                    name=name,
                    modality="text",
                    model=model,
                    profile=profile,
                    route=route,
                    snapshot=snapshot,
                    registered_at=registered_at,
                ),
            )
        except sqlite3.IntegrityError as error:
            _raise_name_conflict(connection, name, error)
    return Job(
        record_id=record_id,
        name=name,
        provider=snapshot.provider,
        provider_job_id=snapshot.id,
        provider_reference=provider_reference,
        routing_fingerprint=route.fingerprint,
        modality="text",
        model=model,
        profile=profile,
        status=snapshot.status,
        request_counts=counts,
        registered_at=registered_at,
        provider_created_at=snapshot.created_at,
        completed_at=snapshot.completed_at,
        expires_at=snapshot.expires_at,
        terminal_at=registered_at if is_terminal_status(snapshot.status) else None,
    )


def adopt_job(
    path: Path,
    *,
    name: str | None,
    profile: str | None,
    route: RegistryRoute,
    snapshot: BatchSnapshot,
    registered_at: datetime,
) -> RegistryJob:
    if name is not None and not is_job_name(name):
        raise ValueError("name must be shell-safe and cannot match a record ID")
    with _open_registry(path, create=True) as connection:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            """
            SELECT * FROM jobs
            WHERE provider = ? AND provider_job_id = ? AND routing_fingerprint = ?
            """,
            (snapshot.provider.value, snapshot.id, route.fingerprint),
        ).fetchone()
        if row is not None:
            if name is not None and row["name"] not in {None, name}:
                raise RegistryNameConflict(name)
            if name is not None and row["name"] is None:
                try:
                    connection.execute(
                        "UPDATE jobs SET name = ? WHERE record_id = ?",
                        (name, row["record_id"]),
                    )
                except sqlite3.IntegrityError as error:
                    _raise_name_conflict(connection, name, error)
            counts = snapshot.request_counts
            connection.execute(
                f"""
                UPDATE jobs SET profile = COALESCE(?, profile), status = ?,
                    request_total = ?, request_completed = ?, request_failed = ?,
                    provider_created_at = COALESCE(provider_created_at, ?),
                    completed_at = ?, expires_at = ?,
                    terminal_at = COALESCE(terminal_at, ?), last_refreshed_at = ?
                WHERE record_id = ?
                    AND (
                        status NOT IN ({_TERMINAL_STATUS_PLACEHOLDERS})
                        OR ?
                    )
                """,
                (
                    profile,
                    snapshot.status.value,
                    counts.total,
                    counts.completed,
                    counts.failed,
                    snapshot.created_at.isoformat() if snapshot.created_at else None,
                    snapshot.completed_at.isoformat() if snapshot.completed_at else None,
                    snapshot.expires_at.isoformat() if snapshot.expires_at else None,
                    registered_at.isoformat() if is_terminal_status(snapshot.status) else None,
                    registered_at.isoformat(),
                    row["record_id"],
                    *_TERMINAL_STATUS_VALUES,
                    is_terminal_status(snapshot.status),
                ),
            )
            row = connection.execute(
                "SELECT * FROM jobs WHERE record_id = ?", (row["record_id"],)
            ).fetchone()
            if row is None:
                raise sqlite3.IntegrityError("adopted job disappeared")
            return _registry_job(row)
        record_id = f"bw_{uuid4().hex}"
        try:
            connection.execute(
                _INSERT_JOB,
                _job_values(
                    record_id=record_id,
                    name=name,
                    modality=None,
                    model=None,
                    profile=profile,
                    route=route,
                    snapshot=snapshot,
                    registered_at=registered_at,
                ),
            )
        except sqlite3.IntegrityError as error:
            _raise_name_conflict(connection, name, error)
        row = connection.execute("SELECT * FROM jobs WHERE record_id = ?", (record_id,)).fetchone()
    if row is None:
        raise sqlite3.IntegrityError("adopted job was not persisted")
    return _registry_job(row)


def _datetime(value: object) -> datetime | None:
    return datetime.fromisoformat(value) if isinstance(value, str) else None


def _registry_job(row: sqlite3.Row) -> RegistryJob:
    job = Job(
        record_id=row["record_id"],
        name=row["name"],
        provider=BatchProvider(row["provider"]),
        provider_job_id=row["provider_job_id"],
        provider_reference=row["provider_reference"],
        routing_fingerprint=row["routing_fingerprint"],
        modality=row["modality"],
        model=row["model"],
        profile=row["profile"],
        status=BatchStatus(row["status"]),
        request_counts=BatchRequestCounts(
            total=row["request_total"],
            completed=row["request_completed"],
            failed=row["request_failed"],
        ),
        registered_at=_datetime(row["registered_at"]),
        provider_created_at=_datetime(row["provider_created_at"]),
        completed_at=_datetime(row["completed_at"]),
        expires_at=_datetime(row["expires_at"]),
        terminal_at=_datetime(row["terminal_at"]),
        last_refreshed_at=_datetime(row["last_refreshed_at"]),
    )
    route = RegistryRoute(
        fingerprint=row["routing_fingerprint"],
        api_key_env=row["api_key_env"],
        base_url=row["base_url"],
        headers=json.loads(row["headers_json"]),
        header_env=json.loads(row["header_env_json"]),
    )
    return RegistryJob(job, route)


def get_job(path: Path, selector: str) -> RegistryJob | None:
    if not path.is_file():
        return None
    with _open_registry(path, create=False) as connection:
        column = "record_id" if _RECORD_ID.fullmatch(selector) else "name"
        row = connection.execute(f"SELECT * FROM jobs WHERE {column} = ?", (selector,)).fetchone()
    return _registry_job(row) if row is not None else None


def update_job(
    path: Path,
    record_id: str,
    snapshot: BatchSnapshot,
    refreshed_at: datetime,
    *,
    profile: str | None = None,
) -> None:
    terminal = is_terminal_status(snapshot.status)
    counts = snapshot.request_counts
    with _open_registry(path, create=False) as connection:
        connection.execute(
            f"""
            UPDATE jobs SET profile = COALESCE(?, profile), status = ?,
                request_total = ?, request_completed = ?,
                request_failed = ?, completed_at = ?, expires_at = ?,
                terminal_at = COALESCE(terminal_at, ?), last_refreshed_at = ?
            WHERE record_id = ?
                AND (
                    status NOT IN ({_TERMINAL_STATUS_PLACEHOLDERS})
                    OR ?
                )
            """,
            (
                profile,
                snapshot.status.value,
                counts.total,
                counts.completed,
                counts.failed,
                snapshot.completed_at.isoformat() if snapshot.completed_at else None,
                snapshot.expires_at.isoformat() if snapshot.expires_at else None,
                refreshed_at.isoformat() if terminal else None,
                refreshed_at.isoformat(),
                record_id,
                *_TERMINAL_STATUS_VALUES,
                terminal,
            ),
        )


def list_registry_jobs(
    path: Path,
    *,
    provider: BatchProvider | None = None,
    modality: str | None = None,
    statuses: Sequence[BatchStatus] = (),
    name: str | None = None,
    limit: int | None = None,
) -> list[RegistryJob]:
    if not path.is_file():
        return []
    clauses: list[str] = []
    parameters: list[str | int] = []
    if provider is not None:
        clauses.append("provider = ?")
        parameters.append(provider.value)
    if modality is not None:
        clauses.append("modality = ?")
        parameters.append(modality)
    if statuses:
        placeholders = ", ".join("?" for _ in statuses)
        clauses.append(f"status IN ({placeholders})")
        parameters.extend(status.value for status in statuses)
    if name is not None:
        clauses.append("name = ?")
        parameters.append(name)
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    limit_clause = " LIMIT ?" if limit is not None else ""
    if limit is not None:
        parameters.append(limit)
    with _open_registry(path, create=False) as connection:
        rows = connection.execute(
            f"SELECT * FROM jobs{where} ORDER BY registered_at DESC, record_id ASC{limit_clause}",
            parameters,
        ).fetchall()
    return [_registry_job(row) for row in rows]


def forget_job(path: Path, selector: str) -> RegistryJob | None:
    if not path.is_file():
        return None
    with _open_registry(path, create=False) as connection:
        connection.execute("BEGIN IMMEDIATE")
        column = "record_id" if is_record_id(selector) else "name"
        row = connection.execute(f"SELECT * FROM jobs WHERE {column} = ?", (selector,)).fetchone()
        if row is None:
            return None
        connection.execute("DELETE FROM jobs WHERE record_id = ?", (row["record_id"],))
    return _registry_job(row)


def prune_jobs(path: Path, cutoff_at: datetime, *, commit: bool) -> int:
    if not path.is_file():
        return 0
    predicate = f"status IN ({_TERMINAL_STATUS_PLACEHOLDERS}) AND terminal_at < ?"
    parameters = (*_TERMINAL_STATUS_VALUES, cutoff_at.isoformat())
    with _open_registry(path, create=False) as connection:
        if not commit:
            row = connection.execute(
                f"SELECT COUNT(*) FROM jobs WHERE {predicate}", parameters
            ).fetchone()
            return int(row[0]) if row is not None else 0
        connection.execute("BEGIN IMMEDIATE")
        cursor = connection.execute(f"DELETE FROM jobs WHERE {predicate}", parameters)
        return cursor.rowcount


def _check_registry_unlocked(path: Path) -> RegistryCheckResult:
    with sqlite3.connect(path, timeout=BUSY_TIMEOUT_MILLISECONDS / 1_000) as connection:
        connection.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MILLISECONDS}")
        version = _user_version(connection)
        if version > CURRENT_SCHEMA_VERSION:
            return RegistryCheckResult(False, version, "unsupported_schema")
        integrity = _integrity(connection, "integrity_check")
        if integrity != "ok":
            return RegistryCheckResult(False, version, integrity)
        has_jobs = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'jobs'"
        ).fetchone()
        if version == 0 and has_jobs is not None:
            return RegistryCheckResult(False, version, "unsupported_schema")
        if version > 0 and has_jobs is None:
            return RegistryCheckResult(False, version, "schema_missing")
    return RegistryCheckResult(True, version, integrity)


def check_registry(path: Path) -> RegistryCheckResult:
    if not path.exists():
        return RegistryCheckResult(ok=True, user_version=0, integrity="missing")
    try:
        with _process_lock(path, exclusive=False, create_lock_file=False):
            return _check_registry_unlocked(path)
    except (OSError, sqlite3.Error) as error:
        return RegistryCheckResult(False, 0, f"open_failed: {error}")


def _recovery_members(path: Path) -> tuple[Path, ...]:
    candidates = (path, Path(f"{path}-wal"), Path(f"{path}-shm"))
    return tuple(candidate for candidate in candidates if candidate.exists())


def _preserve_recovery_set(path: Path) -> Path | None:
    members = _recovery_members(path)
    if not members:
        return None
    recovery = path.with_name(f"{path.name}.recovery-{uuid4().hex}")
    recovery.mkdir(mode=0o700)
    copied: list[Path] = []
    try:
        for source in members:
            destination = recovery / source.name
            shutil.copy2(source, destination)
            copied.append(destination)
    except BaseException as error:
        cleanup_errors: list[str] = []
        for destination in reversed(copied):
            try:
                destination.unlink()
            except OSError as cleanup_error:
                cleanup_errors.append(str(cleanup_error))
        if not cleanup_errors:
            recovery.rmdir()
        detail = f"; cleanup failed: {'; '.join(cleanup_errors)}" if cleanup_errors else ""
        raise OSError(f"could not preserve registry recovery set: {error}{detail}") from error
    return recovery


def _discard_recovery_set(path: Path) -> None:
    for member in path.iterdir():
        member.unlink()
    path.rmdir()


def reset_registry(path: Path) -> RegistryResetResult:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with _process_lock(path, exclusive=True):
        backup_path = _preserve_recovery_set(path)
        try:
            if path.exists():
                connection = sqlite3.connect(path, timeout=BUSY_TIMEOUT_MILLISECONDS / 1_000)
                try:
                    connection.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MILLISECONDS}")
                    try:
                        connection.execute("BEGIN IMMEDIATE")
                    except sqlite3.Error as error:
                        if error.sqlite_errorcode in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED}:
                            raise
                    else:
                        connection.rollback()
                finally:
                    connection.close()
        except BaseException:
            if backup_path is not None:
                _discard_recovery_set(backup_path)
            raise
        for member in _recovery_members(path):
            member.unlink()
        with _open_registry_unlocked(path, create=True):
            pass
    return RegistryResetResult(backup_path, None)
