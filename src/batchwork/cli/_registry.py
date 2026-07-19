"""Metadata-only SQLite continuity for CLI jobs."""

from __future__ import annotations

import json
import os
import re
import sqlite3
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

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
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
    modality TEXT,
    model TEXT,
    profile TEXT,
    status TEXT NOT NULL,
    request_total INTEGER NOT NULL,
    request_completed INTEGER NOT NULL,
    request_failed INTEGER NOT NULL,
    registered_at TEXT NOT NULL,
    provider_created_at TEXT,
    completed_at TEXT,
    expires_at TEXT,
    terminal_at TEXT,
    last_refreshed_at TEXT,
    UNIQUE (provider, provider_job_id, routing_fingerprint)
)
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


def default_registry_path() -> Path:
    configured = os.environ.get("BATCHWORK_REGISTRY")
    if configured:
        return Path(configured)
    data_home = os.environ.get("XDG_DATA_HOME")
    root = Path(data_home) if data_home else Path.home() / ".local" / "share"
    return root / "batchwork" / "registry.sqlite3"


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
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    record_id = f"bw_{uuid4().hex}"
    provider_reference = f"{snapshot.provider.value}:{snapshot.id}"
    counts = snapshot.request_counts
    with sqlite3.connect(path, timeout=5) as connection:
        if os.name == "posix":
            path.chmod(0o600)
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA busy_timeout = 5000")
        connection.execute(SCHEMA)
        connection.execute("PRAGMA user_version = 1")
        connection.execute(
            """
            INSERT INTO jobs (
                record_id, name, provider, provider_job_id, provider_reference,
                routing_fingerprint, api_key_env, base_url, headers_json, header_env_json,
                modality, model, profile, status, request_total, request_completed,
                request_failed, registered_at, provider_created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record_id,
                name,
                snapshot.provider.value,
                snapshot.id,
                provider_reference,
                route.fingerprint,
                route.api_key_env,
                route.base_url,
                json.dumps(
                    route.headers, ensure_ascii=False, separators=(",", ":"), sort_keys=True
                ),
                json.dumps(
                    route.header_env, ensure_ascii=False, separators=(",", ":"), sort_keys=True
                ),
                "text",
                model,
                profile,
                snapshot.status.value,
                counts.total,
                counts.completed,
                counts.failed,
                registered_at.isoformat(),
                snapshot.created_at.isoformat() if snapshot.created_at else None,
            ),
        )
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
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with sqlite3.connect(path, timeout=5) as connection:
        if os.name == "posix":
            path.chmod(0o600)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA busy_timeout = 5000")
        connection.execute(SCHEMA)
        _ensure_columns(connection)
        row = connection.execute(
            """
            SELECT * FROM jobs
            WHERE provider = ? AND provider_job_id = ? AND routing_fingerprint = ?
            """,
            (snapshot.provider.value, snapshot.id, route.fingerprint),
        ).fetchone()
        if row is not None:
            if name is not None and row["name"] not in {None, name}:
                raise sqlite3.IntegrityError("job already has a different name")
            if name is not None and row["name"] is None:
                connection.execute(
                    "UPDATE jobs SET name = ? WHERE record_id = ?", (name, row["record_id"])
                )
                row = connection.execute(
                    "SELECT * FROM jobs WHERE record_id = ?", (row["record_id"],)
                ).fetchone()
            if row is None:
                raise sqlite3.IntegrityError("adopted job disappeared")
            return _registry_job(row)
        record_id = f"bw_{uuid4().hex}"
        counts = snapshot.request_counts
        provider_reference = f"{snapshot.provider.value}:{snapshot.id}"
        connection.execute(
            """
            INSERT INTO jobs (
                record_id, name, provider, provider_job_id, provider_reference,
                routing_fingerprint, api_key_env, base_url, headers_json, header_env_json,
                modality, model, profile, status, request_total, request_completed,
                request_failed, registered_at, provider_created_at, completed_at, expires_at,
                terminal_at, last_refreshed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record_id,
                name,
                snapshot.provider.value,
                snapshot.id,
                provider_reference,
                route.fingerprint,
                route.api_key_env,
                route.base_url,
                json.dumps(route.headers, separators=(",", ":"), sort_keys=True),
                json.dumps(route.header_env, separators=(",", ":"), sort_keys=True),
                "text",
                None,
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
            ),
        )
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


def _ensure_columns(connection: sqlite3.Connection) -> None:
    existing = {row[1] for row in connection.execute("PRAGMA table_info(jobs)")}
    for name in ("completed_at", "expires_at", "terminal_at", "last_refreshed_at"):
        if name not in existing:
            connection.execute(f"ALTER TABLE jobs ADD COLUMN {name} TEXT")


def get_job(path: Path, selector: str) -> RegistryJob | None:
    if not path.is_file():
        return None
    with sqlite3.connect(path, timeout=5) as connection:
        connection.row_factory = sqlite3.Row
        _ensure_columns(connection)
        column = "record_id" if _RECORD_ID.fullmatch(selector) else "name"
        row = connection.execute(f"SELECT * FROM jobs WHERE {column} = ?", (selector,)).fetchone()
    return _registry_job(row) if row is not None else None


def update_job(path: Path, record_id: str, snapshot: BatchSnapshot, refreshed_at: datetime) -> None:
    terminal = is_terminal_status(snapshot.status)
    counts = snapshot.request_counts
    with sqlite3.connect(path, timeout=5) as connection:
        _ensure_columns(connection)
        current = connection.execute(
            "SELECT status FROM jobs WHERE record_id = ?", (record_id,)
        ).fetchone()
        if current is None:
            return
        current_terminal = is_terminal_status(BatchStatus(current[0]))
        if current_terminal and not terminal:
            return
        connection.execute(
            """
            UPDATE jobs SET status = ?, request_total = ?, request_completed = ?,
                request_failed = ?, completed_at = ?, expires_at = ?,
                terminal_at = COALESCE(terminal_at, ?), last_refreshed_at = ?
            WHERE record_id = ?
            """,
            (
                snapshot.status.value,
                counts.total,
                counts.completed,
                counts.failed,
                snapshot.completed_at.isoformat() if snapshot.completed_at else None,
                snapshot.expires_at.isoformat() if snapshot.expires_at else None,
                refreshed_at.isoformat() if terminal else None,
                refreshed_at.isoformat(),
                record_id,
            ),
        )
