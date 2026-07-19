"""Metadata-only SQLite continuity for CLI jobs."""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from batchwork.types import BatchSnapshot

from ._contract import Job

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
    modality TEXT NOT NULL,
    model TEXT NOT NULL,
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


@dataclass(frozen=True, slots=True)
class RegistryRoute:
    fingerprint: str
    api_key_env: str
    base_url: str | None
    headers: dict[str, str]
    header_env: dict[str, str]


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
