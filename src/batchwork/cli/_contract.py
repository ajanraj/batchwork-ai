"""Typed source of truth for the CLI schema-v1 machine contract."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal, TypeAlias

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    field_validator,
    model_validator,
)

from batchwork.types import (
    BatchImage,
    BatchProvider,
    BatchRequestCounts,
    BatchResult,
    BatchResultStatus,
    BatchSnapshot,
    BatchStatus,
)

SCHEMA_VERSION = 1
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_PATH = REPOSITORY_ROOT / "docs/public/schemas/batchwork-cli-v1.schema.json"
FIXTURE_DIRECTORY = REPOSITORY_ROOT / "tests/fixtures/cli-v1"

Modality: TypeAlias = Literal["text", "embeddings", "images"]
RecordId: TypeAlias = Annotated[str, Field(pattern=r"^bw_[0-9a-f]{32}$")]
Sha256Hex: TypeAlias = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
RoutingFingerprint: TypeAlias = Sha256Hex
ErrorCategory: TypeAlias = Literal[
    "internal",
    "usage",
    "configuration",
    "provider_rejection",
    "provider_availability",
    "job_state",
    "wait_timeout",
    "local_state",
    "interrupted",
    "terminated",
]
ExitCode: TypeAlias = Literal[1, 2, 3, 4, 5, 6, 7, 8, 130, 143]


def _reject_non_finite(value: object, location: str = "machine output") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{location} numbers must be finite")
    if isinstance(value, BaseModel):
        _reject_non_finite(value.model_dump(mode="python"), location)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_non_finite(item, f"{location}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        for index, item in enumerate(value):
            _reject_non_finite(item, f"{location}[{index}]")


class ContractModel(BaseModel):
    """Strict immutable producer model for schema-v1 records."""

    model_config = ConfigDict(extra="ignore", frozen=True, validate_default=True)

    @model_validator(mode="after")
    def _finite_json_numbers(self) -> ContractModel:
        _reject_non_finite(self.model_dump(mode="python"))
        return self


class Job(ContractModel):
    record_id: RecordId | None = None
    name: str | None = Field(default=None, min_length=1, max_length=64)
    provider: BatchProvider
    provider_job_id: str = Field(min_length=1)
    provider_reference: str = Field(min_length=3)
    routing_fingerprint: RoutingFingerprint
    modality: Modality | None = None
    model: str | None = None
    profile: str | None = None
    status: BatchStatus | None = None
    request_counts: BatchRequestCounts | None = None
    registered_at: AwareDatetime | None = None
    provider_created_at: AwareDatetime | None = None
    completed_at: AwareDatetime | None = None
    expires_at: AwareDatetime | None = None
    terminal_at: AwareDatetime | None = None
    last_refreshed_at: AwareDatetime | None = None


class Recovery(ContractModel):
    action: str = Field(min_length=1)
    command: list[str] | None = None


class ErrorDetail(ContractModel):
    code: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    category: ErrorCategory
    message: str = Field(min_length=1)
    exit_code: ExitCode
    retryable: bool
    operation: str = Field(min_length=1)
    provider: BatchProvider | None = None
    job: str | None = None
    routing_fingerprint: RoutingFingerprint | None = None
    profile: str | None = None
    config_path: str | None = None
    registry_path: str | None = None
    http_status: int | None = Field(default=None, ge=100, le=599)
    request_id: str | None = None
    retry_after_seconds: float | None = Field(default=None, ge=0)
    submission_outcome: Literal["not_sent", "rejected", "unknown", "accepted"] | None = None
    partial_output: bool | None = None
    records_emitted: int | None = Field(default=None, ge=0)
    item_successes: int | None = Field(default=None, ge=0)
    item_failures: int | None = Field(default=None, ge=0)
    cancel_requested: bool | None = None
    materialized_images: int | None = Field(default=None, ge=0)
    materialized_bytes: int | None = Field(default=None, ge=0)
    recovery: Recovery | None = None


class PathState(ContractModel):
    path: str
    exists: bool


class ConfigProviderView(ContractModel):
    api_key_env: str
    base_url: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    header_env: dict[str, str] = Field(default_factory=dict)


class ImageManifestEntry(ContractModel):
    path: str
    custom_id: str
    image_index: int = Field(ge=1)
    source_kind: Literal["data", "url"]
    media_type: str
    byte_count: int = Field(ge=0)
    sha256: Sha256Hex


class Materialization(ContractModel):
    output_dir: str
    manifest: Literal["manifest.json"] = "manifest.json"
    images: list[ImageManifestEntry]

    @field_validator("output_dir")
    @classmethod
    def _absolute_output_directory(cls, value: str) -> str:
        if not Path(value).is_absolute():
            raise ValueError("output_dir must be an absolute path")
        return value


class JobEnvelope(ContractModel):
    schema_version: Literal[1] = SCHEMA_VERSION
    type: Literal["job"] = "job"
    job: Job


class SnapshotEnvelope(ContractModel):
    schema_version: Literal[1] = SCHEMA_VERSION
    type: Literal["snapshot"] = "snapshot"
    job: str
    routing_fingerprint: RoutingFingerprint | None = None
    snapshot: BatchSnapshot


class ResultEnvelope(ContractModel):
    schema_version: Literal[1] = SCHEMA_VERSION
    type: Literal["result"] = "result"
    job: str
    routing_fingerprint: RoutingFingerprint | None = None
    result: BatchResult
    materialization: Materialization | None = None


class JobListEnvelope(ContractModel):
    schema_version: Literal[1] = SCHEMA_VERSION
    type: Literal["job_list"] = "job_list"
    jobs: list[Job]


class ResultListEnvelope(ContractModel):
    schema_version: Literal[1] = SCHEMA_VERSION
    type: Literal["result_list"] = "result_list"
    job: str
    routing_fingerprint: RoutingFingerprint | None = None
    results: list[BatchResult]
    materialization: Materialization | None = None


class RunEnvelope(ContractModel):
    schema_version: Literal[1] = SCHEMA_VERSION
    type: Literal["run"] = "run"
    job: Job
    snapshot: BatchSnapshot
    results: list[BatchResult]
    materialization: Materialization | None = None


class ErrorEnvelope(ContractModel):
    schema_version: Literal[1] = SCHEMA_VERSION
    type: Literal["error"] = "error"
    error: ErrorDetail


class PathsEnvelope(ContractModel):
    schema_version: Literal[1] = SCHEMA_VERSION
    type: Literal["paths"] = "paths"
    config: PathState
    registry: PathState


class ConfigValidationEnvelope(ContractModel):
    schema_version: Literal[1] = SCHEMA_VERSION
    type: Literal["config_validation"] = "config_validation"
    path: str
    exists: bool
    valid: bool
    config_schema_version: Literal[1] | None = None
    profiles: list[str] = Field(default_factory=list)
    default_profile: str | None = None
    credentials_read: Literal[False] = False


class ConfigViewEnvelope(ContractModel):
    schema_version: Literal[1] = SCHEMA_VERSION
    type: Literal["config_view"] = "config_view"
    path: str
    profile: str | None = None
    models: dict[str, str] = Field(default_factory=dict)
    providers: dict[str, ConfigProviderView] = Field(default_factory=dict)
    credentials_read: Literal[False] = False


class RegistryCheckEnvelope(ContractModel):
    schema_version: Literal[1] = SCHEMA_VERSION
    type: Literal["registry_check"] = "registry_check"
    path: str
    ok: bool
    user_version: int = Field(ge=0)
    integrity: str


class RegistryPrunePlanEnvelope(ContractModel):
    schema_version: Literal[1] = SCHEMA_VERSION
    type: Literal["registry_prune_plan"] = "registry_prune_plan"
    path: str
    older_than: str
    cutoff_at: AwareDatetime
    candidate_records: int = Field(ge=0)
    committed: Literal[False] = False
    remote_jobs_changed: Literal[False] = False


class RegistryChangeEnvelope(ContractModel):
    schema_version: Literal[1] = SCHEMA_VERSION
    type: Literal["registry_change"] = "registry_change"
    operation: Literal["forget", "prune", "reset"]
    path: str
    changed_records: int | None = Field(default=None, ge=0)
    record_id: RecordId | None = None
    provider_reference: str | None = None
    older_than: str | None = None
    backup_path: str | None = None
    records_count_known: bool | None = None
    user_version: int | None = Field(default=None, ge=0)
    remote_jobs_changed: Literal[False] = False


class ImageManifestEnvelope(ContractModel):
    schema_version: Literal[1] = SCHEMA_VERSION
    type: Literal["image_manifest"] = "image_manifest"
    job: str
    routing_fingerprint: RoutingFingerprint | None = None
    images: list[ImageManifestEntry]


Envelope: TypeAlias = Annotated[
    JobEnvelope
    | SnapshotEnvelope
    | ResultEnvelope
    | JobListEnvelope
    | ResultListEnvelope
    | RunEnvelope
    | ErrorEnvelope
    | PathsEnvelope
    | ConfigValidationEnvelope
    | ConfigViewEnvelope
    | RegistryCheckEnvelope
    | RegistryPrunePlanEnvelope
    | RegistryChangeEnvelope
    | ImageManifestEnvelope,
    Field(discriminator="type"),
]


def envelope_adapter() -> TypeAdapter[Envelope]:
    return TypeAdapter(Envelope)


def serialize_envelope(envelope: Envelope) -> str:
    document = envelope_adapter().dump_python(envelope, mode="json", exclude_none=True)
    return (
        json.dumps(
            document,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n"
    )


def _allow_additive_fields(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: _allow_additive_fields(item)
            for key, item in value.items()
            if key != "additionalProperties" or item is not False
        }
    if isinstance(value, list):
        return [_allow_additive_fields(item) for item in value]
    return value


def schema_document() -> str:
    schema = envelope_adapter().json_schema(by_alias=False, mode="serialization")
    schema["$id"] = "https://batchwork.ajanraj.com/schemas/batchwork-cli-v1.schema.json"
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["title"] = "Batchwork CLI machine envelope schema version 1"
    schema = _allow_additive_fields(schema)
    return json.dumps(schema, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _foundation_envelopes() -> dict[str, Envelope]:
    timestamp = datetime(2026, 7, 19, 12, tzinfo=UTC)
    fingerprint = "1" * 64
    record_id = "bw_0123456789abcdef0123456789abcdef"
    counts = BatchRequestCounts(total=2, completed=2, failed=0)
    job = Job(
        record_id=record_id,
        name="example",
        provider=BatchProvider.OPENAI,
        provider_job_id="batch_example",
        provider_reference="openai:batch_example",
        routing_fingerprint=fingerprint,
        modality="images",
        model="openai/gpt-image-1",
        status=BatchStatus.COMPLETED,
        request_counts=counts,
        registered_at=timestamp,
        provider_created_at=timestamp,
        completed_at=timestamp,
        terminal_at=timestamp,
        last_refreshed_at=timestamp,
    )
    snapshot = BatchSnapshot(
        id="batch_example",
        provider=BatchProvider.OPENAI,
        status=BatchStatus.COMPLETED,
        request_counts=counts,
        raw={"providerState": "completed"},
        created_at=timestamp,
        completed_at=timestamp,
    )
    result = BatchResult(
        custom_id="request-0",
        status=BatchResultStatus.SUCCEEDED,
        images=[BatchImage(url="https://example.com/image.png")],
        response={"providerField": "preserved"},
    )
    image_sha = "2" * 64
    custom_id_hash = hashlib.sha256(result.custom_id.encode()).hexdigest()[:12]
    image_entry = ImageManifestEntry(
        path=f"request-0--{custom_id_hash}--1.png",
        custom_id=result.custom_id,
        image_index=1,
        source_kind="url",
        media_type="image/png",
        byte_count=68,
        sha256=image_sha,
    )
    materialization = Materialization(
        output_dir="/home/example/output",
        images=[image_entry],
    )
    return {
        "job.json": JobEnvelope(job=job),
        "snapshot.json": SnapshotEnvelope(job=record_id, snapshot=snapshot),
        "result.json": ResultEnvelope(
            job=record_id,
            result=result,
            materialization=materialization,
        ),
        "job_list.json": JobListEnvelope(jobs=[job]),
        "result_list.json": ResultListEnvelope(
            job=record_id,
            results=[result],
            materialization=materialization,
        ),
        "run.json": RunEnvelope(
            job=job,
            snapshot=snapshot,
            results=[result],
            materialization=materialization,
        ),
        "error.json": ErrorEnvelope(
            error=ErrorDetail(
                code="registry_write_failed_after_submit",
                category="local_state",
                message=(
                    "The provider accepted the batch, but Batchwork could not record it locally."
                ),
                exit_code=8,
                retryable=False,
                operation="submit",
                provider=BatchProvider.OPENAI,
                job="openai:batch_example",
                routing_fingerprint=fingerprint,
                submission_outcome="accepted",
                partial_output=True,
                records_emitted=1,
                recovery=Recovery(
                    action="resume_with_direct_reference",
                    command=["batchwork", "status", "openai:batch_example"],
                ),
            )
        ),
        "paths.json": PathsEnvelope(
            config=PathState(path="/home/example/.config/batchwork/config.toml", exists=True),
            registry=PathState(
                path="/home/example/.local/share/batchwork/registry.sqlite3", exists=True
            ),
        ),
        "config_validation.json": ConfigValidationEnvelope(
            path="/home/example/.config/batchwork/config.toml",
            exists=True,
            valid=True,
            config_schema_version=1,
            profiles=["work"],
            default_profile="work",
        ),
        "config_view.json": ConfigViewEnvelope(
            path="/home/example/.config/batchwork/config.toml",
            profile="work",
            models={"text": "openai/gpt-5.6-sol"},
            providers={
                "openai": ConfigProviderView(
                    api_key_env="WORK_OPENAI_API_KEY",
                    base_url="https://gateway.example.com/v1",
                    headers={"X-Application": "batchwork-cli"},
                    header_env={"Authorization": "WORK_GATEWAY_AUTHORIZATION"},
                )
            },
        ),
        "registry_check.json": RegistryCheckEnvelope(
            path="/home/example/.local/share/batchwork/registry.sqlite3",
            ok=True,
            user_version=1,
            integrity="ok",
        ),
        "registry_prune_plan.json": RegistryPrunePlanEnvelope(
            path="/home/example/.local/share/batchwork/registry.sqlite3",
            older_than="30d",
            cutoff_at=timestamp,
            candidate_records=12,
        ),
        "registry_change.json": RegistryChangeEnvelope(
            operation="forget",
            path="/home/example/.local/share/batchwork/registry.sqlite3",
            changed_records=1,
            record_id=record_id,
            provider_reference="openai:batch_example",
        ),
        "image_manifest.json": ImageManifestEnvelope(
            job=record_id,
            images=[image_entry],
        ),
    }


def fixture_documents() -> dict[str, str]:
    return {
        name: serialize_envelope(envelope) for name, envelope in _foundation_envelopes().items()
    }


def contract_drift() -> list[Path]:
    expected = {SCHEMA_PATH: schema_document()}
    expected.update(
        {FIXTURE_DIRECTORY / name: document for name, document in fixture_documents().items()}
    )
    drifted = [
        path
        for path, document in expected.items()
        if not path.exists() or path.read_text() != document
    ]
    tracked_fixtures = (
        set(FIXTURE_DIRECTORY.glob("*.json")) if FIXTURE_DIRECTORY.exists() else set()
    )
    drifted.extend(sorted(tracked_fixtures - set(expected)))
    return drifted


def write_contract_artifacts() -> None:
    SCHEMA_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    SCHEMA_PATH.write_text(schema_document())
    for name, document in fixture_documents().items():
        (FIXTURE_DIRECTORY / name).write_text(document)
