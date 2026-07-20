import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from batchwork.cli._contract import (
    CANONICAL_ERROR_CODES,
    DOC_PATH,
    ERROR_CODE_CATEGORIES,
    ERROR_EXAMPLE_OPERATIONS,
    EXIT_CODE_BY_CATEGORY,
    FIXTURE_DIRECTORY,
    SCHEMA_PATH,
    ImageManifestEntry,
    Materialization,
    ResultEnvelope,
    envelope_adapter,
    fixture_documents,
    reference_document,
    schema_document,
    serialize_envelope,
)
from batchwork.types import BatchResult, BatchResultStatus


def test_checked_in_schema_and_fixtures_match_typed_contract() -> None:
    assert SCHEMA_PATH.read_text() == schema_document()
    assert DOC_PATH.read_text() == reference_document()

    generated = fixture_documents()
    checked_in = {path.name: path.read_text() for path in sorted(FIXTURE_DIRECTORY.glob("*.json"))}
    assert checked_in == generated


def test_public_schema_is_valid_draft_2020_12() -> None:
    Draft202012Validator.check_schema(json.loads(SCHEMA_PATH.read_text()))


def test_canonical_error_fixtures_cover_exactly_the_stable_catalog() -> None:
    expected_names = {f"error-{code}.json" for code in CANONICAL_ERROR_CODES}
    actual_paths = set(FIXTURE_DIRECTORY.glob("error-*.json"))

    assert {path.name for path in actual_paths} == expected_names
    documents = [json.loads(path.read_text()) for path in actual_paths]
    assert {document["error"]["code"] for document in documents} == set(CANONICAL_ERROR_CODES)
    for document in documents:
        error = document["error"]
        assert error["category"] == ERROR_CODE_CATEGORIES[error["code"]]
        assert error["exit_code"] == EXIT_CODE_BY_CATEGORY[error["category"]]
        assert error["operation"] == ERROR_EXAMPLE_OPERATIONS[error["code"]]


@pytest.mark.parametrize("path", sorted(FIXTURE_DIRECTORY.glob("*.json")))
def test_foundation_fixture_validates_against_models_and_public_schema(path: Path) -> None:
    document = json.loads(path.read_text())
    envelope = envelope_adapter().validate_python(document)
    Draft202012Validator(json.loads(SCHEMA_PATH.read_text())).validate(document)

    assert json.loads(serialize_envelope(envelope)) == document


def test_machine_contract_rejects_unknown_schema_version() -> None:
    document = json.loads(fixture_documents()["job.json"])
    document["schema_version"] = 2

    with pytest.raises(ValidationError):
        envelope_adapter().validate_python(document)


def test_machine_contract_ignores_unknown_additive_fields() -> None:
    document = json.loads(fixture_documents()["job.json"])
    document["future_field"] = {"value": True}

    envelope = envelope_adapter().validate_python(document)

    assert "future_field" not in serialize_envelope(envelope)


def test_machine_contract_rejects_non_finite_numbers() -> None:
    result = BatchResult(
        custom_id="request-0",
        status=BatchResultStatus.SUCCEEDED,
        embedding=[float("nan")],
    )

    with pytest.raises(ValidationError, match="finite"):
        ResultEnvelope(job="bw_0123456789abcdef0123456789abcdef", result=result)


def test_machine_contract_rejects_inconsistent_error_category_or_exit_code() -> None:
    document = json.loads(fixture_documents()["error-wait_timeout.json"])
    document["error"]["exit_code"] = 6

    with pytest.raises(ValidationError, match="requires exit code"):
        envelope_adapter().validate_python(document)


def test_materialization_requires_absolute_output_and_fixed_manifest_name() -> None:
    image = ImageManifestEntry(
        path="request-0--c1185fd39a30--1.png",
        custom_id="request-0",
        image_index=1,
        source_kind="data",
        media_type="image/png",
        byte_count=1,
        sha256="2" * 64,
    )

    with pytest.raises(ValidationError, match="absolute path"):
        Materialization(output_dir="relative", images=[image])
    with pytest.raises(ValidationError):
        Materialization(output_dir="/absolute/output", manifest="other.json", images=[image])


def test_image_manifest_fixture_uses_normative_filename_vector() -> None:
    document = json.loads(fixture_documents()["image_manifest.json"])

    assert document["images"][0]["path"] == "request-0--c1185fd39a30--1.png"


def test_accepted_submission_fixture_has_route_complete_recovery() -> None:
    document = json.loads(fixture_documents()["error.json"])

    command = document["error"]["recovery"]["command"]
    assert command[:3] == ["batchwork", "status", "openai:batch_example"]
    assert command[3:] == ["--api-key-env", "EXAMPLE_OPENAI_API_KEY"]


def test_machine_contract_serializes_utc_timestamps_with_z() -> None:
    document = fixture_documents()["job.json"]

    assert datetime(2026, 7, 19, 12, tzinfo=UTC).isoformat() not in document
    assert "2026-07-19T12:00:00Z" in document
