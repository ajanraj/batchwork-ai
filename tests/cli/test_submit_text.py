from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sysconfig
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar

import pytest
from click.testing import CliRunner

from batchwork.cli._commands import cli


class _FakeOpenAIHandler(BaseHTTPRequestHandler):
    requests: ClassVar[list[tuple[str, dict[str, str], bytes]]] = []

    def do_POST(self) -> None:
        length = int(self.headers["Content-Length"])
        body = self.rfile.read(length)
        self.requests.append((self.path, dict(self.headers), body))
        if self.path == "/v1/files":
            response = {"id": "file-input"}
        elif self.path == "/v1/batches":
            response = {
                "id": "batch_123",
                "status": "validating",
                "request_counts": {"total": 2, "completed": 0, "failed": 0},
            }
        else:
            self.send_error(404)
            return
        encoded = json.dumps(response).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args: object) -> None:
        return


@pytest.fixture
def fake_openai() -> tuple[str, list[tuple[str, dict[str, str], bytes]]]:
    _FakeOpenAIHandler.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeOpenAIHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}/v1", _FakeOpenAIHandler.requests
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def _installed_batchwork() -> Path:
    executable = Path(sysconfig.get_path("scripts")) / (
        "batchwork.exe" if os.name == "nt" else "batchwork"
    )
    assert executable.is_file(), "batchwork console script is not installed"
    return executable


def test_installed_submit_text_emits_job_and_persists_metadata_only(
    tmp_path: Path,
    fake_openai: tuple[str, list[tuple[str, dict[str, str], bytes]]],
) -> None:
    base_url, provider_requests = fake_openai
    source = tmp_path / "requests.jsonl"
    source.write_text(
        "\n".join(
            (
                json.dumps({"custom_id": "first", "prompt": "private prompt"}),
                "",
                json.dumps({"prompt": "second private prompt", "temperature": 0.2}),
            )
        )
        + "\n"
    )
    registry = tmp_path / "registry.sqlite3"
    environment = os.environ.copy()
    environment.update({"TEST_OPENAI_KEY": "top-secret", "NO_COLOR": "1"})

    result = subprocess.run(
        [
            str(_installed_batchwork()),
            "--json",
            "--registry",
            str(registry),
            "submit",
            "text",
            str(source),
            "--model",
            "openai/gpt-test",
            "--base-url",
            base_url,
            "--api-key-env",
            "TEST_OPENAI_KEY",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    envelope = json.loads(result.stdout)
    assert envelope["schema_version"] == 1
    assert envelope["type"] == "job"
    assert envelope["job"] | {
        "record_id": "ignored",
        "routing_fingerprint": "ignored",
        "registered_at": "ignored",
    } == {
        "record_id": "ignored",
        "provider": "openai",
        "provider_job_id": "batch_123",
        "provider_reference": "openai:batch_123",
        "routing_fingerprint": "ignored",
        "modality": "text",
        "model": "openai/gpt-test",
        "status": "validating",
        "request_counts": {"total": 2, "completed": 0, "failed": 0},
        "registered_at": "ignored",
    }
    assert envelope["job"]["record_id"].startswith("bw_")
    assert len(envelope["job"]["routing_fingerprint"]) == 64

    assert [request[0] for request in provider_requests] == ["/v1/files", "/v1/batches"]
    upload = provider_requests[0]
    assert upload[1]["Authorization"] == "Bearer top-secret"
    assert b'"custom_id":"first"' in upload[2]
    assert b'"custom_id":"request-1"' in upload[2]
    assert b'"model":"gpt-test"' in upload[2]

    with sqlite3.connect(registry) as connection:
        row = connection.execute("SELECT * FROM jobs").fetchone()
        columns = [
            description[0] for description in connection.execute("SELECT * FROM jobs").description
        ]
    persisted = dict(zip(columns, row, strict=True))
    assert persisted["record_id"] == envelope["job"]["record_id"]
    assert persisted["provider_job_id"] == "batch_123"
    assert persisted["api_key_env"] == "TEST_OPENAI_KEY"
    assert "private prompt" not in repr(persisted)
    assert "top-secret" not in repr(persisted)
    assert not {"request", "result", "raw", "secret", "source"}.intersection(persisted)
    database = registry.read_bytes()
    assert b"private prompt" not in database
    assert b"top-secret" not in database


@pytest.mark.parametrize(
    ("contents", "message"),
    (
        ('{"prompt":"valid"}\nnot-json\n', "line 2"),
        ('{"prompt":"a","custom_id":"request-1"}\n{"prompt":"b"}\n', "custom_id"),
        ('{"prompt":"a","custom_id":"a","customId":"b"}\n', "cannot both be present"),
        ('{"prompt":"a","temperature":NaN}\n', "non-finite JSON number"),
        ("[]\n", "object"),
    ),
)
def test_submit_text_rejects_whole_invalid_source_before_provider_mutation(
    tmp_path: Path,
    fake_openai: tuple[str, list[tuple[str, dict[str, str], bytes]]],
    contents: str,
    message: str,
) -> None:
    base_url, provider_requests = fake_openai
    source = tmp_path / "requests.jsonl"
    source.write_text(contents)

    result = CliRunner().invoke(
        cli,
        [
            "--registry",
            str(tmp_path / "registry.sqlite3"),
            "submit",
            "text",
            str(source),
            "--model",
            "openai/gpt-test",
            "--base-url",
            base_url,
            "--api-key-env",
            "TEST_OPENAI_KEY",
        ],
        env={"TEST_OPENAI_KEY": "secret"},
    )

    assert result.exit_code == 2
    assert message in result.stderr
    assert provider_requests == []


def test_submit_text_validates_credentials_before_provider_mutation(
    tmp_path: Path,
    fake_openai: tuple[str, list[tuple[str, dict[str, str], bytes]]],
) -> None:
    base_url, provider_requests = fake_openai
    source = tmp_path / "requests.jsonl"
    source.write_text('{"prompt":"private"}\n')

    result = CliRunner().invoke(
        cli,
        [
            "submit",
            "text",
            str(source),
            "--model",
            "openai/gpt-test",
            "--base-url",
            base_url,
            "--api-key-env",
            "MISSING_KEY",
        ],
    )

    assert result.exit_code == 2
    assert "MISSING_KEY" in result.stderr
    assert provider_requests == []
    assert "private" not in result.stderr


def test_submit_text_rejects_large_batch_without_authorization_before_mutation(
    tmp_path: Path,
    fake_openai: tuple[str, list[tuple[str, dict[str, str], bytes]]],
) -> None:
    base_url, provider_requests = fake_openai
    source = tmp_path / "requests.jsonl"
    source.write_text("".join(f'{{"prompt":"{index}"}}\n' for index in range(10_001)))

    result = CliRunner().invoke(
        cli,
        [
            "submit",
            "text",
            str(source),
            "--model",
            "openai/gpt-test",
            "--base-url",
            base_url,
        ],
        env={"OPENAI_API_KEY": "secret"},
    )

    assert result.exit_code == 2
    assert "--allow-large-batch" in result.stderr
    assert provider_requests == []


def test_submit_text_rejects_oversized_jsonl_line_before_provider_mutation(
    tmp_path: Path,
    fake_openai: tuple[str, list[tuple[str, dict[str, str], bytes]]],
) -> None:
    base_url, provider_requests = fake_openai
    source = tmp_path / "requests.jsonl"
    with source.open("wb") as stream:
        stream.write(b"x" * (24 * 1024 * 1024 + 1))

    result = CliRunner().invoke(
        cli,
        [
            "submit",
            "text",
            str(source),
            "--model",
            "openai/gpt-test",
            "--base-url",
            base_url,
        ],
        env={"OPENAI_API_KEY": "secret"},
    )

    assert result.exit_code == 2
    assert "line 1" in result.stderr
    assert "25165824 byte limit" in result.stderr
    assert provider_requests == []


def test_accepted_job_is_emitted_with_recovery_when_registry_write_fails(
    tmp_path: Path,
    fake_openai: tuple[str, list[tuple[str, dict[str, str], bytes]]],
) -> None:
    base_url, provider_requests = fake_openai
    source = tmp_path / "requests.jsonl"
    source.write_text('{"prompt":"hello"}\n')
    blocked_parent = tmp_path / "not-a-directory"
    blocked_parent.write_text("blocked")

    result = CliRunner().invoke(
        cli,
        [
            "--json",
            "--registry",
            str(blocked_parent / "registry.sqlite3"),
            "submit",
            "text",
            str(source),
            "--model",
            "openai/gpt-test",
            "--base-url",
            base_url,
            "--api-key-env",
            "TEST_OPENAI_KEY",
        ],
        env={"TEST_OPENAI_KEY": "secret"},
    )

    assert result.exit_code == 8
    job = json.loads(result.stdout)
    error = json.loads(result.stderr)
    assert job["job"]["provider_reference"] == "openai:batch_123"
    assert "record_id" not in job["job"]
    assert error["error"]["code"] == "registry_write_failed_after_submit"
    assert error["error"]["submission_outcome"] == "accepted"
    assert error["error"]["recovery"]["command"] == [
        "batchwork",
        "status",
        "openai:batch_123",
        "--provider",
        "openai",
        "--api-key-env",
        "TEST_OPENAI_KEY",
        "--base-url",
        base_url,
    ]
    assert [request[0] for request in provider_requests] == ["/v1/files", "/v1/batches"]


def test_submit_text_human_output_contains_copyable_selector(
    tmp_path: Path,
    fake_openai: tuple[str, list[tuple[str, dict[str, str], bytes]]],
) -> None:
    base_url, _ = fake_openai
    source = tmp_path / "requests.jsonl"
    source.write_text('{"prompt":"hello"}\n')

    result = CliRunner().invoke(
        cli,
        [
            "--human",
            "--registry",
            str(tmp_path / "registry.sqlite3"),
            "submit",
            "text",
            str(source),
            "--model",
            "openai/gpt-test",
            "--base-url",
            base_url,
        ],
        env={"OPENAI_API_KEY": "secret"},
    )

    assert result.exit_code == 0, result.stderr
    selector = next(
        line.removeprefix("Job: ")
        for line in result.stdout.splitlines()
        if line.startswith("Job: ")
    )
    assert selector.startswith("bw_")
    assert f"batchwork status {selector}" in result.stdout
    assert result.stderr == ""
