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

import batchwork.cli._submit_text as submit_module
from batchwork.cli._commands import cli
from batchwork.cli._failures import InterruptionRequested, TerminationRequested


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


@pytest.mark.parametrize(
    ("signal_error", "exit_code", "error_code"),
    (
        (InterruptionRequested, 130, "interrupted"),
        (TerminationRequested, 143, "terminated"),
    ),
)
def test_signal_after_acceptance_emits_direct_identity_and_recovery(
    tmp_path: Path,
    fake_openai: tuple[str, list[tuple[str, dict[str, str], bytes]]],
    monkeypatch: pytest.MonkeyPatch,
    signal_error: type[Exception],
    exit_code: int,
    error_code: str,
) -> None:
    base_url, _ = fake_openai
    source = tmp_path / "requests.jsonl"
    source.write_text('{"prompt":"private"}\n')

    def interrupt_insert(*_: object, **__: object) -> object:
        raise signal_error

    monkeypatch.setattr(submit_module, "insert_job", interrupt_insert)
    result = CliRunner().invoke(
        cli,
        [
            "--json",
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

    assert result.exit_code == exit_code
    assert json.loads(result.stdout)["job"]["provider_reference"] == "openai:batch_123"
    error = json.loads(result.stderr)["error"]
    assert error["code"] == error_code
    assert error["submission_outcome"] == "accepted"
    assert error["records_emitted"] == 1
    assert error["recovery"]["command"][:3] == [
        "batchwork",
        "status",
        "openai:batch_123",
    ]


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


def test_submit_text_resolves_default_profile_model_and_route(
    tmp_path: Path,
    fake_openai: tuple[str, list[tuple[str, dict[str, str], bytes]]],
) -> None:
    base_url, provider_requests = fake_openai
    source = tmp_path / "requests.jsonl"
    source.write_text('{"prompt":"hello"}\n')
    config = tmp_path / "config.toml"
    config.write_text(
        f"""\
schema_version = 1
default_profile = "work"

[profiles.work.models]
text = "openai/gpt-profile"

[profiles.work.providers.openai]
api_key_env = "PROFILE_KEY"
base_url = "{base_url}"

[profiles.work.providers.openai.headers]
X-Origin = "profile"

[profiles.work.providers.openai.header_env]
X-Secret = "PROFILE_HEADER"
"""
    )

    result = CliRunner().invoke(
        cli,
        [
            "--json",
            "--config",
            str(config),
            "--registry",
            str(tmp_path / "registry.sqlite3"),
            "submit",
            "text",
            str(source),
        ],
        env={"PROFILE_KEY": "key-secret", "PROFILE_HEADER": "header-secret"},
    )

    assert result.exit_code == 0, result.stderr
    job = json.loads(result.stdout)["job"]
    assert job["model"] == "openai/gpt-profile"
    assert job["profile"] == "work"
    upload_headers = provider_requests[0][1]
    assert upload_headers["Authorization"] == "Bearer key-secret"
    assert upload_headers["x-origin"] == "profile"
    assert upload_headers["x-secret"] == "header-secret"
    assert b'"model":"gpt-profile"' in provider_requests[0][2]


def test_submit_text_flags_override_profile_route_fields(
    tmp_path: Path,
    fake_openai: tuple[str, list[tuple[str, dict[str, str], bytes]]],
) -> None:
    base_url, provider_requests = fake_openai
    source = tmp_path / "requests.jsonl"
    source.write_text('{"prompt":"hello"}\n')
    config = tmp_path / "config.toml"
    config.write_text(
        """\
schema_version = 1
default_profile = "work"
[profiles.work.models]
text = "openai/profile-model"
[profiles.work.providers.openai]
api_key_env = "PROFILE_KEY"
base_url = "https://unused.example/v1"
[profiles.work.providers.openai.headers]
X-Origin = "profile"
[profiles.work.providers.openai.header_env]
X-Secret = "PROFILE_HEADER"
"""
    )

    result = CliRunner().invoke(
        cli,
        [
            "--config",
            str(config),
            "--registry",
            str(tmp_path / "registry.sqlite3"),
            "submit",
            "text",
            str(source),
            "--model",
            "openai/flag-model",
            "--base-url",
            base_url,
            "--api-key-env",
            "FLAG_KEY",
            "--header",
            "X-Origin=flag",
            "--header-env",
            "X-Secret=FLAG_HEADER",
        ],
        env={"FLAG_KEY": "flag-key", "FLAG_HEADER": "flag-header"},
    )

    assert result.exit_code == 0, result.stderr
    headers = provider_requests[0][1]
    assert headers["Authorization"] == "Bearer flag-key"
    assert headers["x-origin"] == "flag"
    assert headers["x-secret"] == "flag-header"
    assert b'"model":"flag-model"' in provider_requests[0][2]


def test_invalid_explicit_config_fails_before_source_or_provider_work(
    tmp_path: Path,
    fake_openai: tuple[str, list[tuple[str, dict[str, str], bytes]]],
) -> None:
    _, provider_requests = fake_openai
    config = tmp_path / "config.toml"
    config.write_text("schema_version = 2\n")

    result = CliRunner().invoke(
        cli,
        [
            "--config",
            str(config),
            "submit",
            "text",
            str(tmp_path / "missing.jsonl"),
            "--model",
            "openai/gpt-test",
        ],
    )

    assert result.exit_code == 3
    assert "schema version 1" in result.stderr
    assert "missing.jsonl" not in result.stderr
    assert provider_requests == []


@pytest.mark.parametrize(
    ("filename", "contents", "input_format"),
    (
        ("requests.json", '{"prompt":"hello"}', None),
        ("requests.jsonl", '{"prompt":"hello"}\n', None),
        ("requests.csv", "prompt,temperature\nhello,0.2\n", None),
        ("requests.txt", "hello\n", None),
        (None, '{"prompt":"hello"}', "json"),
    ),
)
def test_submit_text_accepts_every_transport_and_explicit_stdin(
    tmp_path: Path,
    fake_openai: tuple[str, list[tuple[str, dict[str, str], bytes]]],
    filename: str | None,
    contents: str,
    input_format: str | None,
) -> None:
    base_url, provider_requests = fake_openai
    source = "-"
    if filename is not None:
        path = tmp_path / filename
        path.write_text(contents)
        source = str(path)
    arguments = [
        "--registry",
        str(tmp_path / "registry.sqlite3"),
        "submit",
        "text",
        source,
        "--model",
        "openai/gpt-test",
        "--base-url",
        base_url,
    ]
    if input_format is not None:
        arguments.extend(["--format", input_format])

    result = CliRunner().invoke(
        cli,
        arguments,
        input=contents if filename is None else None,
        env={"OPENAI_API_KEY": "secret"},
    )

    assert result.exit_code == 0, result.stderr
    assert [request[0] for request in provider_requests] == ["/v1/files", "/v1/batches"]
    assert b'"custom_id":"request-0"' in provider_requests[0][2]


@pytest.mark.parametrize(
    ("filename", "contents", "message"),
    (
        (
            "requests.json",
            '[{"prompt":"valid"},{"prompt":"also valid","unknown":true}]',
            "JSON index 1",
        ),
        (
            "requests.csv",
            "prompt,max_output_tokens\nvalid,1\nalso valid,nope\n",
            "row 3, column max_output_tokens",
        ),
    ),
)
def test_submit_text_rejects_invalid_structured_source_before_provider_mutation(
    tmp_path: Path,
    fake_openai: tuple[str, list[tuple[str, dict[str, str], bytes]]],
    filename: str,
    contents: str,
    message: str,
) -> None:
    base_url, provider_requests = fake_openai
    source = tmp_path / filename
    source.write_text(contents)

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
    assert message in result.stderr
    assert provider_requests == []


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

    assert result.exit_code == 3
    error = json.loads(result.stderr)["error"]
    assert error["code"] == "missing_environment_variable"
    assert "MISSING_KEY" in error["message"]
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
        "--api-key-env",
        "TEST_OPENAI_KEY",
        "--base-url",
        base_url,
    ]
    assert [request[0] for request in provider_requests] == ["/v1/files", "/v1/batches"]


def test_registry_failure_recovery_never_emits_literal_header_values(
    tmp_path: Path,
    fake_openai: tuple[str, list[tuple[str, dict[str, str], bytes]]],
) -> None:
    base_url, _ = fake_openai
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
            "--header",
            "X-Tenant=private-tenant",
        ],
        env={"TEST_OPENAI_KEY": "secret"},
    )

    assert result.exit_code == 8
    assert "private-tenant" not in result.stderr
    assert "--header" not in result.stderr


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
