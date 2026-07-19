from __future__ import annotations

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar

import pytest
from click.testing import CliRunner

import batchwork.cli._commands as commands_module
from batchwork.cli._commands import cli


class _FailureHandler(BaseHTTPRequestHandler):
    status: ClassVar[int] = 503
    body: ClassVar[bytes] = b'{"secret":"provider body"}'
    disconnect: ClassVar[bool] = False
    requests: ClassVar[int] = 0

    def _respond(self) -> None:
        type(self).requests += 1
        if self.disconnect:
            self.connection.shutdown(socket.SHUT_RDWR)
            self.connection.close()
            return
        self.send_response(self.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(self.body)))
        self.send_header("X-Request-Id", "req_safe")
        self.send_header("Retry-After", "7200")
        self.send_header("X-Provider-Secret", "secret-header")
        self.end_headers()
        self.wfile.write(self.body)

    def do_GET(self) -> None:
        self._respond()

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        self._respond()

    def log_message(self, format: str, *args: object) -> None:
        return


@pytest.fixture
def failure_provider() -> tuple[str, type[_FailureHandler]]:
    _FailureHandler.status = 503
    _FailureHandler.body = b'{"secret":"provider body"}'
    _FailureHandler.disconnect = False
    _FailureHandler.requests = 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FailureHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}/v1", _FailureHandler
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def _status(base_url: str) -> list[str]:
    return [
        "--json",
        "status",
        "openai:batch_123",
        "--base-url",
        base_url,
        "--api-key-env",
        "TEST_OPENAI_KEY",
    ]


@pytest.mark.parametrize(
    ("status", "code", "category", "exit_code"),
    (
        (401, "authentication_failed", "configuration", 3),
        (403, "authorization_failed", "configuration", 3),
        (404, "provider_job_not_found", "provider_rejection", 4),
        (422, "provider_rejected", "provider_rejection", 4),
        (503, "provider_unavailable", "provider_availability", 5),
    ),
)
def test_provider_http_failures_map_to_safe_exact_envelopes(
    failure_provider: tuple[str, type[_FailureHandler]],
    status: int,
    code: str,
    category: str,
    exit_code: int,
) -> None:
    base_url, handler = failure_provider
    handler.status = status

    result = CliRunner().invoke(
        cli,
        _status(base_url),
        env={"TEST_OPENAI_KEY": "credential-secret"},
    )

    assert result.exit_code == exit_code
    assert result.stdout == ""
    error = json.loads(result.stderr)["error"]
    assert error["code"] == code
    assert error["category"] == category
    assert error["http_status"] == status
    assert error["request_id"] == "req_safe"
    assert error["retry_after_seconds"] == 3600
    assert "provider body" not in result.stderr
    assert "secret-header" not in result.stderr
    assert "credential-secret" not in result.stderr


def test_invalid_provider_json_maps_to_protocol_failure(
    failure_provider: tuple[str, type[_FailureHandler]],
) -> None:
    base_url, handler = failure_provider
    handler.status = 200
    handler.body = b"not-json secret-provider-body"

    result = CliRunner().invoke(
        cli,
        _status(base_url),
        env={"TEST_OPENAI_KEY": "credential-secret"},
    )

    assert result.exit_code == 5
    assert result.stdout == ""
    error = json.loads(result.stderr)["error"]
    assert error["code"] == "provider_protocol_error"
    assert error["http_status"] == 200
    assert error["request_id"] == "req_safe"
    assert "secret-provider-body" not in result.stderr


def test_unknown_submission_outcome_warns_against_blind_resubmission(
    tmp_path: Path,
    failure_provider: tuple[str, type[_FailureHandler]],
) -> None:
    base_url, handler = failure_provider
    handler.disconnect = True
    source = tmp_path / "requests.jsonl"
    source.write_text('{"prompt":"private workload"}\n')

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
        env={"TEST_OPENAI_KEY": "credential-secret"},
    )

    assert result.exit_code == 5
    assert result.stdout == ""
    error = json.loads(result.stderr)["error"]
    assert error["code"] == "provider_unavailable"
    assert error["submission_outcome"] == "unknown"
    assert error["retryable"] is False
    assert error["recovery"] == {"action": "inspect_provider_account"}
    assert "Do not resubmit blindly" in error["message"]
    assert "duplicate work or cost" in error["message"]
    assert "private workload" not in result.stderr
    assert handler.requests == 1


def test_selector_failures_distinguish_invalid_and_missing_local(tmp_path: Path) -> None:
    invalid = CliRunner().invoke(cli, ["--json", "status", "bad/selector"])
    missing = CliRunner().invoke(
        cli,
        [
            "--json",
            "--registry",
            str(tmp_path / "registry.sqlite3"),
            "status",
            "missing-local",
        ],
    )

    assert invalid.exit_code == 2
    assert json.loads(invalid.stderr)["error"]["code"] == "invalid_job_selector"
    assert missing.exit_code == 8
    assert json.loads(missing.stderr)["error"]["code"] == "local_job_not_found"
    assert invalid.stdout == missing.stdout == ""


def test_root_parse_failure_is_one_machine_envelope() -> None:
    result = CliRunner().invoke(cli, ["--json", "--bogus"])

    assert result.exit_code == 2
    assert result.stdout == ""
    assert len(result.stderr.splitlines()) == 1
    error = json.loads(result.stderr)["error"]
    assert error["code"] == "usage_error"
    assert error["operation"] == "cli"
    assert "Usage:" not in result.stderr


@pytest.mark.parametrize(
    ("contents", "code"),
    (
        ("not-json\n", "input_parse_failed"),
        ("{}\n", "input_validation_failed"),
        (
            '{"custom_id":"duplicate","prompt":"one"}\n{"custom_id":"duplicate","prompt":"two"}\n',
            "duplicate_custom_id",
        ),
    ),
)
def test_input_failures_emit_specific_codes(tmp_path: Path, contents: str, code: str) -> None:
    source = tmp_path / "requests.jsonl"
    source.write_text(contents)

    result = CliRunner().invoke(
        cli,
        ["--json", "submit", "text", str(source), "--model", "openai/gpt-test"],
        env={"OPENAI_API_KEY": "test-key"},
    )

    assert result.exit_code == 2
    assert result.stdout == ""
    assert json.loads(result.stderr)["error"]["code"] == code


def test_missing_input_emits_read_failure(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        cli,
        [
            "--json",
            "submit",
            "text",
            str(tmp_path / "missing.jsonl"),
            "--model",
            "openai/gpt-test",
        ],
        env={"OPENAI_API_KEY": "test-key"},
    )

    assert result.exit_code == 2
    assert result.stdout == ""
    assert json.loads(result.stderr)["error"]["code"] == "input_read_failed"


def test_missing_selected_credential_is_configuration_failure(
    failure_provider: tuple[str, type[_FailureHandler]],
) -> None:
    base_url, handler = failure_provider

    result = CliRunner().invoke(
        cli,
        [
            "--json",
            "status",
            "openai:batch_123",
            "--base-url",
            base_url,
            "--api-key-env",
            "MISSING_PROVIDER_KEY",
        ],
    )

    assert result.exit_code == 3
    error = json.loads(result.stderr)["error"]
    assert error["code"] == "missing_environment_variable"
    assert "MISSING_PROVIDER_KEY" in error["message"]
    assert handler.requests == 0


def test_unexpected_machine_failure_is_safe_internal_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail(*_: object, **__: object) -> object:
        raise RuntimeError("private invariant detail")

    monkeypatch.setattr(commands_module, "status_job", fail)

    result = CliRunner().invoke(
        cli,
        [
            "--json",
            "status",
            "openai:batch_123",
            "--api-key-env",
            "TEST_OPENAI_KEY",
        ],
        env={"TEST_OPENAI_KEY": "credential-secret"},
    )

    assert result.exit_code == 1
    assert result.stdout == ""
    assert len(result.stderr.splitlines()) == 1
    error = json.loads(result.stderr)["error"]
    assert error["code"] == "internal_error"
    assert "private invariant detail" not in result.stderr
    assert "Traceback" not in result.stderr
