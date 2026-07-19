from __future__ import annotations

import json
import os
import signal
import socket
import sqlite3
import subprocess
import sysconfig
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar

import pytest


class _LifecycleHandler(BaseHTTPRequestHandler):
    requests: ClassVar[list[tuple[str, str]]] = []
    statuses: ClassVar[list[str]] = []
    result_documents: ClassVar[list[dict[str, object]]] = []
    result_gate: ClassVar[threading.Event | None] = None
    result_status: ClassVar[int] = 200
    snapshot_http_statuses: ClassVar[list[int]] = [200]
    break_result_stream_after_first: ClassVar[bool] = False

    def _json(self, document: object) -> None:
        encoded = json.dumps(document).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_POST(self) -> None:
        self.requests.append(("POST", self.path))
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        if self.path == "/v1/files":
            self._json({"id": "file-input"})
        elif self.path == "/v1/batches":
            self._json(
                {
                    "id": "batch_123",
                    "status": "validating",
                    "request_counts": {"total": 1, "completed": 0, "failed": 0},
                }
            )
        elif self.path == "/v1/batches/batch_123/cancel":
            self._json({})
        else:
            self.send_error(404)

    def do_GET(self) -> None:
        self.requests.append(("GET", self.path))
        if self.path == "/v1/batches/batch_123":
            response_status = (
                self.snapshot_http_statuses.pop(0)
                if len(self.snapshot_http_statuses) > 1
                else self.snapshot_http_statuses[0]
            )
            if response_status != 200:
                self.send_response(response_status)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            status = self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
            terminal = status in {"completed", "failed", "expired", "cancelled"}
            self._json(
                {
                    "id": "batch_123",
                    "status": status,
                    "request_counts": {
                        "total": 1,
                        "completed": 1 if status == "completed" else 0,
                        "failed": 1 if status == "failed" else 0,
                    },
                    **({"output_file_id": "file-output"} if terminal else {}),
                }
            )
        elif self.path == "/v1/files/file-output/content":
            if self.result_status != 200:
                self.send_response(self.result_status)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/jsonl")
            if self.break_result_stream_after_first:
                self.send_header("Content-Length", "9999")
            self.end_headers()
            for index, document in enumerate(self.result_documents):
                self.wfile.write((json.dumps(document) + "\n").encode())
                self.wfile.flush()
                if index == 0 and self.break_result_stream_after_first:
                    self.connection.shutdown(socket.SHUT_RDWR)
                    self.connection.close()
                    return
                if index == 0 and self.result_gate is not None:
                    self.result_gate.wait(timeout=2)
        else:
            self.send_error(404)

    def log_message(self, format: str, *args: object) -> None:
        return


@pytest.fixture
def lifecycle_provider() -> tuple[str, type[_LifecycleHandler]]:
    _LifecycleHandler.requests = []
    _LifecycleHandler.statuses = ["completed"]
    _LifecycleHandler.result_documents = [
        {
            "custom_id": "request-0",
            "response": {
                "status_code": 200,
                "body": {"choices": [{"message": {"content": "hello"}}]},
            },
        }
    ]
    _LifecycleHandler.result_gate = None
    _LifecycleHandler.result_status = 200
    _LifecycleHandler.snapshot_http_statuses = [200]
    _LifecycleHandler.break_result_stream_after_first = False
    server = ThreadingHTTPServer(("127.0.0.1", 0), _LifecycleHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}/v1", _LifecycleHandler
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def _batchwork() -> str:
    executable = Path(sysconfig.get_path("scripts")) / (
        "batchwork.exe" if os.name == "nt" else "batchwork"
    )
    assert executable.is_file()
    return str(executable)


def _environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["TEST_OPENAI_KEY"] = "secret"
    environment["NO_COLOR"] = "1"
    return environment


def _direct(base_url: str) -> list[str]:
    return [
        "openai:batch_123",
        "--base-url",
        base_url,
        "--api-key-env",
        "TEST_OPENAI_KEY",
    ]


@pytest.mark.parametrize(("terminal", "exit_code"), (("completed", 0), ("failed", 6)))
def test_installed_run_text_emits_identity_snapshot_and_terminal_results(
    tmp_path: Path,
    lifecycle_provider: tuple[str, type[_LifecycleHandler]],
    terminal: str,
    exit_code: int,
) -> None:
    base_url, handler = lifecycle_provider
    handler.statuses = ["in_progress", terminal, terminal]
    source = tmp_path / "requests.jsonl"
    source.write_text('{"prompt":"private"}\n')

    result = subprocess.run(
        [
            _batchwork(),
            "--jsonl",
            "--registry",
            str(tmp_path / "registry.sqlite3"),
            "run",
            "text",
            str(source),
            "--model",
            "openai/gpt-test",
            "--base-url",
            base_url,
            "--api-key-env",
            "TEST_OPENAI_KEY",
            "--poll-interval",
            ".01",
        ],
        cwd=tmp_path,
        env=_environment(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == exit_code, result.stderr
    assert [json.loads(line)["type"] for line in result.stdout.splitlines()] == [
        "job",
        "snapshot",
        "result",
    ]
    if terminal == "completed":
        assert result.stderr == ""
    else:
        error = json.loads(result.stderr)["error"]
        assert error["code"] == "terminal_partial_results"
        assert error["item_successes"] == 1
        assert error["item_failures"] == 0
    assert handler.requests == [
        ("POST", "/v1/files"),
        ("POST", "/v1/batches"),
        ("GET", "/v1/batches/batch_123"),
        ("GET", "/v1/batches/batch_123"),
        ("GET", "/v1/batches/batch_123"),
        ("GET", "/v1/files/file-output/content"),
    ]


def test_installed_results_never_waits_for_nonterminal_job(
    tmp_path: Path,
    lifecycle_provider: tuple[str, type[_LifecycleHandler]],
) -> None:
    base_url, handler = lifecycle_provider
    handler.statuses = ["in_progress"]

    result = subprocess.run(
        [_batchwork(), "--json", "results", *_direct(base_url)],
        cwd=tmp_path,
        env=_environment(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 6
    assert result.stdout == ""
    assert json.loads(result.stderr)["error"]["code"] == "results_not_ready"
    assert handler.requests == [("GET", "/v1/batches/batch_123")]


def test_status_observes_failed_terminal_state_with_success_exit(
    tmp_path: Path,
    lifecycle_provider: tuple[str, type[_LifecycleHandler]],
) -> None:
    base_url, handler = lifecycle_provider
    handler.statuses = ["failed"]

    result = subprocess.run(
        [_batchwork(), "--json", "status", *_direct(base_url)],
        cwd=tmp_path,
        env=_environment(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert json.loads(result.stdout)["snapshot"]["status"] == "failed"
    assert result.stderr == ""


def test_wait_emits_failed_terminal_snapshot_then_job_state_error(
    tmp_path: Path,
    lifecycle_provider: tuple[str, type[_LifecycleHandler]],
) -> None:
    base_url, handler = lifecycle_provider
    handler.statuses = ["failed"]

    result = subprocess.run(
        [_batchwork(), "--json", "wait", *_direct(base_url)],
        cwd=tmp_path,
        env=_environment(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 6
    assert json.loads(result.stdout)["snapshot"]["status"] == "failed"
    assert json.loads(result.stderr)["error"]["code"] == "job_failed"


def test_installed_jsonl_results_flush_each_record_as_received(
    tmp_path: Path,
    lifecycle_provider: tuple[str, type[_LifecycleHandler]],
) -> None:
    base_url, handler = lifecycle_provider
    handler.result_documents = [
        *handler.result_documents,
        {
            "custom_id": "request-1",
            "response": {"status_code": 200, "body": {"choices": []}},
        },
    ]
    handler.result_gate = threading.Event()
    process = subprocess.Popen(
        [_batchwork(), "--jsonl", "results", *_direct(base_url)],
        cwd=tmp_path,
        env=_environment(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdout is not None
    first = process.stdout.readline()
    assert json.loads(first)["result"]["custom_id"] == "request-0"
    assert process.poll() is None
    handler.result_gate.set()
    stdout, stderr = process.communicate(timeout=5)
    assert process.returncode == 0, stderr
    assert json.loads(stdout)["result"]["custom_id"] == "request-1"


def test_jsonl_result_stream_failure_preserves_complete_records_and_counts(
    tmp_path: Path,
    lifecycle_provider: tuple[str, type[_LifecycleHandler]],
) -> None:
    base_url, handler = lifecycle_provider
    handler.break_result_stream_after_first = True

    result = subprocess.run(
        [_batchwork(), "--jsonl", "results", *_direct(base_url)],
        cwd=tmp_path,
        env=_environment(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 5
    records = [json.loads(line) for line in result.stdout.splitlines()]
    assert [record["result"]["custom_id"] for record in records] == ["request-0"]
    error = json.loads(result.stderr)["error"]
    assert error["code"] == "result_stream_failed"
    assert error["partial_output"] is True
    assert error["records_emitted"] == 1
    assert error["item_successes"] == 1
    assert error["item_failures"] == 0


def test_installed_cancel_is_noop_for_terminal_job(
    tmp_path: Path,
    lifecycle_provider: tuple[str, type[_LifecycleHandler]],
) -> None:
    base_url, handler = lifecycle_provider
    handler.statuses = ["cancelled"]

    result = subprocess.run(
        [_batchwork(), "--json", "cancel", *_direct(base_url)],
        cwd=tmp_path,
        env=_environment(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["snapshot"]["status"] == "cancelled"
    assert handler.requests == [("GET", "/v1/batches/batch_123")]


def test_cancel_request_with_failed_refresh_reports_uncertain_remote_state(
    tmp_path: Path,
    lifecycle_provider: tuple[str, type[_LifecycleHandler]],
) -> None:
    base_url, handler = lifecycle_provider
    handler.statuses = ["in_progress"]
    handler.snapshot_http_statuses = [200, 503]

    result = subprocess.run(
        [_batchwork(), "--json", "cancel", *_direct(base_url)],
        cwd=tmp_path,
        env=_environment(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 5
    assert result.stdout == ""
    error = json.loads(result.stderr)["error"]
    assert error["code"] == "cancellation_refresh_failed"
    assert error["cancel_requested"] is True
    assert error["recovery"]["command"][:2] == ["batchwork", "status"]
    assert handler.requests == [
        ("GET", "/v1/batches/batch_123"),
        ("POST", "/v1/batches/batch_123/cancel"),
        ("GET", "/v1/batches/batch_123"),
    ]


def test_unsuccessful_terminal_state_precedes_secondary_result_failure(
    tmp_path: Path,
    lifecycle_provider: tuple[str, type[_LifecycleHandler]],
) -> None:
    base_url, handler = lifecycle_provider
    handler.statuses = ["failed"]
    handler.result_status = 503

    result = subprocess.run(
        [_batchwork(), "--json", "results", *_direct(base_url)],
        cwd=tmp_path,
        env=_environment(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 6
    assert result.stdout == ""
    error = json.loads(result.stderr)["error"]
    assert error["code"] == "terminal_partial_results"
    assert error["category"] == "job_state"


def test_completed_result_with_item_failure_emits_data_then_exit_six(
    tmp_path: Path,
    lifecycle_provider: tuple[str, type[_LifecycleHandler]],
) -> None:
    base_url, handler = lifecycle_provider
    handler.statuses = ["completed"]
    handler.result_documents = [
        {
            "custom_id": "request-0",
            "response": {
                "status_code": 400,
                "body": {"error": {"message": "provider item detail"}},
            },
        }
    ]

    result = subprocess.run(
        [_batchwork(), "--json", "results", *_direct(base_url)],
        cwd=tmp_path,
        env=_environment(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 6
    assert json.loads(result.stdout)["results"][0]["status"] == "errored"
    error = json.loads(result.stderr)["error"]
    assert error["code"] == "completed_with_item_failures"
    assert error["item_successes"] == 0
    assert error["item_failures"] == 1


def test_installed_status_accepts_saved_alias_and_bare_explicit_provider(
    tmp_path: Path,
    lifecycle_provider: tuple[str, type[_LifecycleHandler]],
) -> None:
    base_url, handler = lifecycle_provider
    registry = tmp_path / "registry.sqlite3"
    saved = subprocess.run(
        [
            _batchwork(),
            "--json",
            "--registry",
            str(registry),
            "status",
            *_direct(base_url),
            "--save",
            "--name",
            "bw_team",
        ],
        cwd=tmp_path,
        env=_environment(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert saved.returncode == 0, saved.stderr
    assert json.loads(saved.stdout)["job"].startswith("bw_")

    handler.requests = []
    alias = subprocess.run(
        [
            _batchwork(),
            "--json",
            "--registry",
            str(registry),
            "status",
            "bw_team",
        ],
        cwd=tmp_path,
        env=_environment(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert alias.returncode == 0, alias.stderr
    assert handler.requests == [("GET", "/v1/batches/batch_123")]

    handler.requests = []
    bare = subprocess.run(
        [
            _batchwork(),
            "--json",
            "status",
            "batch_123",
            "--provider",
            "openai",
            "--base-url",
            base_url,
            "--api-key-env",
            "TEST_OPENAI_KEY",
        ],
        cwd=tmp_path,
        env=_environment(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert bare.returncode == 0, bare.stderr
    assert handler.requests == [("GET", "/v1/batches/batch_123")]


def test_direct_lifecycle_bypasses_corrupt_registry(
    tmp_path: Path,
    lifecycle_provider: tuple[str, type[_LifecycleHandler]],
) -> None:
    base_url, handler = lifecycle_provider
    registry = tmp_path / "registry.sqlite3"
    registry.write_bytes(b"not sqlite")

    result = subprocess.run(
        [
            _batchwork(),
            "--json",
            "--registry",
            str(registry),
            "status",
            *_direct(base_url),
        ],
        cwd=tmp_path,
        env=_environment(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["snapshot"]["status"] == "completed"
    assert registry.read_bytes() == b"not sqlite"
    assert handler.requests == [("GET", "/v1/batches/batch_123")]


def test_local_lifecycle_fails_closed_on_corrupt_registry(
    tmp_path: Path,
    lifecycle_provider: tuple[str, type[_LifecycleHandler]],
) -> None:
    _, handler = lifecycle_provider
    registry = tmp_path / "registry.sqlite3"
    registry.write_bytes(b"not sqlite")

    result = subprocess.run(
        [
            _batchwork(),
            "--json",
            "--registry",
            str(registry),
            "status",
            "local-name",
        ],
        cwd=tmp_path,
        env=_environment(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 8
    assert result.stdout == ""
    assert json.loads(result.stderr)["error"]["code"] == "registry_unavailable"
    assert registry.read_bytes() == b"not sqlite"
    assert handler.requests == []


def test_adoption_alias_collision_preserves_registry_and_returns_direct_recovery(
    tmp_path: Path,
    lifecycle_provider: tuple[str, type[_LifecycleHandler]],
) -> None:
    base_url, handler = lifecycle_provider
    registry = tmp_path / "registry.sqlite3"
    saved = subprocess.run(
        [
            _batchwork(),
            "--json",
            "--registry",
            str(registry),
            "status",
            *_direct(base_url),
            "--save",
            "--name",
            "taken",
        ],
        cwd=tmp_path,
        env=_environment(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert saved.returncode == 0, saved.stderr

    environment = _environment()
    environment["OTHER_OPENAI_KEY"] = "secret"
    collision = subprocess.run(
        [
            _batchwork(),
            "--json",
            "--registry",
            str(registry),
            "status",
            "openai:batch_123",
            "--base-url",
            base_url,
            "--api-key-env",
            "OTHER_OPENAI_KEY",
            "--save",
            "--name",
            "taken",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert collision.returncode == 8
    assert collision.stdout == ""
    error = json.loads(collision.stderr)["error"]
    assert error["code"] == "registry_unavailable"
    assert error["recovery"]["command"] == [
        "batchwork",
        "--registry",
        str(registry),
        "status",
        "openai:batch_123",
        "--api-key-env",
        "OTHER_OPENAI_KEY",
        "--base-url",
        base_url,
        "--save",
    ]
    with sqlite3.connect(registry) as connection:
        assert connection.execute("SELECT name FROM jobs").fetchall() == [("taken",)]
    assert handler.requests == [
        ("GET", "/v1/batches/batch_123"),
        ("GET", "/v1/batches/batch_123"),
    ]


def test_local_selector_explicit_profile_is_fingerprint_checked_before_network(
    tmp_path: Path,
    lifecycle_provider: tuple[str, type[_LifecycleHandler]],
) -> None:
    base_url, handler = lifecycle_provider
    registry = tmp_path / "registry.sqlite3"
    saved = subprocess.run(
        [
            _batchwork(),
            "--registry",
            str(registry),
            "status",
            *_direct(base_url),
            "--save",
            "--name",
            "local-job",
        ],
        cwd=tmp_path,
        env=_environment(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert saved.returncode == 0, saved.stderr
    config = tmp_path / "config.toml"
    config.write_text(
        """\
schema_version = 1
[profiles.other.providers.openai]
api_key_env = "OTHER_KEY"
base_url = "https://other.example/v1"
"""
    )
    handler.requests = []

    result = subprocess.run(
        [
            _batchwork(),
            "--config",
            str(config),
            "--registry",
            str(registry),
            "--profile",
            "other",
            "status",
            "local-job",
        ],
        cwd=tmp_path,
        env=_environment(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 3
    assert "routing fingerprint" in result.stderr
    assert "openai:batch_123" in result.stderr
    assert "--save" in result.stderr
    assert handler.requests == []


def test_matching_explicit_profile_updates_only_label_after_success(
    tmp_path: Path,
    lifecycle_provider: tuple[str, type[_LifecycleHandler]],
) -> None:
    base_url, _ = lifecycle_provider
    registry = tmp_path / "registry.sqlite3"
    saved = subprocess.run(
        [_batchwork(), "--registry", str(registry), "status", *_direct(base_url), "--save"],
        cwd=tmp_path,
        env=_environment(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert saved.returncode == 0, saved.stderr
    record_id = json.loads(saved.stdout)["job"]
    config = tmp_path / "config.toml"
    config.write_text(
        f"""\
schema_version = 1
[profiles.match.providers.openai]
api_key_env = "TEST_OPENAI_KEY"
base_url = "{base_url}"
"""
    )

    result = subprocess.run(
        [
            _batchwork(),
            "--config",
            str(config),
            "--registry",
            str(registry),
            "--profile",
            "match",
            "status",
            record_id,
        ],
        cwd=tmp_path,
        env=_environment(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    with sqlite3.connect(registry) as connection:
        profile, fingerprint = connection.execute(
            "SELECT profile, routing_fingerprint FROM jobs WHERE record_id = ?", (record_id,)
        ).fetchone()
    assert profile == "match"
    assert fingerprint


def test_installed_run_timeout_preserves_resumable_job_without_cancelling(
    tmp_path: Path,
    lifecycle_provider: tuple[str, type[_LifecycleHandler]],
) -> None:
    base_url, handler = lifecycle_provider
    handler.statuses = ["in_progress"]
    source = tmp_path / "requests.jsonl"
    source.write_text('{"prompt":"private"}\n')
    registry = tmp_path / "registry.sqlite3"

    result = subprocess.run(
        [
            _batchwork(),
            "--jsonl",
            "--registry",
            str(registry),
            "run",
            "text",
            str(source),
            "--model",
            "openai/gpt-test",
            "--base-url",
            base_url,
            "--api-key-env",
            "TEST_OPENAI_KEY",
            "--poll-interval",
            ".01",
            "--timeout",
            ".03s",
        ],
        cwd=tmp_path,
        env=_environment(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 7
    job = json.loads(result.stdout.splitlines()[0])["job"]
    assert json.loads(result.stderr)["error"]["code"] == "wait_timeout"
    assert all(path != "/v1/batches/batch_123/cancel" for _, path in handler.requests)

    handler.requests = []
    handler.statuses = ["completed"]
    resumed = subprocess.run(
        [
            _batchwork(),
            "--json",
            "--registry",
            str(registry),
            "status",
            job["record_id"],
        ],
        cwd=tmp_path,
        env=_environment(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert resumed.returncode == 0, resumed.stderr
    assert json.loads(resumed.stdout)["snapshot"]["status"] == "completed"
    assert handler.requests == [("GET", "/v1/batches/batch_123")]


@pytest.mark.skipif(os.name == "nt", reason="POSIX signal exit contract")
@pytest.mark.parametrize(
    ("process_signal", "exit_code", "error_code"),
    ((signal.SIGINT, 130, "interrupted"), (signal.SIGTERM, 143, "terminated")),
)
def test_installed_wait_signal_preserves_remote_job(
    tmp_path: Path,
    lifecycle_provider: tuple[str, type[_LifecycleHandler]],
    process_signal: signal.Signals,
    exit_code: int,
    error_code: str,
) -> None:
    base_url, handler = lifecycle_provider
    handler.statuses = ["in_progress"]
    process = subprocess.Popen(
        [_batchwork(), "--json", "wait", *_direct(base_url)],
        cwd=tmp_path,
        env=_environment(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.monotonic() + 5
    while not handler.requests and time.monotonic() < deadline:
        time.sleep(0.01)
    assert handler.requests == [("GET", "/v1/batches/batch_123")]

    process.send_signal(process_signal)
    stdout, stderr = process.communicate(timeout=5)

    assert process.returncode == exit_code
    assert stdout == ""
    assert json.loads(stderr)["error"]["code"] == error_code
    assert all(path != "/v1/batches/batch_123/cancel" for _, path in handler.requests)
