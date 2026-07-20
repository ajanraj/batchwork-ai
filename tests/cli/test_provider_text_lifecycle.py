from __future__ import annotations

import json
import os
import subprocess
import sysconfig
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar

import pytest


@dataclass(frozen=True)
class _ProviderCase:
    provider: str
    model: str
    base_path: str
    supports_submission: bool = True


_CASES = (
    _ProviderCase("anthropic", "claude-test", ""),
    _ProviderCase("xai", "grok-test", "/v1"),
    _ProviderCase("google", "gemini-test", "/v1beta"),
    _ProviderCase("groq", "llama-test", "/v1"),
    _ProviderCase("mistral", "mistral-test", "/v1"),
    # Together submission requires a presigned HTTPS upload to a public host; a local
    # HTTP fake must be rejected by its SSRF guard. Direct lifecycle calls share the
    # OpenAI-compatible contract and remain covered here.
    _ProviderCase("together", "model-test", "/v1", supports_submission=False),
)


def _installed_batchwork() -> str:
    executable = Path(sysconfig.get_path("scripts")) / (
        "batchwork.exe" if os.name == "nt" else "batchwork"
    )
    assert executable.is_file(), "batchwork console script is not installed"
    return str(executable)


def _environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["TEST_PROVIDER_KEY"] = "secret"
    environment["NO_COLOR"] = "1"
    return environment


def _generic_snapshot(status: str) -> dict[str, object]:
    return {
        "id": "batch_123",
        "status": status,
        "request_counts": {"total": 1, "completed": status == "completed", "failed": 0},
        **({"output_file_id": "file-output"} if status == "completed" else {}),
    }


def _result_file() -> bytes:
    return (
        json.dumps(
            {
                "custom_id": "request-0",
                "response": {
                    "status_code": 200,
                    "body": {"choices": [{"message": {"content": "hello"}}]},
                },
            }
        )
        + "\n"
    ).encode()


def _xai_results() -> dict[str, object]:
    return {
        "results": [
            {
                "batch_request_id": "request-0",
                "batch_result": {
                    "response": {
                        "chat_completion": {"choices": [{"message": {"content": "hello"}}]}
                    }
                },
            }
        ]
    }


def _anthropic_results() -> bytes:
    return (
        json.dumps(
            {
                "custom_id": "request-0",
                "result": {
                    "type": "succeeded",
                    "message": {"content": [{"type": "text", "text": "hello"}]},
                },
            }
        )
        + "\n"
    ).encode()


def _google_snapshot(*, done: bool) -> dict[str, object]:
    return {
        "name": "batches/batch_123",
        "done": done,
        "metadata": {
            "batchStats": {
                "requestCount": 1,
                "successfulRequestCount": 1 if done else 0,
                "pendingRequestCount": 0 if done else 1,
            }
        },
        **(
            {
                "response": {
                    "inlinedResponses": [
                        {
                            "metadata": {"key": "request-0"},
                            "response": {
                                "candidates": [{"content": {"parts": [{"text": "hello"}]}}]
                            },
                        }
                    ]
                }
            }
            if done
            else {}
        ),
    }


def _mistral_snapshot(*, completed: bool) -> dict[str, object]:
    return {
        "id": "batch_123",
        "status": "SUCCESS" if completed else "QUEUED",
        "succeeded_requests": 1 if completed else 0,
        "failed_requests": 0,
        "total_requests": 1,
        **({"output_file": "file-output"} if completed else {}),
    }


@contextmanager
def _provider_server(case: _ProviderCase) -> Iterator[tuple[str, type[BaseHTTPRequestHandler]]]:
    class Handler(BaseHTTPRequestHandler):
        requests: ClassVar[list[tuple[str, str]]] = []
        cancel_pending: ClassVar[bool] = False
        cancelled: ClassVar[bool] = False

        def _json(self, document: object, status: int = 200) -> None:
            encoded = json.dumps(document).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def _jsonl(self, content: bytes) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "application/jsonl")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def _completed(self) -> bool:
            return not self.cancel_pending or self.cancelled

        def _snapshot(self) -> dict[str, object]:
            completed = self._completed()
            if case.provider == "anthropic":
                return {
                    "id": "batch_123",
                    "processing_status": "ended" if completed else "in_progress",
                    "request_counts": {
                        "succeeded": 1 if completed else 0,
                        "processing": 0 if completed else 1,
                    },
                    **(
                        {
                            "results_url": (
                                f"{self.server.base_url}/v1/messages/batches/batch_123/results"
                            )
                        }
                        if completed
                        else {}
                    ),
                }
            if case.provider == "google":
                return _google_snapshot(done=completed)
            if case.provider == "mistral":
                return _mistral_snapshot(completed=completed)
            if case.provider == "xai":
                return {
                    "batch_id": "batch_123",
                    "state": {
                        "num_requests": 1,
                        "num_pending": 0 if completed else 1,
                        "num_success": 1 if completed else 0,
                        "num_error": 0,
                        "num_cancelled": 0,
                    },
                }
            return _generic_snapshot("completed" if completed else "in_progress")

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            self.rfile.read(length)
            self.requests.append(("POST", self.path))
            path = self.path.split("?", 1)[0]
            if path.endswith("/cancel") or path.endswith(":cancel"):
                type(self).cancelled = True
                self._json({})
            elif case.provider == "anthropic" and path == "/v1/messages/batches":
                self._json(self._snapshot())
            elif case.provider == "google" and path.startswith("/v1beta/models/"):
                self._json(self._snapshot())
            elif case.provider == "mistral" and path == "/v1/files":
                self._json({"id": "file-input"})
            elif case.provider == "mistral" and path == "/v1/batch/jobs":
                self._json(self._snapshot())
            elif case.provider == "xai" and path == "/v1/files":
                self._json({"id": "file-input"})
            elif case.provider == "xai" and path == "/v1/batches":
                self._json(self._snapshot())
            elif case.provider in {"groq", "together"} and path == "/v1/files":
                self._json({"id": "file-input"})
            elif case.provider in {"groq", "together"} and path == "/v1/batches":
                self._json(self._snapshot())
            else:
                self.send_error(404)

        def do_GET(self) -> None:
            self.requests.append(("GET", self.path))
            path = self.path.split("?", 1)[0]
            if case.provider == "anthropic":
                if path == "/v1/messages/batches/batch_123":
                    self._json(self._snapshot())
                elif path == "/v1/messages/batches/batch_123/results":
                    self._jsonl(_anthropic_results())
                else:
                    self.send_error(404)
                return
            if case.provider == "google" and path == "/v1beta/batches/batch_123":
                self._json(self._snapshot())
                return
            if case.provider == "mistral" and path == "/v1/batch/jobs/batch_123":
                self._json(self._snapshot())
                return
            if case.provider == "xai":
                if path == "/v1/batches/batch_123":
                    self._json(self._snapshot())
                elif path == "/v1/batches/batch_123/results":
                    self._json(_xai_results())
                else:
                    self.send_error(404)
                return
            if path == "/v1/batches/batch_123":
                self._json(self._snapshot())
            elif path == "/v1/files/file-output/content":
                self._jsonl(_result_file())
            else:
                self.send_error(404)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    host, port = server.server_address
    server.base_url = f"http://{host}:{port}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"{server.base_url}{case.base_path}", Handler
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def _run(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            _installed_batchwork(),
            "--json",
            "--registry",
            str(tmp_path / f"{args[0]}.sqlite3"),
            *args,
        ],
        cwd=tmp_path,
        env=_environment(),
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.provider)
def test_installed_text_lifecycle_for_each_provider(
    tmp_path: Path,
    case: _ProviderCase,
) -> None:
    source = tmp_path / "requests.jsonl"
    source.write_text('{"prompt":"hello"}\n')
    with _provider_server(case) as (base_url, handler):
        provider_job_id = "batches/batch_123" if case.provider == "google" else "batch_123"
        direct = [
            f"{case.provider}:{provider_job_id}",
            "--base-url",
            base_url,
            "--api-key-env",
            "TEST_PROVIDER_KEY",
        ]

        if case.supports_submission:
            submitted = _run(
                tmp_path,
                "submit",
                "text",
                str(source),
                "--model",
                f"{case.provider}/{case.model}",
                "--base-url",
                base_url,
                "--api-key-env",
                "TEST_PROVIDER_KEY",
            )
            assert submitted.returncode == 0, submitted.stderr
            assert json.loads(submitted.stdout)["job"]["provider_reference"] == direct[0]

            ran = _run(
                tmp_path,
                "run",
                "text",
                str(source),
                "--model",
                f"{case.provider}/{case.model}",
                "--base-url",
                base_url,
                "--api-key-env",
                "TEST_PROVIDER_KEY",
                "--poll-interval",
                ".01",
            )
            assert ran.returncode == 0, ran.stderr
            assert json.loads(ran.stdout)["results"][0]["text"] == "hello"

        status = _run(tmp_path, "status", *direct)
        assert status.returncode == 0, status.stderr
        assert json.loads(status.stdout)["snapshot"]["status"] == "completed"

        waited = _run(tmp_path, "wait", *direct, "--poll-interval", ".01")
        assert waited.returncode == 0, waited.stderr
        assert json.loads(waited.stdout)["snapshot"]["status"] == "completed"

        results = _run(tmp_path, "results", *direct)
        assert results.returncode == 0, results.stderr
        assert json.loads(results.stdout)["results"][0]["text"] == "hello"

        handler.requests = []
        handler.cancel_pending = True
        handler.cancelled = False
        cancelled = _run(tmp_path, "cancel", *direct)
        assert cancelled.returncode == 0, cancelled.stderr
        assert json.loads(cancelled.stdout)["snapshot"]["status"] == "completed"
        assert any(
            method == "POST" and (path.endswith("/cancel") or path.endswith(":cancel"))
            for method, path in handler.requests
        )
