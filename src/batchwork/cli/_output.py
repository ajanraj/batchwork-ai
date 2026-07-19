"""Transactional machine-output helpers."""

from __future__ import annotations

import os
import sys
import tempfile

from pydantic import TypeAdapter

from batchwork.types import BatchResult

from ._contract import ResultEnvelope, ResultListEnvelope, RunEnvelope, serialize_envelope

_RESULTS_MARKER = '"results":[]'
_RESULT_ADAPTER = TypeAdapter(BatchResult)


class JsonResultSpool:
    """Build a result-bearing JSON envelope on private disk before publishing it."""

    def __init__(self, envelope: ResultListEnvelope | RunEnvelope) -> None:
        directory = os.environ.get("TMPDIR")
        self._file = tempfile.TemporaryFile(
            mode="w+",
            encoding="utf-8",
            newline="",
            prefix="batchwork-output-",
            dir=directory,
        )
        document = serialize_envelope(envelope)
        before, marker, after = document.rpartition(_RESULTS_MARKER)
        if not marker:
            self._file.close()
            raise RuntimeError("batchwork: buffered envelope has no results field")
        self._prefix = before + '"results":['
        self._suffix = "]" + after
        self._first = True
        self._file.write(self._prefix)

    def append(self, item: BatchResult) -> None:
        ResultEnvelope(job="buffered-output", result=item)
        if not self._first:
            self._file.write(",")
        encoded = _RESULT_ADAPTER.dump_json(item, exclude_none=True).decode("utf-8")
        self._file.write(encoded)
        self._first = False

    def reset(self) -> None:
        self._file.seek(0)
        self._file.truncate()
        self._file.write(self._prefix)
        self._first = True

    def publish(self) -> None:
        self._file.write(self._suffix)
        self._file.flush()
        self._file.seek(0)
        while chunk := self._file.read(64 * 1024):
            sys.stdout.write(chunk)
        sys.stdout.flush()

    def close(self) -> None:
        self._file.close()
