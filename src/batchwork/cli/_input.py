"""Canonical CLI input transports."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from io import StringIO
from pathlib import Path
from typing import BinaryIO, Generic, TypeVar

import click
from pydantic import ValidationError

from batchwork._limits import MAX_RECORD_BYTES, MAX_REQUESTS, MAX_SOURCE_BYTES
from batchwork.types import BatchEmbeddingRequest, BatchRequest

from ._contract import KnownErrorCode
from ._failures import CliUsageError

_READ_CHUNK_BYTES = 64 * 1024


class InputFormat(StrEnum):
    CSV = "csv"
    JSON = "json"
    JSONL = "jsonl"
    TEXT = "text"


INPUT_FORMATS = tuple(input_format.value for input_format in InputFormat)
_EXTENSION_FORMATS = {
    **{
        f".{input_format.value}": input_format
        for input_format in InputFormat
        if input_format is not InputFormat.TEXT
    },
    ".txt": InputFormat.TEXT,
}


@dataclass(frozen=True, slots=True)
class _CsvField:
    convert: Callable[[str], object]
    expected: str


def _tool_choice(value: str) -> str:
    if value not in {"auto", "none", "required"}:
        raise ValueError
    return value


def _finite_float(value: str) -> float:
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError
    return converted


_CSV_FIELDS = {
    "custom_id": _CsvField(str, "text"),
    "frequency_penalty": _CsvField(_finite_float, "finite number"),
    "max_output_tokens": _CsvField(int, "integer"),
    "presence_penalty": _CsvField(_finite_float, "finite number"),
    "prompt": _CsvField(str, "non-empty text"),
    "seed": _CsvField(int, "integer"),
    "system": _CsvField(str, "text"),
    "temperature": _CsvField(_finite_float, "finite number"),
    "tool_choice": _CsvField(_tool_choice, "auto, none, or required"),
    "top_k": _CsvField(int, "integer"),
    "top_p": _CsvField(_finite_float, "finite number"),
}

_RequestT = TypeVar("_RequestT", BatchRequest, BatchEmbeddingRequest)


@dataclass(frozen=True, slots=True)
class _ParsedRequest(Generic[_RequestT]):
    request: _RequestT
    coordinate: str


def _usage(message: str, *, code: KnownErrorCode = "input_validation_failed") -> click.UsageError:
    return CliUsageError(message, code=code)


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON number {value}")


def _reject_non_finite_numbers(value: object) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite JSON number")
    if isinstance(value, list):
        for item in value:
            _reject_non_finite_numbers(item)
    elif isinstance(value, dict):
        for item in value.values():
            _reject_non_finite_numbers(item)


def _input_format(source: Path, explicit: str | None) -> InputFormat:
    if explicit is not None:
        try:
            return InputFormat(explicit)
        except ValueError as error:
            raise _usage(f'Unsupported input format: "{explicit}".', code="usage_error") from error
    if source == Path("-"):
        raise _usage("stdin requires --format; input content is never sniffed.", code="usage_error")
    inferred = _EXTENSION_FORMATS.get(source.suffix.lower())
    if inferred is None:
        raise _usage(
            f'SOURCE "{source}" has an unknown extension and requires --format.',
            code="usage_error",
        )
    return inferred


def _read_source(source: Path, stdin: BinaryIO | None, input_format: InputFormat) -> str:
    if source == Path("-"):
        if stdin is None:
            raise _usage("stdin is unavailable.", code="input_read_failed")
        stream = stdin
        label = "stdin"
        close = False
    else:
        if not source.exists():
            raise _usage(f'SOURCE does not exist: "{source}".', code="input_read_failed")
        if not source.is_file():
            raise _usage(f'SOURCE must be a regular file: "{source}".', code="input_read_failed")
        try:
            stream = source.open("rb")
        except OSError as error:
            raise _usage(
                f'Could not read SOURCE "{source}": {error}.', code="input_read_failed"
            ) from error
        label = f'SOURCE "{source}"'
        close = True
    encoded = bytearray()
    record_bytes = 0
    record_number = 1
    previous_cr = False
    csv_quoted = False
    csv_pending_quote = False
    try:
        while True:
            remaining_source = MAX_SOURCE_BYTES - len(encoded) + 1
            remaining_record = (
                remaining_source
                if input_format is InputFormat.JSON
                else MAX_RECORD_BYTES - record_bytes + 1
            )
            chunk = stream.read(min(_READ_CHUNK_BYTES, remaining_source, remaining_record))
            if not chunk:
                break
            for byte in chunk:
                encoded.append(byte)
                if len(encoded) > MAX_SOURCE_BYTES:
                    raise _usage(
                        f"{label} exceeds the {MAX_SOURCE_BYTES} byte transport limit.",
                        code="hard_limit_exceeded",
                    )
                if input_format is InputFormat.JSON:
                    continue
                record_bytes += 1
                if input_format is InputFormat.CSV:
                    if byte == ord('"'):
                        if csv_quoted and csv_pending_quote:
                            csv_pending_quote = False
                        elif csv_quoted:
                            csv_pending_quote = True
                        else:
                            csv_quoted = True
                    elif csv_pending_quote:
                        csv_quoted = False
                        csv_pending_quote = False
                    if byte in {ord("\n"), ord("\r")} and not csv_quoted:
                        record_bytes = 0
                        if not (byte == ord("\n") and previous_cr):
                            record_number += 1
                elif byte in {ord("\n"), ord("\r")}:
                    record_bytes = 0
                    if not (byte == ord("\n") and previous_cr):
                        record_number += 1
                if record_bytes > MAX_RECORD_BYTES:
                    unit = "record" if input_format is InputFormat.CSV else "line"
                    raise _usage(
                        f"{label} {unit} {record_number} exceeds the "
                        f"{MAX_RECORD_BYTES} byte limit.",
                        code="hard_limit_exceeded",
                    )
                previous_cr = byte == ord("\r")
    except OSError as error:
        raise _usage(f"Could not read {label}: {error}.", code="input_read_failed") from error
    finally:
        if close:
            stream.close()
    try:
        return bytes(encoded).decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise _usage(
            f"Could not read {label}: input must be UTF-8 with an optional BOM.",
            code="input_read_failed",
        ) from error


def _conflicting_alias(
    document: object, model: type[BatchRequest] | type[BatchEmbeddingRequest] = BatchRequest
) -> tuple[str, str] | None:
    if not isinstance(document, Mapping):
        return None
    for field_name, field in model.model_fields.items():
        alias = field.alias
        if alias is not None and alias != field_name:
            if field_name in document and alias in document:
                return field_name, alias
    return None


def _validate_request(
    document: object,
    coordinate: str,
    model: type[_RequestT],
    *,
    label: str,
) -> _ParsedRequest[_RequestT]:
    if not isinstance(document, dict):
        raise _usage(f"Invalid {label} at {coordinate}: expected a JSON object.")
    conflict = _conflicting_alias(document, model)
    if conflict is not None:
        field_name, alias = conflict
        raise _usage(
            f'Invalid {label} at {coordinate}: "{field_name}" and "{alias}" cannot both be present.'
        )
    try:
        request = model.model_validate(document)
    except ValidationError as error:
        detail = error.errors(include_url=False)[0]["msg"]
        raise _usage(f"Invalid {label} at {coordinate}: {detail}.") from error
    return _ParsedRequest(request, coordinate)


def _validate_text_request(document: object, coordinate: str) -> _ParsedRequest[BatchRequest]:
    return _validate_request(document, coordinate, BatchRequest, label="request")


def _decode_json(contents: str, error_prefix: str) -> object:
    try:
        document = json.loads(contents, parse_constant=_reject_json_constant)
        _reject_non_finite_numbers(document)
    except json.JSONDecodeError as error:
        raise _usage(f"{error_prefix}: {error.msg}.", code="input_parse_failed") from error
    except ValueError as error:
        raise _usage(f"{error_prefix}: {error}.", code="input_parse_failed") from error
    return document


def _requests_from_json(
    contents: str,
    validate: Callable[[object, str], _ParsedRequest[_RequestT]],
    *,
    empty_item: str,
) -> list[_ParsedRequest[_RequestT]]:
    document = _decode_json(contents, "SOURCE is not valid JSON")
    if isinstance(document, list):
        if not document:
            raise _usage(f"SOURCE must contain at least one {empty_item}.")
        if len(document) > MAX_REQUESTS:
            raise _usage(
                f"SOURCE exceeds the {MAX_REQUESTS} request limit.",
                code="hard_limit_exceeded",
            )
        return [validate(item, f"JSON index {index}") for index, item in enumerate(document)]
    return [validate(document, "JSON object")]


def _json_requests(contents: str) -> list[_ParsedRequest[BatchRequest]]:
    return _requests_from_json(
        contents,
        _validate_text_request,
        empty_item="JSON request object",
    )


def _requests_from_jsonl(
    contents: str,
    validate: Callable[[object, str], _ParsedRequest[_RequestT]],
) -> list[_ParsedRequest[_RequestT]]:
    parsed: list[_ParsedRequest[_RequestT]] = []
    for line_number, line in enumerate(_lines(contents), start=1):
        if not line.strip():
            continue
        _ensure_request_slot(parsed, f"JSONL line {line_number}")
        document = _decode_json(line, f"Invalid JSONL object at line {line_number}")
        parsed.append(validate(document, f"JSONL line {line_number}"))
    if not parsed:
        raise _usage("SOURCE must contain at least one non-empty JSONL object.")
    return parsed


def _jsonl_requests(contents: str) -> list[_ParsedRequest[BatchRequest]]:
    return _requests_from_jsonl(contents, _validate_text_request)


def _csv_requests(contents: str) -> list[_ParsedRequest[BatchRequest]]:
    reader = csv.reader(StringIO(contents, newline=""), strict=True)
    try:
        header = next(reader)
    except StopIteration as error:
        raise _usage("SOURCE must contain a CSV header row.") from error
    except csv.Error as error:
        raise _usage(f"Invalid CSV header at row 1: {error}.", code="input_parse_failed") from error
    seen: set[str] = set()
    for column, name in enumerate(header, start=1):
        if name in seen:
            raise _usage(f'Duplicate CSV header "{name}" at row 1, column {column}.')
        seen.add(name)
        if name not in _CSV_FIELDS:
            raise _usage(f'Unknown CSV header "{name}" at row 1, column {column}.')
    if "prompt" not in seen:
        raise _usage('CSV header at row 1, column 1 must include the canonical "prompt" field.')

    parsed: list[_ParsedRequest[BatchRequest]] = []
    try:
        for row_number, row in enumerate(reader, start=2):
            _ensure_request_slot(parsed, f"CSV row {row_number}")
            if len(row) != len(header):
                column = header[len(row)] if len(row) < len(header) else str(len(header) + 1)
                raise _usage(
                    f"Invalid CSV record at row {row_number}, column {column}: "
                    f"expected {len(header)} columns, received {len(row)}."
                )
            document: dict[str, object] = {}
            for name, value in zip(header, row, strict=True):
                if value == "" and name != "prompt":
                    continue
                field = _CSV_FIELDS[name]
                try:
                    if name == "prompt" and value == "":
                        raise ValueError
                    document[name] = field.convert(value)
                except ValueError as error:
                    raise _usage(
                        f"Invalid CSV record at row {row_number}, column {name}: "
                        f"expected {field.expected}."
                    ) from error
            try:
                request = BatchRequest.model_validate(document)
            except ValidationError as error:
                detail = error.errors(include_url=False)[0]
                location = detail["loc"]
                column = location[0] if location and isinstance(location[0], str) else "prompt"
                raise _usage(
                    f"Invalid CSV record at row {row_number}, column {column}: {detail['msg']}."
                ) from error
            parsed.append(_ParsedRequest(request, f"CSV row {row_number}"))
    except csv.Error as error:
        raise _usage(
            f"Invalid CSV record at row {reader.line_num}: {error}.", code="input_parse_failed"
        ) from error
    if not parsed:
        raise _usage("SOURCE must contain at least one CSV request row.")
    return parsed


def _requests_from_text(
    contents: str,
    create: Callable[[str], _RequestT],
) -> list[_ParsedRequest[_RequestT]]:
    parsed: list[_ParsedRequest[_RequestT]] = []
    for line_number, line in enumerate(_lines(contents), start=1):
        if not line.strip():
            continue
        _ensure_request_slot(parsed, f"text line {line_number}")
        parsed.append(_ParsedRequest(create(line), f"text line {line_number}"))
    if not parsed:
        raise _usage("SOURCE must contain at least one non-whitespace text line.")
    return parsed


def _text_requests(contents: str) -> list[_ParsedRequest[BatchRequest]]:
    return _requests_from_text(contents, lambda line: BatchRequest(prompt=line))


def _ensure_request_slot(parsed: Sequence[object], coordinate: str) -> None:
    if len(parsed) == MAX_REQUESTS:
        raise _usage(
            f"SOURCE exceeds the {MAX_REQUESTS} request limit at {coordinate}.",
            code="hard_limit_exceeded",
        )


def _normalize_ids(parsed: Sequence[_ParsedRequest[_RequestT]]) -> list[_RequestT]:
    seen: dict[str, str] = {}
    normalized: list[_RequestT] = []
    for index, item in enumerate(parsed):
        custom_id = (
            item.request.custom_id if item.request.custom_id is not None else f"request-{index}"
        )
        previous = seen.get(custom_id)
        if previous is not None:
            raise _usage(
                f'Invalid custom_id "{custom_id}" at {item.coordinate}: '
                f"already used at {previous}.",
                code="duplicate_custom_id",
            )
        seen[custom_id] = item.coordinate
        normalized.append(item.request.model_copy(update={"custom_id": custom_id}))
    return normalized


def _lines(contents: str) -> list[str]:
    return contents.replace("\r\n", "\n").replace("\r", "\n").split("\n")


def _validate_embedding_request(
    document: object, coordinate: str
) -> _ParsedRequest[BatchEmbeddingRequest]:
    return _validate_request(
        document,
        coordinate,
        BatchEmbeddingRequest,
        label="embedding request",
    )


def _embedding_json_requests(contents: str) -> list[_ParsedRequest[BatchEmbeddingRequest]]:
    return _requests_from_json(
        contents,
        _validate_embedding_request,
        empty_item="JSON embedding request object",
    )


def _embedding_jsonl_requests(contents: str) -> list[_ParsedRequest[BatchEmbeddingRequest]]:
    return _requests_from_jsonl(contents, _validate_embedding_request)


def _embedding_csv_requests(contents: str) -> list[_ParsedRequest[BatchEmbeddingRequest]]:
    reader = csv.reader(StringIO(contents, newline=""), strict=True)
    try:
        header = next(reader)
    except StopIteration as error:
        raise _usage("SOURCE must contain a CSV header row.") from error
    except csv.Error as error:
        raise _usage(f"Invalid CSV header at row 1: {error}.", code="input_parse_failed") from error
    allowed = {"custom_id", "dimensions", "value"}
    seen: set[str] = set()
    for column, name in enumerate(header, start=1):
        if name in seen:
            raise _usage(f'Duplicate CSV header "{name}" at row 1, column {column}.')
        seen.add(name)
        if name not in allowed:
            raise _usage(f'Unknown CSV header "{name}" at row 1, column {column}.')
    if "value" not in seen:
        raise _usage('CSV header at row 1, column 1 must include the canonical "value" field.')
    parsed: list[_ParsedRequest[BatchEmbeddingRequest]] = []
    try:
        for row_number, row in enumerate(reader, start=2):
            _ensure_request_slot(parsed, f"CSV row {row_number}")
            if len(row) != len(header):
                column = header[len(row)] if len(row) < len(header) else str(len(header) + 1)
                raise _usage(
                    f"Invalid CSV record at row {row_number}, column {column}: "
                    f"expected {len(header)} columns, received {len(row)}."
                )
            document: dict[str, object] = {}
            for name, value in zip(header, row, strict=True):
                if value == "" and name != "value":
                    continue
                if name == "dimensions":
                    try:
                        document[name] = int(value)
                    except ValueError as error:
                        raise _usage(
                            f"Invalid CSV record at row {row_number}, column dimensions: "
                            "expected integer."
                        ) from error
                else:
                    document[name] = value
            parsed.append(_validate_embedding_request(document, f"CSV row {row_number}"))
    except csv.Error as error:
        raise _usage(
            f"Invalid CSV record at row {reader.line_num}: {error}.", code="input_parse_failed"
        ) from error
    if not parsed:
        raise _usage("SOURCE must contain at least one CSV request row.")
    return parsed


def _embedding_text_requests(contents: str) -> list[_ParsedRequest[BatchEmbeddingRequest]]:
    return _requests_from_text(contents, lambda line: BatchEmbeddingRequest(value=line))


def load_embedding_requests(
    source: Path,
    input_format: str | None,
    *,
    stdin: BinaryIO | None = None,
) -> list[BatchEmbeddingRequest]:
    """Read, validate, and identify one canonical embedding-request source."""
    selected = _input_format(source, input_format)
    contents = _read_source(source, stdin, selected)
    parsers = {
        InputFormat.CSV: _embedding_csv_requests,
        InputFormat.JSON: _embedding_json_requests,
        InputFormat.JSONL: _embedding_jsonl_requests,
        InputFormat.TEXT: _embedding_text_requests,
    }
    return _normalize_ids(parsers[selected](contents))


def load_text_requests(
    source: Path,
    input_format: str | None,
    *,
    stdin: BinaryIO | None = None,
) -> list[BatchRequest]:
    """Read, validate, and identify one canonical text-request source."""
    selected = _input_format(source, input_format)
    contents = _read_source(source, stdin, selected)
    parsers = {
        InputFormat.CSV: _csv_requests,
        InputFormat.JSON: _json_requests,
        InputFormat.JSONL: _jsonl_requests,
        InputFormat.TEXT: _text_requests,
    }
    return _normalize_ids(parsers[selected](contents))
