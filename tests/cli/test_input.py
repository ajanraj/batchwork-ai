from __future__ import annotations

from io import BytesIO
from pathlib import Path

import click
import pytest

import batchwork.cli._input as input_module
from batchwork.cli._input import load_embedding_requests, load_text_requests


def _write_source(tmp_path: Path, name: str, contents: bytes) -> Path:
    source = tmp_path / name
    source.write_bytes(contents)
    return source


@pytest.mark.parametrize(
    ("name", "contents", "expected_prompts"),
    (
        ("requests.json", b'{"prompt":"one"}', ["one"]),
        ("requests.json", b'[{"prompt":"one"},{"prompt":"two"}]', ["one", "two"]),
        ("requests.jsonl", b'\n{"prompt":"one"}\n\n{"prompt":"two"}\n', ["one", "two"]),
        ("requests.jsonl", b'{"prompt":"one"}\r{"prompt":"two"}\r', ["one", "two"]),
        ("requests.txt", b"  one  \n\t\n two\t\r\n", ["  one  ", " two\t"]),
        ("requests.txt", b"one\rtwo\r", ["one", "two"]),
    ),
)
def test_load_text_requests_accepts_inferred_transports(
    tmp_path: Path,
    name: str,
    contents: bytes,
    expected_prompts: list[str],
) -> None:
    requests = load_text_requests(_write_source(tmp_path, name, contents), None)

    assert [request.prompt for request in requests] == expected_prompts
    assert [request.custom_id for request in requests] == [
        f"request-{index}" for index in range(len(requests))
    ]


def test_load_text_requests_accepts_utf8_bom_and_explicit_stdin_format() -> None:
    requests = load_text_requests(
        Path("-"),
        "json",
        stdin=BytesIO(b'\xef\xbb\xbf{"prompt":"hello"}'),
    )

    assert requests[0].prompt == "hello"


@pytest.mark.parametrize(
    ("name", "contents", "values"),
    (
        ("embeddings.json", b'{"value":"one","dimensions":32}', ["one"]),
        ("embeddings.jsonl", b'{"value":"one"}\n{"value":"two"}\n', ["one", "two"]),
        ("embeddings.csv", b"value,dimensions\none,32\ntwo,\n", ["one", "two"]),
        ("embeddings.txt", b"  one  \n\ntwo\n", ["  one  ", "two"]),
    ),
)
def test_load_embedding_requests_accepts_all_transports(
    tmp_path: Path,
    name: str,
    contents: bytes,
    values: list[str],
) -> None:
    requests = load_embedding_requests(_write_source(tmp_path, name, contents), None)

    assert [request.value for request in requests] == values
    assert [request.custom_id for request in requests] == [
        f"request-{index}" for index in range(len(requests))
    ]


def test_load_embedding_requests_rejects_whole_source_with_coordinates(tmp_path: Path) -> None:
    source = _write_source(
        tmp_path,
        "embeddings.jsonl",
        b'{"value":"one"}\n{"value":"two","dimensions":0}\n',
    )

    with pytest.raises(click.UsageError, match="JSONL line 2"):
        load_embedding_requests(source, None)


@pytest.mark.parametrize("name", ("requests.data", "requests"))
def test_load_text_requests_requires_format_for_unknown_extension(
    tmp_path: Path, name: str
) -> None:
    source = _write_source(tmp_path, name, b'{"prompt":"hello"}')

    with pytest.raises(click.UsageError, match="requires --format"):
        load_text_requests(source, None)


def test_load_text_requests_explicit_format_overrides_extension(tmp_path: Path) -> None:
    source = _write_source(tmp_path, "requests.json", b"first\nsecond\n")

    requests = load_text_requests(source, "text")

    assert [request.prompt for request in requests] == ["first", "second"]


@pytest.mark.parametrize(
    ("contents", "message"),
    (
        (b"[]", "at least one"),
        (b'{"requests":[{"prompt":"one"}]}', "Extra inputs"),
        (b'{"prompt":"one",}', "valid JSON"),
        (b'// comment\n{"prompt":"one"}', "valid JSON"),
        (b'{"prompt":"one","temperature":1e400}', "non-finite JSON number"),
        (b'{"prompt":"one","custom_id":"a","customId":"b"}', "cannot both"),
        (b'[{"prompt":"one"},{"messages":[]}]', "index 1"),
    ),
)
def test_load_text_requests_rejects_invalid_json(
    tmp_path: Path, contents: bytes, message: str
) -> None:
    source = _write_source(tmp_path, "requests.json", contents)

    with pytest.raises(click.UsageError, match=message):
        load_text_requests(source, None)


@pytest.mark.parametrize(
    ("contents", "message"),
    (
        (b'{"prompt":"one"}\n[]\n', "line 2"),
        (b'{"prompt":"one"}\nnot-json\n', "line 2"),
        (b'{"prompt":"one"}\n{"prompt":"two","temperature":1e400}\n', "line 2"),
        (b"\n\t\n", "at least one"),
    ),
)
def test_load_text_requests_reports_jsonl_lines(
    tmp_path: Path, contents: bytes, message: str
) -> None:
    source = _write_source(tmp_path, "requests.jsonl", contents)

    with pytest.raises(click.UsageError, match=message):
        load_text_requests(source, None)


def test_load_text_requests_converts_csv_scalar_fields(tmp_path: Path) -> None:
    source = _write_source(
        tmp_path,
        "requests.csv",
        b"prompt,custom_id,max_output_tokens,temperature,system,tool_choice\r\n"
        b'" hello ",kept,42,0.25,"be useful",required\r\n'
        b"second,,,,,\r\n",
    )

    requests = load_text_requests(source, None)

    assert requests[0].prompt == " hello "
    assert requests[0].custom_id == "kept"
    assert requests[0].max_output_tokens == 42
    assert requests[0].temperature == 0.25
    assert requests[0].system == "be useful"
    assert requests[0].tool_choice == "required"
    assert requests[1].custom_id == "request-1"
    assert requests[1].max_output_tokens is None


@pytest.mark.parametrize(
    ("contents", "message"),
    (
        ("prompt,prompt\none,two\n", "row 1, column 2"),
        ("prompt,unknown\none,two\n", "row 1, column 2"),
        ("prompt,messages\none,[]\n", "row 1, column 2"),
        ("prompt,max_output_tokens\none,nope\n", "row 2, column max_output_tokens"),
        ("prompt,tool_choice\none,sometimes\n", "row 2, column tool_choice"),
        ("prompt,temperature\none,nan\n", "row 2, column temperature"),
        ("prompt\n\n", "row 2, column prompt"),
    ),
)
def test_load_text_requests_rejects_invalid_csv_with_coordinates(
    tmp_path: Path, contents: str, message: str
) -> None:
    source = _write_source(tmp_path, "requests.csv", contents.encode())

    with pytest.raises(click.UsageError, match=message):
        load_text_requests(source, None)


@pytest.mark.parametrize(
    "contents",
    (
        b'{"prompt":"one","custom_id":"request-1"}\n{"prompt":"two"}\n',
        b'{"prompt":"one"}\n{"prompt":"two","custom_id":"request-0"}\n',
        b'{"prompt":"one","custom_id":"same"}\n{"prompt":"two","custom_id":"same"}\n',
    ),
)
def test_load_text_requests_rejects_explicit_and_generated_id_collisions(
    tmp_path: Path, contents: bytes
) -> None:
    source = _write_source(tmp_path, "requests.jsonl", contents)

    with pytest.raises(click.UsageError, match="custom_id"):
        load_text_requests(source, None)


def test_load_text_requests_preserves_explicit_empty_id(tmp_path: Path) -> None:
    source = _write_source(tmp_path, "requests.json", b'{"prompt":"one","custom_id":""}')

    requests = load_text_requests(source, None)

    assert requests[0].custom_id == ""


def test_load_text_requests_rejects_non_regular_file(tmp_path: Path) -> None:
    with pytest.raises(click.UsageError, match="regular file"):
        load_text_requests(tmp_path, "text")


def test_load_text_requests_rejects_non_utf8(tmp_path: Path) -> None:
    source = _write_source(tmp_path, "requests.txt", b"\x96\n")

    with pytest.raises(click.UsageError, match="UTF-8"):
        load_text_requests(source, None)


def test_load_text_requests_stops_at_first_request_over_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _write_source(
        tmp_path,
        "requests.jsonl",
        b'{"prompt":"one"}\n{"prompt":"two"}\n{"prompt":"unparsed","unknown":true}\n',
    )
    monkeypatch.setattr(input_module, "MAX_REQUESTS", 1)

    with pytest.raises(click.UsageError, match="request limit at JSONL line 2"):
        load_text_requests(source, None)
