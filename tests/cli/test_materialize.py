from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path

import pytest

from batchwork.cli._commands import _partial_error_envelope
from batchwork.cli._failures import CliFailure
from batchwork.cli._materialize import ImageMaterializer, prepare_output_directory
from batchwork.types import BatchImage, BatchResult, BatchResultStatus

_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def test_prepare_output_directory_requires_absent_or_empty_target(tmp_path: Path) -> None:
    created = prepare_output_directory(tmp_path / "created", operation="results")
    assert created.is_dir()
    if os.name != "nt":
        assert created.stat().st_mode & 0o777 == 0o700

    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "keep.txt").write_text("keep")
    with pytest.raises(CliFailure) as failure:
        prepare_output_directory(occupied, operation="results")
    assert failure.value.envelope.error.code == "output_directory_invalid"
    assert (occupied / "keep.txt").read_text() == "keep"


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission modes only")
def test_prepare_output_directory_does_not_change_existing_directory_mode(tmp_path: Path) -> None:
    existing = tmp_path / "existing"
    existing.mkdir(mode=0o750)

    prepared = prepare_output_directory(existing, operation="results")

    assert prepared == existing
    assert existing.stat().st_mode & 0o777 == 0o750


@pytest.mark.asyncio
async def test_materializer_prefers_data_and_writes_atomic_manifest(tmp_path: Path) -> None:
    output = prepare_output_directory(tmp_path / "images", operation="results")
    materializer = ImageMaterializer(output, operation="results")
    result = BatchResult(
        custom_id="CON / unsafe",
        status=BatchResultStatus.SUCCEEDED,
        images=[
            BatchImage(
                data=base64.b64encode(_PNG).decode(),
                media_type="image/png",
                url="http://127.0.0.1/never-requested.png",
            )
        ],
    )

    projection = await materializer.materialize_result("bw_" + "a" * 32, None, result)

    assert projection is not None
    assert len(projection.images) == 1
    image_path = output / projection.images[0].path
    assert image_path.read_bytes() == _PNG
    custom_id_hash = hashlib.sha256(result.custom_id.encode()).hexdigest()[:12]
    assert image_path.name == f"CON-unsafe--{custom_id_hash}--1.png"
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["type"] == "image_manifest"
    assert manifest["images"][0]["path"] == image_path.name
    assert not list(output.glob(".*.tmp"))


@pytest.mark.asyncio
async def test_materializer_never_interprets_provider_data_as_local_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "secret.png").write_bytes(_PNG)
    output = prepare_output_directory(tmp_path / "images", operation="results")
    materializer = ImageMaterializer(output, operation="results")
    result = BatchResult(
        custom_id="unsafe",
        status=BatchResultStatus.SUCCEEDED,
        images=[BatchImage(data="secret.png", media_type="image/png")],
    )

    with pytest.raises(CliFailure) as failure:
        await materializer.materialize_result("bw_" + "a" * 32, None, result)

    assert failure.value.envelope.error.code == "output_write_failed"
    assert not list(output.glob("*.png"))


@pytest.mark.asyncio
async def test_materializer_rejects_unknown_bytes_declared_as_supported_image(
    tmp_path: Path,
) -> None:
    output = prepare_output_directory(tmp_path / "images", operation="results")
    materializer = ImageMaterializer(output, operation="results")
    result = BatchResult(
        custom_id="not-an-image",
        status=BatchResultStatus.SUCCEEDED,
        images=[
            BatchImage(
                data=base64.b64encode(b"not actually a png").decode(),
                media_type="image/png",
            )
        ],
    )

    with pytest.raises(CliFailure) as failure:
        await materializer.materialize_result("bw_" + "a" * 32, None, result)

    assert failure.value.envelope.error.materialized_images == 0
    assert failure.value.envelope.error.materialized_bytes == 0
    assert not list(output.glob("*.png"))


@pytest.mark.asyncio
async def test_materializer_preserves_unknown_image_media_with_bin_extension(
    tmp_path: Path,
) -> None:
    output = prepare_output_directory(tmp_path / "images", operation="results")
    materializer = ImageMaterializer(output, operation="results")
    payload = b"<svg xmlns='http://www.w3.org/2000/svg'/>"
    result = BatchResult(
        custom_id="vector",
        status=BatchResultStatus.SUCCEEDED,
        images=[
            BatchImage(
                data=base64.b64encode(payload).decode(),
                media_type="image/svg+xml",
            )
        ],
    )

    projection = await materializer.materialize_result("bw_" + "a" * 32, None, result)

    assert projection is not None
    assert projection.images[0].path.endswith(".bin")
    assert (output / projection.images[0].path).read_bytes() == payload


@pytest.mark.asyncio
async def test_materializer_never_overwrites_a_completed_image(tmp_path: Path) -> None:
    output = prepare_output_directory(tmp_path / "images", operation="results")
    materializer = ImageMaterializer(output, operation="results")
    result = BatchResult(
        custom_id="duplicate",
        status=BatchResultStatus.SUCCEEDED,
        images=[
            BatchImage(
                data=base64.b64encode(_PNG).decode(),
                media_type="image/png",
            )
        ],
    )
    first = await materializer.materialize_result("bw_" + "a" * 32, None, result)
    assert first is not None
    image_path = output / first.images[0].path
    original = image_path.read_bytes()

    with pytest.raises(CliFailure) as failure:
        await materializer.materialize_result("bw_" + "a" * 32, None, result)

    assert image_path.read_bytes() == original
    assert failure.value.envelope.error.partial_output is True
    assert failure.value.envelope.error.materialized_images == 1
    assert failure.value.envelope.error.materialized_bytes == len(_PNG)
    manifest = json.loads((output / "manifest.json").read_text())
    assert len(manifest["images"]) == 1

    enriched = _partial_error_envelope(
        failure.value,
        materializer=materializer,
        records_emitted=2,
    )
    assert enriched.error.partial_output is True
    assert enriched.error.records_emitted == 2
    assert enriched.error.materialized_images == 1
    assert enriched.error.materialized_bytes == len(_PNG)
