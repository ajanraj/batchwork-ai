from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from batchwork.cli._failures import CliFailure
from batchwork.cli._materialize import ImageMaterializer, prepare_output_directory
from batchwork.types import BatchImage, BatchResult, BatchResultStatus

_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def test_prepare_output_directory_requires_absent_or_empty_target(tmp_path: Path) -> None:
    created = prepare_output_directory(tmp_path / "created", operation="results")
    assert created.is_dir()

    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "keep.txt").write_text("keep")
    with pytest.raises(CliFailure) as failure:
        prepare_output_directory(occupied, operation="results")
    assert failure.value.envelope.error.code == "output_directory_invalid"
    assert (occupied / "keep.txt").read_text() == "keep"


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
    assert image_path.name.endswith("-1.png")
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
