"""Explicit, bounded image-result materialization."""

from __future__ import annotations

import base64
import binascii
import hashlib
import os
import re
import tempfile
from pathlib import Path

from batchwork._limits import MAX_AGGREGATE_MEDIA_BYTES, MAX_DECODED_MEDIA_BYTES
from batchwork.media import DefaultMediaResolver
from batchwork.types import BatchResult

from ._contract import (
    ErrorDetail,
    ErrorEnvelope,
    ImageManifestEntry,
    ImageManifestEnvelope,
    KnownErrorCode,
    Materialization,
    serialize_envelope,
)
from ._failures import CliFailure

_SAFE_FILENAME = re.compile(r"[^A-Za-z0-9._-]+")
_WINDOWS_RESERVED = {
    "aux",
    "com1",
    "com2",
    "com3",
    "com4",
    "com5",
    "com6",
    "com7",
    "com8",
    "com9",
    "con",
    "lpt1",
    "lpt2",
    "lpt3",
    "lpt4",
    "lpt5",
    "lpt6",
    "lpt7",
    "lpt8",
    "lpt9",
    "nul",
    "prn",
}
_EXTENSIONS = {
    "image/gif": "gif",
    "image/jpeg": "jpg",
    "image/png": "png",
    "image/webp": "webp",
}


def _failure(
    code: KnownErrorCode,
    message: str,
    *,
    operation: str,
    materialized_images: int | None = None,
    materialized_bytes: int | None = None,
) -> CliFailure:
    return CliFailure(
        ErrorEnvelope(
            error=ErrorDetail(
                code=code,
                category="local_state",
                message=message,
                exit_code=8,
                retryable=False,
                operation=operation,
                partial_output=(True if materialized_images or materialized_bytes else None),
                materialized_images=materialized_images,
                materialized_bytes=materialized_bytes,
            )
        )
    )


def prepare_output_directory(path: Path, *, operation: str) -> Path:
    """Create an absent target or require an empty, non-symlink directory."""
    target = Path.cwd() / path if not path.is_absolute() else path
    target = target.absolute()
    try:
        if target.is_symlink():
            raise _failure(
                "output_directory_invalid",
                f'Output directory "{target}" must not be a symlink.',
                operation=operation,
            )
        if target.exists():
            if not target.is_dir():
                raise _failure(
                    "output_directory_invalid",
                    f'Output directory "{target}" must be a directory.',
                    operation=operation,
                )
            if any(target.iterdir()):
                raise _failure(
                    "output_directory_invalid",
                    f'Output directory "{target}" must be empty.',
                    operation=operation,
                )
        else:
            target.mkdir(mode=0o700, parents=False)
    except CliFailure:
        raise
    except OSError as error:
        raise _failure(
            "output_directory_invalid",
            f'Could not prepare output directory "{target}": {error}.',
            operation=operation,
        ) from error
    return target


class ImageMaterializer:
    """Materialize normalized images and atomically maintain a safe manifest."""

    def __init__(self, output_dir: Path, *, operation: str) -> None:
        self.output_dir = output_dir
        self.operation = operation
        self.entries: list[ImageManifestEntry] = []
        self.byte_count = 0
        self._job: str | None = None
        self._routing_fingerprint: str | None = None
        self._resolver = DefaultMediaResolver(timeout=30.0)

    async def materialize_result(
        self,
        job: str,
        routing_fingerprint: str | None,
        result: BatchResult,
    ) -> Materialization | None:
        if self._job is None:
            self._job = job
            self._routing_fingerprint = routing_fingerprint
        new_entries: list[ImageManifestEntry] = []
        try:
            for index, image in enumerate(result.images or (), start=1):
                source_kind = "data" if image.data is not None else "url"
                source = image.data if image.data is not None else image.url
                if source is None:
                    continue
                if image.data is not None and not image.data.startswith("data:"):
                    if len(image.data) > 4 * ((MAX_DECODED_MEDIA_BYTES + 2) // 3):
                        raise ValueError(
                            f"inline image exceeds the {MAX_DECODED_MEDIA_BYTES} byte limit"
                        )
                    try:
                        decoded = base64.b64decode(image.data, validate=True)
                    except (binascii.Error, ValueError) as error:
                        raise ValueError("provider returned malformed base64 image data") from error
                    resolved = await self._resolver.resolve(
                        decoded,
                        media_type=image.media_type,
                        max_bytes=MAX_DECODED_MEDIA_BYTES,
                    )
                else:
                    resolved = await self._resolver.resolve(
                        source,
                        media_type=image.media_type,
                        max_bytes=MAX_DECODED_MEDIA_BYTES,
                    )
                if not resolved.media_type.startswith("image/"):
                    raise ValueError(
                        f'provider returned non-image media type "{resolved.media_type}"'
                    )
                extension = self._extension(resolved.data, resolved.media_type)
                if self.byte_count + len(resolved.data) > MAX_AGGREGATE_MEDIA_BYTES:
                    raise ValueError(
                        "materialized images exceed the "
                        f"{MAX_AGGREGATE_MEDIA_BYTES} byte aggregate limit"
                    )
                digest = hashlib.sha256(resolved.data).hexdigest()
                filename = self._filename(result.custom_id, index, extension)
                if (self.output_dir / filename).exists():
                    raise ValueError(f'completed image "{filename}" already exists')
                self._atomic_write(self.output_dir / filename, resolved.data)
                entry = ImageManifestEntry(
                    path=filename,
                    custom_id=result.custom_id,
                    image_index=index,
                    source_kind=source_kind,
                    media_type=resolved.media_type,
                    byte_count=len(resolved.data),
                    sha256=digest,
                )
                self.entries.append(entry)
                new_entries.append(entry)
                self.byte_count += len(resolved.data)
                self._write_manifest()
        except CliFailure:
            raise
        except Exception as error:
            raise _failure(
                "output_write_failed",
                f"Could not materialize image result: {error}. Completed files were preserved.",
                operation=self.operation,
                materialized_images=len(self.entries),
                materialized_bytes=self.byte_count,
            ) from error
        if not new_entries:
            return None
        return Materialization(output_dir=str(self.output_dir), images=new_entries)

    def summary(self) -> Materialization:
        return Materialization(output_dir=str(self.output_dir), images=list(self.entries))

    def _filename(self, custom_id: str, index: int, extension: str) -> str:
        safe = _SAFE_FILENAME.sub("-", custom_id).strip(" .-_") or "image"
        safe = safe[:80].rstrip(" .-_") or "image"
        if safe.split(".", 1)[0].casefold() in _WINDOWS_RESERVED:
            safe = f"_{safe}"
        custom_id_hash = hashlib.sha256(custom_id.encode()).hexdigest()[:12]
        return f"{safe}--{custom_id_hash}--{index}.{extension}"

    def _extension(self, data: bytes, media_type: str) -> str:
        detected: str | None = None
        if data.startswith(b"\x89PNG\r\n\x1a\n"):
            detected = "image/png"
        elif data.startswith(b"\xff\xd8\xff"):
            detected = "image/jpeg"
        elif data.startswith((b"GIF87a", b"GIF89a")):
            detected = "image/gif"
        elif data.startswith(b"RIFF") and data[8:12] == b"WEBP":
            detected = "image/webp"
        if detected is None and media_type in _EXTENSIONS:
            raise ValueError(
                f'provider returned bytes that do not match declared type "{media_type}"'
            )
        return _EXTENSIONS.get(detected or media_type, "bin")

    def _atomic_write(self, path: Path, data: bytes) -> None:
        descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=self.output_dir)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                descriptor = -1
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass

    def _write_manifest(self) -> None:
        if self._job is None:
            raise RuntimeError("image materializer has no job identity")
        manifest = ImageManifestEnvelope(
            job=self._job,
            routing_fingerprint=self._routing_fingerprint,
            images=self.entries,
        )
        self._atomic_write(
            self.output_dir / "manifest.json",
            serialize_envelope(manifest).encode("utf-8"),
        )
