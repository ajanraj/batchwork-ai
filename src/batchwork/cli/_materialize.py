"""Explicit, bounded image-result materialization."""

from __future__ import annotations

import base64
import binascii
import hashlib
import os
import re
import stat
import tempfile
import xml.etree.ElementTree as ElementTree
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
FileIdentity = tuple[int, int]


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
        self._output_identity = self._path_identity(output_dir)
        self._manifest_identity: FileIdentity | None = None
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
                try:
                    image_identity = self._atomic_write(
                        self.output_dir / filename,
                        resolved.data,
                        replace=False,
                    )
                except FileExistsError as error:
                    raise ValueError(f'completed image "{filename}" already exists') from error
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
                try:
                    self._write_manifest()
                except Exception:
                    self.entries.pop()
                    self._unlink_if_identity(self.output_dir / filename, image_identity)
                    raise
                new_entries.append(entry)
                self.byte_count += len(resolved.data)
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
        if detected is not None:
            return _EXTENSIONS[detected]
        if media_type in _EXTENSIONS:
            raise ValueError(
                f'provider returned bytes that do not match declared type "{media_type}"'
            )
        if self._valid_other_image(data, media_type):
            return "bin"
        raise ValueError(f'provider returned invalid image media type "{media_type}"')

    def _valid_other_image(self, data: bytes, media_type: str) -> bool:
        signatures = {
            "image/bmp": (b"BM",),
            "image/tiff": (b"II*\x00", b"MM\x00*"),
            "image/x-icon": (b"\x00\x00\x01\x00",),
            "image/vnd.microsoft.icon": (b"\x00\x00\x01\x00",),
        }
        if media_type in signatures:
            return data.startswith(signatures[media_type])
        if media_type == "image/svg+xml":
            try:
                root = ElementTree.fromstring(data)
            except ElementTree.ParseError:
                return False
            return root.tag.rsplit("}", 1)[-1].casefold() == "svg"
        if media_type in {"image/avif", "image/heic", "image/heif"}:
            return (
                len(data) >= 16
                and data[4:8] == b"ftyp"
                and any(brand in data[8:32] for brand in (b"avif", b"avis", b"heic", b"heif"))
            )
        return False

    def _path_identity(self, path: Path) -> FileIdentity:
        metadata = path.stat(follow_symlinks=False)
        return metadata.st_dev, metadata.st_ino

    def _validate_output_directory(self) -> None:
        metadata = self.output_dir.stat(follow_symlinks=False)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or (metadata.st_dev, metadata.st_ino) != self._output_identity
        ):
            raise OSError("output directory changed after validation")

    def _unlink_if_identity(self, path: Path, identity: FileIdentity) -> None:
        try:
            if self._path_identity(path) == identity:
                path.unlink()
        except FileNotFoundError:
            pass

    def _atomic_write(
        self,
        path: Path,
        data: bytes,
        *,
        replace: bool = True,
    ) -> FileIdentity:
        self._validate_output_directory()
        descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=self.output_dir)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                descriptor = -1
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
                metadata = os.fstat(stream.fileno())
                identity = (metadata.st_dev, metadata.st_ino)
            self._validate_output_directory()
            if replace:
                os.replace(temporary, path)
            elif os.name == "nt":
                os.rename(temporary, path)
            else:
                os.link(temporary, path)
                os.unlink(temporary)
            return identity
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
        path = self.output_dir / "manifest.json"
        if self._manifest_identity is not None:
            if self._path_identity(path) != self._manifest_identity:
                raise OSError("manifest changed after Batchwork created it")
        self._manifest_identity = self._atomic_write(
            path,
            serialize_envelope(manifest).encode("utf-8"),
            replace=self._manifest_identity is not None,
        )
