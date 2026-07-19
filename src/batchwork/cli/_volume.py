"""Soft cost-bearing workload authorization."""

from __future__ import annotations

from dataclasses import dataclass

from batchwork._limits import (
    LARGE_BATCH_IMAGES,
    LARGE_BATCH_REQUESTS,
    LARGE_BATCH_UPLOAD_BYTES,
)

from ._failures import CliUsageError


@dataclass(frozen=True, slots=True)
class WorkloadVolume:
    requests: int = 0
    upload_bytes: int = 0
    generated_images: int = 0


def require_large_batch_authorization(volume: WorkloadVolume, *, authorized: bool) -> None:
    if authorized:
        return
    measurements = (
        (volume.requests, LARGE_BATCH_REQUESTS, "requests", "request"),
        (volume.upload_bytes, LARGE_BATCH_UPLOAD_BYTES, "serialized provider upload", "byte"),
        (volume.generated_images, LARGE_BATCH_IMAGES, "requested generated images", "image"),
    )
    for actual, maximum, label, unit in measurements:
        if actual > maximum:
            raise CliUsageError(
                f"The batch contains {actual} {label}, above the {maximum} {unit} soft limit; "
                "authorize it with --allow-large-batch.",
                code="large_batch_not_allowed",
            )
