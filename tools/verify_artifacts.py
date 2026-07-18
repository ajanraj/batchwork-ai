"""Verify release archives contain only the intended Python distribution."""

from __future__ import annotations

import subprocess
import sys
import tarfile
import tempfile
import zipfile
from email.parser import Parser
from pathlib import Path, PurePosixPath

FORBIDDEN_COMPONENTS = frozenset(
    {
        ".coverage",
        ".git",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
        "node_modules",
    }
)


def fail(message: str) -> None:
    raise SystemExit(f"artifact verification failed: {message}")


def normalized_members(names: list[str], archive: Path) -> list[PurePosixPath]:
    members: list[PurePosixPath] = []
    for name in names:
        member = PurePosixPath(name)
        if member.is_absolute() or ".." in member.parts:
            fail(f"{archive.name} contains unsafe path {name!r}")
        if FORBIDDEN_COMPONENTS.intersection(member.parts) or any(
            component.endswith("_cache") for component in member.parts
        ):
            fail(f"{archive.name} contains cache or toolchain path {name!r}")
        members.append(member)
    return members


def verify_wheel(wheel: Path) -> None:
    with zipfile.ZipFile(wheel) as archive:
        members = normalized_members(archive.namelist(), wheel)
        metadata_members = [
            member
            for member in members
            if member.name == "METADATA" and member.parent.name.endswith(".dist-info")
        ]
        if len(metadata_members) != 1:
            fail(f"{wheel.name} must contain exactly one dist-info/METADATA file")
        metadata = Parser().parsestr(archive.read(str(metadata_members[0])).decode())

    distribution_version = metadata.get("Version")
    if not distribution_version:
        fail(f"{wheel.name} metadata does not contain Version")

    invalid = [
        str(member)
        for member in members
        if member.parts
        and member.parts[0] != "batchwork"
        and not member.parts[0].endswith(".dist-info")
    ]
    if invalid:
        fail(f"{wheel.name} contains non-package files: {invalid[:3]}")

    names = {str(member) for member in members}
    if "batchwork/py.typed" not in names:
        fail(f"{wheel.name} does not contain batchwork/py.typed")
    basenames = {member.name for member in members}
    if "LICENSE" not in basenames:
        fail(f"{wheel.name} does not contain LICENSE")

    import_check = """
import importlib.abc
import pathlib
import sys

wheel = pathlib.Path(sys.argv[1]).resolve()
expected_version = sys.argv[2]

class BlockRedis(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "upstash_redis" or fullname.startswith("upstash_redis."):
            raise ImportError("upstash-redis must remain optional")
        return None

sys.meta_path.insert(0, BlockRedis())
sys.path.insert(0, str(wheel))
import batchwork
assert str(wheel) in batchwork.__file__, batchwork.__file__
assert batchwork.__version__ == expected_version, (
    batchwork.__version__,
    expected_version,
)
"""
    with tempfile.TemporaryDirectory() as temporary_directory:
        result = subprocess.run(
            [
                sys.executable,
                "-I",
                "-c",
                import_check,
                str(wheel.resolve()),
                distribution_version,
            ],
            cwd=temporary_directory,
            capture_output=True,
            check=False,
            text=True,
        )
    if result.returncode:
        detail = (
            result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "unknown error"
        )
        fail(f"base wheel import failed with Redis unavailable: {detail}")


def verify_sdist(sdist: Path) -> None:
    with tarfile.open(sdist, "r:gz") as archive:
        members = normalized_members(archive.getnames(), sdist)

    roots = {member.parts[0] for member in members if member.parts}
    if len(roots) != 1:
        fail(f"{sdist.name} must contain exactly one root directory")
    root = next(iter(roots))
    forbidden_roots = {f"{root}/batchwork", f"{root}/docs"}
    for member in members:
        path = str(member)
        if any(path == item or path.startswith(f"{item}/") for item in forbidden_roots):
            fail(f"{sdist.name} contains nested checkout or docs toolchain path {path!r}")


def main() -> None:
    dist = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("dist")
    wheels = sorted(dist.glob("*.whl"))
    sdists = sorted(dist.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        fail(f"expected one wheel and one sdist in {dist}, found {len(wheels)} and {len(sdists)}")
    verify_wheel(wheels[0])
    verify_sdist(sdists[0])
    print(f"verified {wheels[0].name} and {sdists[0].name}")


if __name__ == "__main__":
    main()
