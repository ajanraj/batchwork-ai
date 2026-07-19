"""Verify release archives contain only the intended Python distribution."""

from __future__ import annotations

import configparser
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from email.parser import Parser
from pathlib import Path, PurePosixPath

from batchwork.cli._commands import CLI_HELP_PATHS

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


def run_checked(
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    description: str,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        capture_output=True,
        check=False,
        text=True,
    )
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown error"
        fail(f"{description} failed: {detail.splitlines()[-1]}")
    return result


def verify_installed_cli(wheel: Path, distribution_version: str) -> None:
    uv = shutil.which("uv")
    if uv is None:
        fail("uv is required for installed CLI verification")

    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        bin_directory = root / "bin"
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        environment.update(
            {
                "BATCHWORK_CONFIG": str(root / "missing-config.toml"),
                "BATCHWORK_REGISTRY": str(root / "missing-registry.sqlite3"),
                "HOME": str(root / "home"),
                "NO_COLOR": "1",
                "UV_CACHE_DIR": str(root / "cache"),
                "UV_NO_CONFIG": "1",
                "UV_TOOL_BIN_DIR": str(bin_directory),
                "UV_TOOL_DIR": str(root / "tools"),
                "XDG_CACHE_HOME": str(root / "xdg-cache"),
                "XDG_CONFIG_HOME": str(root / "xdg-config"),
                "XDG_DATA_HOME": str(root / "xdg-data"),
            }
        )
        run_checked(
            [
                uv,
                "tool",
                "install",
                "--force",
                "--no-config",
                "--python",
                sys.executable,
                str(wheel.resolve()),
            ],
            cwd=root,
            environment=environment,
            description="isolated wheel tool installation",
        )

        executable = bin_directory / ("batchwork.exe" if os.name == "nt" else "batchwork")
        if not executable.is_file():
            fail("installed wheel did not create the batchwork executable")

        version_result = run_checked(
            [str(executable), "--version"],
            cwd=root,
            environment=environment,
            description="installed batchwork --version",
        )
        expected_version = f"batchwork, version {distribution_version}"
        if version_result.stdout.strip() != expected_version:
            fail(
                "installed batchwork --version returned "
                f"{version_result.stdout.strip()!r}, expected {expected_version!r}"
            )

        inspection_prefix = [
            str(executable),
            "--config",
            str(root / "explicit-missing-config.toml"),
            "--registry",
            str(root / "explicit-missing-registry.sqlite3"),
        ]
        for path in CLI_HELP_PATHS:
            run_checked(
                [*inspection_prefix, *path, "--help"],
                cwd=root,
                environment=environment,
                description=f"installed help path {' '.join(path) or 'root'}",
            )

        for shell in ("bash", "zsh", "fish"):
            completion_environment = environment | {"_BATCHWORK_COMPLETE": f"{shell}_source"}
            completion = run_checked(
                [str(executable)],
                cwd=root,
                environment=completion_environment,
                description=f"installed {shell} completion setup",
            )
            if not completion.stdout.strip():
                fail(f"installed {shell} completion setup returned empty output")


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
        entry_point_members = [
            member
            for member in members
            if member.name == "entry_points.txt" and member.parent.name.endswith(".dist-info")
        ]
        if len(entry_point_members) != 1:
            fail(f"{wheel.name} must contain exactly one dist-info/entry_points.txt file")
        entry_points = configparser.ConfigParser()
        entry_points.read_string(archive.read(str(entry_point_members[0])).decode())

    distribution_version = metadata.get("Version")
    if not distribution_version:
        fail(f"{wheel.name} metadata does not contain Version")
    requirements = metadata.get_all("Requires-Dist", [])
    if not any(requirement.startswith("click") for requirement in requirements):
        fail(f"{wheel.name} metadata does not require Click")
    console_scripts = dict(entry_points.items("console_scripts"))
    if console_scripts != {"batchwork": "batchwork.cli:main"}:
        fail(f"{wheel.name} has unexpected console scripts: {console_scripts}")

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

    verify_installed_cli(wheel, distribution_version)


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
