from importlib.metadata import version

import pytest
from click.testing import CliRunner

from batchwork.cli._commands import CLI_HELP_PATHS, cli


@pytest.mark.parametrize("path", CLI_HELP_PATHS)
def test_every_foundation_help_path_works_without_operational_state(
    path: tuple[str, ...], tmp_path
) -> None:
    missing_config = tmp_path / "missing.toml"
    missing_registry = tmp_path / "missing" / "registry.sqlite3"

    result = CliRunner().invoke(
        cli,
        ["--config", str(missing_config), "--registry", str(missing_registry), *path, "--help"],
        prog_name="batchwork",
    )

    assert result.exit_code == 0, result.output
    assert result.output.startswith("Usage: batchwork")
    assert result.stderr == ""


def test_version_uses_installed_distribution_metadata() -> None:
    result = CliRunner().invoke(cli, ["--version"], prog_name="batchwork")

    assert result.exit_code == 0
    assert result.output == f"batchwork, version {version('batchwork-ai')}\n"
    assert result.stderr == ""


@pytest.mark.parametrize("shell", ("bash", "zsh", "fish"))
def test_shell_completion_source_bypasses_operational_state(shell: str, tmp_path) -> None:
    result = CliRunner().invoke(
        cli,
        [],
        prog_name="batchwork",
        env={
            "BATCHWORK_CONFIG": str(tmp_path / "missing.toml"),
            "BATCHWORK_REGISTRY": str(tmp_path / "missing.sqlite3"),
            "_BATCHWORK_COMPLETE": f"{shell}_source",
        },
    )

    assert result.exit_code == 0, result.output
    assert result.output
    assert result.stderr == ""


@pytest.mark.parametrize(
    "mode_flags", (("--human", "--json"), ("--human", "--jsonl"), ("--json", "--jsonl"))
)
def test_output_modes_are_mutually_exclusive(mode_flags: tuple[str, str]) -> None:
    result = CliRunner().invoke(cli, [*mode_flags, "status", "job_1"], prog_name="batchwork")

    assert result.exit_code == 2
    assert "mutually exclusive" in result.stderr


def test_color_controls_are_mutually_exclusive() -> None:
    result = CliRunner().invoke(
        cli, ["--color", "--no-color", "status", "job_1"], prog_name="batchwork"
    )

    assert result.exit_code == 2
    assert "mutually exclusive" in result.stderr


@pytest.mark.parametrize("value", ("nan", "inf", "-1", "0"))
def test_poll_interval_requires_positive_finite_seconds(value: str) -> None:
    result = CliRunner().invoke(
        cli, ["wait", "job_1", "--poll-interval", value], prog_name="batchwork"
    )

    assert result.exit_code == 2
    assert "positive finite number" in result.stderr


@pytest.mark.parametrize("value", ("nanm", "infs", "-1s", "0s", "1x"))
def test_duration_requires_positive_finite_number_and_unit(value: str) -> None:
    result = CliRunner().invoke(cli, ["wait", "job_1", "--timeout", value], prog_name="batchwork")

    assert result.exit_code == 2
    assert "positive finite number followed by s, m, h, or d" in result.stderr


@pytest.mark.parametrize(
    ("option", "value"),
    (
        ("--config", "config.toml"),
        ("--registry", "registry.sqlite3"),
        ("--profile", "work"),
        ("--human", None),
        ("--json", None),
        ("--jsonl", None),
        ("--quiet", None),
        ("--progress", None),
        ("--color", None),
        ("--no-color", None),
    ),
)
def test_global_controls_are_root_only(option: str, value: str | None) -> None:
    arguments = ["status", "job_1", option]
    if value is not None:
        arguments.append(value)

    result = CliRunner().invoke(cli, arguments, prog_name="batchwork")

    assert result.exit_code == 2
    assert "No such option" in result.stderr
    assert option in result.stderr
