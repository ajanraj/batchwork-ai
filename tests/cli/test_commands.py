import json
from importlib.metadata import version

import pytest
from click.testing import CliRunner

from batchwork.cli._commands import CLI_HELP_PATHS, cli


def test_machine_usage_failure_is_one_error_envelope() -> None:
    result = CliRunner().invoke(cli, ["--json", "status"], prog_name="batchwork")

    assert result.exit_code == 2
    assert result.stdout == ""
    assert len(result.stderr.splitlines()) == 1
    error = json.loads(result.stderr)["error"]
    assert error["code"] == "usage_error"
    assert error["operation"] == "status"


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


def test_root_help_explains_output_defaults_and_route_complete_selectors() -> None:
    result = CliRunner().invoke(cli, ["--help"], prog_name="batchwork")

    assert result.exit_code == 0
    assert "Interactive stdout uses human output" in result.output
    assert "JSONL for streaming results and run" in result.output
    assert "local alias" in result.output
    assert "provider:provider-job-id" in result.output


@pytest.mark.parametrize("command", ("status", "wait", "results", "cancel"))
def test_lifecycle_help_explains_direct_routing_and_adoption(command: str) -> None:
    result = CliRunner().invoke(cli, [command, "--help"], prog_name="batchwork")

    assert result.exit_code == 0
    assert "Qualify a bare provider job ID" in result.output
    assert "Adopt a successful direct operation locally" in result.output
    assert "--header-env NAME=ENV_VAR" in result.output


def test_creation_help_explains_source_transport_and_safe_headers() -> None:
    result = CliRunner().invoke(cli, ["submit", "text", "--help"], prog_name="batchwork")

    assert result.exit_code == 0
    assert 'SOURCE is one regular file or "-"' in result.output
    assert "Stdin and unknown" in result.output
    assert "Repeatable non-secret literal provider header" in result.output
    assert "Authorize work above the soft volume gate" in result.output


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
