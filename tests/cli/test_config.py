from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from click.testing import CliRunner

from batchwork.cli._commands import cli

VALID_CONFIG = """\
schema_version = 1
default_profile = "work"

[profiles.work.models]
text = "openai/gpt-test"

[profiles.work.providers.openai]
api_key_env = "WORK_OPENAI_KEY"
base_url = "https://gateway.example.com/v1/"

[profiles.work.providers.openai.headers]
X-Application = "batchwork-cli"

[profiles.work.providers.openai.header_env]
Authorization = "WORK_AUTHORIZATION"
"""


def test_config_path_uses_independent_explicit_environment_and_default_precedence(
    tmp_path: Path,
) -> None:
    explicit_config = tmp_path / "explicit.toml"
    explicit_registry = tmp_path / "explicit.sqlite3"
    environment_config = tmp_path / "environment.toml"
    environment_registry = tmp_path / "environment.sqlite3"

    result = CliRunner().invoke(
        cli,
        [
            "--json",
            "--config",
            str(explicit_config),
            "--registry",
            str(explicit_registry),
            "config",
            "path",
        ],
        env={
            "BATCHWORK_CONFIG": str(environment_config),
            "BATCHWORK_REGISTRY": str(environment_registry),
        },
    )

    assert result.exit_code == 0, result.stderr
    document = json.loads(result.stdout)
    assert document["config"] == {"path": str(explicit_config), "exists": False}
    assert document["registry"] == {"path": str(explicit_registry), "exists": False}


def test_absent_default_config_is_valid(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        cli,
        ["--json", "config", "validate"],
        env={"XDG_CONFIG_HOME": str(tmp_path / "config")},
    )

    assert result.exit_code == 0, result.stderr
    document = json.loads(result.stdout)
    assert document["exists"] is False
    assert document["valid"] is True
    assert "config_schema_version" not in document
    assert document["credentials_read"] is False


def test_explicit_missing_config_fails() -> None:
    result = CliRunner().invoke(
        cli,
        ["--config", "missing.toml", "config", "validate"],
    )

    assert result.exit_code == 3
    assert 'Configuration file "missing.toml" does not exist.' in result.stderr


def test_machine_config_failure_uses_structured_configuration_error() -> None:
    result = CliRunner().invoke(
        cli,
        ["--json", "--config", "missing.toml", "config", "validate"],
    )

    assert result.exit_code == 3
    assert result.stdout == ""
    error = json.loads(result.stderr)["error"]
    assert error["code"] == "invalid_configuration"
    assert error["category"] == "configuration"
    assert error["exit_code"] == 3


@pytest.mark.parametrize(
    ("contents", "message"),
    (
        ("default_profile = 'work'\n", "schema_version"),
        ("schema_version = 2\n", "schema version 1"),
        ("schema_version = 1\nunknown = true\n", "unknown key"),
        (
            "schema_version = 1\n[profiles.work.models]\ntext = 'gpt-test'\n",
            'model must use the "provider/model" form',
        ),
        (
            "schema_version = 1\n[profiles.work.providers.openai]\n"
            "base_url = 'http://example.com/v1'\n",
            "absolute HTTPS",
        ),
        (
            "schema_version = 1\n[profiles.work.providers.openai.headers]\n"
            "Authorization = 'secret'\n",
            "may contain secrets",
        ),
    ),
)
def test_config_validate_rejects_invalid_configuration(
    tmp_path: Path, contents: str, message: str
) -> None:
    path = tmp_path / "config.toml"
    path.write_text(contents)

    result = CliRunner().invoke(cli, ["--config", str(path), "config", "validate"])

    assert result.exit_code == 3
    assert message in result.stderr


@pytest.mark.skipif(os.name != "posix", reason="POSIX permissions only")
def test_config_validate_rejects_group_writable_file(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text("schema_version = 1\n")
    path.chmod(0o620)

    result = CliRunner().invoke(cli, ["--config", str(path), "config", "validate"])

    assert result.exit_code == 3
    assert "group or other writable" in result.stderr


def test_config_show_is_normalized_and_does_not_read_credentials(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(VALID_CONFIG)

    result = CliRunner().invoke(
        cli,
        ["--json", "--config", str(path), "config", "show"],
    )

    assert result.exit_code == 0, result.stderr
    document = json.loads(result.stdout)
    assert document == {
        "schema_version": 1,
        "type": "config_view",
        "path": str(path),
        "profile": "work",
        "models": {"text": "openai/gpt-test"},
        "providers": {
            "openai": {
                "api_key_env": "WORK_OPENAI_KEY",
                "base_url": "https://gateway.example.com/v1",
                "headers": {"x-application": "batchwork-cli"},
                "header_env": {"authorization": "WORK_AUTHORIZATION"},
            }
        },
        "credentials_read": False,
    }


def test_explicit_profile_precedes_environment_profile(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        """\
schema_version = 1
default_profile = "default"
[profiles.default.models]
text = "openai/default"
[profiles.environment.models]
text = "openai/environment"
[profiles.explicit.models]
text = "openai/explicit"
"""
    )

    environment = CliRunner().invoke(
        cli,
        ["--json", "--config", str(path), "config", "show"],
        env={"BATCHWORK_PROFILE": "environment"},
    )
    explicit = CliRunner().invoke(
        cli,
        [
            "--json",
            "--config",
            str(path),
            "--profile",
            "explicit",
            "config",
            "show",
        ],
        env={"BATCHWORK_PROFILE": "environment"},
    )

    assert json.loads(environment.stdout)["profile"] == "environment"
    assert json.loads(explicit.stdout)["profile"] == "explicit"
