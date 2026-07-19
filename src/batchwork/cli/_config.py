"""Strict non-secret CLI configuration and path resolution."""

from __future__ import annotations

import os
import re
import stat
import sys
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from ipaddress import ip_address
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit, urlunsplit

import click

from batchwork.errors import BatchworkError
from batchwork.types import BatchProvider, resolve_model

CONFIG_SCHEMA_VERSION: Literal[1] = 1
PROFILE_ENV = "BATCHWORK_PROFILE"
ENVIRONMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
HEADER_NAME = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")
SENSITIVE_HEADERS = frozenset(
    {"authorization", "cookie", "proxy-authorization", "x-api-key", "api-key"}
)
API_KEY_ENV = {
    BatchProvider.ANTHROPIC: "ANTHROPIC_API_KEY",
    BatchProvider.GOOGLE: "GOOGLE_GENERATIVE_AI_API_KEY",
    BatchProvider.GROQ: "GROQ_API_KEY",
    BatchProvider.MISTRAL: "MISTRAL_API_KEY",
    BatchProvider.OPENAI: "OPENAI_API_KEY",
    BatchProvider.TOGETHER: "TOGETHER_API_KEY",
    BatchProvider.XAI: "XAI_API_KEY",
}
BASE_URL_ENV = {
    provider: f"{name.removesuffix('_API_KEY')}_BASE_URL" for provider, name in API_KEY_ENV.items()
}


class ConfigError(click.ClickException):
    exit_code = 3

    def __init__(self, message: str, *, code: str = "invalid_configuration") -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class ProviderConfig:
    api_key_env: str | None = None
    base_url: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    header_env: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ProfileConfig:
    models: dict[str, str] = field(default_factory=dict)
    providers: dict[BatchProvider, ProviderConfig] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ConfigDocument:
    schema_version: Literal[1] | None = None
    default_profile: str | None = None
    profiles: dict[str, ProfileConfig] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LoadedConfig:
    path: Path
    exists: bool
    document: ConfigDocument


def default_config_path(environment: Mapping[str, str] = os.environ) -> Path:
    if root := environment.get("XDG_CONFIG_HOME"):
        return Path(root) / "batchwork" / "config.toml"
    if os.name == "nt":
        root = environment.get("APPDATA")
        base = Path(root) if root else Path.home() / "AppData" / "Roaming"
        return base / "batchwork" / "config.toml"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "batchwork" / "config.toml"
    return Path.home() / ".config" / "batchwork" / "config.toml"


def default_registry_path(environment: Mapping[str, str] = os.environ) -> Path:
    if root := environment.get("XDG_DATA_HOME"):
        return Path(root) / "batchwork" / "registry.sqlite3"
    if os.name == "nt":
        root = environment.get("LOCALAPPDATA")
        base = Path(root) if root else Path.home() / "AppData" / "Local"
        return base / "batchwork" / "registry.sqlite3"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "batchwork" / "registry.sqlite3"
    return Path.home() / ".local" / "share" / "batchwork" / "registry.sqlite3"


def _selected_path(
    explicit: Path | None,
    variable: str,
    default: Path,
    environment: Mapping[str, str],
) -> tuple[Path, bool]:
    if explicit is not None:
        return explicit, True
    if variable in environment:
        configured = environment[variable]
        if not configured:
            raise ConfigError(f"{variable} selects an empty path.")
        return Path(configured), True
    return default, False


def config_path(
    explicit: Path | None, environment: Mapping[str, str] = os.environ
) -> tuple[Path, bool]:
    return _selected_path(
        explicit, "BATCHWORK_CONFIG", default_config_path(environment), environment
    )


def registry_path(explicit: Path | None, environment: Mapping[str, str] = os.environ) -> Path:
    path, _ = _selected_path(
        explicit, "BATCHWORK_REGISTRY", default_registry_path(environment), environment
    )
    return path


def normalize_base_url(value: str | None, label: str = "base URL") -> str | None:
    if value is None:
        return None
    parsed = urlsplit(value)
    hostname = parsed.hostname
    try:
        _ = parsed.port
    except ValueError as error:
        raise ConfigError(f"{label} contains an invalid port.") from error
    loopback = hostname == "localhost"
    if hostname is not None:
        try:
            loopback = loopback or ip_address(hostname).is_loopback
        except ValueError:
            pass
    if (
        not hostname
        or any(character.isspace() for character in parsed.netloc)
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or (parsed.scheme != "https" and not (parsed.scheme == "http" and loopback))
    ):
        raise ConfigError(
            f"{label} must be absolute HTTPS (HTTP allowed for loopback), without "
            "userinfo, query, or fragment."
        )
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", ""))


def validate_environment_name(value: str, label: str) -> str:
    if not ENVIRONMENT_NAME.fullmatch(value):
        raise ConfigError(f'{label} environment variable name is invalid: "{value}".')
    return value


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ConfigError(f"{label} must be a TOML table.")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ConfigError(f"{label} contains a non-string key.")
        result[key] = item
    return result


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{label} must be a non-empty string.")
    return value


def _unknown_keys(table: Mapping[str, object], allowed: set[str], label: str) -> None:
    unknown = sorted(table.keys() - allowed)
    if unknown:
        raise ConfigError(f'{label} contains unknown key "{unknown[0]}".')


def _headers(value: object, label: str, *, literal: bool) -> dict[str, str]:
    table = _mapping(value, label)
    result: dict[str, str] = {}
    for name, item in table.items():
        normalized = name.lower()
        if not HEADER_NAME.fullmatch(name):
            raise ConfigError(f'{label} contains invalid header name "{name}".')
        if normalized in result:
            raise ConfigError(f'{label} contains duplicate header name "{name}".')
        if literal and normalized in SENSITIVE_HEADERS:
            raise ConfigError(f'Header "{name}" may contain secrets; use header_env.')
        parsed = _string(item, f'{label}."{name}"')
        result[normalized] = (
            parsed if literal else validate_environment_name(parsed, f'Header "{name}"')
        )
    return result


def _provider(value: object, profile: str, provider: BatchProvider) -> ProviderConfig:
    label = f'profiles."{profile}".providers.{provider.value}'
    table = _mapping(value, label)
    _unknown_keys(table, {"api_key_env", "base_url", "headers", "header_env"}, label)
    api_key_env = None
    if "api_key_env" in table:
        api_key_env = validate_environment_name(
            _string(table["api_key_env"], f"{label}.api_key_env"), "API key"
        )
    base_url = None
    if "base_url" in table:
        base_url = normalize_base_url(
            _string(table["base_url"], f"{label}.base_url"), f"{label}.base_url"
        )
    headers = _headers(table.get("headers", {}), f"{label}.headers", literal=True)
    header_env = _headers(table.get("header_env", {}), f"{label}.header_env", literal=False)
    overlap = headers.keys() & header_env.keys()
    if overlap:
        raise ConfigError(f'Header "{sorted(overlap)[0]}" is configured more than once.')
    return ProviderConfig(api_key_env, base_url, headers, header_env)


def _profile(value: object, name: str) -> ProfileConfig:
    label = f'profiles."{name}"'
    table = _mapping(value, label)
    _unknown_keys(table, {"models", "providers"}, label)
    models_table = _mapping(table.get("models", {}), f"{label}.models")
    _unknown_keys(models_table, {"text", "embeddings", "images"}, f"{label}.models")
    models: dict[str, str] = {}
    for modality, value in models_table.items():
        model = _string(value, f"{label}.models.{modality}")
        try:
            resolve_model(model)
        except (BatchworkError, ValueError) as error:
            raise ConfigError(f"{label}.models.{modality}: {error}") from error
        models[modality] = model
    providers_table = _mapping(table.get("providers", {}), f"{label}.providers")
    providers: dict[BatchProvider, ProviderConfig] = {}
    for provider_name, provider_value in providers_table.items():
        try:
            provider = BatchProvider(provider_name)
        except ValueError as error:
            raise ConfigError(
                f'{label}.providers contains unknown provider "{provider_name}".'
            ) from error
        providers[provider] = _provider(provider_value, name, provider)
    return ProfileConfig(models, providers)


def _parse(document: object) -> ConfigDocument:
    root = _mapping(document, "configuration")
    _unknown_keys(root, {"schema_version", "default_profile", "profiles"}, "configuration")
    version = root.get("schema_version")
    if not isinstance(version, int) or isinstance(version, bool):
        raise ConfigError("configuration.schema_version must be integer schema version 1.")
    if version != CONFIG_SCHEMA_VERSION:
        raise ConfigError("configuration must use schema version 1.")
    profiles_table = _mapping(root.get("profiles", {}), "configuration.profiles")
    profiles: dict[str, ProfileConfig] = {}
    for name, value in profiles_table.items():
        if not name:
            raise ConfigError("Profile names must not be empty.")
        profiles[name] = _profile(value, name)
    default_profile = None
    if "default_profile" in root:
        default_profile = _string(root["default_profile"], "configuration.default_profile")
        if default_profile not in profiles:
            raise ConfigError(
                f'Default profile "{default_profile}" is not defined in configuration.profiles.'
            )
    return ConfigDocument(version, default_profile, profiles)


def load_config(explicit: Path | None, environment: Mapping[str, str] = os.environ) -> LoadedConfig:
    path, required = config_path(explicit, environment)
    if not path.exists():
        if required:
            raise ConfigError(f'Configuration file "{path}" does not exist.')
        return LoadedConfig(path, False, ConfigDocument())
    try:
        info = path.lstat()
    except OSError as error:
        raise ConfigError(f'Could not inspect configuration file "{path}": {error}.') from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ConfigError(f'Configuration file "{path}" must be a regular non-symlink file.')
    if os.name == "posix":
        if info.st_uid != os.geteuid():
            raise ConfigError(f'Configuration file "{path}" must be owned by the current user.')
        if info.st_mode & 0o022:
            raise ConfigError(f'Configuration file "{path}" must not be group or other writable.')
    try:
        with path.open("rb") as stream:
            parsed = tomllib.load(stream)
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f'Configuration file "{path}" is malformed TOML: {error}.') from error
    except OSError as error:
        raise ConfigError(f'Could not read configuration file "{path}": {error}.') from error
    return LoadedConfig(path, True, _parse(parsed))


def select_profile(
    loaded: LoadedConfig,
    explicit: str | None,
    environment: Mapping[str, str] = os.environ,
    *,
    ambient: bool = True,
) -> tuple[str | None, ProfileConfig | None]:
    name = explicit
    if name is None and ambient:
        if PROFILE_ENV in environment:
            name = environment[PROFILE_ENV]
            if not name:
                raise ConfigError(f"{PROFILE_ENV} selects an empty profile name.")
        else:
            name = loaded.document.default_profile
    if name is None:
        return None, None
    profile = loaded.document.profiles.get(name)
    if profile is None:
        raise ConfigError(f'Profile "{name}" is not defined in configuration.')
    return name, profile
