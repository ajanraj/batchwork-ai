"""Framework-neutral text submission workflow."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import click
from pydantic import TypeAdapter, ValidationError

from batchwork.body import build_text_bodies, validate_request_count
from batchwork.client import Batchwork
from batchwork.errors import BatchworkError
from batchwork.types import (
    BatchDefaults,
    BatchLimits,
    BatchProvider,
    JsonValue,
    ModelKind,
    ModelSpec,
    ProviderOptions,
    resolve_model,
)

from ._contract import (
    ErrorDetail,
    ErrorEnvelope,
    Job,
    JobEnvelope,
    Recovery,
    serialize_envelope,
)
from ._input import load_text_requests
from ._registry import RegistryRoute, default_registry_path, insert_job
from ._state import OutputMode, RootOptions

_ENVIRONMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_JOB_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_RECORD_ID = re.compile(r"^bw_[0-9a-f]{32}$")
_SENSITIVE_HEADERS = frozenset(
    {"authorization", "cookie", "proxy-authorization", "x-api-key", "api-key"}
)
_API_KEY_ENV = {
    BatchProvider.ANTHROPIC: "ANTHROPIC_API_KEY",
    BatchProvider.GOOGLE: "GOOGLE_GENERATIVE_AI_API_KEY",
    BatchProvider.GROQ: "GROQ_API_KEY",
    BatchProvider.MISTRAL: "MISTRAL_API_KEY",
    BatchProvider.OPENAI: "OPENAI_API_KEY",
    BatchProvider.TOGETHER: "TOGETHER_API_KEY",
    BatchProvider.XAI: "XAI_API_KEY",
}
_BASE_URL_ENV = {
    provider: f"{name.removesuffix('_API_KEY')}_BASE_URL" for provider, name in _API_KEY_ENV.items()
}
_JSON_OBJECT = TypeAdapter(dict[str, JsonValue])
_MAX_PROVIDER_OPTIONS_BYTES = 1024 * 1024


@dataclass(frozen=True, slots=True)
class ResolvedRoute:
    api_key: str
    base_url: str | None
    headers: dict[str, str]
    registry: RegistryRoute


@dataclass(frozen=True, slots=True)
class SubmissionResult:
    job: Job
    error: ErrorEnvelope | None = None


@dataclass(frozen=True, slots=True)
class SubmitTextOptions:
    source: Path
    model: str | None
    input_format: str | None
    name: str | None
    batch_metadata: Sequence[str]
    provider_options: str | None
    provider_options_file: Path | None
    allow_large_batch: bool
    base_url: str | None
    api_key_env: str | None
    header: Sequence[str]
    header_env: Sequence[str]
    system: str | None
    max_output_tokens: int | None
    temperature: float | None
    top_p: float | None
    top_k: int | None
    seed: int | None
    frequency_penalty: float | None
    presence_penalty: float | None
    stop: Sequence[str]
    tool_choice: str | None
    endpoint: str | None


def _usage(message: str) -> click.UsageError:
    return click.UsageError(message)


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON number {value}")


def _environment_value(name: str, purpose: str) -> str:
    if not _ENVIRONMENT_NAME.fullmatch(name):
        raise _usage(f'{purpose} environment variable name is invalid: "{name}".')
    value = os.environ.get(name)
    if not value:
        raise _usage(f'{purpose} environment variable "{name}" is missing or empty.')
    return value


def _normalized_base_url(value: str | None) -> str | None:
    if value is None:
        return None
    parsed = urlsplit(value)
    loopback = parsed.hostname in {"127.0.0.1", "::1", "localhost"}
    if (
        not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or (parsed.scheme != "https" and not (parsed.scheme == "http" and loopback))
    ):
        raise _usage(
            "--base-url must be absolute HTTPS (HTTP allowed for loopback), without "
            "userinfo, query, or fragment."
        )
    path = parsed.path.rstrip("/")
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def _key_value(values: Sequence[str], label: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        name, separator, item = value.partition("=")
        normalized = name.strip().lower()
        if not separator or not normalized or not item:
            raise _usage(f'{label} must use non-empty NAME=VALUE syntax: "{value}".')
        if normalized in parsed:
            raise _usage(f'Duplicate {label} name: "{name}".')
        parsed[normalized] = item
    return parsed


def _resolve_route(
    provider: BatchProvider,
    *,
    api_key_env: str | None,
    base_url: str | None,
    header: Sequence[str],
    header_env: Sequence[str],
) -> ResolvedRoute:
    selected_key_env = api_key_env or _API_KEY_ENV[provider]
    if provider is BatchProvider.GOOGLE and api_key_env is None:
        if not os.environ.get(selected_key_env) and os.environ.get("GEMINI_API_KEY"):
            selected_key_env = "GEMINI_API_KEY"
    api_key = _environment_value(selected_key_env, "API key")
    selected_base_url = base_url or os.environ.get(_BASE_URL_ENV[provider])
    normalized_base_url = _normalized_base_url(selected_base_url)
    literal_headers = _key_value(header, "--header")
    forbidden = _SENSITIVE_HEADERS.intersection(literal_headers)
    if forbidden:
        name = sorted(forbidden)[0]
        raise _usage(f'Header "{name}" may contain secrets; use --header-env.')
    header_variables = _key_value(header_env, "--header-env")
    overlap = literal_headers.keys() & header_variables.keys()
    if overlap:
        raise _usage(f'Header "{sorted(overlap)[0]}" is configured more than once.')
    resolved_headers = dict(literal_headers)
    for name, variable in header_variables.items():
        resolved_headers[name] = _environment_value(variable, f'Header "{name}"')
    descriptor = {
        "provider": provider.value,
        "base_url": normalized_base_url or "default",
        "api_key_env": selected_key_env,
        "headers": literal_headers,
        "header_env": header_variables,
    }
    canonical = json.dumps(
        descriptor, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode()
    registry_route = RegistryRoute(
        fingerprint=hashlib.sha256(canonical).hexdigest(),
        api_key_env=selected_key_env,
        base_url=normalized_base_url,
        headers=literal_headers,
        header_env=header_variables,
    )
    return ResolvedRoute(api_key, normalized_base_url, resolved_headers, registry_route)


def resolve_registered_route(provider: BatchProvider, route: RegistryRoute) -> ResolvedRoute:
    api_key = _environment_value(route.api_key_env, "API key")
    resolved_headers = dict(route.headers)
    for name, variable in route.header_env.items():
        resolved_headers[name] = _environment_value(variable, f'Header "{name}"')
    return ResolvedRoute(api_key, route.base_url, resolved_headers, route)


def _parse_json_object(document: str, label: str) -> dict[str, JsonValue]:
    if len(document.encode()) > _MAX_PROVIDER_OPTIONS_BYTES:
        raise _usage(f"{label} exceeds the {_MAX_PROVIDER_OPTIONS_BYTES} byte limit.")
    try:
        json.loads(document, parse_constant=_reject_json_constant)
        return _JSON_OBJECT.validate_json(document)
    except json.JSONDecodeError as error:
        raise _usage(f"{label} is not valid JSON: {error.msg}.") from error
    except ValidationError as error:
        raise _usage(f"{label} must be a JSON object containing valid JSON values.") from error
    except ValueError as error:
        raise _usage(f"{label} is not valid JSON: {error}.") from error


def _provider_options(
    provider: BatchProvider,
    inline: str | None,
    source: Path | None,
) -> ProviderOptions | None:
    if inline is not None and source is not None:
        raise _usage("--provider-options and --provider-options-file are mutually exclusive.")
    if source is not None:
        try:
            with source.open("rb") as stream:
                encoded = stream.read(_MAX_PROVIDER_OPTIONS_BYTES + 1)
            if len(encoded) > _MAX_PROVIDER_OPTIONS_BYTES:
                raise _usage(
                    f"Provider options exceed the {_MAX_PROVIDER_OPTIONS_BYTES} byte limit."
                )
            document = encoded.decode("utf-8-sig")
        except click.UsageError:
            raise
        except (OSError, UnicodeError) as error:
            raise _usage(f'Could not read --provider-options-file "{source}": {error}.') from error
    elif inline is not None:
        document = inline
    else:
        return None
    selected = _parse_json_object(document, "Provider options")
    return {provider.value: selected}


def _metadata(values: Sequence[str]) -> dict[str, str] | None:
    parsed = _key_value(values, "--batch-metadata")
    return parsed or None


def _model_spec(model: str, endpoint: str | None) -> ModelSpec:
    try:
        resolved = resolve_model(model)
    except (BatchworkError, ValueError) as error:
        raise _usage(str(error)) from error
    if endpoint is None:
        return resolved
    kinds = {
        "chat-completions": ModelKind.CHAT,
        "responses": ModelKind.RESPONSES,
        "completions": ModelKind.COMPLETION,
    }
    return resolved.model_copy(update={"kind": kinds[endpoint]})


def _defaults(
    *,
    provider_options: ProviderOptions | None,
    system: str | None,
    max_output_tokens: int | None,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    seed: int | None,
    frequency_penalty: float | None,
    presence_penalty: float | None,
    stop: Sequence[str],
    tool_choice: str | None,
) -> BatchDefaults:
    try:
        return BatchDefaults(
            provider_options=provider_options,
            system=system,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop_sequences=list(stop) or None,
            tool_choice=tool_choice,
        )
    except ValidationError as error:
        raise _usage(
            f"Invalid text defaults: {error.errors(include_url=False)[0]['msg']}."
        ) from error


async def submit_text(
    root: RootOptions,
    options: SubmitTextOptions,
) -> SubmissionResult:
    if options.model is None:
        raise _usage("--model is required for submit text.")
    spec = _model_spec(options.model, options.endpoint)
    if options.name is not None and (
        not _JOB_NAME.fullmatch(options.name) or _RECORD_ID.fullmatch(options.name)
    ):
        raise _usage("--name must be 1-64 shell-safe characters and cannot be a record ID.")
    route = _resolve_route(
        spec.provider,
        api_key_env=options.api_key_env,
        base_url=options.base_url,
        header=options.header,
        header_env=options.header_env,
    )
    requests = load_text_requests(
        options.source,
        options.input_format,
        stdin=click.get_binary_stream("stdin") if options.source == Path("-") else None,
    )
    defaults = _defaults(
        provider_options=_provider_options(
            spec.provider, options.provider_options, options.provider_options_file
        ),
        system=options.system,
        max_output_tokens=options.max_output_tokens,
        temperature=options.temperature,
        top_p=options.top_p,
        top_k=options.top_k,
        seed=options.seed,
        frequency_penalty=options.frequency_penalty,
        presence_penalty=options.presence_penalty,
        stop=options.stop,
        tool_choice=options.tool_choice,
    )
    limits = BatchLimits(
        max_upload_bytes=(200 * 1024 * 1024 if options.allow_large_batch else 50 * 1024 * 1024)
    )
    try:
        validate_request_count(requests, limits)
        built = build_text_bodies(
            spec.provider, spec.model_id, requests, defaults, limits, kind=spec.kind
        )
    except (BatchworkError, ValueError) as error:
        raise _usage(str(error)) from error
    if len(built) > 10_000 and not options.allow_large_batch:
        raise _usage(
            "More than 10,000 requests requires explicit cost authorization with "
            "--allow-large-batch."
        )

    async with Batchwork() as client:
        job = await client.batch(
            model=spec,
            requests=requests,
            defaults=defaults,
            metadata=_metadata(options.batch_metadata),
            limits=limits,
            api_key=route.api_key,
            base_url=route.base_url,
            headers=route.headers,
        )
    registered_at = datetime.now(UTC)
    registry_path = root.registry or default_registry_path()
    canonical_model = f"{spec.provider.value}/{spec.model_id}"
    try:
        registered_job = insert_job(
            registry_path,
            name=options.name,
            model=canonical_model,
            profile=root.profile,
            route=route.registry,
            snapshot=job.snapshot,
            registered_at=registered_at,
        )
    except (OSError, sqlite3.Error):
        provider_reference = f"{job.provider.value}:{job.id}"
        direct_job = Job(
            provider=job.provider,
            provider_job_id=job.id,
            provider_reference=provider_reference,
            routing_fingerprint=route.registry.fingerprint,
            modality="text",
            model=canonical_model,
            profile=root.profile,
            status=job.status,
            request_counts=job.request_counts,
            provider_created_at=job.snapshot.created_at,
        )
        error = ErrorEnvelope(
            error=ErrorDetail(
                code="registry_write_failed_after_submit",
                category="local_state",
                message=(
                    "The provider accepted the batch, but Batchwork could not record it locally."
                ),
                exit_code=8,
                retryable=False,
                operation="submit",
                provider=job.provider,
                job=provider_reference,
                routing_fingerprint=route.registry.fingerprint,
                profile=root.profile,
                registry_path=str(registry_path),
                submission_outcome="accepted",
                partial_output=True,
                records_emitted=1,
                recovery=Recovery(
                    action="resume_with_direct_reference",
                    command=_direct_recovery_command(
                        provider_reference, job.provider, root, route.registry
                    ),
                ),
            )
        )
        return SubmissionResult(direct_job, error)
    return SubmissionResult(registered_job)


def _direct_recovery_command(
    provider_reference: str,
    provider: BatchProvider,
    root: RootOptions,
    route: RegistryRoute,
) -> list[str]:
    command = ["batchwork"]
    if root.profile:
        command.extend(["--profile", root.profile])
    command.extend(["status", provider_reference])
    if not root.profile:
        command.extend(["--api-key-env", route.api_key_env])
        if route.base_url:
            command.extend(["--base-url", route.base_url])
        for name, value in sorted(route.headers.items()):
            command.extend(["--header", f"{name}={value}"])
        for name, variable in sorted(route.header_env.items()):
            command.extend(["--header-env", f"{name}={variable}"])
    return command


def render_job(job: Job, mode: OutputMode) -> str:
    if mode in {OutputMode.JSON, OutputMode.JSONL}:
        return serialize_envelope(JobEnvelope(job=job))
    selector = job.record_id or job.provider_reference
    lines = [
        "Submitted text batch",
        f"Job: {selector}",
        f"Provider: {job.provider.value}",
        f"Reference: {job.provider_reference}",
        f"Status: {job.status.value if job.status else 'unknown'}",
        f"Resume: batchwork status {selector}",
    ]
    return "\n".join(lines) + "\n"


def render_error(error: ErrorEnvelope, mode: OutputMode) -> str:
    if mode in {OutputMode.JSON, OutputMode.JSONL}:
        return serialize_envelope(error)
    recovery = error.error.recovery
    command = " ".join(recovery.command) if recovery and recovery.command else "unavailable"
    return f"Error: {error.error.message}\nRecovery: {command}\n"
