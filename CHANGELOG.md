# Changelog

All notable changes to `batchwork-ai` are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [semantic versioning](https://semver.org/).

## [Unreleased]

## [0.2.3] - 2026-07-23

### Changed

- Python `Batchwork` submissions now reject unsupported canonical settings, provider options, collisions, and non-empty unsupported submission-level batch metadata instead of silently discarding them. Empty metadata mappings are treated as omitted, and low-level body builders retain their explicit `strict` control.
- `BatchJob.wait()` now requires a positive finite polling interval and a finite non-negative timeout, and retries transient provider reads up to three total attempts while honoring bounded `Retry-After` delays.
- Terminal `BatchJob` snapshots are reused when retrieving results, avoiding redundant provider status requests.

### Fixed

- Malformed media `Content-Length` headers now fall back to the streamed-byte limit instead of leaking `ValueError`.
- xAI cancellation markers are normalized correctly, result pagination rejects repeated continuation tokens and duplicate result IDs, and a 10,000-page ceiling stops unique-token loops before another request.
- Unknown, missing, or malformed OpenAI-compatible and Mistral batch statuses now fail as non-retryable provider protocol errors instead of being treated as in progress.

### Security

- Intentionally hardened trusted custom provider base URLs: HTTPS is required except when an HTTP host is exactly `localhost` or a literal loopback IP; userinfo, query, and fragment components are rejected, and validation errors no longer echo rejected credential-bearing routing values.

## [0.2.2] - 2026-07-20

### Changed

- `batchwork list` human output now shows when each job was submitted and completed (UTC), in both the wide table and the narrow block layout. Machine output is unchanged; it already carried these timestamps.

## [0.2.1] - 2026-07-20

### Fixed

- Compressed provider responses (for example `Content-Encoding: gzip`) are no longer decompressed twice, which raised `httpx.DecodingError: incorrect header check` and made every affected request fail — including OpenAI file uploads during batch submission. The bounded response rebuild now drops the original `content-encoding` and `content-length` headers because the body is already decoded.

## [0.2.0] - 2026-07-20

This release adds the installable `batchwork` command-line tool and the portable Agent Skill, so batches can be submitted and managed from the terminal, from scripts, and by coding agents without writing Python.

### Added

- `batchwork` CLI installed with the package (`uv tool install batchwork-ai`), with `submit`, `run`, `status`, `wait`, `results`, and `cancel` for text, embedding, and image batches on all seven providers.
- Machine output contract (schema version 1): `--json` and `--jsonl` envelopes with snake_case keys, UTC timestamps, stable symbolic error codes, and stable process exit categories. The generated JSON Schema ships at `docs/public/schemas/batchwork-cli-v1.schema.json`.
- Local job registry: a private per-user SQLite database with immutable `bw_` record IDs, optional aliases, route-complete identity fingerprints, and `list`, `forget`, `prune`, and `registry check`/`registry reset --backup` commands.
- Non-secret TOML profiles (`config path`, `config validate`, `config show`) for model defaults, credential variable names, base URLs, and headers. Secrets stay environment-only.
- Input transports for creation commands: canonical JSON and JSONL, constrained CSV, and one-value-per-line text, with whole-source validation before any provider request.
- Explicit image materialization through `--output-dir`: atomic writes, deterministic filenames, a schema-v1 `manifest.json`, bounded HTTPS downloads, and no files without the flag.
- Soft volume gate (`--allow-large-batch`) above 10,000 requests, 50 MiB serialized upload, or 100 requested images, on top of the existing hard limits.
- Provider-neutral embedding settings and defaults in the Python API, including canonical `dimensions` support.
- The `batchwork-ai` Agent Skill (0.1.0), installable with `npx skills add ajanraj/batchwork-ai@batchwork-ai`, which teaches coding agents the CLI contract, authorization boundaries, and recovery flows.
- CLI documentation: quick starts, command selection, selectors, input formats, machine schema reference, exit catalog, and configuration/registry reference.

### Changed

- README and documentation now cover the CLI and Agent Skill alongside the Python API.

### Security

- Resolved credentials and secret headers are never persisted in configuration, the registry, or machine output; sensitive literal header names are rejected.
- Custom base URLs require HTTPS (loopback HTTP excepted) and reject userinfo, query, and fragment components.
- Image downloads revalidate redirect targets and resolved addresses, and never forward provider credentials or configured headers.

## [0.1.1] - 2026-07-18

First release published to PyPI as `batchwork-ai`.

### Added

- Typed async Python API for provider-native batch jobs on OpenAI, Anthropic, Google Gemini, Groq, Mistral, Together AI, and xAI.
- Text, embedding, and image batch workloads with normalized jobs, snapshots, results, usage, and errors correlated by `custom_id`.
- Messages, tools, structured content, and local/remote media inputs with safe media resolution.
- `BatchPoller`, in-memory and Upstash Redis stores, and signed completion webhooks for production polling.

## [0.1.0] - 2026-07-18

Initial tagged version; superseded by 0.1.1 for the PyPI name change.

[0.2.3]: https://github.com/ajanraj/batchwork-ai/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/ajanraj/batchwork-ai/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/ajanraj/batchwork-ai/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/ajanraj/batchwork-ai/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/ajanraj/batchwork-ai/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/ajanraj/batchwork-ai/releases/tag/v0.1.0
