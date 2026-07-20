# OpenAI Build Week — judge demo

This folder is a self-contained demo project for the **Batchwork** submission to [OpenAI Build Week](https://openai.devpost.com/) (track: **Developer Tools**). It installs the published [`batchwork-ai`](https://pypi.org/project/batchwork-ai/) package from PyPI, so you can test the project here without touching or building the main repository. It exists only for the hackathon and will be removed afterwards.

Batchwork is a unified async batch API for OpenAI, Anthropic, Google Gemini, Groq, Mistral, Together AI, and xAI — one typed Python interface and one CLI for provider-native batch jobs, which providers price up to 50% below synchronous calls. Full docs: [batchwork.ajanraj.com](https://batchwork.ajanraj.com).

## Setup

Requirements: Python 3.11+ and [uv](https://docs.astral.sh/uv/) on macOS, Linux, or Windows.

```bash
cd hackathon
uv sync
cp .env.example .env   # then paste your OPENAI_API_KEY
```

Both demos default to OpenAI with `gpt-5.4-nano`; total token cost is a fraction of a cent. Batch processing is asynchronous on the provider side — it usually completes within a few minutes, but providers allow up to 24 hours. Interrupting a demo never cancels the remote job.

## Demo 1: the CLI

Submit four prompts as one batch, then check on it:

```bash
uv run --env-file .env batchwork submit text prompts.txt \
  --model openai/gpt-5.4-nano --name demo
uv run --env-file .env batchwork status demo
uv run --env-file .env batchwork results demo
```

Or do all three in one blocking command:

```bash
uv run --env-file .env batchwork run text prompts.txt --model openai/gpt-5.4-nano
```

The CLI is agent-friendly by design: pipe-aware output (human when interactive, JSON/JSONL when redirected), a versioned machine schema, a local job registry with aliases, and a documented exit-code catalog. Try `uv run batchwork --help`.

### Or install the CLI globally

The same CLI works as a standalone tool, no project or Python code required:

```bash
uv tool install batchwork-ai
export OPENAI_API_KEY="sk-..."

batchwork run text prompts.txt --model openai/gpt-5.4-nano
```

`run` submits, waits, and prints results in one blocking command; local interruption never cancels the remote job, and `batchwork list` finds it again.

## Demo 2: the Python package

```bash
uv run --env-file .env python demo_package.py
```

Submits the same kind of tiny batch through the typed async API, streams status while polling, and prints normalized results correlated by `custom_id`.

For a larger real-world example — a 300-request spam-classification evaluation on public data for about $0.007 — see the [classification guide](https://batchwork.ajanraj.com/docs/guides/classification).

## How Codex and GPT-5.6 built this project

The entire package was built in the Codex CLI with **GPT-5.6-sol** (with GPT-5.6-luna and GPT-5.6-terra subagents) over roughly five days. The main session — where the core functionality was built — is the one referenced by the `/feedback` session ID in the submission form (Codex session `019f67dd-3cab-75b1-b125-cf7f7d10dbf4`, ~55 prompts over three days). Its arc:

1. **Plan.** Starting prompt: port an existing TypeScript batch SDK to Python with complete parity. Codex split read-only discovery across three parallel Explore subagents (API surface, behavior, packaging), then ran a structured interview to lock every product decision — no assumptions — before producing the plan.
2. **Implement.** GPT-5.6-sol implemented the full package: seven provider adapters, text/embedding/image workloads, normalized jobs/results/usage/errors, stores, polling, and signed webhooks. A subagent migrated type checking from mypy to ty with nothing suppressed.
3. **Harden.** Repeated automated-review loops surfaced and fixed real P1/P2 issues: SSRF protection on injected media transports, Together's presigned upload protocol, credential-stripping on borrowed HTTP clients, Google inline embedding result parsing, webhook retry semantics, and timeout edge cases.
4. **Verify live.** Codex tested against real OpenAI and Gemini keys, ran the 300-comment classification evaluation end to end (90.3% accuracy, ~$0.007), and reproduced it from a clean temporary uv project installing only the published wheel.
5. **Ship.** Codex drove repo creation, PyPI trusted-publishing setup, the docs site, and Cloudflare Pages deployment.

Later GPT-5.6 sessions in the same workspace built the rest of the product, end to end: the `batchwork` CLI (submit/run/status/wait/results/cancel, the schema-versioned machine output contract, the SQLite job registry, TOML profiles, the exit-code catalog), the `batchwork-ai` Agent Skill, the documentation site, and the CI release pipeline. Codex handled the breadth (seven providers × three modalities × serialization quirks, plus a full CLI surface) that would normally make a project like this a multi-week effort; the human role was direction, review feedback, and live-key approval.

## Where key decisions were made

- **Provider-neutral core, vendor adapters:** all serialization quirks isolated per provider, decided during the planning interview.
- **Security as a feature:** SSRF-pinned media fetching, signed webhooks, and credential hygiene came out of the Codex review loops rather than being bolted on later.
- **Agent-first CLI:** machine schema versioning, never auto-retrying acceptance-ambiguous submissions, and explicit spend guards were designed for coding agents as first-class users.
