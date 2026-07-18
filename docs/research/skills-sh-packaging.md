# skills.sh skill packaging and discovery

Research for [batchwork-ai issue #3](https://github.com/ajanraj/batchwork-ai/issues/3). As of 2026-07-18. Research only; no Batchwork skill authored.

Upstream baselines:

- `skills` CLI `v1.5.19`, commit [`777599e`](https://github.com/vercel-labs/skills/tree/777599e1159e401b11ce4c8a57c20f09a8f1596e), released 2026-07-16.
- Agent Skills specification commit [`38a2ff8`](https://github.com/agentskills/agentskills/tree/38a2ff82958afee88dadf4831509e6f7e9d8ef4e).
- Vercel agent-skills example repository commit [`f8a72b9`](https://github.com/vercel-labs/agent-skills/tree/f8a72b9603728bb92a217a879b7e62e43ad76c81).

## Recommended next decision

Adopt this future repository shape, but defer authoring until the CLI contract is settled:

```text
skills/
└── batchwork-ai/
    ├── SKILL.md
    ├── references/   # detailed CLI contracts and safe operating guidance, if needed
    ├── scripts/      # only deterministic helpers that materially improve reliability
    └── assets/       # only templates/static inputs agents need
```

Decide now:

1. Public skill identifier: `batchwork-ai`; directory and `name` must match exactly.
2. Public install command: `npx skills add ajanraj/batchwork-ai@batchwork-ai`; also document `--skill batchwork-ai` form.
3. Default release channel: repository default branch. Offer Git-tag-pinned installs for reproducibility; do not treat `metadata.version` as package-manager versioning.
4. Portability target: normative Agent Skills fields and plain relative references only. Avoid hooks, `context: fork`, and reliance on `allowed-tools` enforcement.
5. Publication trigger: public GitHub repository plus a successful non-CI CLI install. No manual skills.sh submission step.
6. Validation gate: full Agent Skills reference validation plus `npx skills add ./skills --list`. The CLI discovery check alone is insufficient.

Do **not** place `SKILL.md` at repository root. On the normal clone path, the CLI recursively copies a root skill directory, which would package the whole Batchwork repository. A nested `skills/batchwork-ai/` directory scopes installation to the skill payload. See [`add.ts`](https://github.com/vercel-labs/skills/blob/777599e1159e401b11ce4c8a57c20f09a8f1596e/src/add.ts#L1718-L1727) and the root-snapshot exception in [`blob.ts`](https://github.com/vercel-labs/skills/blob/777599e1159e401b11ce4c8a57c20f09a8f1596e/src/blob.ts#L561-L569).

## Format and metadata constraints

The open [Agent Skills specification](https://agentskills.io/specification) is the normative format. A skill is a directory containing `SKILL.md`; `scripts/`, `references/`, `assets/`, and arbitrary additional files are allowed. See the [commit-pinned specification](https://github.com/agentskills/agentskills/blob/38a2ff82958afee88dadf4831509e6f7e9d8ef4e/docs/specification.mdx#L6-L17).

| Field | Status | Constraint / implication |
| --- | --- | --- |
| `name` | Required | 1–64 characters; lowercase alphanumeric plus single hyphens; no leading, trailing, or consecutive hyphens; must equal the parent directory name. `batchwork-ai` is valid. |
| `description` | Required | 1–1024 characters; state both capability and when to use it; include trigger keywords because clients load it during discovery. |
| `license` | Optional | Short license identifier or reference to a bundled license file. Use the repository license unless skill-specific terms are required. |
| `compatibility` | Optional | 1–500 characters; only for actual environment requirements such as the future `batchwork-ai` CLI, network access, credentials, or supported operating systems. |
| `metadata` | Optional | String-to-string extension map. Namespaced keys reduce collisions. `version` is only conventional metadata, not a skills CLI release selector. |
| `allowed-tools` | Optional, experimental | Space-separated hint. Support and enforcement vary by client; never use it as a security boundary. |

The reference validator rejects unknown top-level frontmatter fields. Put extensions under `metadata`, not beside the standard fields. See [`validator.py`](https://github.com/agentskills/agentskills/blob/38a2ff82958afee88dadf4831509e6f7e9d8ef4e/skills-ref/src/skills_ref/validator.py#L10-L22) and its [name/directory checks](https://github.com/agentskills/agentskills/blob/38a2ff82958afee88dadf4831509e6f7e9d8ef4e/skills-ref/src/skills_ref/validator.py#L25-L67).

The Vercel CLI is intentionally more permissive: discovery only requires string `name` and `description`; it does not enforce the normative name syntax, directory match, allowed-field set, or length limits. See [`parseSkillMd`](https://github.com/vercel-labs/skills/blob/777599e1159e401b11ce4c8a57c20f09a8f1596e/src/skills.ts#L69-L105). Therefore an installable skill can still be non-conformant and fail in stricter clients.

## Supporting files and progressive disclosure

The specification's context model is:

1. Clients load `name` and `description` for discovery, approximately 100 tokens.
2. Clients load the entire `SKILL.md` body after activation; under 5,000 tokens is recommended.
3. Clients load scripts, references, and assets only as needed.

Keep `SKILL.md` under 500 lines. Use relative paths from the skill root and keep references one level deep; avoid chains of references. See [progressive disclosure and file-reference guidance](https://github.com/agentskills/agentskills/blob/38a2ff82958afee88dadf4831509e6f7e9d8ef4e/docs/specification.mdx#L214-L245) and the [Agent Skills overview](https://agentskills.io/what-are-skills).

For Batchwork:

- Keep activation criteria, safety boundaries, and the normal CLI workflow in `SKILL.md`.
- Move provider/modality matrices, input schemas, output examples, and troubleshooting into focused `references/` files only after those CLI contracts exist.
- Avoid duplicating CLI orchestration in scripts. Add a script only when it is deterministic, portable, testable, and materially safer than direct CLI usage.
- Keep all support files inside `skills/batchwork-ai/`; the installer recursively copies that directory and hashes all files for updates. See [`add.ts`](https://github.com/vercel-labs/skills/blob/777599e1159e401b11ce4c8a57c20f09a8f1596e/src/add.ts#L1718-L1727) and [`local-lock.ts`](https://github.com/vercel-labs/skills/blob/777599e1159e401b11ce4c8a57c20f09a8f1596e/src/local-lock.ts#L112-L156).

## Repository discovery and installation

The current CLI accepts GitHub shorthand, full GitHub URLs, direct repository subpaths, GitLab URLs, arbitrary git URLs, local paths, and well-known web sources. See the [official CLI README](https://github.com/vercel-labs/skills/blob/777599e1159e401b11ce4c8a57c20f09a8f1596e/README.md#L28-L88) and [`source-parser.ts`](https://github.com/vercel-labs/skills/blob/777599e1159e401b11ce4c8a57c20f09a8f1596e/src/source-parser.ts#L139-L408).

Default repository discovery:

- A root `SKILL.md` is treated as the sole skill unless `--full-depth` is used.
- Standard containers include `skills/`, `skills/.curated/`, `skills/.experimental/`, `skills/.system/`, and many agent-specific project directories.
- Standard containers support `container/<skill>/SKILL.md` and `container/<category>/<skill>/SKILL.md`.
- A shallower `SKILL.md` shadows nested skills below it.
- If standard discovery finds nothing, the CLI recursively searches to depth five. `--full-depth` forces recursive search too.
- Claude plugin manifests can declare deeper skill paths.
- Duplicate frontmatter names are deduplicated by first discovery order.
- Project-installed skills recorded in `skills-lock.json` are skipped as authored-source candidates.

Source: [`discoverSkills`](https://github.com/vercel-labs/skills/blob/777599e1159e401b11ce4c8a57c20f09a8f1596e/src/skills.ts#L150-L295) and the [documented search locations](https://github.com/vercel-labs/skills/blob/777599e1159e401b11ce4c8a57c20f09a8f1596e/README.md#L369-L458).

`skills/batchwork-ai/SKILL.md` is therefore the least surprising layout: standard, directly selectable, compatible with a multi-skill repository later, and isolated from installed third-party skills.

Installation behavior:

- Project scope is default; `-g` selects global scope.
- `--agent` targets clients; `--skill` selects one or more skills; `--list` discovers without installing; `--copy` avoids symlinks; `-y` is non-interactive.
- The installer uses `.agents/skills/<name>` as the canonical project copy for most clients, then creates agent-specific links/copies as needed. Claude Code's project path is `.claude/skills/`; Eve uses `agent/skills/`.
- Project installation writes `skills-lock.json`, intended for version control. It records source, optional ref, source type, `SKILL.md` path, and a whole-folder content hash.
- Global installation uses `$XDG_STATE_HOME/skills/.skill-lock.json` or `~/.agents/.skill-lock.json`.

Sources: [installation options](https://github.com/vercel-labs/skills/blob/777599e1159e401b11ce4c8a57c20f09a8f1596e/README.md#L50-L105), [agent paths](https://github.com/vercel-labs/skills/blob/777599e1159e401b11ce4c8a57c20f09a8f1596e/README.md#L238-L323), [`installer.ts`](https://github.com/vercel-labs/skills/blob/777599e1159e401b11ce4c8a57c20f09a8f1596e/src/installer.ts#L285-L380), [`local-lock.ts`](https://github.com/vercel-labs/skills/blob/777599e1159e401b11ce4c8a57c20f09a8f1596e/src/local-lock.ts#L5-L59), and [`skill-lock.ts`](https://github.com/vercel-labs/skills/blob/777599e1159e401b11ce4c8a57c20f09a8f1596e/src/skill-lock.ts#L62-L103).

The CLI also supports domain-hosted publication through `https://<host>/.well-known/agent-skills/index.json`, with the older `/.well-known/skills/index.json` as fallback. The current v0.2.0 index uses schema `https://schemas.agentskills.io/discovery/0.2.0/schema.json` and per-skill artifact URL plus digest. GitHub is simpler for Batchwork and already matches repository ownership. See [`wellknown.ts`](https://github.com/vercel-labs/skills/blob/777599e1159e401b11ce4c8a57c20f09a8f1596e/src/providers/wellknown.ts#L7-L99).

## skills.sh publication and discovery

There is no upload, package-publish, or self-service registration API. [skills.sh About](https://skills.sh/about) states that it indexes every public skill that ships through the open CLI and ranks skills from anonymous, deduplicated install counts. A Vercel maintainer confirms: run `npx skills add <your-repo>`; indexing is automatic and no manual listing request is needed ([issue #1017](https://github.com/vercel-labs/skills/issues/1017#issuecomment-4331100908)).

The source path is concrete:

- A successful public remote install sends source, skill names, target agents, and relative `SKILL.md` paths to the telemetry endpoint. Private GitHub repositories are excluded. See [`add.ts`](https://github.com/vercel-labs/skills/blob/777599e1159e401b11ce4c8a57c20f09a8f1596e/src/add.ts#L1743-L1805).
- Telemetry can be disabled with `DISABLE_TELEMETRY` or `DO_NOT_TRACK`; CI is marked but current source still sends unless explicitly disabled. The README says telemetry is automatically disabled in CI, but `isEnabled()` does not inspect CI; treat this as a documentation/source discrepancy. See [`telemetry.ts`](https://github.com/vercel-labs/skills/blob/777599e1159e401b11ce4c8a57c20f09a8f1596e/src/telemetry.ts#L71-L86) and [README telemetry wording](https://github.com/vercel-labs/skills/blob/777599e1159e401b11ce4c8a57c20f09a8f1596e/README.md#L488-L505).
- `npx skills find <query>` searches skills.sh and sorts by installs. The current CLI uses the unauthenticated legacy endpoint `/api/search`. See [`find.ts`](https://github.com/vercel-labs/skills/blob/777599e1159e401b11ce4c8a57c20f09a8f1596e/src/find.ts#L16-L112).
- The documented `/api/v1/` catalog API supports leaderboard, search, curated skills, detail, and audit reads, but requires a Vercel OIDC token. It has no write/publication endpoint. See [skills.sh API docs](https://skills.sh/docs/api).

Concrete publication flow for the future skill:

1. Merge `skills/batchwork-ai/` to the public default branch.
2. Validate locally.
3. Run a real local `npx skills add ajanraj/batchwork-ai@batchwork-ai --list`, then install in a temporary project or global test environment without telemetry opt-out to trigger indexing.
4. Confirm the skill detail page and `npx skills find batchwork` result after ingestion; timing is not guaranteed or documented.
5. Add a skills.sh badge/link to the repository README only if desired.

Optional `skills.sh.json` at repository root customizes listing groups only; it does not affect CLI discovery or installation. It requires 1–50 groups, each with a title and 1–500 skill slugs; unknown skills are ignored. A single Batchwork skill does not justify this file yet. Sources: [customization docs](https://skills.sh/docs/customize) and [JSON schema](https://skills.sh/schemas/skills.sh.schema.json).

Vercel exposes two discovery surfaces:

- `npx skills find <query>` searches the open skills.sh directory.
- `vercel skills [query]` searches the catalog; without a query it detects the framework and scans `package.json` for curated notable dependencies before recommending skills. Batchwork is a Python package, so do not rely on automatic dependency matching; explicit docs and skills.sh search remain primary. See [Vercel Agent Skills](https://vercel.com/docs/agent-resources/skills) and [`vercel skills`](https://vercel.com/docs/cli/skills).

## Naming and versioning

Naming:

- Use `batchwork-ai`, matching the repository/package identity and normative directory rule.
- Avoid a generic `batchwork` alias because install selection and catalog identity should remain unambiguous.
- Make the description include likely triggers such as batch inference, asynchronous provider jobs, submit, retrieve, wait, results, cancel, embeddings, and images—but finalize only after CLI commands are decided.

Versioning:

- The Agent Skills spec defines no normative package version or dependency-resolution field.
- `metadata.version` is an arbitrary client extension and is not used by the skills CLI for selection or updates.
- The CLI accepts branch/tag/commit refs via `#ref` and persists the ref in project/global lock entries. Updates reconstruct the source with the same ref. See [`source-parser.ts`](https://github.com/vercel-labs/skills/blob/777599e1159e401b11ce4c8a57c20f09a8f1596e/src/source-parser.ts#L202-L238), [`update-source.ts`](https://github.com/vercel-labs/skills/blob/777599e1159e401b11ce4c8a57c20f09a8f1596e/src/update-source.ts#L17-L21), and [lock fields](https://github.com/vercel-labs/skills/blob/777599e1159e401b11ce4c8a57c20f09a8f1596e/src/local-lock.ts#L15-L44).
- Unpinned installs follow the repository's current default-branch content when refreshed. Pinned installs remain pinned; they do not float to a newer release tag.

Recommended policy: keep the default install command unpinned for normal users; publish matching Git tags with Batchwork releases and document a deterministic form such as `npx skills add ajanraj/batchwork-ai#v0.1.0@batchwork-ai` for automation or reproducible environments. Do not create an independent skill version unless the skill lifecycle demonstrably diverges from the CLI.

## Cross-agent compatibility

The shared format provides broad basic portability, but runtime semantics are client-owned. The current skills CLI advertises 70+ target agents and documents feature differences: basic skills are widely supported, `allowed-tools` is not universal, and Claude-specific `context: fork` and hooks are not portable. See the [compatibility table](https://github.com/vercel-labs/skills/blob/777599e1159e401b11ce4c8a57c20f09a8f1596e/README.md#L460-L470) and the [Agent Skills client showcase](https://agentskills.io/clients).

Batchwork compatibility rules:

- Write portable Markdown instructions first.
- State actual executable/network/credential requirements in `compatibility`; do not encode agent brand preference unless required.
- Treat environment variables and provider credentials as runtime prerequisites; never bundle secrets.
- Avoid client-specific hooks, subagents, fork context, slash commands, or tool names in the portable core.
- If an agent-specific enhancement becomes necessary later, isolate it as an optional adapter rather than changing the portable skill contract.

## Validation evidence

Temporary fixture verification against CLI `1.5.19` and the specification reference implementation:

```text
$ npx --yes skills@1.5.19 add /tmp/batchwork-skill-validation --list
Found 2 skills
Bad_Name
valid-skill
```

The CLI accepted `Bad_Name`, a mismatched directory, and unknown `extra-field` because it only checks string `name` and `description`.

```text
$ uv run --project /tmp/batchwork-agent-skills-spec/skills-ref skills-ref validate .../valid-skill
Valid skill

$ uv run --project /tmp/batchwork-agent-skills-spec/skills-ref skills-ref validate .../mismatched-directory
Validation failed:
- Unexpected fields in frontmatter: extra-field
- Skill name 'Bad_Name' must be lowercase
- Skill name 'Bad_Name' contains invalid characters
- Directory name 'mismatched-directory' must match skill name 'Bad_Name'
```

The `skills-ref` repository labels itself demonstration-only, not production software. Pin its upstream revision if used in CI, or encode the small normative checks locally later. See its [README](https://github.com/agentskills/agentskills/blob/38a2ff82958afee88dadf4831509e6f7e9d8ef4e/skills-ref/README.md#L1-L7).

Additional checks:

- `npx --yes skills@1.5.19 add vercel-labs/agent-skills --list`: passed; discovered nine current Vercel skills.
- `GET https://skills.sh/api/search?q=web-design&limit=2`: passed; returned ranked skill records.
- `GET https://skills.sh/api/v1/skills/search?q=web-design&limit=2` without OIDC: failed as documented with `HTTP 401` and `authentication_required`.

## Constraints and blockers

- skills.sh ingestion/index refresh latency is not documented; automatic publication is not guaranteed to be immediate.
- The CLI's permissive parser does not prove conformance. Always run normative validation separately.
- The reference validator is demonstration-only and currently accepts lowercase `skill.md`, while the specification names uppercase `SKILL.md`; author uppercase for maximum compatibility.
- skills.sh ranking depends on telemetry; installs with telemetry disabled cannot contribute to indexing/ranking.
- The skills.sh `/api/v1` API is read-only and OIDC-authenticated; it cannot publish or force re-indexing.
- The future skill's exact references, examples, compatibility string, and triggers remain blocked on the CLI command/input/output and persistence decisions in the parent wayfinding map.

## Authoritative sources

- [Agent Skills specification](https://agentskills.io/specification)
- [Agent Skills overview](https://agentskills.io/what-are-skills)
- [Agent Skills source and reference validator](https://github.com/agentskills/agentskills/tree/38a2ff82958afee88dadf4831509e6f7e9d8ef4e)
- [skills CLI README and source, v1.5.19](https://github.com/vercel-labs/skills/tree/777599e1159e401b11ce4c8a57c20f09a8f1596e)
- [skills.sh About](https://skills.sh/about)
- [skills.sh overview](https://skills.sh/docs)
- [skills.sh CLI docs](https://skills.sh/docs/cli)
- [skills.sh customization docs](https://skills.sh/docs/customize)
- [skills.sh API docs](https://skills.sh/docs/api)
- [skills.sh FAQ](https://skills.sh/docs/faq)
- [skills.sh JSON schema](https://skills.sh/schemas/skills.sh.schema.json)
- [Vercel Agent Skills docs](https://vercel.com/docs/agent-resources/skills)
- [Vercel `vercel skills` docs](https://vercel.com/docs/cli/skills)
- [Vercel's official agent-skills repository](https://github.com/vercel-labs/agent-skills/tree/f8a72b9603728bb92a217a879b7e62e43ad76c81)
- [Maintainer confirmation of automatic skills.sh indexing](https://github.com/vercel-labs/skills/issues/1017)
