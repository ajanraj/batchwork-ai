# Repository Guidelines

## Project Structure & Module Organization

The Python package uses a `src/` layout. Public APIs live in `src/batchwork/`; provider adapters belong in `providers/`, server transport in `server/`, and persistence in `stores/`. Mirror these areas under `tests/`; keep shared fixtures in `tests/fixtures/` and credential-backed cases in `tests/live/`. The Blume site lives in `docs/`: MDX content under `docs/docs/`, Astro components under `docs/components/`, and static assets under `docs/public/`.

## Build, Test, and Development Commands

- `uv sync --dev`: install the package and locked development dependencies.
- `uv run pytest`: run the default quiet test suite.
- `uv run pytest tests/providers/test_adapters.py`: run one focused test module.
- `uv run ruff check .` and `uv run ruff format --check .`: lint and verify formatting.
- `uv run ty check`: type-check `src/`.
- `uv build`: produce source and wheel artifacts.
- `cd docs && bun install && bun run dev`: serve documentation locally.
- `cd docs && bun run build`: validate the production documentation build.
- `bunx prettier@3.6.2 --prose-wrap never --write README.md "docs/docs/**/*.mdx" tests/live/README.md`: format all tracked Markdown and MDX documentation after editing it.

Live provider tests require `BATCHWORK_RUN_LIVE=1`, provider credentials, and model variables documented in `tests/live/README.md`; they may incur cost.

## Coding Style & Naming Conventions

Target Python 3.11+ and preserve the typed public interface. Use four-space indentation, a 100-character limit, `snake_case` for functions/modules, and `PascalCase` for classes. Ruff enforces imports, modernization, bugbear, and async rules. Keep shared behavior provider-neutral; isolate vendor serialization in its adapter.

## Testing Guidelines

Use pytest and `pytest-asyncio` (`asyncio_mode = "auto"`). Name files `test_<subject>.py` and tests `test_<behavior>`. Add regression coverage beside the closest existing test. Mark external acceptance cases `live` and Redis-dependent cases `redis`; keep ordinary tests deterministic and credential-free.

## Releases & Deployments

Documentation deploys automatically to Cloudflare Pages on push; do not run `bun run deploy` during routine contribution. Format and build docs before pushing.

PyPI publishing is tag-driven through `.github/workflows/publish.yml`. Set the project version, then run `uv build`, `uvx twine check dist/*`, and `uv run python tools/verify_artifacts.py`. Push a matching `v<version>` tag (for example, `v0.1.2`); a mismatch aborts publishing. GitHub Actions publishes through trusted OIDC in the `pypi` environment. Do not upload manually or store PyPI tokens.

## Commit & Pull Request Guidelines

Use Conventional Commits with an imperative summary, such as `fix: preserve provider error details`. Pull requests should explain behavior and rationale, link issues, list verification commands, and note API/provider/docs impact. Include screenshots for visible docs changes. Never commit credentials or secret-bearing live-test output.

## Agent skills

### Issue tracker

Issues and PRDs are tracked in GitHub Issues for this repository. See `docs/agents/issue-tracker.md`.

### Triage labels

Use the canonical `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, and `wontfix` labels. See `docs/agents/triage-labels.md`.

### Domain docs

Use the single-context domain-doc layout. See `docs/agents/domain.md`.
