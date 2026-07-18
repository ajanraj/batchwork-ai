# uv tool packaging and executable conventions

Research for [issue #2](https://github.com/ajanraj/batchwork-ai/issues/2), part of [map #1](https://github.com/ajanraj/batchwork-ai/issues/1). Researched 2026-07-18 against uv 0.11.29 and current Python Packaging Authority specifications. Implementation intentionally deferred.

## Recommended decision

Make the default published distribution directly installable as a tool:

```toml
[project.scripts]
batchwork = "batchwork.cli:main"
```

Then support and document these canonical forms:

```console
uv tool install batchwork-ai
batchwork --version

uvx --from batchwork-ai batchwork --version
uvx --from 'batchwork-ai==0.1.1' batchwork --version
```

Do not require a `cli` extra for the command to start. Any runtime dependency imported by `batchwork.cli` should be a normal `[project].dependencies` item. Preserve feature-specific dependencies such as Redis as extras:

```console
uv tool install 'batchwork-ai[redis]'
uvx --from 'batchwork-ai[redis]' batchwork ...
```

Reason: `uv tool install batchwork-ai` is the natural package-oriented installation command, while `uvx` is command-oriented and otherwise infers a distribution named `batchwork`. PyPI currently has no `batchwork` distribution, and `uvx batchwork` fails. The explicit `--from` form permanently binds the `batchwork` executable to the `batchwork-ai` distribution.

## Packaging contract

- Distribution name, import package, and executable name are independent:
  - distribution: `batchwork-ai`
  - import package: `batchwork`
  - executable: `batchwork`
- `[project.scripts]` maps the executable name to a `module:callable` object reference and produces a `console_scripts` entry point. The wrapper calls the function with no arguments. An integer return becomes the process exit status; `None` means success. [PyPA pyproject specification](https://packaging.python.org/en/latest/specifications/pyproject-toml/#entry-points), [entry-points specification](https://packaging.python.org/en/latest/specifications/entry-points/)
- `batchwork` is a valid and conventional command name. New entry-point names should use letters, numbers, underscores, dots, or dashes. Cross-distribution command collisions are installer-defined. [PyPA entry-points specification](https://packaging.python.org/en/latest/specifications/entry-points/)
- Distribution lookup normalizes case and every run of `.`, `-`, or `_` to `-`. Thus `batchwork-ai`, `batchwork_ai`, and case variants identify the same distribution, but this normalization does not rename the executable. [PyPA name-normalization specification](https://packaging.python.org/en/latest/specifications/name-normalization/)
- Entry-point extras such as `module:function [extra]` exist in the underlying entry-point format but are not recommended for new publishing use and consumers may ignore them. Do not use them to make CLI dependencies implicit. [PyPA entry-points specification](https://packaging.python.org/en/latest/specifications/entry-points/)
- Packaging creates the command wrapper, not a `--version` option. The CLI must define `batchwork --version`. Prefer installed distribution metadata as the version source instead of adding a third hard-coded copy beside `pyproject.toml` and `batchwork.__version__`.

## `uv tool install`

`uv tool install` operates on a distribution and installs every executable provided by that distribution into uv's executable directory. The package gets a persistent, isolated virtual environment. Its importable modules are not added to the active project environment. [uv tools guide, 0.11.29](https://github.com/astral-sh/uv/blob/0.11.29/docs/guides/tools.md), [uv tools concepts, 0.11.29](https://github.com/astral-sh/uv/blob/0.11.29/docs/concepts/tools.md)

Canonical commands:

```console
uv tool install batchwork-ai
uv tool install 'batchwork-ai==0.1.1'
uv tool install 'batchwork-ai[redis]'
uv tool install --python 3.11 batchwork-ai
```

Constraints and edge cases:

- Without a constraint, installation resolves the latest available release.
- The executable directory must be on `PATH`; uv warns otherwise. `uv tool update-shell` updates supported shell configuration. `uv tool dir --bin` reports the directory.
- uv refuses to overwrite an executable in its tool bin directory that uv did not install. `--force` overrides this protection. This matters if pipx or another package already owns `batchwork`.
- Only executables from the primary distribution are exposed. Executables from dependencies are not exposed unless explicitly requested with `--with-executables-from`.
- `--with PACKAGE` adds an extra distribution to the tool environment but does not expose that distribution's commands. It is distinct from requesting a target extra with `batchwork-ai[redis]`.
- `-e/--editable` supports local development installs. Release acceptance should test a built wheel, not only an editable source tree.
- A package with no executables is not a tool. On the current repository, uv 0.11.29 installs dependencies, then removes the environment and reports:

  ```text
  No executables are provided by package `batchwork-ai`; removing tool
  error: Failed to install entrypoints for `batchwork-ai`
  ```

- Tool commands ignore project-local uv configuration and use user/system configuration, unless a specific `--config-file` is supplied. Therefore repository-local index/source settings are not an end-user installation contract. [uv configuration files](https://docs.astral.sh/uv/concepts/configuration-files/)

## `uvx` / `uv tool run`

`uvx` and `uv tool run` are equivalent. They create a dependency environment isolated from the current project and cache it in uv's cache as disposable state. `uv cache clean` may remove it; uv recreates it on demand. [uv tools concepts, 0.11.29](https://github.com/astral-sh/uv/blob/0.11.29/docs/concepts/tools.md)

Because the executable and distribution names differ, use `--from`:

```console
uvx --from batchwork-ai batchwork ...
uvx --from 'batchwork-ai==0.1.1' batchwork ...
uvx --from 'batchwork-ai[redis]' batchwork ...
```

Do not document either of these as canonical:

```console
uvx batchwork
uvx batchwork-ai
```

The first asks uv to resolve a distribution named `batchwork`; PyPI returns 404 for that name as of 2026-07-18, and uv 0.11.29 reports that `batchwork` was not found. The second uses `batchwork-ai` as the inferred command name, but the intended entry point is `batchwork`. uv documents `uvx --from httpie http` for exactly this package/command mismatch. [uv tools guide, 0.11.29](https://github.com/astral-sh/uv/blob/0.11.29/docs/guides/tools.md), [PyPI `batchwork` endpoint](https://pypi.org/pypi/batchwork/json)

Version behavior:

- First unconstrained `uvx` execution resolves the latest available version, then normally reuses its cached environment.
- Once a tool is persistently installed, `uvx` reuses that installed version by default.
- `@latest` refreshes and explicitly requests the latest release when command and distribution names match, for example `uvx ruff@latest`. For Batchwork, the unambiguous exact-version form is `uvx --from 'batchwork-ai==0.1.1' batchwork`.
- `--isolated` ignores a persistently installed tool, but `@latest` is the explicit freshness request. Do not describe ordinary `uvx` as automatically fresh on every invocation.
- Use `uv run`, not `uvx`, when a command must import the current project or use its locked project environment. Batchwork's installed CLI should not depend on the caller's project environment.

Isolation here means Python dependency-environment isolation, not an operating-system sandbox. The command still has ordinary process access to inherited environment variables, the working directory, filesystem permissions, and network. Provider credentials can therefore remain environment-based; uv does not store or isolate those secrets.

## Upgrades

```console
uv tool upgrade batchwork-ai
uv tool upgrade --all
uv tool upgrade --python 3.12 batchwork-ai
```

- Upgrade identity is the normalized distribution name, not the executable name. Use `batchwork-ai`, not `batchwork`.
- Upgrades preserve the installation's version constraints and settings, including prerelease policy. An exact install remains exact.
- To replace constraints, reinstall with a new requirement, for example `uv tool install 'batchwork-ai>=0.2'`.
- `--reinstall` rebuilds the full environment; `--reinstall-package NAME` targets one package.
- No automatic upgrade occurs merely because a newer release exists.
- Each tool environment is tied to a Python interpreter. `--python` can select or change it. Removing that interpreter can break the tool.

[uv tools concepts, 0.11.29](https://github.com/astral-sh/uv/blob/0.11.29/docs/concepts/tools.md), [uv tools guide, 0.11.29](https://github.com/astral-sh/uv/blob/0.11.29/docs/guides/tools.md)

## Optional dependencies and lock boundaries

- `[project.optional-dependencies]` is published wheel metadata. Extras are activated only when requested with requirement syntax such as `batchwork-ai[redis]`. Multiple extras form a union. [PyPA pyproject specification](https://packaging.python.org/en/latest/specifications/pyproject-toml/#dependencies-optional-dependencies), [dependency-specifier specification](https://packaging.python.org/en/latest/specifications/dependency-specifiers/)
- `[dependency-groups].dev` is development-only and is not included as installable distribution metadata. It cannot supply CLI runtime dependencies to `uv tool install` or `uvx`. [PyPA dependency-groups specification](https://packaging.python.org/en/latest/specifications/dependency-groups/)
- Current published metadata exposes only the `redis` extra. The wheel has no `entry_points.txt` and therefore no executable. [PyPI `batchwork-ai` metadata](https://pypi.org/pypi/batchwork-ai/json)
- Tool environments are independently resolved from the requested distribution metadata and retained installation settings. The repository's `uv.lock` governs project commands such as `uv run`/`uv sync`; it is not an end-user tool lock. This follows from the isolated tool model and the absence of project-lock flags on `uv tool install`/`uv tool run` in uv 0.11.29.
- Result: published dependency bounds are the compatibility contract for tool users. Release verification must test resolution from built/published metadata, not rely only on the repository `.venv`.

## Python-version implications

The distribution declares `Requires-Python: >=3.11`; installers must honor that package metadata. Separately, uv's tool interpreter discovery ignores the current directory's project-local `.python-version` and `requires-python`. Users may select a compatible interpreter with `--python`. [uv tools concepts, 0.11.29](https://github.com/astral-sh/uv/blob/0.11.29/docs/concepts/tools.md), [PyPA pyproject specification](https://packaging.python.org/en/latest/specifications/pyproject-toml/#requires-python)

Implications:

- Do not assume running `uv tool install` from this repository forces Python 3.11.
- Test the executable on the supported Python floor and current upper supported versions.
- Keep console startup compatible with Python 3.11 even if development happens on a newer interpreter.

## Repository evidence

Current [`pyproject.toml`](../../pyproject.toml):

- distribution `batchwork-ai`, version `0.1.1`
- import module explicitly mapped to `batchwork`
- Python `>=3.11`
- normal dependencies: `httpcore`, `httpx`, `pydantic`
- optional extra: `redis`
- no `[project.scripts]`
- development dependencies correctly isolated in `[dependency-groups].dev`

Artifact inspection on 2026-07-18:

```text
wheel=batchwork_ai-0.1.1-py3-none-any.whl
entry_points_files=[]
Name: batchwork-ai
Version: 0.1.1
Requires-Python: >=3.11
Provides-Extra: redis
```

A disposable uv-build fixture verified that a differently named distribution can expose `demo-command`, that `uv tool install 'distribution[extra] @ file://…'` installs the requested extra, and that `uvx --from ... demo-command` invokes it successfully.

## Implementation acceptance criteria for the later ticket

1. Built wheel contains one intended `console_scripts` entry named `batchwork`.
2. With isolated `UV_TOOL_DIR`, `UV_TOOL_BIN_DIR`, and `UV_CACHE_DIR`:
   - `uv tool install <wheel-or-index-requirement>` succeeds.
   - `batchwork --version` exits 0 and reports the installed distribution version.
   - `uv tool list` records tool identity `batchwork-ai` and executable `batchwork`.
3. `uvx --from <wheel-or-index-requirement> batchwork --version` succeeds without importing the checkout or project `.venv`.
4. Plain installation starts without optional extras; Redis-only behavior gives a concrete recovery instruction to install `batchwork-ai[redis]` when needed.
5. Test Python 3.11 and at least one newer supported version; include Windows console-wrapper coverage in CI if the CLI is declared cross-platform.
6. Built artifacts pass `twine check`; verify wheel entry-point and dependency metadata directly.
7. Documentation uses distribution identity for install/upgrade and executable identity for invocation.

## Primary sources

- [uv 0.11.29 release, 2026-07-15](https://github.com/astral-sh/uv/releases/tag/0.11.29)
- [uv tools concepts, tag 0.11.29](https://github.com/astral-sh/uv/blob/0.11.29/docs/concepts/tools.md)
- [uv tools guide, tag 0.11.29](https://github.com/astral-sh/uv/blob/0.11.29/docs/guides/tools.md)
- [uv CLI reference](https://docs.astral.sh/uv/reference/cli/)
- [uv configuration-file rules](https://docs.astral.sh/uv/concepts/configuration-files/)
- [uv storage locations](https://docs.astral.sh/uv/reference/storage/)
- [PyPA `pyproject.toml` specification](https://packaging.python.org/en/latest/specifications/pyproject-toml/)
- [PyPA entry-points specification](https://packaging.python.org/en/latest/specifications/entry-points/)
- [PyPA dependency-specifier specification](https://packaging.python.org/en/latest/specifications/dependency-specifiers/)
- [PyPA dependency-groups specification](https://packaging.python.org/en/latest/specifications/dependency-groups/)
- [PyPA name-normalization specification](https://packaging.python.org/en/latest/specifications/name-normalization/)
- [PyPI `batchwork-ai` JSON metadata](https://pypi.org/pypi/batchwork-ai/json)
- [PyPI `batchwork` JSON endpoint (404 as researched)](https://pypi.org/pypi/batchwork/json)
