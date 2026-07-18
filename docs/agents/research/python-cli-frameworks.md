# Python CLI framework options

Research snapshot: 2026-07-18. Scope: issue [#4](https://github.com/ajanraj/batchwork-ai/issues/4), supporting the Wayfinder map in [#1](https://github.com/ajanraj/batchwork-ai/issues/1). This note compares viable choices; it does not select a framework or define product behavior.

## Repository constraints

- Python `>=3.11`; asyncio-classified, typed package.
- Runtime dependencies today: `httpcore`, `httpx`, and `pydantic`.
- The CLI must install cleanly as a uv tool, expose nested commands, provide useful help/completion, preserve strict typing, call an async core, support deterministic tests, and keep machine-readable output stable.
- There is no current `[project.scripts]` entry point or CLI implementation.

Source: [`pyproject.toml`](../../../pyproject.toml).

## Current shortlist

Versions and dependency metadata below are current PyPI publisher metadata as of the snapshot date.

| Choice | Current release / baseline | Native async command | Declaration typing | Nested commands | Completion | Runtime footprint on Python 3.11+ / non-Windows |
| --- | --- | --- | --- | --- | --- | --- |
| `argparse` | Python 3.11 stdlib | No; explicit `asyncio.run()` boundary | Parser declarations produce a dynamic `Namespace` | Yes, manual `add_subparsers()` trees | None in stdlib; optional Argcomplete adds Bash/Zsh | 0 |
| Click | 8.4.2, 2026-06-24 | No | Typed package, but decorator replaces callback symbol with `Command`; option declaration duplicates callback types | Arbitrary `Group` nesting | Bash 4.4+, Zsh, Fish | 1 distribution; only conditional `colorama` dependency on Windows |
| Typer | 0.27.0, 2026-07-15 | No | Parameters derived from annotations; decorated function keeps its callable type | Arbitrary nested `Typer` apps | Bash, Zsh, Fish, PowerShell / `pwsh`; install/show commands | 3 direct dependencies; 7 installed distributions in an isolated probe; vendored Click code |
| Cyclopts | 4.21.1, 2026-07-16 | Yes; asyncio default, `run_async()` for an existing loop | Parameters derived from annotations; decorated function keeps its callable type | Fully recursive `App` composition | Bash, Zsh, Fish; install/generate APIs | 4 direct dependencies; 8 installed distributions in an isolated probe |
| AsyncClick | 8.4.2.1, 2026-06-30 | Yes; `asyncio.run()` fallback, optional AnyIO/Trio backend, awaitable `main()` | Click-style declarations; decorated symbol is a command object | Click-compatible arbitrary groups | Bash, Zsh, Fish | 1 distribution; only conditional `colorama` dependency on Windows |

Release and dependency sources: [Click PyPI JSON](https://pypi.org/pypi/click/json), [Typer PyPI JSON](https://pypi.org/pypi/typer/json), [Cyclopts PyPI JSON](https://pypi.org/pypi/cyclopts/json), [AsyncClick PyPI JSON](https://pypi.org/pypi/asyncclick/json). All four repositories were active and unarchived at the snapshot: [Click](https://github.com/pallets/click), [Typer](https://github.com/fastapi/typer), [Cyclopts](https://github.com/BrianPugh/cyclopts), [AsyncClick](https://github.com/python-trio/asyncclick).

### 1. Standard-library `argparse`

Strengths:

- No added package or supply-chain surface.
- Automatic usage/help and invalid-argument errors; `format_help()` and `format_usage()` support deterministic help assertions.
- `add_subparsers()` provides command dispatch, aliases, and required subcommands. Nested trees are possible by adding subparsers to child parsers.
- `parse_args([...])` accepts explicit tokens, useful for unit tests; parser exit/error behavior can be overridden or tested through subprocesses.

Constraints:

- Parsed values are returned through `argparse.Namespace`. In the repository's `ty==0.0.59` probe, `args` was `Namespace` and `args.count` was `Any`; strict typing therefore needs a typed conversion boundary, typed command objects, or disciplined extraction after parsing.
- Async execution is application-owned. A synchronous entry point can call `asyncio.run()` once, but `asyncio.run()` cannot run while another loop is active in the same thread.
- The stdlib module has no shell-completion facility. [Argcomplete 3.7.0](https://pypi.org/project/argcomplete/) is a maintained, dependency-free companion for Bash and Zsh, but it adds package/activation work, executes application startup during completion, requires careful placement of `autocomplete()`, and documents first-1-KiB marker and executable-name constraints.
- More code is required for reusable options, nested composition, consistent diagnostics, and rich help than with the framework choices.

Sources: [Python 3.11 `argparse`](https://docs.python.org/3.11/library/argparse.html), [`asyncio.run()`](https://docs.python.org/3/library/asyncio-runner.html#asyncio.run), [Argcomplete docs](https://kislyuk.github.io/argcomplete/), [Argcomplete metadata](https://pypi.org/pypi/argcomplete/json).

### 2. Click

Strengths:

- Mature command/group/context model with arbitrary group nesting, delayed command registration, generated help, scoped options, environment/default-map support, and explicit exception/exit-code conventions.
- Built-in completion for Bash 4.4+, Zsh, and Fish. It supports command, option, type, and custom value completion. Completion requires an installed entry point, matching the intended uv-tool packaging model.
- `click.testing.CliRunner` covers argument invocation, stdin, environment, isolated filesystems, exit codes, exceptions, and stdout/stderr capture.
- Smallest third-party footprint among the synchronous frameworks: no non-Windows runtime dependency beyond Click itself.
- Ships a `py.typed` marker.

Constraints:

- Click invokes callbacks as `callback(*args, **kwargs)` and does not detect or await coroutine callbacks. A local probe returned an unexecuted coroutine with exit code 0. Commands must therefore remain synchronous and call the async core through one explicit `asyncio.run()` boundary.
- Click's decorator changes the bound symbol from the callback function to a `Command`. The `ty` probe revealed `Command`, while Typer and Cyclopts preserved the original callable signature. Callback parameter annotations remain useful internally, but Click options and callback types are separate declarations that must stay aligned.
- `CliRunner` changes interpreter-global state and is explicitly not thread-safe; concurrent tests need subprocess isolation or serialization.
- Default help/errors are human-oriented, not a machine JSON contract.

Sources: [commands and groups](https://click.palletsprojects.com/en/stable/commands-and-groups/), [shell completion](https://click.palletsprojects.com/en/stable/shell-completion/), [testing](https://click.palletsprojects.com/en/stable/testing/), [exceptions and exit codes](https://click.palletsprojects.com/en/stable/exceptions/), [`Context.invoke` at 8.4.2](https://github.com/pallets/click/blob/8.4.2/src/click/core.py#L853-L907), [`py.typed`](https://github.com/pallets/click/blob/8.4.2/src/click/py.typed).

### 3. Typer

Strengths:

- Generates CLI parameters from normal Python annotations and `Annotated[...]` metadata. The command decorator registers but does not replace the original function, preserving strict callable types in the probe.
- Supports deeply nested subcommands by mounting `Typer` apps with `add_typer()`.
- Rich generated help and errors; packaged apps expose `--install-completion` and `--show-completion`. Current source registers Bash, Zsh, Fish, PowerShell, and `pwsh` completion classes.
- `typer.testing.CliRunner` supports pytest-style token, stdin, exit-code, stdout, and stderr assertions.
- Ships a `py.typed` marker.

Constraints:

- Typer 0.26.0 vendored Click and removed the external Click dependency/integration surface. Current 0.27.0 has three direct dependencies (`shellingham`, `rich`, `annotated-doc`), but the isolated environment contained seven distributions because Rich adds `markdown-it-py`, `mdurl`, and `pygments`.
- Vendoring reduces a separate distribution but increases Typer's owned code surface and means Click-specific integrations are no longer supported. This matters if the CLI expects the Click plugin ecosystem.
- The vendored `Context.invoke` still directly returns `callback(*args, **kwargs)` with no coroutine detection. The local probe returned an unexecuted coroutine, so commands need the same explicit synchronous-to-async boundary as Click.
- Rich help/error rendering improves human UX but must be kept out of stdout in machine mode. Current Typer termination docs do not themselves guarantee all custom diagnostics use stderr; application code must enforce the stream contract.

Sources: [Typer release notes](https://typer.tiangolo.com/release-notes/#0260-2026-05-26), [nested subcommands](https://typer.tiangolo.com/tutorial/subcommands/nested-subcommands/), [packaging/completion](https://typer.tiangolo.com/tutorial/package/), [testing](https://typer.tiangolo.com/tutorial/testing/), [current completion registrations](https://github.com/fastapi/typer/blob/0.27.0/typer/_completion_classes.py#L224-L229), [vendored callback invocation](https://github.com/fastapi/typer/blob/0.27.0/typer/_click/core.py#L474-L489), [`py.typed`](https://github.com/fastapi/typer/blob/0.27.0/typer/py.typed).

### 4. Cyclopts

Strengths:

- Annotation-driven declaration preserves the original callable type and supports typed user classes/dataclasses.
- Native async commands: Cyclopts creates an event loop automatically, defaults to asyncio, and exposes `await app.run_async(...)` when already inside an event loop.
- Command composition is fully recursive; child apps inherit parent configuration.
- Generated `-h`/`--help` supports Markdown by default plus Rich, RST, and plaintext modes.
- Completion for Bash, Zsh, and Fish includes install and script-generation APIs. Static generated completion avoids importing Python on every tab press.
- Testing docs cover direct parsing, return values, pytest capture, environment variables, file configuration, exit codes, and deterministic Rich help fixtures.
- Ships a `py.typed` marker.

Constraints:

- Largest new dependency footprint in this shortlist: `attrs`, `docstring-parser`, `rich-rst`, and `rich`, eight installed distributions in the isolated probe. `attrs` is an additional data-model package beside Batchwork's existing Pydantic dependency, even though Cyclopts does not require Pydantic at runtime.
- Default `result_action="print_non_int_sys_exit"` prints non-integer return values to stdout and maps integers/booleans to process status. Stable JSON needs an explicit result policy such as `return_value`, explicit serialization, and separate error console handling.
- Completion officially covers Bash, Zsh, and Fish, not PowerShell.
- Maintained known issues document partial limitations for some postponed-annotation/dataclass-inheritance scenarios across modules. Any planned use of inherited structured parameter models should be tested against the exact shape.
- Newer/smaller ecosystem than Click/Typer; the choice trades ecosystem maturity for native async and a more type-driven API.

Sources: [commands and async](https://cyclopts.readthedocs.io/en/stable/commands.html), [shell completion](https://cyclopts.readthedocs.io/en/stable/shell_completion.html), [help](https://cyclopts.readthedocs.io/en/stable/help.html), [unit testing source](https://github.com/BrianPugh/cyclopts/blob/v4.21.1/docs/source/cookbook/unit_testing.rst), [result actions/API source](https://github.com/BrianPugh/cyclopts/blob/v4.21.1/docs/source/api.rst), [known issues](https://cyclopts.readthedocs.io/en/stable/known_issues.html), [`py.typed`](https://github.com/BrianPugh/cyclopts/blob/v4.21.1/cyclopts/py.typed).

### 5. AsyncClick

Strengths:

- A current, maintained async fork of Click with the same small installed footprint on non-Windows.
- Async callbacks are awaited. Calling the command creates an event loop; current source prefers AnyIO if installed, otherwise falls back to Trio when requested or stdlib `asyncio.run()`. Existing async applications can `await command.main()`.
- Retains Click's command/group/help/completion model. Current source includes Bash, Zsh, and Fish completion implementations.
- Async `CliRunner.invoke()` supports stdin, environment, return values, exceptions, stdout/stderr capture, and isolated filesystems.
- Python requirement is exactly compatible with Batchwork: `>=3.11`; ships `py.typed`.

Constraints:

- It is a fork, not an extension that can coexist with Click: upstream documentation says Click and AsyncClick cannot be used in the same program. Adopting it accepts fork-sync and ecosystem-compatibility risk.
- Advanced Click methods such as `Command.main` and `Context.invoke` become async, so integrations assuming Click's synchronous internals need adaptation.
- The test runner's `invoke()` is async and changes process-global state; its source warns it only works without concurrency.
- Documentation is primarily fork README/source plus inherited Click concepts, less cohesive than Click/Typer/Cyclopts documentation.
- Completion officially covers Bash, Zsh, and Fish, not PowerShell.

Sources: [tagged README](https://github.com/python-trio/asyncclick/blob/8.4.2.1%2Basync/README.md), [event-loop implementation](https://github.com/python-trio/asyncclick/blob/8.4.2.1%2Basync/src/asyncclick/core.py#L1626-L1666), [completion implementation](https://github.com/python-trio/asyncclick/blob/8.4.2.1%2Basync/src/asyncclick/shell_completion.py), [testing implementation](https://github.com/python-trio/asyncclick/blob/8.4.2.1%2Basync/src/asyncclick/testing.py), [`py.typed`](https://github.com/python-trio/asyncclick/blob/8.4.2.1%2Basync/src/asyncclick/py.typed).

## Cross-cutting implications

### uv installation and packaging

Framework choice does not materially change uv installation. The package must add a `[project.scripts]` entry whose object is called without arguments. Installers create a wrapper command; `uv tool install batchwork-ai` installs provided executables in an isolated environment, while `uvx --from batchwork-ai <command>` runs one temporarily. Shell completion generally depends on invoking that installed executable rather than `python module.py`.

Sources: [PyPA entry-point specification](https://packaging.python.org/en/latest/specifications/entry-points/#use-for-scripts), [uv tools guide](https://docs.astral.sh/uv/guides/tools/).

### Stable JSON output

No assessed parser/framework defines Batchwork's JSON schema or stability policy. Treat output as a separate product contract:

1. Successful machine output: JSON or JSONL only on stdout.
2. Diagnostics, warnings, progress, and human errors: stderr only.
3. Explicit JSON serialization policy, including ordering and whitespace. Python documents `sort_keys=True` for comparable serialization and `separators=(",", ":")` for compact output.
4. Explicit error-envelope and exit-code mapping if machine consumers need structured failures; framework-native error text is human-facing and can change.
5. Disable color/Rich formatting and automatic result printing in machine mode. Cyclopts especially needs a non-default `result_action` or commands that return `None` after explicit serialization.
6. Test stdout bytes, stderr bytes, exit status, and schema independently. Use subprocess acceptance tests for the installed entry point in addition to framework runners.

Source: [Python `json`](https://docs.python.org/3/library/json.html).

### Async boundary

There are two viable architecture shapes; this research does not select one:

```diagram
Synchronous CLI boundary                 Native async CLI callback

parser/framework callback                async framework callback
        │                                          │
        ├─ validate CLI input                       ├─ validate CLI input
        └─ asyncio.run(async command service)       └─ await async command service
                    │                                          │
                    └──────── existing Batchwork async core ───┘
```

- `argparse`, Click, and Typer require the left shape.
- Cyclopts and AsyncClick support the right shape and also define an API for callers already inside an event loop.
- A normal installed CLI process owns its event loop, so the synchronous boundary is technically sound if invoked once. Native async mainly removes wrapper repetition and makes in-process async embedding/testing more direct.

## Decision criteria

The next product decision should answer these in order, without selecting by aesthetics alone:

1. **Is native async callback support a hard requirement?** If yes, the direct comparison narrows to Cyclopts versus AsyncClick. If no, all choices remain viable because one outer `asyncio.run()` boundary is correct for a normal CLI process.
2. **Is PowerShell completion required in v1?** Typer explicitly registers it. The assessed Click, Cyclopts, AsyncClick, and maintained Argcomplete paths officially cover narrower shell sets.
3. **What is the runtime dependency budget?** Decide whether zero/minimal packages outweigh annotation-derived declarations and richer help.
4. **How strict must CLI-layer typing be?** `argparse.Namespace` yields `Any`; Click replaces callback symbols with `Command`; Typer and Cyclopts preserve ordinary callable signatures.
5. **Is Click ecosystem compatibility valuable?** Click has the broadest mature extension surface; Typer 0.26+ explicitly stopped supporting Click-specific integrations; AsyncClick is a fork with async internal APIs.
6. **Should help be plain and stable or richly rendered?** `argparse`/Click default toward plain text; Typer/Cyclopts favor Rich output but can be constrained. Machine output must bypass either presentation layer.
7. **What test concurrency model is expected?** Click-family runners mutate global state; Cyclopts docs favor direct parse/call plus pytest capture. Installed-entry-point subprocess tests remain framework-neutral.
8. **Will structured dataclass-like CLI parameters be central?** If yes, validate Cyclopts's documented postponed-annotation edge cases and compare the benefit against Typer's simpler annotation model. If no, this criterion should carry little weight.

## Recommended next decision

Decide two hard gates before any prototype or product specification work:

- native async callbacks: required or optional;
- PowerShell completion in v1: required or optional.

Then set a maximum acceptable runtime dependency footprint. Those three answers reduce the shortlist mechanically without prematurely choosing product direction. Afterward, a separate ticket can run a disposable contract spike against the remaining two or three choices using the same nested command, JSON/stdout-stderr, completion, async, and test fixtures.

## Verification evidence

Commands executed from the isolated research branch included:

```text
gh issue view 1 --repo ajanraj/batchwork-ai --comments
gh issue view 4 --repo ajanraj/batchwork-ai --comments
git switch -c research/python-cli-frameworks
python <PyPI JSON metadata probe>
gh api repos/<upstream-repository>
uv run --no-project --isolated --with <framework> python <dependency/async probes>
uv venv <temporary-directory>/.venv
uv pip install --python <temporary-directory>/.venv/bin/python ty==0.0.59 click==8.4.2 typer==0.27.0 cyclopts==4.21.1
<temporary-directory>/.venv/bin/ty check --python <temporary-directory>/.venv/bin/python <probe files>
```

Observed async probe:

```text
click: 0 '' coroutine True
typer: 0 '' coroutine True
awaited
cyclopts: 'done' str
awaited
argparse: 'done'
awaited
asyncclick: 'done' str
```

Observed `ty==0.0.59` reveal types:

```text
argparse args: Namespace
argparse args.count: Any
Click decorated command: Command
Typer decorated command: def command(count: int) -> None
Cyclopts decorated command: def command(count: int) -> None
```

The temporary probes were removed after execution. No CLI code, dependency, package metadata, issue, or product behavior was changed.
