---
name: verify
summary: Verify the built Batchwork CLI outside the checkout.
---

1. Build artifacts: `uv build`.
2. Run `uv run python tools/verify_artifacts.py`; it installs the wheel into an isolated `uv tool` directory, then checks version, every help path, and Bash/Zsh/Fish completion without valid config or registry state.
3. For parser changes, install the wheel into a retained temporary `UV_TOOL_DIR`/`UV_TOOL_BIN_DIR` and invoke that `batchwork` executable directly. Probe root-only options, conflicting output/color flags, and malformed polling/duration values.
4. Run `uv run python tools/generate_cli_contract.py --check` for checked-in schema/fixture drift.

Use isolated `HOME`, XDG directories, `UV_CACHE_DIR`, `BATCHWORK_CONFIG`, and `BATCHWORK_REGISTRY`. Remove temporary directories with `trash`.
