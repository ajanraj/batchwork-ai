"""Generate or check the public CLI schema and credential-free fixtures."""

from __future__ import annotations

import argparse

from batchwork.cli._contract import contract_drift, write_contract_artifacts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Fail when generated artifacts drift.")
    arguments = parser.parse_args()

    if arguments.check:
        drifted = contract_drift()
        if drifted:
            paths = "\n".join(f"- {path}" for path in drifted)
            raise SystemExit(f"CLI contract artifacts are out of date:\n{paths}")
        return

    write_contract_artifacts()


if __name__ == "__main__":
    main()
