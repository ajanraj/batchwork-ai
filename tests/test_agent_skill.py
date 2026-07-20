import json
import re
from pathlib import Path

SKILL_ROOT = Path(__file__).parents[1] / "skills" / "batchwork-ai"
SKILL_PATH = SKILL_ROOT / "SKILL.md"
REFERENCE_NAMES = {
    "commands.md",
    "configuration-and-secrets.md",
    "failures-and-recovery.md",
    "image-materialization.md",
    "inputs-and-options.md",
    "machine-output.md",
}
ASSET_NAMES = {"config.example.toml", "provider-options.example.json"}


def test_skill_has_portable_shallow_structure() -> None:
    assert SKILL_PATH.is_file()
    assert {path.name for path in (SKILL_ROOT / "references").iterdir()} == REFERENCE_NAMES
    assert {path.name for path in (SKILL_ROOT / "assets").iterdir()} == ASSET_NAMES
    assert not (SKILL_ROOT / "scripts").exists()

    skill_text = SKILL_PATH.read_text()
    assert len(skill_text.splitlines()) < 500

    shipped_files = [SKILL_PATH]
    shipped_files.extend((SKILL_ROOT / "references").iterdir())
    shipped_files.extend((SKILL_ROOT / "assets").iterdir())
    for path in shipped_files:
        text = path.read_text()
        assert not re.search(r"(?<![\w.])/(?:Users|home|tmp)/", text), path
        assert "github.com/ajanraj/batchwork-ai/issues/" not in text, path
        assert not re.search(r"(?:sk-[A-Za-z0-9]{16}|AIza[A-Za-z0-9_-]{16})", text), path
        for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", text):
            if "://" not in target:
                resolved = SKILL_ROOT / target
                assert resolved.is_file(), (path, target)
                assert resolved.parent in {
                    SKILL_ROOT / "references",
                    SKILL_ROOT / "assets",
                }


def test_skill_declares_exact_release_compatibility() -> None:
    text = SKILL_PATH.read_text()
    frontmatter = text.split("---", 2)[1]

    assert "name: batchwork-ai" in frontmatter
    assert "license: MIT" in frontmatter
    assert 'version: "0.1.0"' in frontmatter
    assert "Batchwork >=0.2,<0.3" in frontmatter
    assert "schema_version 1" in frontmatter
    assert "allowed-tools:" not in frontmatter


def test_skill_encodes_direct_invocation_and_safety_contract() -> None:
    text = SKILL_PATH.read_text()
    required = (
        "batchwork --json --quiet status JOB",
        "Never infer a provider",
        "Never retry submission",
        "Never blindly resubmit",
        "--allow-large-batch",
        "explicit authorization",
        "Timeout, interruption, termination, broken pipe, and local failures never cancel",
        "record_id",
        "provider_reference",
        "--output-dir",
        "schema_version",
        "custom_id",
    )
    for phrase in required:
        assert phrase in text

    assert "scripts/" not in text
    assert "wrapper script" not in text.lower()


def test_skill_references_cover_behavioral_recovery_edges() -> None:
    combined = "\n".join(path.read_text() for path in (SKILL_ROOT / "references").iterdir())
    skill = SKILL_PATH.read_text()

    assert "help differs from this contract, stop all operational work" in skill
    assert "mode `0600`" in skill
    assert "raw provider response bodies" in combined
    assert "`submit` → `job`" in combined
    assert "`candidate_records` as a nonnegative integer, including zero" in combined
    assert "`records_count_known` is false" in combined
    assert re.search(
        r"`changed_records`\s+must be\s+absent; never infer or invent it",
        combined,
    )
    assert "Require `remote_jobs_changed: false`" in combined


def test_skill_assets_are_non_secret_and_selected_provider_only() -> None:
    config = (SKILL_ROOT / "assets" / "config.example.toml").read_text()
    options = json.loads((SKILL_ROOT / "assets" / "provider-options.example.json").read_text())

    assert "schema_version = 1" in config
    assert "api_key_env" in config
    assert "api_key =" not in config
    assert set(options) == {"reasoningEffort", "textVerbosity"}
