from pathlib import Path

from batchwork._text_validation import _OPENAI_ENDPOINT_OPTIONS, _TEXT_PROVIDER_OPTIONS
from batchwork.providers._image import _allowed_options
from batchwork.types import BatchProvider

ROOT = Path(__file__).parents[2]
DOCS = ROOT / "docs" / "docs"


def test_cli_documentation_covers_public_operating_contract() -> None:
    cli = (DOCS / "cli.mdx").read_text()
    registry = (DOCS / "reference" / "cli-configuration-registry.mdx").read_text()
    readme = (ROOT / "README.md").read_text()

    for phrase in (
        "Human quick start",
        "Agent quick start",
        "Command selection",
        "Selectors",
        "Input formats",
        "Output modes",
        "Recovery",
        "--output-dir",
    ):
        assert phrase in cli
    for phrase in (
        "Path precedence",
        "OS defaults",
        "schema_version = 1",
        "Endpoint trust",
        "Registry privacy",
        "Migration and backup",
        "Direct-reference recovery",
    ):
        assert phrase in registry
    assert "batchwork submit text" in readme
    assert "batchwork --jsonl --quiet run text" in readme


def test_provider_pages_list_every_implemented_exact_option_key() -> None:
    for provider, keys in _TEXT_PROVIDER_OPTIONS.items():
        page = (DOCS / "providers" / f"{provider.value}.mdx").read_text()
        for key in keys:
            assert f"`{key}`" in page

    openai = (DOCS / "providers" / "openai.mdx").read_text()
    for keys in _OPENAI_ENDPOINT_OPTIONS.values():
        for key in keys:
            assert f"`{key}`" in openai

    for provider in (BatchProvider.OPENAI, BatchProvider.GOOGLE, BatchProvider.XAI):
        page = (DOCS / "providers" / f"{provider.value}.mdx").read_text()
        for key in _allowed_options(provider):
            assert f"`{key}`" in page


def test_embedding_provider_options_are_documented_exactly() -> None:
    expected = {
        BatchProvider.OPENAI: {"dimensions", "user"},
        BatchProvider.GOOGLE: {"content", "outputDimensionality", "taskType", "title"},
    }
    for provider, keys in expected.items():
        page = (DOCS / "providers" / f"{provider.value}.mdx").read_text()
        for key in keys:
            assert f"`{key}`" in page

    mistral = (DOCS / "providers" / "mistral.mdx").read_text()
    assert "CLI embedding provider options: none" in mistral
