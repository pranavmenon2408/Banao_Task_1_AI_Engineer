"""Headless render of the Streamlit results page from a saved API result (no backend required)."""

import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
RESULT = json.loads((ROOT / "tests/fixtures/sample_result.json").read_text(encoding="utf-8"))


@pytest.fixture
def app(monkeypatch) -> AppTest:
    monkeypatch.setenv("API_URL", "http://127.0.0.1:9")  # nothing listens here: the UI must degrade gracefully
    at = AppTest.from_file(str(ROOT / "ui/streamlit_app.py"), default_timeout=30)
    at.session_state["result"] = RESULT
    at.session_state["base_criteria"] = RESULT["criteria"]
    return at.run()


def test_results_page_renders_without_errors(app):
    assert not app.exception
    assert [t.label.split(" ", 1)[1] for t in app.tabs] == [
        "Per-criterion assessment",
        "Extracted profile",
        "Run metrics",
    ]
    cards = [m.value for m in app.markdown if 'class="crit-title"' in m.value]
    assert len(cards) == len(RESULT["criteria"])


def test_html_cards_are_single_blocks(app):
    """Blank or indented lines inside a card make Markdown render the rest of it as a code block."""
    for block in (m.value for m in app.markdown if 'class="card' in m.value):
        assert "\n\n" not in block and not any(line.startswith("    ") for line in block.splitlines())


def test_unreachable_api_is_reported(app):
    assert any("API unreachable" in e.value for e in app.error)
