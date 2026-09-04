"""Regression checks for production-safe Streamlit UI behavior."""

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAIN_MODULE = PROJECT_ROOT / "app" / "main.py"


def _main_source() -> str:
    return MAIN_MODULE.read_text(encoding="utf-8")


def test_streamlit_ui_does_not_render_raw_exceptions():
    """Detailed exceptions must stay in server logs, not reach end users."""

    source = _main_source()
    tree = ast.parse(source)

    raw_exception_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "st"
        and node.func.attr == "exception"
    ]

    assert raw_exception_calls == []
    assert "could not be rendered: {exc}" not in source


def test_streamlit_ui_keeps_bounded_session_only_follow_up_context():
    """Follow-up context must be bounded, resettable, and planner-only."""

    source = _main_source()
    normalized_source = "".join(source.split())

    assert 'if "conversation_history" not in st.session_state:' in source
    assert "conversation_context=(" in source
    assert "history[-MAX_CONVERSATION_TURNS:]" in normalized_source
    assert "Clear follow-up context" in source

    # It is reset when either the active dataset or just the context is
    # cleared, and when a fresh upload successfully replaces the dataset.
    assert source.count(
        "st.session_state.conversation_history = []"
    ) >= 3
