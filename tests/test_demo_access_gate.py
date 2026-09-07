"""Regression tests for the optional public-demo access gate."""

from pathlib import Path

from streamlit.testing.v1 import AppTest


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAIN_FILE = PROJECT_ROOT / "app" / "main.py"


def test_configured_access_code_blocks_the_main_application(monkeypatch) -> None:
    """A configured password must block the normal interface until unlocked."""

    monkeypatch.setenv("APP_ACCESS_PASSWORD", "test-demo-code")

    app = AppTest.from_file(str(MAIN_FILE))
    app.run(timeout=15)

    assert not app.exception
    assert app.title[0].value == "🔒 Private Portfolio Demo"
    assert len(app.text_input) == 1
    assert len(app.button) == 1
    assert len(app.text_area) == 0


def test_correct_access_code_unlocks_the_main_application(monkeypatch) -> None:
    """A correct code must allow the normal no-dataset screen to render."""

    monkeypatch.setenv("APP_ACCESS_PASSWORD", "test-demo-code")

    app = AppTest.from_file(str(MAIN_FILE))
    app.run(timeout=15)
    app.text_input[0].input("test-demo-code")
    app.button[0].click().run(timeout=15)

    assert not app.exception
    assert app.title[0].value == "📊 AI Data Analyst Agent"
    assert len(app.text_area) == 0

def test_production_without_password_stays_locked(monkeypatch):
    from dataclasses import replace
    import app.config as config_module
    monkeypatch.delenv('APP_ACCESS_PASSWORD', raising=False)
    monkeypatch.setattr(config_module, 'config', replace(config_module.config, app_env='production'))
    app = AppTest.from_file(str(MAIN_FILE)).run(timeout=15)
    assert not app.exception
    assert any('locked' in item.value for item in app.error)
    assert not app.text_area


def test_wrong_password_does_not_unlock(monkeypatch):
    monkeypatch.setenv('APP_ACCESS_PASSWORD', 'test-demo-code')
    app = AppTest.from_file(str(MAIN_FILE)).run(timeout=15)
    app.text_input[0].input('wrong')
    app.button[0].click().run(timeout=15)
    assert not app.exception
    assert app.error
    assert not app.session_state.demo_access_granted
