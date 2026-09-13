"""Runtime UI tests. Only the upload widget and remote model are stubbed.

AppTest cannot attach files; supply an in-memory CSV at the widget boundary.
CSV loading, profiling, real agent execution and result rendering still run.
"""
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from app.core import agent as agent_module
from app.core.agent import (AgentCapacityError, AgentPlanningError,
    AgentValidationError, AgentExecutionError)
from app.core.query_planner import QueryPlanner
from app.core.explainer import ExplainerLLMError

MAIN = Path(__file__).resolve().parents[1] / 'app/main.py'


def button(app, label):
    return next(b for b in app.button if label in b.label)


@pytest.fixture
def uploaded_app(monkeypatch):
    monkeypatch.delenv('APP_ACCESS_PASSWORD', raising=False)
    upload = BytesIO(b'product,sales\nA,100\nB,200\nA,300\n')
    upload.name = 'sales.csv'
    upload.size = len(upload.getvalue())
    upload.file_id = 'first'
    current = [upload]
    monkeypatch.setattr(st,'file_uploader',lambda *a,**kw:current[0])
    model = Mock()
    model.generate_text.return_value = 'SELECT product, SUM(sales) AS total_sales FROM dataset GROUP BY product ORDER BY product'
    monkeypatch.setattr(agent_module,'QueryPlanner',lambda:QueryPlanner(llm_client=model))
    monkeypatch.setattr(agent_module,'explain_query_result',lambda **kw:'A has total sales of 400; B has 200.')
    app = AppTest.from_file(str(MAIN)).run(timeout=15)
    assert not app.exception
    return app, current, model


def analyze(app):
    app.text_area[0].input('What are total sales by product?')
    button(app,'Analyze').click().run(timeout=15)
    assert not app.exception


def test_upload_analysis_results_chart_and_context(uploaded_app):
    app, current, model = uploaded_app
    analyze(app)
    result = app.session_state.last_result
    assert result.query_result.dataframe.to_dict('records') == [
        {'product':'A','total_sales':400},{'product':'B','total_sales':200}]
    assert any('A has total sales' in item.value for item in app.markdown)
    assert len(app.get('plotly_chart')) + len(app.get('arrow_vega_lite_chart')) > 0
    assert len(app.session_state.conversation_history) == 1
    button(app,'Clear follow-up context').click().run()
    assert app.session_state.conversation_history == []
    assert app.session_state.dataframe is not None
    # The cleared uploader returns no file on the next rerun.
    current[0] = None
    button(app,'Clear active dataset').click().run()
    assert app.session_state.last_result is None
    assert app.session_state.agent is None
    assert app.session_state.dataframe is None


@pytest.mark.parametrize('error,message', [
    (AgentCapacityError,'busy processing'),(AgentPlanningError,'valid analysis plan'),
    (AgentValidationError,'security checks'),(AgentExecutionError,'could not be executed'),
    (RuntimeError,'could not be completed')])

def test_analysis_errors_are_safe_and_clear_stale_result(uploaded_app,error,message):
    app, _, _ = uploaded_app
    analyze(app)
    history = list(app.session_state.conversation_history)
    app.session_state.agent = Mock(run=Mock(side_effect=error('private-token-and-data')))
    analyze(app)
    displayed = ' '.join(e.value for e in list(app.error)+list(app.warning))
    assert message in displayed
    assert 'private-token-and-data' not in displayed
    assert app.session_state.last_result is None
    assert app.session_state.conversation_history == history

def test_replacing_same_filename_resets_analysis(uploaded_app):
    app, current, _ = uploaded_app
    analyze(app)
    replacement = BytesIO(b'product,sales\nC,50\n')
    replacement.name = 'sales.csv'
    replacement.size = len(replacement.getvalue())
    replacement.file_id = 'replacement'
    current[0] = replacement
    app.run()
    assert not app.exception
    assert app.session_state.dataframe['product'].tolist() == ['C']
    assert app.session_state.last_result is None
    assert app.session_state.conversation_history == []

def test_chart_failure_preserves_successful_result(uploaded_app, monkeypatch):
    app, _, _ = uploaded_app

    monkeypatch.setattr(
        agent_module,
        'generate_chart',
        Mock(side_effect=ValueError('bad chart spec')),
    )

    analyze(app)

    assert not app.exception
    displayed = ' '.join(e.value for e in list(app.error) + list(app.warning))
    assert 'visualization could not be generated' in displayed
    assert 'could not be completed' not in displayed
    assert app.session_state.last_result is not None
    assert any('A has total sales' in item.value for item in app.markdown)

def test_explanation_failure_preserves_successful_result(uploaded_app, monkeypatch):
    app, _, _ = uploaded_app

    monkeypatch.setattr(
        agent_module,
        'explain_query_result',
        Mock(side_effect=ExplainerLLMError('boom')),
    )

    analyze(app)

    assert not app.exception
    displayed = ' '.join(e.value for e in list(app.error) + list(app.warning))
    assert 'explanation could not be generated' in displayed
    assert 'could not be completed' not in displayed
    assert app.session_state.last_result is not None
    assert app.session_state.last_result.query_result.dataframe.to_dict('records') == [
        {'product': 'A', 'total_sales': 400}, {'product': 'B', 'total_sales': 200}]