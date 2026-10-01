"""Behavioral regressions for the project-wide correctness/reliability review."""
from dataclasses import replace
from io import BytesIO, StringIO
from pathlib import Path
import gc
import logging
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from app.core.agent import DataAnalystAgent, AgentDataError
from app.core.chart_generator import generate_chart
from app.core.data_loader import load_csv, CSVParsingError
from app.core.data_profiler import profile_dataframe
from app.core.execution_preparation import prepare_execution_dataframe
from app.core.explainer import explain_query_result, _build_result_summary, ExplainerLLMError
from app.core.query_planner import QueryPlanner, QueryPlanningError
from app.core.semantic_checks import check_conversion_fidelity, SemanticConversionError
from app.models.schemas import QueryResult
from app.utils.dataset_resources import DatasetMemoryBudget, DatasetResourceError, dataset_preparation_slot
from app.utils.logger import get_logger
from app.utils.security import validate_sql


def upload(text):
    file = BytesIO(text.encode())
    file.name = "synthetic.csv"
    file.size = len(file.getvalue())
    file.file_id = str(id(file))
    return file


def qr(frame):
    return QueryResult(dataframe=frame, row_count=len(frame), truncated=False)


def test_ingestion_preserves_ids_labels_precision_and_explicit_missing():
    raw = load_csv(upload('id,region,amount,missing\n00123,NA,9007199254740993,\n00124,NULL,0.1234567890123456789,\n'))
    assert raw.id.tolist() == ['00123', '00124']
    assert raw.region.tolist() == ['NA', 'NULL']
    assert raw.amount.tolist() == ['9007199254740993', '0.1234567890123456789']
    assert raw.missing.isna().all()
    before = raw.copy(deep=True)
    typed = prepare_execution_dataframe(raw)
    assert typed.id.tolist() == raw.id.tolist()
    assert typed.amount.tolist() == raw.amount.tolist()
    assert_frame_equal(raw, before)


@pytest.mark.parametrize('text', ['a,a\n1,2\n', 'a,A\n1,2\n', 'a,b\n1,2,3\n4,5,6\n', 'a,b\n1\n'])
def test_invalid_original_headers_and_ragged_rows_are_rejected(text):
    with pytest.raises(CSVParsingError):
        load_csv(upload(text))


def test_loaded_numeric_data_still_aggregates_without_changing_preview():
    raw = load_csv(upload('product,sales\nA,100\nA,300\nB,200\n'))
    planner = MagicMock()
    planner.plan.return_value = 'SELECT product, SUM(sales) AS total FROM dataset GROUP BY product ORDER BY product'
    agent = DataAnalystAgent(raw, profile_dataframe(raw), query_planner=planner)
    result = agent.run('Total sales by product?', False, True)
    assert result.query_result.dataframe.to_dict('records') == [{'product':'A','total':400.0},{'product':'B','total':200.0}]
    assert raw.sales.tolist() == ['100','300','200']
    assert agent.execution_profile.columns[1].inferred_type == 'integer'


@pytest.mark.parametrize('values', [['1',None,'2'], ['0.1','1.25',None], ['-12','10']])
def test_numeric_preparation_preserves_nulls_and_canonical_values(values):
    raw = pd.DataFrame({'value':values})
    typed = prepare_execution_dataframe(raw)
    assert typed.value.isna().equals(raw.value.isna())
    assert pd.api.types.is_numeric_dtype(typed.value)
    assert raw.value.tolist() == values


@pytest.mark.parametrize('values', [['01','02'], ['1','unknown'], ['0.1234567890123456789','1.0'], ['9223372036854775808','1'], ['2020','2021']])
def test_unsafe_numeric_preparation_declines(values):
    raw = pd.DataFrame({'value':values})
    assert_frame_equal(prepare_execution_dataframe(raw),raw)


@pytest.mark.parametrize('values,sql', [
    (['12/7/2023','1/2/2022'], 'SELECT * FROM dataset WHERE TRY_CAST(value AS DATE) > DATE \'2022-12-31\''),
    (['1','unknown'], 'SELECT AVG(TRY_CAST(value AS DOUBLE)) FROM dataset'),
    (['9007199254740993','1'], 'SELECT CAST(value AS DOUBLE) FROM dataset'),
    (['1.5','2'], 'SELECT CAST(value AS INTEGER) FROM dataset'),
    (['128','1'], 'SELECT CAST(value AS TINYINT) FROM dataset'),
    (['1.2345','2'], 'SELECT CAST(value AS DECIMAL(5,2)) FROM dataset'),
    (['-1','1'], 'SELECT CAST(value AS UBIGINT) FROM dataset'),
    (['256','1'], 'SELECT CAST(value AS UTINYINT) FROM dataset'),
])
def test_unsafe_conversions_stop_before_sql_execution(values,sql):
    raw = pd.DataFrame({'value':values})
    planner = MagicMock(); planner.plan.return_value=sql
    agent=DataAnalystAgent(raw,profile_dataframe(raw),query_planner=planner)
    with patch('app.core.agent.SQLExecutor.execute') as execute:
        with pytest.raises(AgentDataError, match='conversion|Date'):
            agent.run('Filter or aggregate values',False,False)
        execute.assert_not_called()


@pytest.mark.parametrize('sql', [
    'SELECT TRY_CAST(REPLACE(REPLACE(value, \'$\', \'\'), \',\', \'\') AS DOUBLE) FROM dataset',
    'SELECT CAST(value AS DECIMAL(10,2)) FROM dataset',
])
def test_supported_lossless_casts_pass(sql):
    values = ['$1,200.50',None,'$2,000.25'] if 'REPLACE' in sql else ['1200.50',None,'2000.25']
    check_conversion_fidelity(sql,pd.DataFrame({'value':values}))


def test_calendar_literals_and_prepared_temporal_columns_pass():
    frame = pd.DataFrame({'date':pd.to_datetime(['2023-12-07',None])})
    check_conversion_fidelity('SELECT * FROM dataset WHERE CAST(date AS DATE) > DATE \'2022-12-31\'',frame)


def test_exact_large_decimal_and_hugeint_conversions_pass():
    frame = pd.DataFrame({'value':['123456789012345678901234567890.12']})
    check_conversion_fidelity('SELECT CAST(value AS DECIMAL(38,2)) FROM dataset',frame)
    frame = pd.DataFrame({'value':['123456789012345678901234567890']})
    check_conversion_fidelity('SELECT CAST(value AS HUGEINT) FROM dataset',frame)


def test_derived_alias_is_never_checked_against_original_column_with_same_name():
    frame = pd.DataFrame({'value':[123]})
    with pytest.raises(SemanticConversionError,match='cannot be checked'):
        check_conversion_fidelity('WITH changed AS (SELECT value / 100.0 AS value FROM dataset) SELECT CAST(value AS INTEGER) FROM changed',frame)


def test_nonadditive_sql_uses_bar_instead_of_pie():
    result = qr(pd.DataFrame({'category':['A','B'],'metric':[100.,200.]}))
    sql = 'SELECT category, AVG(salary) AS metric FROM dataset GROUP BY category'
    assert generate_chart(result,sql=sql)['chart_type']=='bar'
    assert generate_chart(result,chart_type='pie',sql=sql) is None


def test_average_chart_never_sums_other_and_multidimension_result_is_declined():
    averages = qr(pd.DataFrame({'department':['A','B','C'],'average_salary':[100.,200.,300.]}))
    spec=generate_chart(averages,chart_type='bar',max_categories=2,sql='SELECT department, AVG(salary) AS average_salary FROM dataset GROUP BY department')
    assert spec['has_other'] is False
    assert spec['omitted_rows'] == 1
    assert [row['value'] for row in spec['data']] == [300.,200.]
    multidim=qr(pd.DataFrame({'region':['East','East'],'year':['2022','2023'],'average_salary':[100.,200.]}))
    assert generate_chart(multidim) is None


@pytest.mark.parametrize('aggregate', ['AVG(value)', 'COUNT(DISTINCT value)', 'SUM(value)/COUNT(*)'])
def test_nonadditive_metrics_never_gain_other(aggregate):
    result=qr(pd.DataFrame({'category':['A','B','C'],'metric':[1.,2.,3.]}))
    spec=generate_chart(result,chart_type='bar',max_categories=2,sql=f'SELECT category, {aggregate} AS metric FROM dataset GROUP BY category')
    assert not spec['has_other']
    assert spec['omitted_rows'] == 1


def test_duplicate_unproven_categories_decline_chart_and_null_category_survives():
    assert generate_chart(qr(pd.DataFrame({'category':['A','A'],'average':[1.,2.]}))) is None
    spec=generate_chart(qr(pd.DataFrame({'category':['A',None],'metric':[1.,2.]})),chart_type='bar')
    assert 'Missing (NULL)' in [row['category'] for row in spec['data']]
    assert spec['omitted_rows'] == 0


def test_genuine_other_category_is_not_collided_with_summary():
    result=qr(pd.DataFrame({'category':['Other','A','B'],'metric':[10.,9.,8.]}))
    spec=generate_chart(result,chart_type='bar',max_categories=2,sql='SELECT category, SUM(value) AS metric FROM dataset GROUP BY category')
    assert not spec['has_other']
    assert spec['omitted_rows'] == 1


def test_additive_other_preserves_missing_metrics_and_categorical_nulls():
    frame = pd.DataFrame({'category':pd.Categorical(['A','B',None]),'revenue':[10.,5.,3.],'cost':[2.,None,None]})
    spec = generate_chart(qr(frame), chart_type='grouped_bar', max_categories=2,
                          sql='SELECT category, SUM(revenue) AS revenue, SUM(cost) AS cost FROM dataset GROUP BY category')
    assert spec['has_other']
    assert spec['data'][-1]['revenue'] == 8.0
    assert spec['data'][-1]['cost'] is None


@pytest.mark.parametrize('chart_type', ['line','scatter'])
def test_sampling_includes_full_range_and_discloses_omissions(chart_type):
    frame=pd.DataFrame({'x':pd.date_range('2020-01-01',periods=36,freq='MS') if chart_type=='line' else range(36),'y':range(36)})
    spec=generate_chart(qr(frame),chart_type=chart_type,max_rows=20)
    assert spec['sampled']
    assert spec['omitted_rows']==16
    assert spec['displayed_rows']==20
    assert spec['data'][-1]['y']==35
    assert spec['data'][0]['y']==0


def test_question_and_total_prompt_limits_stop_requests(monkeypatch):
    import app.core.query_planner as qm
    model=MagicMock()
    planner=QueryPlanner(llm_client=model)
    profile=profile_dataframe(pd.DataFrame({'value':[1]}))
    with pytest.raises(QueryPlanningError,match='too long'):
        planner.plan('x'*2001,profile)
    monkeypatch.setattr(qm,'config',replace(qm.config,max_question_chars=2_000))
    profile.columns[0].name='x'*50_000
    with pytest.raises(QueryPlanningError,match='input size'):
        planner.plan('Count rows',profile)
    model.generate_text.assert_not_called()


def test_sample_values_are_bounded_identifiers_preserved_and_notice_included():
    frame=pd.DataFrame({'Exact Column':['a'*200_000]})
    profile=profile_dataframe(frame)
    prompt=QueryPlanner._build_prompt('Count rows',profile)
    assert len(prompt)<10_000
    assert 'Exact Column' in prompt
    assert 'sample shortened' in prompt
    assert len(profile.sample_rows[0]['Exact Column'])==200_000
    summary=_build_result_summary(qr(frame),20)
    assert len(summary)<500
    assert 'sample shortened' in summary


def test_explanation_budget_rejects_oversized_schema_before_model():
    model=MagicMock()
    result=qr(pd.DataFrame({'x'*50_000:['value']}))
    with pytest.raises(ExplainerLLMError,match='input size'):
        explain_query_result('SELECT * FROM dataset',result,llm_client=model)
    model.generate_text.assert_not_called()


def test_retained_dataset_budget_and_finalizer_release(monkeypatch):
    import app.core.agent as am
    budget=DatasetMemoryBudget(2000)
    monkeypatch.setattr(am,'dataset_memory_budget',budget)
    frame=pd.DataFrame({'value':['text']})
    agent=DataAnalystAgent(frame,profile_dataframe(frame),query_planner=MagicMock())
    assert budget.used_bytes>0
    del agent
    gc.collect()
    assert budget.used_bytes==0
    lease=budget.reserve(1500)
    with pytest.raises(DatasetResourceError):budget.reserve(501)
    lease.release();lease.release()
    assert budget.used_bytes==0


def test_preparation_gate_serializes_threads_and_is_reentrant():
    import threading
    errors=[]
    def attempt():
        try:
            with dataset_preparation_slot():pass
        except DatasetResourceError:errors.append(True)
    with dataset_preparation_slot():
        with dataset_preparation_slot():pass
        thread=threading.Thread(target=attempt);thread.start();thread.join(timeout=2)
    assert errors==[True]
    with dataset_preparation_slot():pass


@pytest.mark.parametrize('setting,value,text',[
    ('max_dataset_rows',1,'a\n1\n2\n'),
    ('max_dataset_columns',1,'a,b\n1,2\n'),
    ('max_dataset_memory_mb',1,'a\n'+'x'*600_000+'\n'),
], ids=['rows','columns','memory'])
def test_loader_budgets_reject_before_pandas_allocation(monkeypatch,setting,value,text):
    import app.core.data_loader as dl
    monkeypatch.setattr(dl,'config',replace(dl.config,**{setting:value}))
    with patch.object(dl.pd,'read_csv') as parse:
        with pytest.raises((DatasetResourceError,CSVParsingError)):
            load_csv(upload(text))
        parse.assert_not_called()


def test_sql_literals_and_exception_details_do_not_enter_application_logs():
    from app.utils import security
    stream=StringIO()
    logger=security.logger
    handler=logging.StreamHandler(stream)
    for f in logger.handlers[0].filters:handler.addFilter(f)
    with patch.object(logger,'handlers',[handler]):
        validate_sql("SELECT * FROM dataset WHERE name = 'private@example.test'",['dataset'])
        try:raise RuntimeError('private-token-and-data')
        except RuntimeError:logger.exception('Safe error summary')
    assert 'private@example.test' not in stream.getvalue()
    assert 'private-token-and-data' not in stream.getvalue()
    assert 'sql_length=' in stream.getvalue()
    assert 'Safe error summary' in stream.getvalue()
