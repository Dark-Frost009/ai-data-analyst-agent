"""Full-dataset correctness and resource lifecycle of large CSV preparation."""
import io
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from app.core import disk_dataset as module
from app.core.disk_dataset import DiskDataset, DiskSQLExecutor, disk_budget
from app.core.data_loader import CSVParsingError, FileTooLargeError
from app.core.semantic_checks import SemanticConversionError
from app.utils.dataset_resources import DatasetResourceError
from app.utils.security import validate_sql


def query(store, sql):
    with DiskSQLExecutor(store) as executor:
        return executor.execute(validate_sql(sql, allowed_tables=['dataset'], max_rows=10000)).dataframe


def test_full_dataset_dates_and_sample_metadata():
    # US ordering evidence arrives beyond the first batch and preview.
    content = 'id,Join_Date,code\n' + '1,12/7/2023,001\n' * 5000 + '2,12/31/2021,NA\n'
    original = content.encode()
    upload = io.BytesIO(original)
    before = disk_budget.used_bytes
    store = DiskDataset(upload)
    root = store.root
    try:
        assert upload.getvalue() == original
        assert store.raw.read_bytes() == original
        assert store.types == ['Int64', 'us', 'text']
        assert len(store.preview) == 1000
        assert store.execution_profile.row_count == 5001
        assert store.execution_profile.statistics_sampled
        assert store.execution_profile.columns[1].pandas_dtype == 'datetime64[ns]'
        result = query(store, 'SELECT COUNT(*) AS n FROM dataset WHERE "Join_Date" > DATE \'2022-12-31\'')
        assert result.iloc[0, 0] == 5000
        assert query(store, 'SELECT code FROM dataset WHERE id = 2').iloc[0, 0] == 'NA'
    finally:
        store.close()
    assert not root.exists()
    assert disk_budget.used_bytes == before


@pytest.mark.parametrize('last', ['invalid', '31/12/2023', '2023-12-31'])
def test_late_mixed_values_decline_conversion(last):
    store = DiskDataset(io.BytesIO(('d\n' + '12/31/2023\n' * 5000 + last + '\n').encode()))
    try:
        assert store.types == ['text']
        with pytest.raises(SemanticConversionError):
            store.check_conversions('SELECT TRY_CAST(d AS DATE) FROM dataset')
    finally:
        store.close()


@pytest.mark.parametrize('values,kind', [('01/02/2023\n03/04/2024', 'text'), ('2022\n2023', 'text'), ('2023-01-31\n2024-02-29', 'iso')])
def test_conservative_dates(values, kind):
    store = DiskDataset(io.BytesIO(('d\n' + values + '\n').encode()))
    try:
        assert store.types == [kind]
    finally:
        store.close()


def test_nulls_identifiers_and_big_numbers():
    store = DiskDataset(io.BytesIO(b'id,n,d,empty\n001,9007199254740993,2023-01-31,\nNA,,2024-02-29,\nNULL,2,,\n'))
    try:
        assert store.types == ['text', 'Int64', 'iso', 'text']
        result = query(store, 'SELECT * FROM dataset')
        assert list(result.id) == ['001', 'NA', 'NULL']
        assert result.n.iloc[0] == 9007199254740993
        assert result.n.isna().sum() == 1
        assert result.d.isna().sum() == 1
        assert result['empty'].isna().all()
        assert store.raw_profile.columns[1].null_count == 1
    finally:
        store.close()


@pytest.mark.parametrize('content', [b'x,x\n1,2\n', b'x,y\n1\n', b'x\n"unfinished\n'])
def test_bad_csv_releases_disk_capacity(content):
    before = disk_budget.used_bytes
    with pytest.raises(CSVParsingError):
        DiskDataset(io.BytesIO(content))
    assert disk_budget.used_bytes == before


def test_clear_defers_delete_until_executor_closed():
    store = DiskDataset(io.BytesIO(b'n\n1\n2\n'))
    executor = DiskSQLExecutor(store)
    store.close()
    assert store.root.exists()
    validation = validate_sql('SELECT SUM(n) AS n FROM dataset', allowed_tables=['dataset'])
    assert executor.execute(validation).dataframe.iloc[0, 0] == 3
    executor.close()
    executor.close()
    assert not store.root.exists()


def test_external_access_and_read_only_protection():
    store = DiskDataset(io.BytesIO(b'n\n1\n'))
    try:
        with DiskSQLExecutor(store) as executor:
            assert executor._conn.execute("SELECT current_setting('enable_external_access')").fetchone()[0] is False
            assert executor._conn.execute("SELECT current_setting('temp_directory')").fetchone()[0] == ''
            with pytest.raises(Exception):
                executor._conn.execute('DELETE FROM dataset')
            assert not validate_sql("SELECT * FROM read_csv_auto('/etc/passwd')", allowed_tables=['dataset']).is_valid
    finally:
        store.close()


@pytest.mark.parametrize('setting,value,error', [('max_upload_size_mb', 0, FileTooLargeError), ('max_disk_dataset_rows', 1, DatasetResourceError), ('max_dataset_disk_mb', 0, DatasetResourceError), ('dataset_load_timeout_seconds', 0, DatasetResourceError)])
def test_limits_cleanup(monkeypatch, setting, value, error):
    settings = dict(vars(module.config))
    settings[setting] = value
    monkeypatch.setattr(module, 'config', SimpleNamespace(**settings))
    before = disk_budget.used_bytes
    with pytest.raises(error):
        DiskDataset(io.BytesIO(b'n\n1\n2\n'))
    assert disk_budget.used_bytes == before


def test_agent_uses_entire_disk_dataset():
    from app.core.agent import DataAnalystAgent
    from unittest.mock import Mock
    store = DiskDataset(io.BytesIO(('n\n' + '2\n' * 6000).encode()))
    planner = Mock()
    planner.plan.return_value = 'SELECT SUM(n) AS total FROM dataset'
    agent = DataAnalystAgent(store.preview, store.raw_profile, query_planner=planner, disk_dataset=store)
    try:
        result = agent.run('total', generate_explanation=False, generate_chart_spec=False)
        assert result.query_result.dataframe.iloc[0, 0] == 12000
        assert planner.plan.call_args.kwargs['dataset_profile'].row_count == 6000
    finally:
        agent.release_dataset()
    assert not store.root.exists()


def test_wide_results_are_bounded():
    from app.core.sql_executor import QueryExecutionError
    store = DiskDataset(io.BytesIO(('note\n' + ('x' * 200000 + '\n') * 100).encode()))
    try:
        with pytest.raises(QueryExecutionError, match='too large'):
            query(store, 'SELECT note FROM dataset')
    finally:
        store.close()


def test_timeout_preserves_files_until_worker_finishes():
    import threading
    from app.core.sql_executor import QueryTimeoutError
    store = DiskDataset(io.BytesIO(b'n\n1\n'))
    executor = DiskSQLExecutor(store)
    original = executor._conn
    entered, finish = threading.Event(), threading.Event()
    class SlowConnection:
        def execute(self, sql):
            entered.set()
            finish.wait(5)
            return original.execute(sql)
        def interrupt(self):
            pass
        def close(self):
            original.close()
    executor._conn = SlowConnection()
    executor._timeout_seconds = 0.01
    validation = validate_sql('SELECT n FROM dataset', allowed_tables=['dataset'])
    try:
        with pytest.raises(QueryTimeoutError):
            executor.execute(validation)
        assert entered.is_set()
        store.close()
        executor.close()
        assert store.root.exists()
    finally:
        finish.set()
    for _ in range(100):
        if not store.root.exists():
            break
        threading.Event().wait(0.01)
    assert not store.root.exists()


def test_garbage_collection_removes_private_files():
    import gc
    before = disk_budget.used_bytes
    store = DiskDataset(io.BytesIO(b'n\n1\n'))
    root = store.root
    del store
    gc.collect()
    assert not root.exists()
    assert disk_budget.used_bytes == before


def test_shared_disk_budget_rejects_before_creating_files(monkeypatch):
    from app.utils.dataset_resources import DatasetMemoryBudget
    monkeypatch.setattr(module, 'disk_budget', DatasetMemoryBudget(1))
    with pytest.raises(DatasetResourceError):
        DiskDataset(io.BytesIO(b'n\n1\n'))
    assert module.disk_budget.used_bytes == 0


def test_oversized_field_is_rejected_and_cleaned():
    before = disk_budget.used_bytes
    with pytest.raises(CSVParsingError):
        DiskDataset(io.BytesIO(b'text\n' + b'x' * (1024**2 + 1) + b'\n'))
    assert disk_budget.used_bytes == before
