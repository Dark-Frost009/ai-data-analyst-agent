"""Resource isolation, provider failure recovery and crash cleanup."""
from concurrent.futures import ThreadPoolExecutor
import io
from pathlib import Path
import threading
from unittest.mock import Mock

import pytest

from app.core.agent import DataAnalystAgent, AgentCapacityError, AgentPlanningError
from app.core.disk_dataset import DiskDataset, disk_budget
from app.core.llm_client import LLMCredentialsError, LLMAccessDeniedError, LLMThrottlingError, LLMAPIError
from app.core.query_planner import QueryPlanningError
from app.ui.analysis import _planning_error_message
from app.utils.request_budget import RequestBudget, RequestBudgetExceeded
from app.utils.temporary_storage import MARKER, _lock, reap_orphaned_runtimes


def test_hourly_budget_resets_and_counts_concurrent_requests():
    now = [0]
    budget = RequestBudget(3, window_seconds=10, clock=lambda: now[0])
    def attempt(_):
        try:
            budget.acquire()
            return True
        except RequestBudgetExceeded:
            return False
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert sum(pool.map(attempt, range(20))) == 3
    now[0] = 10
    budget.acquire()


def test_exhausted_budget_prevents_provider_request(monkeypatch):
    from app.core import llm_client
    monkeypatch.setenv('GROQ_API_KEY', 'synthetic-test-only')
    budget = RequestBudget(1)
    budget.acquire()
    sdk = Mock()
    monkeypatch.setattr(llm_client, 'provider_request_budget', budget)
    monkeypatch.setattr(llm_client.groq, 'Groq', sdk)
    with pytest.raises(LLMThrottlingError):
        llm_client.GroqClient().generate_text('synthetic question')
    sdk.assert_not_called()


@pytest.mark.parametrize('error,fragment', [(LLMCredentialsError,'credentials'), (LLMAccessDeniedError,'model'), (LLMThrottlingError,'quota'), (LLMAPIError,'reached')])
def test_provider_failure_messages_are_specific_and_private(error, fragment):
    outer = AgentPlanningError('private question')
    outer.__cause__ = QueryPlanningError('private SQL')
    outer.__cause__.__cause__ = error('private provider payload')
    message = _planning_error_message(outer)
    assert fragment in message
    assert 'private' not in message.replace('private deployment settings', '')


def test_concurrent_sessions_fail_busy_then_recover_without_crossing_data():
    entered, release = threading.Event(), threading.Event()
    first = DiskDataset(io.BytesIO(b'amount\n10\n20\n'))
    second = DiskDataset(io.BytesIO(b'amount\n100\n200\n'))
    planner = Mock()
    def blocked_plan(**kwargs):
        entered.set()
        assert release.wait(5)
        raise QueryPlanningError('synthetic failure')
    planner.plan.side_effect = blocked_plan
    a = DataAnalystAgent(first.preview, first.raw_profile, query_planner=planner, disk_dataset=first)
    bplanner = Mock()
    bplanner.plan.return_value = 'SELECT SUM(amount) AS total FROM dataset'
    b = DataAnalystAgent(second.preview, second.raw_profile, query_planner=bplanner, disk_dataset=second)
    before = disk_budget.used_bytes
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            pending = pool.submit(a.run, 'total', False, False)
            assert entered.wait(5)
            with pytest.raises(AgentCapacityError):
                b.run('total', False, False)
            bplanner.plan.assert_not_called()
            release.set()
            with pytest.raises(AgentPlanningError):
                pending.result(5)
        assert b.run('total', False, False).query_result.dataframe.iloc[0, 0] == 300
        planner.plan.side_effect = None
        planner.plan.return_value = 'SELECT SUM(amount) AS total FROM dataset'
        assert a.run('total', False, False).query_result.dataframe.iloc[0, 0] == 30
        assert disk_budget.used_bytes == before
    finally:
        release.set()
        a.release_dataset()
        b.release_dataset()


def test_cleanup_removes_only_unlocked_marked_runtimes(tmp_path):
    orphan = tmp_path/'analyst-runtime-orphan'
    orphan.mkdir()
    (orphan/'owner.lock').write_bytes(MARKER)
    (orphan/'private.csv').write_text('synthetic')
    live = tmp_path/'analyst-runtime-live'
    live.mkdir()
    owner = (live/'owner.lock').open('w+b')
    owner.write(MARKER)
    owner.flush()
    _lock(owner)
    unrelated = tmp_path/'analyst-runtime-unrelated'
    unrelated.mkdir()
    (unrelated/'owner.lock').write_bytes(b'not our application')
    try:
        assert reap_orphaned_runtimes(tmp_path) == 1
        assert not orphan.exists()
        assert live.exists()
        assert unrelated.exists()
    finally:
        owner.close()
    assert reap_orphaned_runtimes(tmp_path) == 1
    assert unrelated.exists()
