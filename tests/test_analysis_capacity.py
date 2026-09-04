"""Tests for the in-process analysis concurrency guard."""

import pytest

from app.utils.analysis_capacity import (
    AnalysisCapacityExceededError,
    AnalysisCapacityLimiter,
)


def test_limiter_rejects_work_when_all_slots_are_reserved():
    limiter = AnalysisCapacityLimiter(max_concurrent_analyses=1)

    with limiter.acquire():
        with pytest.raises(AnalysisCapacityExceededError):
            with limiter.acquire():
                pass


def test_limiter_releases_a_slot_after_the_analysis_finishes():
    limiter = AnalysisCapacityLimiter(max_concurrent_analyses=1)

    with limiter.acquire():
        pass

    with limiter.acquire():
        pass


@pytest.mark.parametrize("limit", [0, -1])
def test_limiter_rejects_non_positive_capacity(limit):
    with pytest.raises(ValueError):
        AnalysisCapacityLimiter(max_concurrent_analyses=limit)