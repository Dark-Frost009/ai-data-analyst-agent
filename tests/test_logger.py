import logging
from uuid import uuid4

from app.utils.logger import (
    analysis_log_context,
    get_analysis_id,
    get_logger,
)


def test_analysis_log_context_sets_and_resets_identifier():
    assert get_analysis_id() == "-"

    with analysis_log_context() as analysis_id:
        assert len(analysis_id) == 12
        assert get_analysis_id() == analysis_id

    assert get_analysis_id() == "-"


def test_nested_analysis_log_context_restores_outer_identifier():
    with analysis_log_context() as outer_id:
        with analysis_log_context() as inner_id:
            assert inner_id != outer_id
            assert get_analysis_id() == inner_id

        assert get_analysis_id() == outer_id


def test_log_handler_attaches_current_analysis_identifier():
    logger = get_logger(f"tests.logger.{uuid4().hex}")
    handler = logger.handlers[0]

    with analysis_log_context() as analysis_id:
        record = logger.makeRecord(
            logger.name,
            logging.INFO,
            __file__,
            1,
            "test message",
            (),
            None,
        )

        for log_filter in handler.filters:
            log_filter.filter(record)

    assert record.analysis_id == analysis_id