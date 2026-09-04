"""Per-process concurrency control for complete analysis runs."""

from contextlib import contextmanager
import threading
from typing import Iterator

from app.config import config


class AnalysisCapacityExceededError(Exception):
    """Raised when this process has no free analysis capacity."""


class AnalysisCapacityLimiter:
    """A non-blocking, thread-safe limiter for in-flight analyses."""

    def __init__(self, max_concurrent_analyses: int) -> None:
        if max_concurrent_analyses <= 0:
            raise ValueError(
                "max_concurrent_analyses must be greater than zero"
            )

        self._semaphore = threading.BoundedSemaphore(
            max_concurrent_analyses
        )

    @contextmanager
    def acquire(self) -> Iterator[None]:
        """Reserve one slot or fail immediately without queuing work."""
        if not self._semaphore.acquire(blocking=False):
            raise AnalysisCapacityExceededError(
                "All analysis slots are currently in use."
            )

        try:
            yield
        finally:
            self._semaphore.release()


_analysis_limiter = AnalysisCapacityLimiter(
    config.max_concurrent_analyses
)


def acquire_analysis_slot() -> Iterator[None]:
    """Return a context manager reserving one process-wide analysis slot."""
    return _analysis_limiter.acquire()