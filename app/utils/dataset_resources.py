"""Bound dataset preparation and retained DataFrames across browser sessions."""
from contextlib import contextmanager
import threading

from app.config import config


class DatasetResourceError(Exception):
    """A dataset exceeds the configured hosting budget."""


class DatasetMemoryBudget:
    def __init__(self, maximum_bytes: int):
        self.maximum_bytes = maximum_bytes
        self.used_bytes = 0
        self._lock = threading.Lock()

    def reserve(self, size: int):
        with self._lock:
            if size > self.maximum_bytes - self.used_bytes:
                raise DatasetResourceError(
                    "Dataset memory capacity is full. Clear unused datasets or upload a smaller file."
                )
            self.used_bytes += size
        return DatasetMemoryLease(self, size)


class DatasetMemoryLease:
    def __init__(self, budget, size):
        self.budget = budget
        self.size = size

    def release(self):
        with self.budget._lock:
            self.budget.used_bytes -= self.size
            self.size = 0


dataset_memory_budget = DatasetMemoryBudget(config.max_total_dataset_memory_mb * 1024 * 1024)
_preparation_lock = threading.Lock()
_preparation_local = threading.local()


@contextmanager
def dataset_preparation_slot():
    """Serialize loading/profiling/copying; nested calls in one thread are safe."""
    nested = getattr(_preparation_local, "active", False)
    if not nested and not _preparation_lock.acquire(blocking=False):
        raise DatasetResourceError("Another dataset is being prepared. Please try uploading again shortly.")
    _preparation_local.active = True
    try:
        yield
    finally:
        _preparation_local.active = nested
        if not nested:
            _preparation_lock.release()


def check_dataframe_limits(dataframe):
    if len(dataframe) > config.max_dataset_rows or len(dataframe.columns) > config.max_dataset_columns:
        raise DatasetResourceError(
            f"Dataset exceeds the supported {config.max_dataset_rows:,} rows or "
            f"{config.max_dataset_columns} columns. Upload a smaller subset."
        )
    size = int(dataframe.memory_usage(index=True, deep=True).sum()) + int(dataframe.columns.memory_usage(deep=True))
    # Conservative reservation covers raw plus execution copy, even when strings share memory.
    if size * 2 > config.max_dataset_memory_mb * 1024 * 1024:
        raise DatasetResourceError("Dataset is too large in memory. Upload a smaller subset.")
    return size * 2
