"""Process-wide request throttling without retries or external infrastructure."""
from collections import deque
import threading
import time

from app.config import config


class RequestBudgetExceeded(Exception):
    """The configured rolling request budget is exhausted."""


class RequestBudget:
    def __init__(self, maximum, window_seconds=3600, clock=time.monotonic):
        if maximum <= 0 or window_seconds <= 0:
            raise ValueError("Budget and window must be positive")
        self.maximum = maximum
        self.window = window_seconds
        self.clock = clock
        self._requests = deque()
        self._lock = threading.Lock()

    def acquire(self):
        now = self.clock()
        with self._lock:
            while self._requests and self._requests[0] <= now - self.window:
                self._requests.popleft()
            if len(self._requests) >= self.maximum:
                raise RequestBudgetExceeded("The service's hourly AI request budget is full. Try again later.")
            self._requests.append(now)


provider_request_budget = RequestBudget(config.max_llm_requests_per_hour)
