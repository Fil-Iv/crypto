# rate_limiter.py
from __future__ import annotations
import time
import threading
from contextlib import contextmanager

class RateLimiter:
    """Simple token-bucket rate limiter (thread-safe).
    Example: RateLimiter(rate=10, per=1.0)  # 10 ops / second
    """
    def __init__(self, rate: int = 10, per: float = 1.0):
        self.capacity = max(1, int(rate))
        self.per = float(per)
        self._tokens = float(self.capacity)
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last
        self._last = now
        # tokens/second = capacity / per
        self._tokens = min(self.capacity, self._tokens + elapsed * (self.capacity / self.per))

    def acquire(self, tokens: int = 1, timeout: float | None = None) -> bool:
        """Try to acquire `tokens`. Block up to `timeout` seconds if provided."""
        deadline = None if timeout is None else (time.monotonic() + timeout)
        while True:
            with self._lock:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return True
            if deadline is not None and time.monotonic() >= deadline:
                return False
            time.sleep(0.01)

    @contextmanager
    def limit(self, tokens: int = 1, timeout: float | None = None):
        ok = self.acquire(tokens=tokens, timeout=timeout)
        if not ok:
            raise TimeoutError("RateLimiter timeout")
        try:
            yield
        finally:
            pass
