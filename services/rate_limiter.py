"""Cross-process rate limiting for third-party APIs.

Alpha Vantage answers a throttled request with HTTP 200 and an explanatory
body, so exceeding the quota does not look like an error at the transport
layer. Before ``NewsUnavailable`` existed that surfaced as a symbol silently
getting no news; it now surfaces as a loud failure, which is better but still
a failed run. This module prevents the condition instead of reporting it.

The limiter is file-backed rather than a module-level object because the
callers do not share a process: the web app serves the UI, while scheduled
jobs shell out to ``scripts/daily_analysis.py``. Two independent in-process
buckets would each happily use the full quota and together exceed it. A small
JSON file under an ``flock`` gives every process on the box one shared view.

The state file holds the timestamps of recent calls. Acquiring prunes anything
older than the window, and either records a new call or sleeps until the
oldest one ages out.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_STATE_DIR = Path(os.getenv("RATE_LIMIT_STATE_DIR", "cache/rate_limits"))


class RateLimitTimeout(RuntimeError):
    """Waiting for a slot exceeded the caller's patience."""


class TokenBucket:
    """A fixed number of calls per rolling window, shared across processes.

    Args:
        name: Identifies the quota. Callers sharing a quota must share a name.
        limit: Calls permitted per window. Values below 1 disable limiting.
        window_s: Length of the rolling window in seconds.
        state_dir: Where to keep the shared state file.
    """

    def __init__(self, name: str, limit: int, window_s: float = 60.0,
                 state_dir: Path | None = None):
        self.name = name
        self.limit = int(limit)
        self.window_s = float(window_s)
        self._dir = Path(state_dir) if state_dir else _STATE_DIR
        self._path = self._dir / f"{name}.json"
        if self.limit > 0:
            self._dir.mkdir(parents=True, exist_ok=True)

    # -- state file ------------------------------------------------------
    def _read(self, fh) -> list[float]:
        fh.seek(0)
        raw = fh.read()
        if not raw.strip():
            return []
        try:
            calls = json.loads(raw)
            return [float(t) for t in calls] if isinstance(calls, list) else []
        except (json.JSONDecodeError, TypeError, ValueError):
            # A torn write should cost one window of throughput, not a crash.
            logger.warning("rate limiter %s: unreadable state, resetting", self.name)
            return []

    def _write(self, fh, calls: list[float]) -> None:
        fh.seek(0)
        fh.truncate()
        fh.write(json.dumps(calls))
        fh.flush()
        os.fsync(fh.fileno())

    # -- public ----------------------------------------------------------
    def acquire(self, timeout: float = 120.0) -> float:
        """Block until a call is permitted. Returns seconds spent waiting.

        Raises:
            RateLimitTimeout: no slot became free within ``timeout``.
        """
        if self.limit <= 0:
            return 0.0

        deadline = time.monotonic() + timeout
        waited = 0.0
        while True:
            with open(self._path, "a+") as fh:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
                try:
                    now = time.time()
                    calls = [t for t in self._read(fh) if now - t < self.window_s]
                    if len(calls) < self.limit:
                        calls.append(now)
                        self._write(fh, calls)
                        return waited
                    # Sleep only as long as the oldest call needs to expire,
                    # computed while holding the lock so it cannot go stale.
                    sleep_for = self.window_s - (now - min(calls)) + 0.01
                finally:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)

            if time.monotonic() + sleep_for > deadline:
                raise RateLimitTimeout(
                    f"{self.name}: no slot within {timeout:.0f}s "
                    f"({self.limit} calls per {self.window_s:.0f}s)")
            if waited == 0.0:
                logger.info("rate limiter %s: at capacity, pacing requests",
                            self.name)
            time.sleep(sleep_for)
            waited += sleep_for

    def reset(self) -> None:
        """Drop recorded calls. For tests and manual recovery."""
        if self._path.exists():
            self._path.unlink()


_buckets: dict[str, TokenBucket] = {}


def get_bucket(name: str, limit: int, window_s: float = 60.0) -> TokenBucket:
    """Return the shared bucket for ``name``, creating it on first use."""
    key = f"{name}:{limit}:{window_s}"
    if key not in _buckets:
        _buckets[key] = TokenBucket(name, limit, window_s)
    return _buckets[key]


def alpha_vantage_bucket() -> TokenBucket:
    """The quota shared by every Alpha Vantage caller in this deployment."""
    from config import API
    return get_bucket("alpha_vantage", API.ALPHA_VANTAGE_CALLS_PER_MIN, 60.0)
