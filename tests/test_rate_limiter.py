"""Guards for the shared Alpha Vantage quota.

A run over a 20-symbol watchlist issues paginated fetches back to back. Alpha
Vantage answers an over-quota request with HTTP 200 and an explanatory body,
which is why exceeding the limit used to read as "this symbol had no news".
NewsUnavailable turned that into a loud failure; the limiter stops it
happening.

The cross-process case is the one worth pinning: the web app and the scheduler
subprocess are separate processes, so a module-level bucket would let each use
the full quota and together double it.

FileSemaphore is the same problem counted in concurrent calls rather than
calls per minute, and it exists for the same reason: a manual run's model
stage is a forked background-callback subprocess, so a threading.Semaphore in
the web process bounded one run's own fan-out and nothing at all between two
people running at once. The tests that matter are therefore the two the file
buys — a second PROCESS waits, and a holder that dies frees its slot.
"""

import multiprocessing as mp
import os
import signal
import time

import pytest

from services.rate_limiter import FileSemaphore, RateLimitTimeout, TokenBucket


@pytest.fixture
def bucket_dir(tmp_path):
    return tmp_path / "limits"


def _burst(state_dir, n, window, limit, q):
    """Run in a child process: spend n calls against the shared bucket."""
    b = TokenBucket("shared", limit=limit, window_s=window, state_dir=state_dir)
    start = time.time()
    for _ in range(n):
        b.acquire(timeout=30)
    q.put(time.time() - start)


def _hold(state_dir, seconds, q):
    """Run in a child process: take the one slot and keep it."""
    sem = FileSemaphore("shared", 1, state_dir=state_dir, poll_s=0.02)
    sem.acquire()
    q.put(os.getpid())
    time.sleep(seconds)
    sem.release()


class TestFileSemaphore:
    def test_a_second_process_waits_for_the_slot(self, bucket_dir):
        q = mp.Queue()
        child = mp.Process(target=_hold, args=(bucket_dir, 1.0, q))
        child.start()
        assert q.get(timeout=30)                     # the child holds it now

        sem = FileSemaphore("shared", 1, state_dir=bucket_dir, poll_s=0.02)
        start = time.time()
        sem.acquire()
        waited = time.time() - start
        sem.release()
        child.join(timeout=30)
        assert waited > 0.4, (
            f"acquired in {waited:.2f}s - the slot is not shared across "
            f"processes")

    def test_a_killed_holder_frees_its_slot(self, bucket_dir):
        """The reason this is flock and not a pid file: the scheduler kills a
        run that overruns, and a slot stranded by that would shrink the
        ceiling for every later run until the box restarted."""
        q = mp.Queue()
        child = mp.Process(target=_hold, args=(bucket_dir, 60.0, q))
        child.start()
        pid = q.get(timeout=30)
        os.kill(pid, signal.SIGKILL)
        child.join(timeout=30)

        sem = FileSemaphore("shared", 1, state_dir=bucket_dir, poll_s=0.02)
        start = time.time()
        sem.acquire()
        assert time.time() - start < 1.0
        sem.release()

    def test_holders_up_to_the_count_do_not_wait(self, bucket_dir):
        sem = FileSemaphore("t", 3, state_dir=bucket_dir, poll_s=0.02)
        start = time.time()
        for _ in range(3):
            sem.acquire()
        assert time.time() - start < 0.5
        for _ in range(3):
            sem.release()

    def test_zero_slots_disables_the_bound(self, bucket_dir):
        """The escape hatch, and the shape a missing state directory falls
        back to: unbounded searching is a cost problem, a crashed run is a
        correctness one."""
        sem = FileSemaphore("off", 0, state_dir=bucket_dir)
        start = time.time()
        for _ in range(50):
            sem.acquire()
        assert time.time() - start < 0.5
        for _ in range(50):
            sem.release()
        assert not bucket_dir.exists()

    def test_releasing_what_this_thread_never_took_is_an_error(self, bucket_dir):
        sem = FileSemaphore("t", 2, state_dir=bucket_dir)
        with pytest.raises(RuntimeError, match="holds no slot"):
            sem.release()


class TestPacing:
    def test_calls_within_the_limit_do_not_wait(self, bucket_dir):
        b = TokenBucket("t", limit=5, window_s=2.0, state_dir=bucket_dir)
        start = time.time()
        for _ in range(5):
            b.acquire(timeout=10)
        assert time.time() - start < 0.5

    def test_the_call_past_the_limit_waits_for_the_window(self, bucket_dir):
        b = TokenBucket("t", limit=3, window_s=1.5, state_dir=bucket_dir)
        for _ in range(3):
            b.acquire(timeout=10)
        start = time.time()
        b.acquire(timeout=10)
        waited = time.time() - start
        assert 1.0 < waited < 2.5, f"waited {waited:.2f}s"

    def test_a_full_bucket_drains_as_the_window_slides(self, bucket_dir):
        b = TokenBucket("t", limit=2, window_s=1.0, state_dir=bucket_dir)
        b.acquire(); b.acquire()
        time.sleep(1.1)
        start = time.time()
        b.acquire(timeout=5)
        assert time.time() - start < 0.3, "expired calls should free slots"


class TestCrossProcess:
    def test_two_processes_share_one_quota(self, bucket_dir):
        """The reason this is file-backed rather than a module global."""
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        procs = [ctx.Process(target=_burst, args=(bucket_dir, 4, 2.0, 4, q))
                 for _ in range(2)]
        start = time.time()
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=60)
        total = time.time() - start

        # 8 calls against 4-per-2s must cross a window boundary. Two
        # independent in-process buckets would finish immediately.
        assert total > 1.5, (
            f"8 calls at 4/2s finished in {total:.2f}s - quota not shared")


class TestEdges:
    def test_timeout_raises_rather_than_blocking_forever(self, bucket_dir):
        b = TokenBucket("t", limit=1, window_s=60.0, state_dir=bucket_dir)
        b.acquire()
        with pytest.raises(RateLimitTimeout, match="no slot"):
            b.acquire(timeout=0.5)

    def test_zero_limit_disables_limiting(self, bucket_dir):
        """Free-tier users cap by day, not minute, and opt out with 0."""
        b = TokenBucket("off", limit=0, state_dir=bucket_dir)
        start = time.time()
        for _ in range(100):
            b.acquire()
        assert time.time() - start < 0.5

    def test_corrupt_state_does_not_crash_the_caller(self, bucket_dir):
        """A torn write should cost throughput, not the run."""
        b = TokenBucket("t", limit=2, window_s=1.0, state_dir=bucket_dir)
        b.acquire()
        b._path.write_text("{not json")
        assert b.acquire(timeout=5) >= 0.0

    def test_reset_clears_recorded_calls(self, bucket_dir):
        b = TokenBucket("t", limit=1, window_s=60.0, state_dir=bucket_dir)
        b.acquire()
        b.reset()
        assert b.acquire(timeout=1) == 0.0
