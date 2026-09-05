import time
import threading
from collections import deque
from typing import Deque, Tuple

_lock = threading.Lock()
_requests: Deque[Tuple[float, str]] = deque()

WINDOW_SECONDS = 30


def record_request(uuid: str) -> None:
    """Record a request timestamp for OPS-API-2's rolling window count."""
    with _lock:
        _requests.append((time.time(), uuid))
        _prune_locked()


def _prune_locked() -> None:
    """Remove entries older than WINDOW_SECONDS. Caller must hold _lock."""
    cutoff = time.time() - WINDOW_SECONDS
    while _requests and _requests[0][0] < cutoff:
        _requests.popleft()


def count_recent_requests() -> int:
    """Number of requests recorded in the last WINDOW_SECONDS."""
    with _lock:
        _prune_locked()
        return len(_requests)


def count_recent_unique_users() -> int:
    """Number of distinct UUIDs seen in the last WINDOW_SECONDS."""
    with _lock:
        _prune_locked()
        return len({uuid for _, uuid in _requests})