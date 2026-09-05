import time
import threading
from typing import Optional, Tuple, Dict, Any

_lock = threading.Lock()
_cache: Dict[Tuple[str, int], Tuple[float, Any]] = {}

TTL_SECONDS = 8

def get_cached(uuid: str, n: int) -> Optional[Any]:
    """Return a cached response for this (uuid, n) pair if still fresh."""
    key = (uuid, n)
    with _lock:
        entry = _cache.get(key)
        if entry is None:
            return None
        timestamp, value = entry
        if time.time() - timestamp > TTL_SECONDS:
            del _cache[key]
            return None
        return value


def set_cached(uuid: str, n: int, value: Any) -> None:
    key = (uuid, n)
    with _lock:
        _cache[key] = (time.time(), value)