import time
import threading
from typing import Dict, Optional

_lock = threading.Lock()
_cache: Dict[str, Dict] = {}


def update_carpark(carpark_id: str, available_spaces: int, confidence_score: float) -> None:
    """Store the latest known state for a car park, called opportunistically
    whenever find-carparks or annotate-carpark successfully processes it."""
    with _lock:
        _cache[carpark_id] = {
            "carpark_id": carpark_id,
            "available_spaces": available_spaces,
            "confidence_score": confidence_score,
            "last_updated": time.time(),
        }


def get_all_known() -> Dict[str, Dict]:
    """Return a snapshot of all cached car park states."""
    with _lock:
        return dict(_cache)


def get_carpark(carpark_id: str) -> Optional[Dict]:
    with _lock:
        return _cache.get(carpark_id)