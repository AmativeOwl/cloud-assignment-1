import os
import base64
import random
import logging
from typing import List

import httpx

logger = logging.getLogger("smartpark.carpark_client")

CARPARK_SIM_URL = os.environ.get("CARPARK_SIM_URL", "http://localhost:9000")
NUM_CARPARKS = int(os.environ.get("NUM_CARPARKS", "20"))
CARPARK_IDS = [f"CBD_{i:03d}" for i in range(1, NUM_CARPARKS + 1)]


def get_all_carpark_ids() -> List[str]:
    return CARPARK_IDS


def sample_carpark_ids(count: int) -> List[str]:
    count = min(count, len(CARPARK_IDS))
    return random.sample(CARPARK_IDS, count)


async def fetch_photo(client: httpx.AsyncClient, carpark_id: str) -> bytes:
    """Calls the simulator's /api/takephoto endpoint. Raises on failure."""
    url = f"{CARPARK_SIM_URL}/api/takephoto"
    response = await client.get(url, params={"carpark_id": carpark_id}, timeout=5.0)
    response.raise_for_status()
    data = response.json()
    return base64.b64decode(data["image_base64"])