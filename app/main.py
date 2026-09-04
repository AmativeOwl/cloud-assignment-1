import io
import time
import logging
import asyncio

import httpx
from fastapi import FastAPI, Query, HTTPException
from fastapi.concurrency import run_in_threadpool
from PIL import Image

from app.models import CarparkResult, FindCarparksResponse, AnnotateCarparkResponse
from app import inference
from app import carpark_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
logger = logging.getLogger("smartpark.main")

app = FastAPI(title="SmartPark Core Platform")


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/api/find-carparks", response_model=FindCarparksResponse)
async def find_carparks(
    uuid: str = Query(..., description="Unique identifier for the requesting client"),
    n: int = Query(3, ge=1, description="Number of top car parks to return"),
):
    start_time = time.time()

    max_n = len(carpark_client.get_all_carpark_ids())
    effective_n = min(n, max_n)  # what-if n > 100? cap it silently, no error
    query_count = min(effective_n * 2, max_n)  # spec: query >= 2n before returning n
    sampled_ids = carpark_client.sample_carpark_ids(query_count)

    logger.info(f"[{uuid}] find-carparks: n={n}, querying {len(sampled_ids)} car parks")

    async def process_carpark(client: httpx.AsyncClient, carpark_id: str):
        try:
            image_bytes = await carpark_client.fetch_photo(client, carpark_id)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            detection = await run_in_threadpool(inference.detect, image)
            return CarparkResult(
                carpark_id=carpark_id,
                available_spaces=detection.available_spaces,
                confidence_score=detection.avg_confidence,
            )
        except Exception as e:
            logger.error(f"[{uuid}] Failed to process {carpark_id}: {e}")
            return None

    async with httpx.AsyncClient() as client:
        tasks = [process_carpark(client, cid) for cid in sampled_ids]
        raw_results = await asyncio.gather(*tasks)

    valid_results = [r for r in raw_results if r is not None]
    valid_results.sort(key=lambda r: r.available_spaces, reverse=True)
    top_results = valid_results[:effective_n]

    elapsed_ms = round((time.time() - start_time) * 1000, 2)

    if not top_results:
        return FindCarparksResponse(
            uuid=uuid, status="error", msg="No car park data could be retrieved.",
            speed_inference=f"{elapsed_ms} ms", requested_n=n, results=[],
        )

    return FindCarparksResponse(
        uuid=uuid, status="success", msg="success",
        speed_inference=f"{elapsed_ms} ms", requested_n=n, results=top_results,
    )


@app.get("/api/annotate-carpark", response_model=AnnotateCarparkResponse)
async def annotate_carpark(carpark_id: str = Query(..., description="Car park ID to annotate")):
    try:
        async with httpx.AsyncClient() as client:
            image_bytes = await carpark_client.fetch_photo(client, carpark_id)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        detection = await run_in_threadpool(inference.detect, image)
        encoded = inference.encode_image_base64(detection.annotated_image_bytes)

        return AnnotateCarparkResponse(
            carpark_id=carpark_id, status="success", msg="success", image_base64=encoded,
        )
    except Exception as e:
        logger.error(f"Failed to annotate {carpark_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to annotate car park {carpark_id}: {str(e)}")