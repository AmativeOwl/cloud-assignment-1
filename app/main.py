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
from app import carpark_cache
from app import request_tracker
from app import response_cache
from app import dashboard
from fastapi.responses import Response

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
logger = logging.getLogger("smartpark.main")

app = FastAPI(title="SmartPark Core Platform")


@app.get("/health")
def health():
    return {"status": "healthy"}


from app import response_cache

@app.get("/api/find-carparks", response_model=FindCarparksResponse)
async def find_carparks(
    uuid: str = Query(..., description="Unique identifier for the requesting client"),
    n: int = Query(3, ge=1, description="Number of top car parks to return"),
):
    request_tracker.record_request(uuid)

    cached = response_cache.get_cached(uuid, n)
    if cached is not None:
        logger.info(f"[{uuid}] find-carparks: served from cache (n={n})")
        return cached

    start_time = time.time()

    max_n = len(carpark_client.get_all_carpark_ids())
    effective_n = min(n, max_n)
    query_count = min(effective_n * 2, max_n)
    sampled_ids = carpark_client.sample_carpark_ids(query_count)

    logger.info(f"[{uuid}] find-carparks: n={n}, querying {len(sampled_ids)} car parks")

    async def process_carpark(client: httpx.AsyncClient, carpark_id: str):
        try:
            image_bytes = await carpark_client.fetch_photo(client, carpark_id)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            detection = await run_in_threadpool(inference.detect, image)

            carpark_cache.update_carpark(
                carpark_id=carpark_id,
                available_spaces=detection.available_spaces,
                confidence_score=detection.avg_confidence,
            )

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
        response = FindCarparksResponse(
            uuid=uuid, status="error", msg="No car park data could be retrieved.",
            speed_inference=f"{elapsed_ms} ms", requested_n=n, results=[],
        )
    else:
        response = FindCarparksResponse(
            uuid=uuid, status="success", msg="success",
            speed_inference=f"{elapsed_ms} ms", requested_n=n, results=top_results,
        )

    response_cache.set_cached(uuid, n, response)
    return response

@app.get("/api/annotate-carpark", response_model=AnnotateCarparkResponse)
async def annotate_carpark(carpark_id: str = Query(..., description="Car park ID to annotate")):
    try:
        async with httpx.AsyncClient() as client:
            image_bytes = await carpark_client.fetch_photo(client, carpark_id)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        detection = await run_in_threadpool(inference.detect, image)
        encoded = inference.encode_image_base64(detection.annotated_image_bytes)

        carpark_cache.update_carpark(
            carpark_id=carpark_id,
            available_spaces=detection.available_spaces,
            confidence_score=detection.avg_confidence,
        )

        return AnnotateCarparkResponse(
            carpark_id=carpark_id, status="success", msg="success", image_base64=encoded,
        )
    except Exception as e:
        logger.error(f"Failed to annotate {carpark_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to annotate car park {carpark_id}: {str(e)}")


@app.get("/api/ops/carparks")
def ops_list_carparks():
    """
    OPS-API-1: Lists all car parks with their IDs and last-known available
    spaces. Data reflects the most recent successful detection for each
    car park (via find-carparks or annotate-carpark); car parks not yet
    queried show as having no data.
    """
    all_ids = carpark_client.get_all_carpark_ids()
    known = carpark_cache.get_all_known()

    results = []
    for cid in all_ids:
        if cid in known:
            entry = known[cid]
            results.append({
                "carpark_id": cid,
                "available_spaces": entry["available_spaces"],
                "confidence_score": entry["confidence_score"],
                "last_updated": entry["last_updated"],
            })
        else:
            results.append({
                "carpark_id": cid,
                "available_spaces": None,
                "confidence_score": None,
                "last_updated": None,
            })

    return {
        "status": "success",
        "msg": "success",
        "total_carparks": len(all_ids),
        "carparks": results,
    }


@app.get("/api/ops/recent-activity")
def ops_recent_activity():
    """
    OPS-API-2: Number of requests and distinct users in the last 30 seconds,
    derived from the in-memory request log populated by find-carparks.
    """
    return {
        "status": "success",
        "msg": "success",
        "window_seconds": request_tracker.WINDOW_SECONDS,
        "recent_request_count": request_tracker.count_recent_requests(),
        "recent_unique_users": request_tracker.count_recent_unique_users(),
    }

@app.get("/api/ops/dashboard")
def ops_dashboard():
    """
    OPS-REQ-2: On-demand operational dashboard showing current car park
    availability and recent platform activity, rendered as a PNG image.
    """
    png_bytes = dashboard.generate_dashboard_png()
    return Response(content=png_bytes, media_type="image/png")