import os
import random
import base64
import logging

from fastapi import FastAPI, HTTPException, Query

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger("carpark_sim")

app = FastAPI(title="SmartPark Car Park Simulator")

IMAGES_DIR = os.environ.get("IMAGES_DIR", "./sample_images")

# Load the list of available image filenames once at startup.
# Each call to /api/takephoto picks one at random, simulating a fresh
# camera capture for whichever car park is requested (per assignment
# assumption: "The camera returns an image for the whole car park.")
try:
    IMAGE_FILES = [
        f for f in os.listdir(IMAGES_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    if not IMAGE_FILES:
        raise FileNotFoundError(f"No image files found in {IMAGES_DIR}")
    logger.info(f"Loaded {len(IMAGE_FILES)} sample images from {IMAGES_DIR}")
except Exception as e:
    logger.error(f"Failed to load sample images: {e}")
    IMAGE_FILES = []


@app.get("/health")
def health():
    return {"status": "healthy", "images_loaded": len(IMAGE_FILES)}


@app.get("/api/takephoto")
def take_photo(carpark_id: str = Query(..., description="ID of the car park to photograph")):
    """
    Simulates a camera capture request for a given car park.
    Returns a randomly selected base64-encoded image, regardless of
    which specific car park ID was requested, since each car park's
    camera independently captures its own lot at request time.
    """
    if not IMAGE_FILES:
        logger.error("No images available to serve.")
        raise HTTPException(status_code=503, detail="No sample images available on this simulator.")

    try:
        chosen_file = random.choice(IMAGE_FILES)
        image_path = os.path.join(IMAGES_DIR, chosen_file)

        with open(image_path, "rb") as f:
            image_bytes = f.read()

        encoded = base64.b64encode(image_bytes).decode("utf-8")

        logger.info(f"carpark_id={carpark_id} served image={chosen_file}")

        return {
            "carpark_id": carpark_id,
            "status": "success",
            "msg": "success",
            "image_base64": encoded,
        }

    except Exception as e:
        logger.error(f"Failed to serve photo for carpark_id={carpark_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to capture photo: {str(e)}")