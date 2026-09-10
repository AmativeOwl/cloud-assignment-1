import os
import logging
import requests

logger = logging.getLogger("smartpark.model_loader")

MODEL_PATH = os.environ.get("MODEL_PATH", "/app/models/model.pt")
GCS_BUCKET = os.environ.get("GCS_BUCKET")
GCS_MODEL_BLOB = os.environ.get("GCS_MODEL_BLOB", "model.pt")


def ensure_model_downloaded():
    """
    Downloads the model from a public GCS URL if not already present locally,
    following the same public-object-access pattern demonstrated in the
    Week 3 static web hosting lab (storage.googleapis.com/[BUCKET]/[OBJECT]).
    This allows the model to be updated by re-uploading to the bucket and
    restarting pods, without rebuilding the container image.
    """
    if os.path.exists(MODEL_PATH):
        logger.info(f"Model already present at {MODEL_PATH}, skipping download.")
        return

    if not GCS_BUCKET:
        raise RuntimeError(
            "MODEL_PATH does not exist and GCS_BUCKET is not set. Cannot load model."
        )

    url = f"https://storage.googleapis.com/{GCS_BUCKET}/{GCS_MODEL_BLOB}"
    logger.info(f"Downloading model from {url} to {MODEL_PATH}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    response = requests.get(url, timeout=60)
    response.raise_for_status()

    with open(MODEL_PATH, "wb") as f:
        f.write(response.content)

    logger.info("Model download complete.")