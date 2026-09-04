import os
import io
import base64
import logging

from PIL import Image
from ultralytics import YOLO

logger = logging.getLogger("smartpark.inference")

# Path where the startup script (start.sh) downloads the model from GCS
MODEL_PATH = os.environ.get("MODEL_PATH", "/app/models/model.pt")

# Class names (case-insensitive) that count as an AVAILABLE parking space.
# IMPORTANT: check `model.names` after loading (see log line below) and update
# this env var / default to match your actual model's class labels.
AVAILABLE_CLASS_NAMES = {
    name.strip().lower()
    for name in os.environ.get("AVAILABLE_CLASS_NAMES", "empty").split(",")
}

logger.info(f"Loading YOLO model from {MODEL_PATH}")
model = YOLO(MODEL_PATH)
logger.info(f"Model loaded. Classes: {model.names}")


class DetectionResult:
    def __init__(self, available_spaces: int, total_detections: int,
                 avg_confidence: float, annotated_image_bytes: bytes):
        self.available_spaces = available_spaces
        self.total_detections = total_detections
        self.avg_confidence = avg_confidence
        self.annotated_image_bytes = annotated_image_bytes


def _run_prediction(image: Image.Image) -> DetectionResult:
    """
    Synchronous, CPU-bound YOLO inference.
    Must be called via run_in_threadpool from async routes to avoid
    blocking the FastAPI event loop (see main.py).
    """
    results = model.predict(image, verbose=False)
    result = results[0]

    available_confidences = []
    all_confidences = []

    for box in result.boxes:
        class_id = int(box.cls[0].item())
        label = result.names[class_id].strip().lower()
        confidence = float(box.conf[0].item())
        all_confidences.append(confidence)
        if label in AVAILABLE_CLASS_NAMES:
            available_confidences.append(confidence)

    avg_confidence = (
        sum(available_confidences) / len(available_confidences)
        if available_confidences
        else (sum(all_confidences) / len(all_confidences) if all_confidences else 0.0)
    )

    annotated_array = result.plot()[:, :, ::-1]
    annotated_image = Image.fromarray(annotated_array)
    buf = io.BytesIO()
    annotated_image.save(buf, format="JPEG")

    return DetectionResult(
        available_spaces=len(available_confidences),
        total_detections=len(result.boxes),
        avg_confidence=round(avg_confidence, 4),
        annotated_image_bytes=buf.getvalue(),
    )


def detect(image: Image.Image) -> DetectionResult:
    """Public entry point — call via run_in_threadpool from async routes."""
    return _run_prediction(image)


def encode_image_base64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")