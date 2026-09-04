from pydantic import BaseModel
from typing import List, Optional


class CarparkResult(BaseModel):
    carpark_id: str
    name: Optional[str] = None
    available_spaces: int
    confidence_score: float


class FindCarparksResponse(BaseModel):
    uuid: str
    status: str
    msg: str
    speed_inference: str
    requested_n: int
    results: List[CarparkResult]


class AnnotateCarparkResponse(BaseModel):
    carpark_id: str
    status: str
    msg: str
    image_base64: Optional[str] = None