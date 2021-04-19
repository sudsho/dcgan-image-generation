"""Response models for the FastAPI app."""
from pydantic import BaseModel


class GenerateResponse(BaseModel):
    n: int
    png_b64: str


class HealthResponse(BaseModel):
    status: str
    ckpt_present: bool
