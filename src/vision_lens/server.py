"""FastAPI server for VisionLens."""
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .config import settings
from .engines import get_vision_engine

logger = logging.getLogger(__name__)

app = FastAPI(
    title="VisionLens Service",
    description="Standalone computer vision API for OCR, captioning and object detection.",
    version="0.1.0"
)

# Enable CORS for cross-origin orchestrator integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global engine singleton
_ENGINE = None


def get_engine():
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = get_vision_engine()
    return _ENGINE


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "engine_mode": settings.VISION_ENGINE_MODE,
        "gemini_model": settings.GEMINI_MODEL,
        "gemini_api_configured": bool(settings.GEMINI_API_KEY)
    }
