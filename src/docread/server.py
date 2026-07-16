import io
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, Response
from PIL import Image

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

# Mount static files hosting
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def read_root():
    """Redirect root path to the index.html page."""
    return RedirectResponse(url="/static/index.html")

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    """Return inline SVG favicon to eliminate 404 browser requests."""
    svg_content = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><text y=".9em" font-size="90">📖</text></svg>'
    return Response(content=svg_content, media_type="image/svg+xml")

# Global engine singleton
_ENGINE = None


def get_engine():
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = get_vision_engine()
    return _ENGINE


def load_image_from_bytes(data: bytes) -> Image.Image:
    """Safely open PIL Image from uploaded bytes."""
    try:
        img = Image.open(io.BytesIO(data))
        img.verify()  # verify it's a valid image
        # PIL Image.open is lazy, reopen to start reading
        return Image.open(io.BytesIO(data))
    except Exception as exc:
        logger.error("Invalid image upload: %s", exc)
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")


# ---------------------------------------------------------------------------
# Core Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "engine_mode": settings.VISION_ENGINE_MODE,
        "gemini_model": settings.GEMINI_MODEL,
        "gemini_api_configured": bool(settings.GEMINI_API_KEY)
    }


@app.post("/api/vision/ocr")
async def ocr(file: UploadFile = File(...)):
    """Extract plain text and layout blocks from uploaded image document."""
    data = await file.read()
    image = load_image_from_bytes(data)
    engine = get_engine()

    logger.info("Executing 2-Stage OCR on file: %s", file.filename)
    res = engine.ocr(image)
    if isinstance(res, dict):
        text = res.get("text", "")
        blocks = res.get("blocks", [])
    else:
        text = str(res)
        blocks = []

    return {
        "filename": file.filename,
        "text": text,
        "blocks": blocks,
        "engine": settings.VISION_ENGINE_MODE
    }


@app.post("/api/vision/describe")
async def describe(
    file: UploadFile = File(...),
    prompt: Optional[str] = Form(None)
):
    """Generate description / caption for uploaded image."""
    data = await file.read()
    image = load_image_from_bytes(data)
    engine = get_engine()

    logger.info("Executing Describe on file: %s with prompt: %s", file.filename, prompt)
    description = engine.describe(image, prompt or "")
    return {
        "filename": file.filename,
        "description": description,
        "engine": settings.VISION_ENGINE_MODE
    }


@app.post("/api/vision/detect")
async def detect(file: UploadFile = File(...)):
    """Detect objects in image and return bounding boxes."""
    data = await file.read()
    image = load_image_from_bytes(data)
    engine = get_engine()

    logger.info("Executing Object Detection on file: %s", file.filename)
    objects = engine.detect_objects(image)
    return {
        "filename": file.filename,
        "objects": objects,
        "engine": settings.VISION_ENGINE_MODE
    }
