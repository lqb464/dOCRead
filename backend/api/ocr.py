"""OCR endpoints router."""
import logging
from fastapi import APIRouter, File, UploadFile
from backend.services.ocr_service import get_ocr_service, load_image_from_bytes

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/vision", tags=["vision"])


@router.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    """Extract text + layout blocks from an uploaded image (server CUDA->CPU)."""
    data = await file.read()
    image = load_image_from_bytes(data)
    engine = get_ocr_service()

    logger.info("OCR request file=%s", file.filename)
    res = engine.ocr(image)
    metrics = res.get("metrics", {})
    return {
        "filename": file.filename,
        "text": res.get("text", ""),
        "blocks": res.get("blocks", []),
        "engine": metrics.get("provider", "server"),
        "metrics": metrics,
    }

