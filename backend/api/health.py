"""Health check API router."""
from fastapi import APIRouter
from backend.services.ocr_service import get_ocr_service

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health")
def health():
    try:
        service = get_ocr_service()
        info = service.health_info()
        return {
            "status": "ok",
            "service": "docread-ocr",
            "engine_mode": info.get("provider", "unknown"),
            **info,
        }
    except Exception as exc:
        return {
            "status": "degraded",
            "service": "docread-ocr",
            "engine_mode": "unavailable",
            "error": str(exc),
        }
