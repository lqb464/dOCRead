"""Factory module to load the appropriate vision engine."""
import logging
from ..config import settings
from .base import BaseVisionEngine

logger = logging.getLogger(__name__)


def get_vision_engine() -> BaseVisionEngine:
    """Return configured vision engine instance."""
    mode = settings.VISION_ENGINE_MODE
    logger.info("Initializing VisionLens in '%s' mode.", mode)

    if mode == "local":
        from .local_engine import LocalVisionEngine  # noqa: PLC0415
        return LocalVisionEngine()
    else:
        from .cloud_engine import CloudVisionEngine  # noqa: PLC0415
        return CloudVisionEngine()
