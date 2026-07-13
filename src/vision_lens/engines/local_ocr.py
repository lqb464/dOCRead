"""Local OCR implementation using easyocr."""
import logging
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)

_EASYOCR_READER = None


def get_easyocr_reader():
    """Lazily load easyocr to avoid slow imports at start."""
    global _EASYOCR_READER
    if _EASYOCR_READER is None:
        import easyocr  # noqa: PLC0415

        logger.info("Initializing easyocr reader (English + Vietnamese)...")
        # Downloads model parameters on first use (~100MB)
        _EASYOCR_READER = easyocr.Reader(["en", "vi"], gpu=False)
        logger.info("easyocr reader loaded.")
    return _EASYOCR_READER


def run_local_ocr(image: Image.Image) -> str:
    """Run local OCR using easyocr."""
    try:
        reader = get_easyocr_reader()
        # Convert PIL to numpy array
        img_np = np.array(image)
        results = reader.readtext(img_np)
        # Results are tuples: (bbox, text, confidence)
        lines = [r[1] for r in results]
        return "\n".join(lines)
    except Exception as exc:
        logger.error("Local OCR failed: %s", exc)
        return f"Error executing local OCR: {exc}"
