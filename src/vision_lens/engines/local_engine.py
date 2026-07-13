"""Local vision engine using open-source models (BLIP, YOLOS, EasyOCR)."""
import logging
from typing import Dict, List, Any
from PIL import Image

from .base import BaseVisionEngine
from .local_ocr import run_local_ocr

logger = logging.getLogger(__name__)

_CAPTION_PIPELINE = None
_DETECTION_PIPELINE = None


def get_caption_pipeline():
    """Lazily load BLIP captioning pipeline."""
    global _CAPTION_PIPELINE
    if _CAPTION_PIPELINE is None:
        from transformers import pipeline  # noqa: PLC0415

        logger.info("Loading BLIP captioning model (Salesforce/blip-image-captioning-base)...")
        # BLIP base is ~990MB, but let's use BLIP-base since it works well on CPU
        _CAPTION_PIPELINE = pipeline(
            "image-to-text",
            model="Salesforce/blip-image-captioning-base",
        )
        logger.info("BLIP model loaded.")
    return _CAPTION_PIPELINE


def get_detection_pipeline():
    """Lazily load object detection pipeline."""
    global _DETECTION_PIPELINE
    if _DETECTION_PIPELINE is None:
        from transformers import pipeline  # noqa: PLC0415

        logger.info("Loading YOLOS object detection model (hustvl/yolos-tiny)...")
        # yolos-tiny is extremely lightweight (~30MB)
        _DETECTION_PIPELINE = pipeline(
            "object-detection",
            model="hustvl/yolos-tiny",
        )
        logger.info("YOLOS model loaded.")
    return _DETECTION_PIPELINE


class LocalVisionEngine(BaseVisionEngine):
    """
    Implements all vision operations locally using lightweight,
    CPU-friendly open-source models.
    """

    def ocr(self, image: Image.Image) -> str:
        return run_local_ocr(image)

    def describe(self, image: Image.Image, prompt: str = "") -> str:
        try:
            pipeline = get_caption_pipeline()
            results = pipeline(image)
            # Result is a list: [{"generated_text": "description of image"}]
            caption = results[0]["generated_text"] if results else "No description generated."
            if prompt:
                caption = f"{caption} (Note: user prompt '{prompt}' was ignored by local BLIP model)."
            return caption
        except Exception as exc:
            logger.error("Local description failed: %s", exc)
            return f"Error executing local description: {exc}"

    def detect_objects(self, image: Image.Image) -> List[Dict[str, Any]]:
        try:
            pipeline = get_detection_pipeline()
            width, height = image.size
            results = pipeline(image)

            # pipeline results structure:
            # [{"score": float, "label": str, "box": {"xmin": int, "ymin": int, "xmax": int, "ymax": int}}]
            output = []
            for r in results:
                box = r["box"]
                # Convert absolute pixel boxes into normalized coordinates [ymin, xmin, ymax, xmax] (0.0 to 1.0)
                ymin = box["ymin"] / height
                xmin = box["xmin"] / width
                ymax = box["ymax"] / height
                xmax = box["xmax"] / width

                output.append({
                    "label": r["label"],
                    "confidence": float(r["score"]),
                    "box": [
                        round(ymin, 4),
                        round(xmin, 4),
                        round(ymax, 4),
                        round(xmax, 4)
                    ]
                })
            return output
        except Exception as exc:
            logger.error("Local object detection failed: %s", exc)
            return []
