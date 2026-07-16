"""Cloud vision engine using OpenAI-compatible Gemini API."""
import base64
import io
import json
import logging
from typing import Dict, List, Any
from PIL import Image
from openai import OpenAI

from ..config import settings
from .base import BaseVisionEngine

logger = logging.getLogger(__name__)


class CloudVisionEngine(BaseVisionEngine):
    """
    Implements vision operations by calling the Gemini Multimodal API
    using the OpenAI compatible interface.
    """

    def __init__(self):
        # Initialized lazily or at start. If key is missing, mock it
        self.api_key = settings.GEMINI_API_KEY
        if self.api_key:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=settings.GEMINI_BASE_URL,
            )
        else:
            self.client = None
            logger.warning(
                "GEMINI_API_KEY is not set. CloudVisionEngine will run in MOCK mode."
            )

    def _to_base64_jpeg(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 JPEG string."""
        buffered = io.BytesIO()
        if image.mode in ("RGBA", "P"):
            image = image.convert("RGB")
        image.save(buffered, format="JPEG", quality=85)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def _fallback_local(self, operation: str, image: Image.Image, prompt: str = "") -> Any:
        """Automatically fallback to LocalVisionEngine when Cloud API fails or returns 503."""
        try:
            logger.warning("Cloud Gemini API unavailable/503. Switching to Local offline vision engine...")
            if not hasattr(self, "_local_engine"):
                from .local_engine import LocalVisionEngine  # noqa: PLC0415
                self._local_engine = LocalVisionEngine()
            
            if operation == "ocr":
                res = self._local_engine.ocr(image)
                if isinstance(res, dict):
                    res["text"] = f"[ℹ️ Tự động Fallback sang Local 2-Stage OCR do Cloud 503 Overload]\n\n{res.get('text', '')}"
                    return res
                return {"text": f"[ℹ️ Fallback]\n{res}", "blocks": []}
            elif operation == "describe":
                res = self._local_engine.describe(image, prompt)
                return f"[ℹ️ Tự động Fallback sang Local Vision AI do Cloud 503 Overload]\n\n{res}"
            elif operation == "detect":
                return self._local_engine.detect_objects(image)
        except Exception as local_exc:
            logger.error("Local fallback failed after Cloud failure: %s", local_exc)
            if operation == "ocr":
                return {"text": f"Error: Cloud vision service overloaded (503) and local engine failed ({local_exc})", "blocks": []}
            if operation == "describe":
                return f"Error: Cloud vision service overloaded (503) and local engine failed ({local_exc})"
            return []

    def _call_gemini_multimodal(self, base64_image: str, prompt: str, image: Image.Image = None, operation: str = "describe", max_tokens: int = 4096) -> Any:
        """Call Gemini model with image and prompt, with retry and automatic local fallback."""
        if not self.client:
            if operation == "ocr":
                return {"text": "Mock Cloud Output: Please configure GEMINI_API_KEY in your .env file.", "blocks": []}
            return "Mock Cloud Output: Please configure GEMINI_API_KEY in your .env file."

        import time
        max_retries = 2
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=settings.GEMINI_MODEL,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    },
                                },
                            ],
                        }
                    ],
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content or ""
            except Exception as exc:
                exc_str = str(exc)
                logger.warning("Attempt %d/%d Cloud vision call failed: %s", attempt + 1, max_retries, exc_str)
                if ("503" in exc_str or "429" in exc_str or "high demand" in exc_str.lower()) and attempt < max_retries - 1:
                    time.sleep(1.5)
                    continue
                
                if image is not None:
                    return self._fallback_local(operation, image, prompt)
                if operation == "ocr":
                    return {"text": f"Error calling Cloud vision service: {exc}", "blocks": []}
                return f"Error calling Cloud vision service: {exc}"

    def ocr(self, image: Image.Image) -> Dict[str, Any]:
        base64_img = self._to_base64_jpeg(image)
        prompt = (
            "Perform a comprehensive 2-Stage OCR and layout analysis on this image:\n"
            "1. Detect the bounding box of every text line, paragraph, title, or table block.\n"
            "2. Extract the exact character text inside each block.\n"
            "Return the output strictly in valid JSON format with two keys:\n"
            '{"text": "Full extracted text combined nicely with newlines", '
            '"blocks": [{"box": [ymin, xmin, ymax, xmax], "text": "exact text inside box", "confidence": 0.98, "label": "short preview max 18 chars"}]}\n'
            "Coordinates [ymin, xmin, ymax, xmax] must be normalized between 0.0 and 1.0."
        )
        raw_output = self._call_gemini_multimodal(base64_img, prompt, image=image, operation="ocr", max_tokens=4096)
        
        if isinstance(raw_output, dict):
            return raw_output
            
        try:
            cleaned = raw_output.strip()
            if cleaned.startswith("```"):
                lines = cleaned.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                cleaned = "\n".join(lines).strip()
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict) and ("text" in parsed or "blocks" in parsed):
                blocks = parsed.get("blocks", [])
                text = parsed.get("text", "\n".join([b.get("text", "") for b in blocks]))
                # If parsed successfully and we have blocks, return them!
                if blocks and len(blocks) > 0:
                    return {"text": text, "blocks": blocks}
        except Exception:
            logger.warning("Failed to parse Cloud 2-stage OCR JSON response: %r", raw_output[:200])
            
        # If cloud returned invalid JSON or truncated/empty layout boxes, run true 2-stage local OCR fallback!
        logger.info("Cloud OCR layout boxes empty or truncated. Executing True 2-Stage Local OCR (EasyOCR Layout Detection + Recognition)...")
        return self._fallback_local("ocr", image)

    def describe(self, image: Image.Image, prompt: str = "") -> str:
        base64_img = self._to_base64_jpeg(image)
        default_prompt = (
            "Provide a comprehensive, detailed description of this image. "
            "Describe the key elements, colors, subject matter, and layout."
        )
        final_prompt = f"{default_prompt} Additional focus: {prompt}" if prompt else default_prompt
        return self._call_gemini_multimodal(base64_img, final_prompt, image=image, operation="describe")

    def detect_objects(self, image: Image.Image) -> List[Dict[str, Any]]:
        base64_img = self._to_base64_jpeg(image)
        prompt = (
            "Detect the main objects in this image. "
            "Return the output strictly in a valid JSON list of objects. "
            "Each object must have 'label', 'confidence' (float 0.0 to 1.0), "
            "and 'box' as [ymin, xmin, ymax, xmax] coordinates normalized between 0.0 and 1.0. "
            "Format example: "
            '[{"label": "dog", "confidence": 0.95, "box": [0.1, 0.2, 0.6, 0.8]}]'
        )
        raw_output = self._call_gemini_multimodal(base64_img, prompt, image=image, operation="detect")
        
        # If fallback returned list directly
        if isinstance(raw_output, list):
            return raw_output

        # Parse JSON from response
        try:
            cleaned = raw_output.strip()
            if cleaned.startswith("```"):
                lines = cleaned.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                cleaned = "\n".join(lines).strip()

            parsed = json.loads(cleaned)
            if isinstance(parsed, list):
                return parsed
            return []
        except Exception:
            logger.warning("Failed to parse object detection response as JSON: %r", raw_output)
            # Try fallback to local detector if cloud JSON parsing failed
            return self._fallback_local("detect", image)
class_map = {"cloud": CloudVisionEngine}
