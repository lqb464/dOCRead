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
        # Convert to RGB if needed to save as JPEG
        if image.mode in ("RGBA", "P"):
            image = image.convert("RGB")
        image.save(buffered, format="JPEG", quality=85)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def _call_gemini_multimodal(self, base64_image: str, prompt: str) -> str:
        """Call Gemini model with image and prompt."""
        if not self.client:
            return "Mock Cloud Output: Please configure GEMINI_API_KEY in your .env file."

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
                max_tokens=800,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            logger.error("Cloud vision API call failed: %s", exc)
            return f"Error calling Cloud vision service: {exc}"

    def ocr(self, image: Image.Image) -> str:
        base64_img = self._to_base64_jpeg(image)
        prompt = (
            "Extract all text from this image exactly as it appears. "
            "Return ONLY the extracted text. If no text is found, return an empty string."
        )
        return self._call_gemini_multimodal(base64_img, prompt)

    def describe(self, image: Image.Image, prompt: str = "") -> str:
        base64_img = self._to_base64_jpeg(image)
        default_prompt = (
            "Provide a comprehensive, detailed description of this image. "
            "Describe the key elements, colors, subject matter, and layout."
        )
        final_prompt = f"{default_prompt} Additional focus: {prompt}" if prompt else default_prompt
        return self._call_gemini_multimodal(base64_img, final_prompt)

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
        raw_output = self._call_gemini_multimodal(base64_img, prompt)

        # Parse JSON from response
        try:
            # Clean potential markdown wrapping (e.g. ```json ... ```)
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
            logger.warning("Failed to parse Gemini object detection response as JSON: %r", raw_output)
            # Fallback mock object if parsing fails to keep app functional
            return [
                {
                    "label": "unparsed_object",
                    "confidence": 0.5,
                    "box": [0.2, 0.2, 0.8, 0.8],
                    "raw": raw_output[:100],
                }
            ]
class_map = {"cloud": CloudVisionEngine}
