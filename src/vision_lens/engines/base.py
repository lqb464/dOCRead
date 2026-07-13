"""Base abstract engine classes for vision operations."""
from abc import ABC, abstractmethod
from typing import Dict, List, Any
from PIL import Image


class BaseVisionEngine(ABC):
    """Abstract interface defining the computer vision capabilities."""

    @abstractmethod
    def ocr(self, image: Image.Image) -> str:
        """
        Perform Optical Character Recognition on an image.

        Returns:
            Extracted text as a plain string.
        """
        pass

    @abstractmethod
    def describe(self, image: Image.Image, prompt: str = "") -> str:
        """
        Generate a textual description or answer questions about an image.

        Args:
            image: PIL Image object.
            prompt: Optional guiding prompt (e.g. "Focus on details").

        Returns:
            Description text.
        """
        pass

    @abstractmethod
    def detect_objects(self, image: Image.Image) -> List[Dict[str, Any]]:
        """
        Detect objects in an image.

        Returns:
            List of dicts representing detected objects:
            [
                {
                    "label": "person",
                    "confidence": 0.95,
                    "box": [ymin, xmin, ymax, xmax]  # normalized coords 0.0 to 1.0
                }
            ]
        """
        pass
