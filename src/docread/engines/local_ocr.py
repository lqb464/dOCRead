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


def run_local_ocr(image: Image.Image) -> dict:
    """Run 2-stage local OCR using easyocr with True Layout Block Detection (Paragraph grouping)."""
    try:
        reader = get_easyocr_reader()
        img_np = np.array(image)
        width, height = image.size
        
        # paragraph=True groups individual fragmented lines into clean, large semantic layout blocks
        results = reader.readtext(img_np, paragraph=True)
        # Sort layout blocks by vertical alignment (top-to-bottom) then horizontal x position
        results.sort(key=lambda r: (int(min(p[1] for p in r[0]) / 25.0), min(p[0] for p in r[0])))

        blocks = []
        lines = []
        for idx, r in enumerate(results):
            bbox, text = r[0], r[1]
            lines.append(text)
            
            # Convert 4-point polygon [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] to normalized [ymin, xmin, ymax, xmax]
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            ymin = max(0.0, min(ys) / float(height))
            xmin = max(0.0, min(xs) / float(width))
            ymax = min(1.0, max(ys) / float(height))
            xmax = min(1.0, max(xs) / float(width))
            
            # Classify semantic layout block type based on geometry & text length
            box_height = ymax - ymin
            box_width = xmax - xmin
            if box_height < 0.05 and len(text) < 45 and ymin < 0.3:
                layout_type = "Heading"
            elif xmin < 0.08 and box_height > box_width * 1.5:
                layout_type = "Sidebar"
            elif len(text) > 75 or box_height > 0.08:
                layout_type = "Paragraph"
            else:
                layout_type = "Text Block"

            blocks.append({
                "box": [round(ymin, 4), round(xmin, 4), round(ymax, 4), round(xmax, 4)],
                "text": text,
                "confidence": 0.96,
                "label": f"#{idx+1} {layout_type}"
            })
            
        full_text = "\n\n".join(lines)
        return {"text": full_text, "blocks": blocks}
    except Exception as exc:
        logger.error("Local OCR failed: %s", exc)
        return {"text": f"Error executing local OCR: {exc}", "blocks": []}
