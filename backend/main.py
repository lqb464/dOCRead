import logging
from io import BytesIO
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# Import our custom trained models
from src.models.trocr import TrOCRWrapper

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="DOCRead - Custom Deep Learning OCR")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the fine-tuned TrOCR model globally
# In a real scenario, this would load weights from our MLflow/checkpoint directory
logger.info("Loading Custom TrOCR Model for Printed & Handwritten OCR...")
try:
    ocr_model = TrOCRWrapper()
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load OCR model: {e}")
    ocr_model = None

@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": ocr_model is not None,
        "mode": "TrOCR (Printed + Handwritten Support)"
    }

@app.post("/api/vision/ocr")
async def perform_ocr(file: UploadFile = File(...)):
    """
    Receives an image file, processes it using our custom trained TrOCR model,
    and returns the predicted text.
    """
    if not ocr_model:
        raise HTTPException(status_code=500, detail="OCR Model is not loaded on the server.")
        
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        
        # Run Inference
        predicted_text = ocr_model.predict(image)
        
        return {
            "status": "success",
            "results": [
                {
                    "text": predicted_text,
                    "confidence": 0.95 # Simulated confidence score
                }
            ]
        }
        
    except Exception as e:
        logger.error(f"OCR Inference Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
