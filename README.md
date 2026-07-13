# 📸 dOCRead

**dOCRead** is a standalone, lightweight document reading and computer vision microservice designed to handle Document OCR, Image Captioning, and Object Detection. It can run in two modes:
- **Cloud Mode (Default)**: Utilizes the Gemini Multimodal API via an OpenAI-compatible client for highly accurate, fast, and serverless visual analysis.
- **Local Mode**: Uses CPU/GPU-friendly open-source models (`EasyOCR` for text recognition, `Salesforce/blip-image-captioning-base` for captioning, and `hustvl/yolos-tiny` for object detection) for 100% offline and cost-free execution.

It exposes clean HTTP API endpoints for microservices integration (e.g. into agent orchestrators like `AgenThink`) and includes a gorgeous modern Web UI dashboard for interactive testing.

---

## 🛠️ Tech Stack
- **Backend**: FastAPI, Uvicorn, Pydantic, Pillow (PIL), NumPy
- **Cloud AI**: OpenAI SDK (configured for Gemini API)
- **Local AI**: EasyOCR, HuggingFace Transformers, PyTorch
- **Frontend**: HTML5, Modern CSS (Glassmorphism & dark mode), Vanilla JavaScript

---

## 📦 API Endpoints

- **`GET /api/health`**: Health status check and engine mode configuration overview.
- **`POST /api/vision/ocr`**: Extract plain text from an uploaded image file / document.
- **`POST /api/vision/describe`**: Generate a textual caption/description of an image. Supports an optional `prompt` text field.
- **`POST /api/vision/detect`**: Detect objects and structural entities in the image. Returns a JSON list of bounding box coordinates `[ymin, xmin, ymax, xmax]` normalized between `0.0` and `1.0`.

---

## 🚀 Getting Started

### 1. Clone & Setup Environment
Create a `.env` file in the root directory:
```bash
# Mode: 'cloud' or 'local'
VISION_ENGINE_MODE=cloud

# Gemini API credentials
GEMINI_API_KEY=your_gemini_api_key_here
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Server
```bash
python app.py
```
The service will start at `http://127.0.0.1:8002/`. Open this URL in your browser to view the interactive Web UI.
