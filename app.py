"""Startup script for VisionLens."""
import uvicorn
from src.vision_lens.config import settings

if __name__ == "__main__":
    uvicorn.run(
        "src.vision_lens.server:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
    )
