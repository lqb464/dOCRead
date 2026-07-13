"""Startup script for dOCRead."""
import uvicorn
from src.docread.config import settings

if __name__ == "__main__":
    uvicorn.run(
        "src.docread.server:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
    )
