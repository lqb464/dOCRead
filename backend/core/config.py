"""Configuration for dOCRead OCR microservice."""
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    HOST: str = "127.0.0.1"
    PORT: int = 8002

    # auto = CUDA if available else CPU (server path)
    OCR_RUNTIME: Literal["auto", "cuda", "cpu"] = "auto"
    # en and/or vi — vi maps to RapidOCR latin rec model
    OCR_LANG: str = "en,vi"
    OCR_MAX_SIDE: int = 1280
    OCR_INTRA_OP_THREADS: int = 4
    OCR_VERSION: str = "PP-OCRv4"
    OCR_MODEL_TYPE: str = "mobile"

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


settings = Settings()
