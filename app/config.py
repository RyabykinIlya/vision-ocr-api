"""Runtime configuration for the vision-ocr-api service.

All settings are overridable via environment variables with the ``VISION_``
prefix, e.g. ``VISION_MODEL=gemma3:4b``.
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VISION_", env_file=".env", extra="ignore")

    # Where Ollama is reachable from inside this container.
    ollama_url: str = "http://ollama:11434"

    # Default vision model. Must already be pulled in the Ollama instance
    # (see README) - this service does not pull models on your behalf.
    model: str = "qwen2.5vl:7b-q4_K_M"

    # How long Ollama keeps the model resident in VRAM after a request.
    # Keep this short so a GPU-sharing service (e.g. Whisper) can reclaim
    # memory between calls.
    keep_alive: str = "60s"

    # Generation parameters.
    num_ctx: int = 8192
    num_predict: int = 2048
    temperature: float = 0.0

    # How long we wait for Ollama to answer a single request.
    request_timeout_seconds: float = 300.0

    # Image handling. Downscaling keeps VRAM/latency predictable; set
    # max_image_side=0 to disable resizing entirely.
    max_image_side: int = 1600
    jpeg_quality: int = 95

    # Request limits.
    max_upload_mb: float = 25.0
    max_images_per_request: int = 8

    # Only one request is sent to Ollama at a time by default, since a
    # single consumer GPU gains nothing from parallel decode and it avoids
    # VRAM contention with other services (e.g. Whisper) sharing the card.
    max_concurrent_requests: int = 1


settings = Settings()
