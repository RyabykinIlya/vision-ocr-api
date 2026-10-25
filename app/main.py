"""FastAPI entrypoint for vision-ocr-api."""
from __future__ import annotations

import asyncio
import base64
import binascii
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .config import settings
from .imaging import ImageProcessingError, process_image
from .ollama_client import OllamaClient, OllamaError
from .prompts import DEFAULT_PRESET, PRESETS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("vision_ocr_api")

_client: OllamaClient | None = None
_semaphore = asyncio.Semaphore(max(1, settings.max_concurrent_requests))


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _client
    _client = OllamaClient(settings)
    log.info(
        "starting: ollama=%s model=%s keep_alive=%s num_ctx=%d",
        settings.ollama_url,
        settings.model,
        settings.keep_alive,
        settings.num_ctx,
    )
    try:
        yield
    finally:
        if _client is not None:
            await _client.aclose()


app = FastAPI(
    title="vision-ocr-api",
    version="1.0.0",
    summary="Image-to-text API backed by a local Ollama vision model",
    lifespan=lifespan,
)


# --------------------------------------------------------------------------
# schemas
# --------------------------------------------------------------------------
class ImageInfo(BaseModel):
    name: str
    width: int
    height: int
    size_bytes: int


class RecognizeResponse(BaseModel):
    text: str
    model: str
    preset: str | None = None
    elapsed_ms: int
    images: list[ImageInfo] = Field(default_factory=list)


class Base64Request(BaseModel):
    images: list[str] = Field(..., description="Base64 image data, optionally as a data URI")
    preset: str | None = Field(default=None, description=f"One of: {', '.join(PRESETS)}")
    prompt: str | None = Field(default=None, description="Custom prompt, overrides preset")
    model: str | None = Field(default=None, description="Override the configured model")
    max_side: int | None = Field(default=None, description="Override image downscaling")


class HealthResponse(BaseModel):
    status: str
    ollama: bool
    model: str


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _resolve_prompt(preset: str | None, prompt: str | None) -> tuple[str, str | None]:
    if preset and prompt:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="pass either 'preset' or 'prompt', not both",
        )
    if prompt:
        return prompt, None
    chosen = preset or DEFAULT_PRESET
    if chosen not in PRESETS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"unknown preset '{chosen}'; available: {', '.join(PRESETS)}",
        )
    return PRESETS[chosen], chosen


def _decode_b64(raw: str) -> bytes:
    payload = raw.strip()
    if payload.startswith("data:"):
        _, _, payload = payload.partition(",")
    try:
        return base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"invalid base64 image data: {exc}",
        ) from exc


async def _recognize(
    *,
    blobs: list[tuple[str, bytes]],
    preset: str | None,
    prompt: str | None,
    model: str | None,
    max_side: int | None,
) -> RecognizeResponse:
    if not blobs:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="at least one image is required",
        )
    if len(blobs) > settings.max_images_per_request:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"too many images: {len(blobs)} > {settings.max_images_per_request}",
        )

    limit = int(settings.max_upload_mb * 1024 * 1024)
    for name, payload in blobs:
        if len(payload) > limit:
            raise HTTPException(
                status_code=413,
                detail=f"'{name}' is larger than {settings.max_upload_mb} MB",
            )

    resolved_prompt, resolved_preset = _resolve_prompt(preset, prompt)
    chosen_model = model or settings.model
    side = settings.max_image_side if max_side is None else max_side

    processed = []
    for name, payload in blobs:
        try:
            processed.append(
                process_image(
                    payload,
                    name=name,
                    max_side=side,
                    jpeg_quality=settings.jpeg_quality,
                )
            )
        except ImageProcessingError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
            ) from exc

    assert _client is not None  # set in lifespan

    started = time.perf_counter()
    async with _semaphore:
        try:
            text = await _client.chat_with_images(
                model=chosen_model,
                prompt=resolved_prompt,
                images_b64=[img.base64_data for img in processed],
            )
        except OllamaError as exc:
            log.error("ollama error: %s", exc)
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    elapsed_ms = int((time.perf_counter() - started) * 1000)

    log.info(
        "recognized %d image(s) model=%s preset=%s in %d ms",
        len(processed),
        chosen_model,
        resolved_preset or "custom",
        elapsed_ms,
    )

    return RecognizeResponse(
        text=text,
        model=chosen_model,
        preset=resolved_preset,
        elapsed_ms=elapsed_ms,
        images=[
            ImageInfo(
                name=img.name,
                width=img.width,
                height=img.height,
                size_bytes=img.size_bytes,
            )
            for img in processed
        ],
    )


# --------------------------------------------------------------------------
# routes
# --------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["meta"])
async def health() -> HealthResponse:
    ok = _client is not None and await _client.healthy()
    return HealthResponse(status="ok" if ok else "degraded", ollama=ok, model=settings.model)


@app.get("/presets", tags=["meta"])
async def list_presets() -> dict[str, str]:
    return PRESETS


@app.get("/models", tags=["meta"])
async def list_models() -> dict[str, list[str]]:
    assert _client is not None
    try:
        return {"models": await _client.list_models()}
    except OllamaError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc


@app.post(
    "/v1/recognize",
    response_model=RecognizeResponse,
    tags=["vision"],
)
async def recognize_upload(
    files: list[UploadFile] = File(..., description="One or more image files"),
    preset: str | None = Form(default=None),
    prompt: str | None = Form(default=None),
    model: str | None = Form(default=None),
    max_side: int | None = Form(default=None),
) -> RecognizeResponse:
    blobs = [(f.filename or f"image-{i}", await f.read()) for i, f in enumerate(files)]
    await _close_files(files)
    return await _recognize(
        blobs=blobs, preset=preset, prompt=prompt, model=model, max_side=max_side
    )


@app.post(
    "/v1/recognize/base64",
    response_model=RecognizeResponse,
    tags=["vision"],
)
async def recognize_base64(body: Base64Request) -> RecognizeResponse:
    blobs = [
        (f"image-{i}", _decode_b64(data)) for i, data in enumerate(body.images)
    ]
    return await _recognize(
        blobs=blobs,
        preset=body.preset,
        prompt=body.prompt,
        model=body.model,
        max_side=body.max_side,
    )


async def _close_files(files: list[UploadFile]) -> None:
    for f in files:
        await f.close()


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc: Exception) -> JSONResponse:
    log.exception("unhandled error on %s: %s", request.url.path, exc)
    return JSONResponse(status_code=500, content={"detail": "internal server error"})
