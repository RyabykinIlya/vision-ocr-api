"""Thin async client for the Ollama chat API."""
from __future__ import annotations

import asyncio
import logging

import httpx

from .config import Settings

log = logging.getLogger("vision_ocr_api.ollama")

# Markers of the Ollama runner bug where a reused slot poisons generation
# and every request aborts with "prediction aborted, token repeat limit
# reached" (https://github.com/ollama/ollama/issues/18609). Unloading the
# model and retrying heals it.
_REPEAT_ABORT_MARKERS = ("repeat limit reached", "prediction aborted")


class OllamaError(RuntimeError):
    """Raised when Ollama is unreachable or returns an error."""


def _is_repeat_abort(exc: OllamaError) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in _REPEAT_ABORT_MARKERS)


class OllamaClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = httpx.AsyncClient(
            base_url=settings.ollama_url.rstrip("/"),
            timeout=httpx.Timeout(settings.request_timeout_seconds, connect=10.0),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def healthy(self) -> bool:
        try:
            resp = await self._client.get("/api/tags", timeout=5.0)
        except httpx.HTTPError:
            return False
        return resp.status_code == 200

    async def list_models(self) -> list[str]:
        try:
            resp = await self._client.get("/api/tags", timeout=10.0)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise OllamaError(f"could not list models: {exc}") from exc
        data = resp.json()
        return [m.get("name", "") for m in data.get("models", []) if m.get("name")]

    async def unload_model(self, model: str) -> None:
        """Ask Ollama to evict the model (keep_alive=0) and wait briefly.

        Used to recover from the runner-side repeat-abort bug; a subsequent
        request starts with a freshly loaded model.
        """
        try:
            await self._client.post(
                "/api/generate", json={"model": model, "keep_alive": 0, "stream": False}
            )
        except httpx.HTTPError as exc:
            log.warning("could not unload model %r: %s", model, exc)
            return
        await asyncio.sleep(1.0)

    async def chat_with_images(
        self,
        *,
        model: str,
        prompt: str,
        images_b64: list[str],
    ) -> str:
        attempts = max(1, self._settings.abort_retries + 1)
        last_exc: OllamaError | None = None
        for attempt in range(1, attempts + 1):
            try:
                return await self._chat_once(
                    model=model, prompt=prompt, images_b64=images_b64
                )
            except OllamaError as exc:
                last_exc = exc
                if attempt >= attempts or not _is_repeat_abort(exc):
                    raise
                log.warning(
                    "ollama repeat-abort (attempt %d/%d) on model %r; "
                    "unloading model and retrying",
                    attempt,
                    attempts,
                    model,
                )
                await self.unload_model(model)
        raise last_exc  # pragma: no cover - loop always returns or raises

    async def _chat_once(
        self,
        *,
        model: str,
        prompt: str,
        images_b64: list[str],
    ) -> str:
        payload = {
            "model": model,
            "stream": False,
            "keep_alive": self._settings.keep_alive,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": images_b64,
                }
            ],
            "options": {
                "temperature": self._settings.temperature,
                "num_ctx": self._settings.num_ctx,
                "num_predict": self._settings.num_predict,
            },
        }

        try:
            resp = await self._client.post("/api/chat", json=payload)
        except httpx.TimeoutException as exc:
            raise OllamaError(
                f"Ollama did not answer within {self._settings.request_timeout_seconds}s"
            ) from exc
        except httpx.HTTPError as exc:
            raise OllamaError(f"could not reach Ollama: {exc}") from exc

        if resp.status_code >= 400:
            detail = resp.text.strip()[:800]
            raise OllamaError(f"Ollama returned HTTP {resp.status_code}: {detail}")

        try:
            data = resp.json()
        except ValueError as exc:
            raise OllamaError("Ollama returned a non-JSON response") from exc

        if isinstance(data.get("error"), str):
            raise OllamaError(data["error"])

        message = data.get("message") or {}
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise OllamaError("Ollama returned an empty response")
        return content.strip()
