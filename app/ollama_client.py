"""Thin async client for the Ollama chat API."""
from __future__ import annotations

import httpx

from .config import Settings


class OllamaError(RuntimeError):
    """Raised when Ollama is unreachable or returns an error."""


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

    async def chat_with_images(
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
