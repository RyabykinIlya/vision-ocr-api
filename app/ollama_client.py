"""Thin async client for the Ollama chat API."""
from __future__ import annotations

import asyncio
import json
import logging
import re

import httpx

from .config import Settings

log = logging.getLogger("vision_ocr_api.ollama")

# Markers of the Ollama runner bug where a reused slot poisons generation
# and every request aborts with "prediction aborted, token repeat limit
# reached" (https://github.com/ollama/ollama/issues/18609). Unloading the
# model and retrying heals it.
_REPEAT_ABORT_MARKERS = (
    "repeat limit reached",
    "prediction aborted",
    # A reasoning model can spend the whole num_predict budget on hidden
    # reasoning and finish with no visible content. It is transient (the
    # next attempt usually answers), so treat it as retryable.
    "empty response",
)


def _detect_loop(text: str, min_unit: int = 8, max_unit: int = 90, reps: int = 4) -> bool:
    """True if the text tail is the same fragment repeated over and over.

    Two independent checks, because degenerate generations come in two
    shapes:

    * exact repetition, possibly cut mid-unit (the stream ends or
      num_predict runs out inside the repeated phrase) — handled by
      testing suffixes with a small offset;
    * the same *sentence* recurring a few times in the tail — handled by a
      simple frequency count over sentence-sized segments.
    """
    # 1) suffix repetition, tolerating a partial trailing unit
    window = text[-(max_unit * reps + max_unit):]
    for unit_len in range(min_unit, max_unit + 1):
        for offset in range(0, min(unit_len, 40)):
            tail = window[: len(window) - offset] if offset else window
            if len(tail) < unit_len * reps:
                continue
            unit = tail[-unit_len:]
            count = 0
            i = len(tail)
            while i >= unit_len and tail[i - unit_len:i] == unit:
                count += 1
                i -= unit_len
            if count >= reps:
                return True

    # 2) same sentence repeated in the tail
    tail = text[-800:]
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n", tail) if len(s.strip()) >= 15]
    if sentences:
        counts: dict[str, int] = {}
        for s in sentences:
            counts[s] = counts.get(s, 0) + 1
        if max(counts.values()) >= 3:
            return True
    return False


class OllamaError(RuntimeError):
    """Raised when Ollama is unreachable or returns an error."""


def _is_repeat_abort(exc: OllamaError) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in ("repeat limit reached", "prediction aborted"))


def _is_empty_answer(exc: OllamaError) -> bool:
    """A reasoning model burned its budget on hidden reasoning."""
    return "empty response" in str(exc).lower()


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
        budget = self._settings.num_predict
        for attempt in range(1, attempts + 1):
            try:
                return await self._chat_streaming(
                    model=model, prompt=prompt, images_b64=images_b64, budget=budget
                )
            except OllamaError as exc:
                last_exc = exc
                if attempt >= attempts:
                    raise
                if _is_repeat_abort(exc):
                    # Runner state is poisoned: unload so the retry starts clean.
                    log.warning(
                        "ollama repeat-abort (attempt %d/%d) on model %r; "
                        "unloading model and retrying",
                        attempt,
                        attempts,
                        model,
                    )
                    await self.unload_model(model)
                elif _is_empty_answer(exc):
                    # The reasoning model spent the whole token budget on
                    # hidden reasoning. The model is fine, just give the retry
                    # more room (only for this request, not globally).
                    new_budget = min(budget * 2, 16384)
                    log.warning(
                        "empty answer (attempt %d/%d) on model %r: budget likely "
                        "spent on reasoning; retrying with num_predict %d",
                        attempt,
                        attempts,
                        model,
                        new_budget,
                    )
                    budget = new_budget
                else:
                    raise
        raise last_exc  # pragma: no cover - loop always returns or raises

    async def _chat_streaming(
        self,
        *,
        model: str,
        prompt: str,
        images_b64: list[str],
        budget: int | None = None,
    ) -> str:
        """Stream the answer so a runaway generation can be cut early.

        Consumes ollama's NDJSON stream, watches the accumulated text and, if
        the same fragment repeats consecutively (a degenerate loop small VLMs
        fall into), drops the connection. Ollama stops generating on
        disconnect, so the GPU is freed immediately instead of grinding to
        num_predict. As a bonus the read timeout now applies per token batch,
        not to the whole generation.
        """
        payload = {
            "model": model,
            "stream": True,
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
                "num_predict": budget if budget is not None else self._settings.num_predict,
            },
        }
        if self._settings.think is not None:
            # Reasoning models: false makes ollama hide the reasoning trace.
            payload["think"] = self._settings.think

        parts: list[str] = []
        checked_at = 0

        try:
            async with self._client.stream("POST", "/api/chat", json=payload) as resp:
                if resp.status_code >= 400:
                    body = (await resp.aread()).decode(errors="replace")
                    raise OllamaError(
                        f"Ollama returned HTTP {resp.status_code}: {body.strip()[:800]}"
                    )
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(data.get("error"), str):
                        raise OllamaError(data["error"])
                    chunk = (data.get("message") or {}).get("content")
                    if chunk:
                        parts.append(chunk)
                    if data.get("done"):
                        break
                    text = "".join(parts)
                    if (
                        self._settings.loop_guard
                        and len(text) - checked_at >= 32
                        and _detect_loop(text)
                    ):
                        log.warning(
                            "generation loop detected after %d chars on %r; "
                            "cutting the stream",
                            len(text),
                            model,
                        )
                        if len(text) < 40:
                            # Looped right from the start: the text is
                            # useless. Retry through the repeat-abort path
                            # (unload + fresh attempt).
                            raise OllamaError(
                                "local repeat limit reached: looped from the start"
                            )
                        return text.strip()  # partial but usable
                    checked_at = len(text)
        except httpx.TimeoutException as exc:
            if parts:
                # Tokens were flowing but stalled mid-generation; keep what
                # we have rather than failing the whole request.
                log.warning("stream stalled mid-generation, keeping %d chars", len("".join(parts)))
                return "".join(parts).strip()
            raise OllamaError(
                f"Ollama did not answer within {self._settings.request_timeout_seconds}s"
            ) from exc
        except httpx.HTTPError as exc:
            raise OllamaError(f"could not reach Ollama: {exc}") from exc

        text = "".join(parts).strip()
        if not text:
            raise OllamaError("Ollama returned an empty response")
        return text
