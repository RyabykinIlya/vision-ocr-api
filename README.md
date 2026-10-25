# vision-ocr-api

Small HTTP service that turns images into text using a local
[Ollama](https://ollama.com) vision model. One `POST`, plain text back — no auth,
so anything on your network can call it. Built to run in Docker (WSL2) next to
other GPU workloads on a single consumer card.

```
POST /v1/recognize         multipart/form-data (one or more files)
POST /v1/recognize/base64  JSON {"images": ["<base64>", ...]}
```

## Why it exists

Ollama's API takes base64 images and returns chat-style JSON, which is awkward to
wire into agents and scripts. This wraps that in a boring endpoint: resize the
image, pick a prompt, return `{text, model, elapsed_ms}`. Used as the backend for
an agent skill, so any agent can hand over a screenshot and get text back.

## Requirements

- Docker with Compose v2
- For GPU: WSL2 + NVIDIA driver on Windows, `nvidia-container-toolkit` in WSL
- ~7 GB VRAM free for the default 7B model (models below are 2–8 GB)

Check GPU passthrough:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

## Quick start

```bash
git clone https://github.com/RyabykinIlya/vision-ocr-api.git
cd vision-ocr-api
cp .env.example .env

docker compose up -d --build
docker compose exec ollama ollama pull qwen2.5vl:7b-q4_K_M   # ~6 GB

curl -s localhost:8080/health
```

First recognition call after a cold start includes model load (10–30 s). After
that `keep_alive` keeps it warm for 60 s.

## Usage

```bash
# multipart — simplest from shell
curl -s -X POST localhost:8080/v1/recognize \
  -F "files=@screenshot.png" -F "preset=ocr"

# base64 — simplest from code
curl -s -X POST localhost:8080/v1/recognize/base64 \
  -H 'Content-Type: application/json' \
  -d '{"images": ["'"$(base64 -w0 screenshot.png)"'"], "preset": "ocr_translate"}'
```

```json
{
  "text": "Invoice #4711\nTotal: 1 240.00 RUB",
  "model": "qwen2.5vl:7b-q4_K_M",
  "preset": "ocr",
  "elapsed_ms": 4820,
  "images": [{"name": "screenshot.png", "width": 1600, "height": 900, "size_bytes": 184320}]
}
```

Multiple images in one call are treated as one document (e.g. pages of a scan) —
the model sees all of them together.

See `examples/recognize.py` (stdlib only) and `examples/smoke-test.sh`.

### Presets

| Preset | Output |
|---|---|
| `ocr` (default) | Verbatim transcription, structure preserved, no commentary |
| `ocr_translate` | Transcription translated into Russian |
| `describe` | Prose description in Russian |
| `table` | Markdown tables only |
| `diagram` | Explanation of a chart/diagram in Russian |

Pass `"prompt": "..."` instead of `"preset"` for anything else. Passing both is a
422.

### Endpoints

| Method | Path | Notes |
|---|---|---|
| `GET` | `/health` | Always 200; `ollama: false` means upstream is down |
| `GET` | `/presets` | Available preset names and prompts |
| `GET` | `/models` | Models pulled in the Ollama instance |
| `POST` | `/v1/recognize` | `files` (repeated), `preset`, `prompt`, `model`, `max_side` |
| `POST` | `/v1/recognize/base64` | JSON body, same optional fields |

Errors: `422` bad input/preset/image, `413` over `VISION_MAX_UPLOAD_MB`,
`502` Ollama unreachable or returned an error (its message is passed through).
Interactive docs at `/docs`.

## Configuration

Every setting is an env var with the `VISION_` prefix; see `.env.example`.

| Variable | Default | Purpose |
|---|---|---|
| `VISION_OLLAMA_URL` | `http://ollama:11434` | Upstream Ollama |
| `VISION_MODEL` | `qwen2.5vl:7b-q4_K_M` | Default model |
| `VISION_KEEP_ALIVE` | `60s` | How long Ollama holds the model in VRAM |
| `VISION_NUM_CTX` | `8192` | Context window; vision tokens eat this fast |
| `VISION_MAX_IMAGE_SIDE` | `1600` | Long-edge downscale, `0` disables |
| `VISION_MAX_CONCURRENT_REQUESTS` | `1` | In-flight requests (one GPU = no gain from more) |
| `VISION_MAX_UPLOAD_MB` | `25` | Per-image limit |
| `VISION_HOST_PORT` | `8080` | Published port (Compose) |

## Model choice

Tuned for an 8 GB card (RTX 3070 Ti and similar). Q4 quant, VRAM ≈ weights.

| Model | VRAM | Best at |
|---|---|---|
| `qwen2.5vl:7b-q4_K_M` | ~6.0 GB | Best all-round: OCR, documents, charts |
| `minicpm-v:8b-2.6-q4_K_M` | ~5.5 GB | Dense small text, screenshots |
| `gemma3:4b` | ~3.3 GB | Fast general questions, good Russian |
| `qwen2.5vl:3b` | ~3.2 GB | Fast compromise |
| `granite3.2-vision:2b` | ~2.5 GB | OCR on minimal VRAM |
| `moondream` | ~1.7 GB | Simple descriptions only |

`llama3.2-vision:11b` technically fits at ~7.9 GB but leaves no room for context
and part of it spills to CPU — not recommended.

Swap with `VISION_MODEL=gemma3:4b`, then `docker compose up -d vision`.

## Sharing the GPU with Whisper

Both squeeze into 8 GB only if they take turns. In the bundled `ollama` service:

```yaml
OLLAMA_KEEP_ALIVE: 60s        # unload the VLM shortly after each request
OLLAMA_MAX_LOADED_MODELS: "1"
```

With `faster-whisper` `medium` (~1.5 GB) resident, a 7B VLM will not fit
alongside — either drop Whisper to `small`, or run the VLM at 3–4B while
transcription is active. If the VLM does go resident next to Whisper you'll see
`CUDA out of memory` in `docker logs ollama-vision`; lowering `VISION_MAX_IMAGE_SIDE`
to 1280 and `VISION_NUM_CTX` to 4096 buys a few hundred MB.

## Using an existing Ollama

Already running Ollama natively in WSL? Delete the `ollama` service from
`docker-compose.yml` and set:

```ini
VISION_OLLAMA_URL=http://host.docker.internal:11434
```

`extra_hosts: host.docker.internal:host-gateway` is already configured for that.

## Security

No authentication by design — anyone who can reach port 8080 can spend your GPU
time and read images they send. Keep it on localhost or a trusted LAN; do not
publish it to the internet without putting a reverse proxy with auth in front.

## Notes

- Images are re-encoded: JPEG (quality 95) unless the source has alpha, then PNG.
  EXIF orientation is applied, so phone photos are not sideways.
- Nothing is written to disk — uploads are processed in memory.
- Requests are serialized by default; a single GPU gains nothing from parallel
  decode, and queueing keeps VRAM predictable.

## License

MIT — see [LICENSE](LICENSE).
