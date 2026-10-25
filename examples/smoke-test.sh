#!/usr/bin/env bash
# Smoke test for a running vision-ocr-api.
set -euo pipefail

API="${API:-http://localhost:8080}"
IMAGE="${1:-./examples/sample.png}"
PRESET="${2:-ocr}"

if [ ! -f "$IMAGE" ]; then
  echo "no such file: $IMAGE" >&2
  echo "usage: $0 [image] [preset]" >&2
  exit 1
fi

echo "== health =="
curl -fsS "$API/health"; echo
echo
echo "== $PRESET: $IMAGE =="
curl -fsS -X POST "$API/v1/recognize" \
  -F "files=@$IMAGE" \
  -F "preset=$PRESET" \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["text"]); print(f"\n[{d[\"model\"]}, {d[\"elapsed_ms\"]} ms]", file=sys.stderr)'
