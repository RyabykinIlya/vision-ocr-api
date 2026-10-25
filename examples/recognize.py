#!/usr/bin/env python3
"""Minimal client for vision-ocr-api."""
import base64
import json
import os
import sys
import urllib.request

API = os.environ.get("VISION_API_URL", "http://localhost:8080")


def b64(path: str) -> str:
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode("ascii")


def recognize(path: str, preset: str = "ocr") -> dict:
    body = json.dumps({"images": [b64(path)], "preset": preset}).encode()
    req = urllib.request.Request(
        f"{API}/v1/recognize/base64",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.load(resp)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: recognize.py <image> [preset]")
    result = recognize(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "ocr")
    print(result["text"])
    print(f"\n[{result['model']}, {result['elapsed_ms']} ms]", file=sys.stderr)
