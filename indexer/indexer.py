"""Nightly image indexer for the family photo archive.

Walks a media directory, sends every new/changed image through the
vision-ocr-api service with one or more prompts, and stores the answers
in an SQLite database with an FTS5 full-text index.

Run modes:
  (no args)          daemon: sleep until the nightly window, index, sleep on
  --once --limit N   one bounded pass, ignoring the window (trial runs)

Everything is configured through INDEXER_* environment variables; see
.env.example and indexer/README.md.
"""
from __future__ import annotations

import argparse
import logging
import os
import random
import re
import signal
import sqlite3
import sys
import time
from datetime import datetime, timedelta, time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

from prompts import get_prompts

log = logging.getLogger("indexer")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif", ".gif", ".bmp", ".tif", ".tiff", ".avif"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
    path       TEXT PRIMARY KEY,   -- relative to the media root
    mtime      REAL NOT NULL,
    size       INTEGER NOT NULL,
    status     TEXT NOT NULL,      -- ok | failed
    attempts   INTEGER NOT NULL DEFAULT 0,
    model      TEXT,               -- model that produced the stored texts
    error      TEXT,
    indexed_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS texts (
    path    TEXT NOT NULL,
    prompt  TEXT NOT NULL,
    content TEXT NOT NULL,
    PRIMARY KEY (path, prompt)
);
CREATE VIRTUAL TABLE IF NOT EXISTS search_fts USING fts5(
    path UNINDEXED, prompt UNINDEXED, content,
    tokenize = 'unicode61 remove_diacritics 2'
);
"""

STOP = False


def _collapse_repeats(raw: str, max_reps: int = 1) -> str:
    """Cut text that repeats the same sentence/line over and over.

    Second line of defence after the service-side loop guard: a degenerate
    generation must not reach the index even if it slipped through. Keeps
    the first `max_reps` occurrences and drops the rest, preserving order.
    """
    segs = re.split(r"(?<=[.!?])\s+|\n", raw)
    counts: dict[str, int] = {}
    out: list[str] = []
    dropped = 0
    for seg in segs:
        key = seg.strip()
        if not key:
            continue
        counts[key] = counts.get(key, 0) + 1
        if counts[key] <= max_reps:
            out.append(key)
        else:
            dropped += 1
    if dropped >= 3:
        log.warning("collapsed %d repeated segment(s) in generated text", dropped)
    return "\n".join(out)


def _normalize_objects(raw: str) -> str:
    """Bring the objects answer to one-item-per-line, deduplicated.

    Handles the two-section prompt output ("Люди:" / "Объекты:") as well as
    plain comma-separated runs the model sometimes falls back to. Items cut
    mid-word by the loop guard are dropped.
    """
    lines: list[str] = []
    for line in raw.splitlines():
        stripped = line.strip().strip("*#-").strip()
        if not stripped:
            continue
        # section headers are kept as-is, they are useful in the index
        if stripped.rstrip(":").lower() in ("люди", "объекты", "предметы", "people", "objects"):
            lines.append(stripped if stripped.endswith(":") else stripped + ":")
            continue
        parts = [p.strip(" .;") for p in re.split(r"[,;]| \| ", stripped) if p.strip(" .;,|")]
        for p in parts:
            p = re.sub(r"^\d+[).]?\s*", "", p)
            if p:
                lines.append(p)

    seen: set[str] = set()
    out: list[str] = []
    for it in lines:
        if it.endswith(":") and it.lower().rstrip(":") in ("люди", "объекты", "предметы"):
            out.append(it)
            continue
        it = it.rstrip(" .;")
        # drop fragments cut mid-word (loop guard): an item ends with a letter
        if not it or not it[-1].isalpha():
            continue
        key = it.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(it)

    # a truncated tail like "челов" is a prefix of a real item ("человек")
    stems = {i.lower() for i in out if not i.endswith(":")}
    out = [
        i for i in out
        if i.endswith(":")
        or not any(o != i.lower() and o.startswith(i.lower()) and len(o) > len(i) for o in stems)
    ]

    # terminal alternation loop (A-B-A-B-...): keep one pass over the cycle
    if len(out) >= 7:
        tail = out[-6:]
        if len(set(tail)) <= 2:
            cycle = out[-2:]
            cut = len(out)
            while cut >= 2 and out[cut - 2:cut] == cycle:
                cut -= 2
            if cut < len(out) - 2:
                log.warning(
                    "collapsed alternation loop in objects list (%d -> %d items)",
                    len(out),
                    cut + 2,
                )
                out = out[: cut + 2]

    return "\n".join(out[:24])


def _handle_signal(signum, _frame):
    global STOP
    log.info("received %s, will stop after the current image", signum)
    STOP = True


class Indexer:
    def __init__(self) -> None:
        self.media_root = Path(os.environ.get("INDEXER_MEDIA_DIR", "/media")).resolve()
        self.db_path = Path(os.environ.get("INDEXER_DB_PATH", "/data/index.db"))
        self.api_url = os.environ.get("INDEXER_API_URL", "http://vision:8080").rstrip("/")
        self.model = os.environ.get("INDEXER_MODEL", "qwen2.5vl:3b")
        self.fallback_model = os.environ.get("INDEXER_FALLBACK_MODEL", "").strip()
        self.prompts = get_prompts(os.environ.get("INDEXER_PROMPTS", "scene,objects"))
        self.tz = ZoneInfo(os.environ.get("INDEXER_TZ", "Europe/Moscow"))
        win = os.environ.get("INDEXER_WINDOW", "00:00-07:00")
        self.window_start, self.window_end = self._parse_window(win)
        self.session = requests.Session()
        self.db = self._open_db()

    # ------------------------------------------------------------------ db
    def _open_db(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.db_path, timeout=30)
        db.row_factory = sqlite3.Row
        db.executescript(SCHEMA)
        db.execute("PRAGMA journal_mode=WAL")
        db.commit()
        return db

    # ------------------------------------------------------------- scanning
    def scan(self) -> dict[str, tuple[float, int]]:
        """Return {relative_path: (mtime, size)} of every image on disk."""
        found: dict[str, tuple[float, int]] = {}
        t0 = time.monotonic()
        for dirpath, dirnames, filenames in os.walk(self.media_root):
            dirnames[:] = [d for d in dirnames if not d.startswith((".", "_"))]
            for name in filenames:
                p = Path(dirpath) / name
                if p.suffix.lower() not in IMAGE_EXTS:
                    continue
                try:
                    st = p.stat()
                except OSError:
                    continue
                found[str(p.relative_to(self.media_root))] = (st.st_mtime, st.st_size)
        log.info("scan: %d images on disk in %.1fs", len(found), time.monotonic() - t0)
        return found

    def pending(self, limit: int, sample_seed: int | None = None) -> list[tuple[str, float, int]]:
        """Images that need indexing: new files and changed files.

        Returns a list of (relative_path, mtime, size).
        """
        disk = self.scan()
        known = {
            r["path"]: (r["mtime"], r["size"])
            for r in self.db.execute("SELECT path, mtime, size FROM images")
        }
        todo = [p for p, sig in disk.items() if known.get(p) != sig]
        if sample_seed is not None:
            random.Random(sample_seed).shuffle(todo)
        return [(p, disk[p][0], disk[p][1]) for p in todo[:limit]]

    # ------------------------------------------------------------ indexing
    def index_one(self, rel: str, mtime: float, size: int) -> None:
        abs_path = self.media_root / rel
        texts: dict[str, str] = {}
        errors: list[str] = []
        used_model = self.model
        t0 = time.monotonic()
        for key, prompt in self.prompts.items():
            content, err = self._ask(prompt, self.model, abs_path)
            if content is None and self.fallback_model and self.fallback_model != self.model:
                log.warning("%s: %s failed on %s, trying fallback %s", rel, key, err, self.fallback_model)
                used_model = self.fallback_model
                content, err = self._ask(prompt, self.fallback_model, abs_path)
            if content is None:
                errors.append(f"{key}: {err}")
            elif key == "objects":
                texts[key] = _normalize_objects(content)
            else:
                texts[key] = _collapse_repeats(content, max_reps=1)
        elapsed = time.monotonic() - t0

        with self.db:  # one transaction per image
            self.db.execute(
                "INSERT INTO images(path, mtime, size, status, attempts, model, error, indexed_at) "
                "VALUES(?,?,?,?,1,?,?,?) "
                "ON CONFLICT(path) DO UPDATE SET mtime=excluded.mtime, size=excluded.size, "
                "status=excluded.status, attempts=attempts+1, model=excluded.model, "
                "error=excluded.error, indexed_at=excluded.indexed_at",
                (
                    rel,
                    mtime,
                    size,
                    "ok" if texts else "failed",
                    used_model if texts else self.model,
                    "; ".join(errors) if errors else None,
                    datetime.now(self.tz).isoformat(timespec="seconds"),
                ),
            )
            self.db.execute("DELETE FROM texts WHERE path = ?", (rel,))
            self.db.execute("DELETE FROM search_fts WHERE path = ?", (rel,))
            for key, content in texts.items():
                self.db.execute(
                    "INSERT INTO texts(path, prompt, content) VALUES(?,?,?)",
                    (rel, key, content),
                )
                self.db.execute(
                    "INSERT INTO search_fts(path, prompt, content) VALUES(?,?,?)",
                    (rel, key, content),
                )
        status = "ok" if texts else "FAILED"
        log.info("%s %s in %.1fs (%d prompt(s))%s", status, rel, elapsed, len(texts),
                 f" errors: {'; '.join(errors)[:200]}" if errors else "")

    def _ask(self, prompt: str, model: str, abs_path: Path) -> tuple[str | None, str]:
        """One recognition call; returns (text, None) or (None, error)."""
        try:
            raw = abs_path.read_bytes()
        except OSError as exc:
            return None, f"read error: {exc}"
        try:
            resp = self.session.post(
                f"{self.api_url}/v1/recognize",
                files={"files": (abs_path.name, raw)},
                data={"prompt": prompt, "model": model},
                timeout=(30, 600),
            )
        except requests.RequestException as exc:
            return None, f"request error: {exc}"
        try:
            data = resp.json()
        except ValueError:
            return None, f"HTTP {resp.status_code}: non-JSON response"
        if resp.status_code != 200:
            return None, f"HTTP {resp.status_code}: {str(data.get('detail'))[:200]}"
        return data.get("text", ""), None

    # ------------------------------------------------------------- windows
    @staticmethod
    def _parse_window(spec: str) -> tuple[dtime, dtime]:
        try:
            start_s, end_s = spec.split("-")
            sh, sm = map(int, start_s.split(":"))
            eh, em = map(int, end_s.split(":"))
            return dtime(sh, sm), dtime(eh, em)
        except Exception as exc:
            raise SystemExit(f"bad INDEXER_WINDOW {spec!r}: {exc}") from exc

    def in_window(self, now: datetime | None = None) -> bool:
        now = now or datetime.now(self.tz)
        return self.window_start <= now.time() < self.window_end

    def seconds_until_start(self, now: datetime | None = None) -> float:
        now = now or datetime.now(self.tz)
        start = now.replace(hour=self.window_start.hour, minute=self.window_start.minute,
                            second=0, microsecond=0)
        if now.time() >= self.window_end or now.time() >= self.window_start:
            start += timedelta(days=1)
        return max(0.0, (start - now).total_seconds())

    def window_left(self, now: datetime | None = None) -> float:
        now = now or datetime.now(self.tz)
        end = now.replace(hour=self.window_end.hour, minute=self.window_end.minute,
                          second=0, microsecond=0)
        return max(0.0, (end - now).total_seconds())

    # ---------------------------------------------------------------- pass
    def run_pass(self, limit: int | None = None, sample_seed: int | None = None) -> None:
        todo = self.pending(limit=limit or 10**9, sample_seed=sample_seed)
        total = len(todo)
        log.info("pass: %d image(s) to index", total)
        for i, (rel, mtime, size) in enumerate(todo, 1):
            if STOP:
                log.info("stop requested, %d image(s) left for the next pass", total - i + 1)
                return
            if limit is None and not self.in_window():
                log.info("window closed, %d image(s) left for the next pass", total - i + 1)
                return
            try:
                self.index_one(rel, mtime, size)
            except Exception:  # noqa: BLE001 - one bad image must not kill the pass
                log.exception("unexpected error on %s", rel)
            if i % 25 == 0 or i == total:
                log.info("progress: %d/%d", i, total)

    def stats(self) -> str:
        rows = self.db.execute(
            "SELECT status, COUNT(*) AS n FROM images GROUP BY status"
        ).fetchall()
        parts = [f"{r['status']}={r['n']}" for r in rows]
        return "db: " + (", ".join(parts) if parts else "empty")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true",
                        help="run one bounded pass and exit (ignores the window)")
    parser.add_argument("--limit", type=int, default=None,
                        help="max images for this run (with --once)")
    parser.add_argument("--sample", type=int, default=None, metavar="SEED",
                        help="shuffle the queue with this seed before limiting (trial runs)")
    parser.add_argument("--fresh", action="store_true",
                        help="delete the database first (trial runs)")
    parser.add_argument("--retry-failed", action="store_true",
                        help="reset failed images so they are retried")
    parser.add_argument("--only", nargs="+", default=None, metavar="PATH",
                        help="index exactly these paths (relative to the media root)")
    parser.add_argument("--redo", action="store_true",
                        help="re-index everything already in the db (refresh prompts)")
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    if args.fresh and Path(os.environ.get("INDEXER_DB_PATH", "/data/index.db")).exists():
        Path(os.environ["INDEXER_DB_PATH"]).unlink()
        log.info("removed old database")

    ix = Indexer()
    log.info(
        "indexer up: media=%s db=%s api=%s model=%s fallback=%s prompts=%s window=%02d:%02d-%02d:%02d %s",
        ix.media_root, ix.db_path, ix.api_url, ix.model,
        ix.fallback_model or "-", ", ".join(ix.prompts),
        ix.window_start.hour, ix.window_start.minute,
        ix.window_end.hour, ix.window_end.minute, ix.tz,
    )

    if args.retry_failed:
        with ix.db:
            n = ix.db.execute("UPDATE images SET status='none' WHERE status='failed'").rowcount
        log.info("reset %d failed image(s) for retry", n)

    if args.once:
        if args.only:
            todo = []
            disk = ix.scan()
            for p in args.only:
                if p in disk:
                    todo.append((p, disk[p][0], disk[p][1]))
                else:
                    log.warning("not found on disk: %s", p)
            log.info("pass (--only): %d image(s)", len(todo))
            for i, (rel, mtime, size) in enumerate(todo, 1):
                if STOP:
                    break
                try:
                    ix.index_one(rel, mtime, size)
                except Exception:  # noqa: BLE001
                    log.exception("unexpected error on %s", rel)
        elif args.redo:
            disk = ix.scan()
            known = [r["path"] for r in ix.db.execute("SELECT path FROM images ORDER BY path")]
            todo = [(p, disk[p][0], disk[p][1]) for p in known if p in disk]
            log.info("pass (--redo): %d already-indexed image(s) to refresh", len(todo))
            for i, (rel, mtime, size) in enumerate(todo, 1):
                if STOP:
                    break
                try:
                    ix.index_one(rel, mtime, size)
                except Exception:  # noqa: BLE001
                    log.exception("unexpected error on %s", rel)
                if i % 10 == 0 or i == len(todo):
                    log.info("progress: %d/%d", i, len(todo))
        else:
            ix.run_pass(limit=args.limit, sample_seed=args.sample)
        log.info(ix.stats())
        return

    # daemon mode: run inside the nightly window, sleep outside it
    enabled = os.environ.get("INDEXER_ENABLED", "false").lower() in ("1", "true", "yes")
    if not enabled:
        log.warning("INDEXER_ENABLED is false — sleeping forever, nothing will be indexed")
    while True:
        if not enabled or STOP:
            time.sleep(3600)
            continue
        if not ix.in_window():
            wait = min(ix.seconds_until_start(), 3600)
            log.info("outside the window, sleeping %.0fs", wait)
            time.sleep(wait)
            continue
        ix.run_pass()
        log.info(ix.stats())


if __name__ == "__main__":
    main()
