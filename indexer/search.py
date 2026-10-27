#!/usr/bin/env python3
"""Full-text search over the indexer's SQLite database.

Usage (inside the container, or anywhere with python3 + the db file):

  python search.py кошк*                    # prefix search
  python search.py "кошка" --prompt scene   # exact token
  python search.py море пляж --limit 50     # any of the words
  python search.py --stats                  # index stats

Words without * get a prefix star appended automatically (russian
morphology friendly: "кошк*" matches кошка/кошки/кошкой). Words of
<= 2 letters are left as-is (prepositions).
"""
from __future__ import annotations

import argparse
import os
import re
import sqlite3
import sys

DB_DEFAULT = os.environ.get("INDEXER_DB_PATH", "/data/index.db")


def build_match(query: str) -> str:
    """Build an FTS5 MATCH expression, russian-morphology friendly.

    "кошка" -> (кошка* OR кошк*)  exact form plus the truncated stem
    "карта" -> (карта* OR карт*)  catches карте/карты/картой
    Explicit "*", quotes and short words pass through untouched.
    """
    tokens = re.findall(r"[\w*]+", query, flags=re.UNICODE)
    out = []
    for t in tokens:
        if "*" in t or len(t) <= 2:
            out.append(t)
            continue
        stem = t[:-2] if len(t) >= 6 else t[:-1] if len(t) >= 4 else t
        if stem and stem != t:
            out.append(f"({t}* OR {stem}*)")
        else:
            out.append(f"{t}*")
    return " AND ".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("query", nargs="*", help="search words")
    parser.add_argument("--db", default=DB_DEFAULT)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--prompt", default=None,
                        help="only search this prompt key (scene/objects/text)")
    parser.add_argument("--full", action="store_true",
                        help="print the full stored text, not a snippet")
    parser.add_argument("--stats", action="store_true", help="show index stats and exit")
    args = parser.parse_args()

    if not os.path.exists(args.db):
        print(f"no database at {args.db}", file=sys.stderr)
        return 1
    db = sqlite3.connect(args.db)
    db.row_factory = sqlite3.Row

    if args.stats:
        for r in db.execute("SELECT status, COUNT(*) n FROM images GROUP BY status"):
            print(f"{r['status']:8s} {r['n']}")
        n_texts = db.execute("SELECT COUNT(*) n FROM texts").fetchone()["n"]
        n_paths = db.execute("SELECT COUNT(DISTINCT path) n FROM texts").fetchone()["n"]
        print(f"texts    {n_texts} over {n_paths} images")
        return 0

    if not args.query:
        parser.error("empty query")

    match = build_match(" ".join(args.query))
    sql = """
        SELECT fts.path AS path, fts.prompt AS prompt,
               snippet(search_fts, 2, '>>', '<<', ' … ', 18) AS snip,
               bm25(search_fts) AS rank,
               images.status AS status
        FROM search_fts fts
        JOIN images ON images.path = fts.path
        WHERE search_fts MATCH ?
    """
    params: list = [match]
    if args.prompt:
        sql += " AND fts.prompt = ?"
        params.append(args.prompt)
    sql += " ORDER BY rank LIMIT ?"
    params.append(args.limit)

    rows = db.execute(sql, params).fetchall()
    if not rows:
        print("ничего не найдено")
        return 0
    for r in rows:
        print(f"{r['path']}")
        print(f"  [{r['prompt']}] {r['snip'] if not args.full else ''}")
        if args.full:
            content = db.execute(
                "SELECT content FROM texts WHERE path=? AND prompt=?",
                (r["path"], r["prompt"]),
            ).fetchone()["content"]
            for line in content.splitlines():
                print(f"    {line}")
        print()
    print(f"{len(rows)} совпадение(й)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
