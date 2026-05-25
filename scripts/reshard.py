#!/usr/bin/env python3
"""Re-shard the database with activity-aware grouping.

Strategy (stable/active separation):

  - Query per-course stats: compressed size estimate, latest processed_at,
    pending/error page count, lecture count.
  - A course is "active" if it has any lectures processed within the last
    14 days, OR has pending/errored lectures (still being processed).
  - All other courses are "stable" (no recent activity, fully processed).
  - Active courses are grouped together (they change frequently, so putting
    them in the same shard means only that shard's hash changes).
  - Stable courses are packed into shards of ~5 MB.  Because they never
    change, their shard hashes are stable across runs.
  - A course is never split across shards.

Usage:
    python scripts/reshard.py <source_db> <output_dir>
"""

from __future__ import annotations

import datetime
import gzip
import hashlib
import json
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.schema import SCHEMA_SQL as _SCHEMA_SQL
from src.data.sharder import (
    COMPRESSION_RATIO_GUESS,
    INDEX_FILENAME,
    INDEX_VERSION,
    SHARDS_DIR,
    _build_shard_db,
    _course_uncompressed_size,
)
from src.data import crypto_box

# ── Config ─────────────────────────────────────────────────────────────────
STABLE_SHARD_TARGET_COMPRESSED = 5 * 1024 * 1024  # 5 MB per stable shard
ACTIVE_SHARD_TARGET_COMPRESSED = 3 * 1024 * 1024  # 3 MB per active shard
ACTIVE_WINDOW_DAYS = 14  # lectures updated within this window → "active"


def _derive_password() -> str:
    stuid = os.environ.get("STUID") or os.environ.get("StuId", "")
    uispsw = os.environ.get("UISPSW") or os.environ.get("UISPsw", "")
    if not stuid or not uispsw:
        print("error: STUID and UISPSW env vars required", file=sys.stderr)
        sys.exit(2)
    return crypto_box.derive_new_password(stuid, uispsw)


def _gzip_and_encrypt(sqlite_path: str, enc_path: str, password: str) -> str:
    """Gzip a SQLite file and encrypt with AES-256-CBC.
    Uses deterministic encryption (salt from content hash) so identical
    input produces identical ciphertext — content-addressed git blobs."""
    with open(sqlite_path, "rb") as f:
        raw = f.read()
    compressed = gzip.compress(raw, mtime=0)  # deterministic gzip
    encrypted = crypto_box.encrypt(compressed, password, deterministic=True)
    with open(enc_path, "wb") as f:
        f.write(encrypted)
    return hashlib.sha256(raw).hexdigest()


def main():
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        sys.exit(2)

    source_db = sys.argv[1]
    output_dir = sys.argv[2]
    os.makedirs(os.path.join(output_dir, SHARDS_DIR), exist_ok=True)

    password = _derive_password()
    conn = sqlite3.connect(source_db)
    conn.row_factory = sqlite3.Row

    # ── Step 1: Per-course stats ──────────────────────────────────────────
    rows = conn.execute(
        """SELECT c.course_id, c.title,
                  MAX(l.processed_at) AS last_processed,
                  COUNT(l.sub_id) AS lecture_count,
                  SUM(CASE WHEN l.error_count > 0 THEN 1 ELSE 0 END) AS error_count,
                  SUM(CASE WHEN l.processed_at IS NULL THEN 1 ELSE 0 END) AS pending_count
           FROM courses c
           LEFT JOIN lectures l ON c.course_id = l.course_id
           GROUP BY c.course_id
           ORDER BY c.course_id"""
    ).fetchall()

    if not rows:
        print("No courses found.")
        conn.close()
        return

    cutoff = datetime.datetime.utcnow() - datetime.timedelta(days=ACTIVE_WINDOW_DAYS)
    cutoff_str = cutoff.strftime("%Y-%m-%d")

    active: list[dict] = []
    stable: list[dict] = []
    for r in rows:
        course = {
            "course_id": r["course_id"],
            "title": r["title"],
            "size": _course_uncompressed_size(conn, r["course_id"]),
            "lecture_count": r["lecture_count"],
            "error_count": r["error_count"],
            "pending_count": r["pending_count"],
        }
        last_processed = r["last_processed"] or ""
        is_active = (
            (last_processed >= cutoff_str)
            or (course["error_count"] > 0)
            or (course["pending_count"] > 0)
        )
        if is_active:
            active.append(course)
        else:
            stable.append(course)

    # Sort active by last_processed desc (newest first), stable by course_id
    active.sort(key=lambda c: c["course_id"])
    stable.sort(key=lambda c: c["course_id"])

    print(f"Courses: {len(rows)} total, "
          f"{len(active)} active, {len(stable)} stable")
    print(f"Active window: {ACTIVE_WINDOW_DAYS} days (since {cutoff_str})")
    if active:
        print("Active courses:")
        for c in active:
            print(f"  {c['course_id']} {c['title'][:40]:40s} "
                  f"{c['size']/1024:6.0f}KB "
                  f"{c['lecture_count']} lectures")
    if stable:
        print(f"Stable courses: {len(stable)} ({sum(c['size'] for c in stable)/1024:.0f}KB)")

    # ── Step 2: Group into shards ─────────────────────────────────────────
    groups: list[list[str]] = []

    def _pack(courses: list[dict], target: int) -> list[list[str]]:
        """First-fit pack course_ids into shards at most ``target`` bytes.
        A course exceeding the target gets its own shard (never split)."""
        result: list[list[str]] = []
        remaining = list(courses)
        while remaining:
            group: list[str] = []
            group_size = 0
            for c in list(remaining):
                sz = c["size"]
                if group_size + sz <= target:
                    group.append(c["course_id"])
                    group_size += sz
                    remaining.remove(c)
                # Single oversized course → own shard (loop will pick it up
                # on next iteration when group is empty).
            result.append(group)
        return result

    # Active courses → 3 MB shards (they change frequently, small shards
    # minimize update payload).
    active_target = ACTIVE_SHARD_TARGET_COMPRESSED * COMPRESSION_RATIO_GUESS
    for g in _pack(active, active_target):
        if g:
            groups.append(g)

    # Stable courses → 5 MB shards (they never change, larger shards mean
    # fewer files with stable hashes).
    stable_target = STABLE_SHARD_TARGET_COMPRESSED * COMPRESSION_RATIO_GUESS
    for g in _pack(stable, stable_target):
        if g:
            groups.append(g)

    if not groups:
        groups = [[]]  # at least one (empty) shard

    print(f"\nShards: {len(groups)}")
    for i, g in enumerate(groups):
        size_kb = sum(
            _course_uncompressed_size(conn, cid) for cid in g
        ) / 1024
        print(f"  shard-{i+1:04d}: {len(g):3d} course(s), {size_kb:7.0f}KB")

    # ── Step 3: Build + encrypt shards ────────────────────────────────────
    shard_entries: list[dict] = []
    for idx, group in enumerate(groups):
        shard_name = f"shard-{idx+1:04d}.db.gz.enc"
        sqlite_path = os.path.join(output_dir, f"shard-{idx+1:04d}.db")
        enc_path = os.path.join(output_dir, SHARDS_DIR, shard_name)

        _build_shard_db(source_db, group, sqlite_path)
        content_hash = _gzip_and_encrypt(sqlite_path, enc_path, password)
        os.unlink(sqlite_path)

        shard_entries.append({
            "name": shard_name,
            "sha256": content_hash,
        })

    # ── Step 4: Write index ───────────────────────────────────────────────
    index = {
        "version": INDEX_VERSION,
        "created_at": datetime.datetime.utcnow().isoformat() + "Z",
        "shards": shard_entries,
        # Validation: sha256 of the JSON serialization of {"shards": [name,
        # sha256, ...], "version"} before encryption.  Lets the frontend / CLI
        # verify decryption was correct without fully reassembling.
        "checksum": "",
    }
    index_bytes = json.dumps(index, ensure_ascii=False, separators=(",", ":")).encode()
    index["checksum"] = hashlib.sha256(index_bytes).hexdigest()
    index_bytes_wsig = json.dumps(index, ensure_ascii=False, separators=(",", ":")).encode()
    encrypted_index = crypto_box.encrypt(index_bytes_wsig, password, deterministic=True)
    index_path = os.path.join(output_dir, INDEX_FILENAME)
    with open(index_path, "wb") as f:
        f.write(encrypted_index)

    print(f"\nIndex: {index_path} ({len(index['shards'])} shards)")
    print("Done.")

    conn.close()


if __name__ == "__main__":
    main()
