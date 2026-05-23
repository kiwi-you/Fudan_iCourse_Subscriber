"""Centralised progress / debug reporter.

All multi-line orchestration output funnels through one object so format
changes happen in exactly one file.  Two reasons:
  1. The main run is long (hours).  Inconsistent log formats make it hard
     to grep / read after the fact.
  2. The image-progress and CPU-snapshot lines have to be *throttled*
     (every 30 pics; every 60 s) — putting the throttle policy in callers
     leaks state everywhere.

Thread safety: all emission methods take ``self._lock`` so concurrent
workers (image downloads, OCR, ResourceMonitor) interleave cleanly.
"""

from __future__ import annotations

import threading
import time


class Reporter:
    """Single sink for orchestration logs.

    Caller pattern: every place that used to ``print()`` orchestration
    text now calls ``reporter.<method>(...)``.  The reporter holds tiny
    throttling state (last-emit timestamps per kind/sub_id) so callers
    can fire freely without thinking about cadence.
    """

    # ── Throttling cadences ──
    IMAGE_PROGRESS_EVERY_PICS = 30  # emit a line every 30 finished images
    CPU_SNAPSHOT_INTERVAL_SEC = 60.0  # emit a snapshot at most every 60 s

    def __init__(self):
        self._lock = threading.Lock()
        # sub_id -> (last_done_emitted_at_count, t0, last_print_t)
        self._image_progress_state: dict[str, dict] = {}
        self._last_cpu_snapshot_t: float = 0.0

    # ── Lifecycle ────────────────────────────────────────────────────────

    def run_header(self):
        with self._lock:
            bar = "=" * 60
            print(bar)
            print("iCourse Subscriber — starting run")
            print(bar, flush=True)

    def run_footer(self):
        with self._lock:
            print(f"\n{'=' * 60}")
            print("Run complete.", flush=True)

    # ── Course-level ─────────────────────────────────────────────────────

    def course_header(self, course_id: str, title: str, teacher: str,
                      total: int, playback: int):
        with self._lock:
            print(f"\n{'─' * 50}")
            print(f"[Course] {course_id}")
            print(f"  Title: {title} (Teacher: {teacher})")
            print(f"  Total lectures: {total} ({playback} with playback)",
                  flush=True)

    def course_dedup_skip(self, sub_title: str, sub_id):
        with self._lock:
            print(f"  [Dedup] Skipping duplicate: {sub_title} "
                  f"(sub_id={sub_id})", flush=True)

    def course_new_count(self, n: int):
        with self._lock:
            print(f"  New/retry lectures: {n}", flush=True)
            if n == 0:
                print("  No new lectures, skipping.", flush=True)

    def course_enumeration_error(self, course_id: str):
        with self._lock:
            print(f"  ERROR enumerating course {course_id}:", flush=True)

    # ── Lecture-level ────────────────────────────────────────────────────

    def lecture_start(self, course_title: str, sub_title: str, date: str):
        """Lecture header — explicitly includes course_title so users can
        correlate a Phase-2 'Processing' line back to its course even after
        the Phase-1 enumeration scrolled away."""
        with self._lock:
            print(f"\n  -- [{course_title}] {sub_title} ({date})")
            print(f"    [Time] Start: {time.strftime('%Y-%m-%d %H:%M:%S')}",
                  flush=True)

    def lecture_skip_v2_done(self, sub_title: str, summary_chars: int):
        with self._lock:
            print(f"    v2 summary exists ({summary_chars} chars) and PPT "
                  f"pages present — skipping.", flush=True)

    def lecture_skip_no_video(self, sub_title: str):
        with self._lock:
            print(f"    No video URL — skipping.", flush=True)

    def lecture_done(self, course_title: str, sub_title: str, elapsed: float):
        with self._lock:
            print(f"    [Time] Done at {time.strftime('%H:%M:%S')}: "
                  f"[{course_title}] {sub_title} (total {elapsed:.0f}s)",
                  flush=True)

    def lecture_error(self, sub_id: str):
        with self._lock:
            print(f"    ERROR processing {sub_id}:", flush=True)

    # ── PPT pipeline ─────────────────────────────────────────────────────

    def ppt_pages_registered(self, total: int, inserted: int):
        with self._lock:
            print(f"    PPT pages: {total} total ({inserted} newly "
                  f"registered)", flush=True)

    def ppt_list_failed(self, exc_type: str, msg: str):
        with self._lock:
            print(f"    [WARN] PPT list fetch failed: {exc_type}: {msg}",
                  flush=True)

    def ppt_pipeline_summary(self, done: int, dedupped: int, invalid: int,
                             failed: int):
        with self._lock:
            print(f"    PPT pipeline: {done} done, {dedupped} dedup'd, "
                  f"{invalid} invalid, {failed} failed", flush=True)

    # ── Image-download progress (throttled every IMAGE_PROGRESS_EVERY_PICS) ──

    def image_progress_start(self, sub_id: str, total: int):
        """Record t0 for a sub_id so per-30-pic rate can be computed."""
        with self._lock:
            self._image_progress_state[sub_id] = {
                "total": total,
                "done": 0,
                "t0": time.time(),
                "last_emit_at_count": 0,
            }

    def image_progress_tick(self, sub_id: str):
        """Call once per finished image. Emits a line every 30 (and at end)."""
        with self._lock:
            st = self._image_progress_state.get(sub_id)
            if st is None:
                return
            st["done"] += 1
            done = st["done"]
            total = st["total"]
            is_final = (done == total)
            cross = (done - st["last_emit_at_count"]) >= self.IMAGE_PROGRESS_EVERY_PICS
            if not (cross or is_final):
                return
            st["last_emit_at_count"] = done
            elapsed = max(time.time() - st["t0"], 0.001)
            rate = done / elapsed
            bar = self._bar(done, total)
            print(f"    [Images {sub_id}] {bar} {done}/{total} "
                  f"({rate:.1f} pic/s)", flush=True)
            if is_final:
                self._image_progress_state.pop(sub_id, None)

    def image_progress_abort(self, sub_id: str):
        with self._lock:
            self._image_progress_state.pop(sub_id, None)

    @staticmethod
    def _bar(done: int, total: int, width: int = 20) -> str:
        if total <= 0:
            return "[" + " " * width + "]"
        filled = int(width * done / total)
        return "[" + "#" * filled + "-" * (width - filled) + "]"

    # ── Resource snapshot (throttled CPU / pool state) ──────────────────

    def cpu_snapshot(self, cpu_pct: float,
                     ocr_busy: int, ocr_target: int, ocr_max: int,
                     image_busy: int, image_max: int,
                     audio_busy: int, audio_max: int):
        """ResourceMonitor calls this every poll; we throttle to one print
        per CPU_SNAPSHOT_INTERVAL_SEC.  Always re-evaluates on call, so the
        printed value is the most recent reading at the print instant."""
        with self._lock:
            now = time.time()
            if now - self._last_cpu_snapshot_t < self.CPU_SNAPSHOT_INTERVAL_SEC:
                return
            self._last_cpu_snapshot_t = now
            print(f"    [Resource] CPU {cpu_pct:5.1f}%  "
                  f"OCR {ocr_busy}/{ocr_target} (max {ocr_max})  "
                  f"img {image_busy}/{image_max}  "
                  f"audio {audio_busy}/{audio_max}", flush=True)

    def cpu_target_changed(self, old: int, new: int, cpu_pct: float):
        """Always-emit log line: ResourceMonitor changed the OCR target.
        Less frequent than snapshots so it doesn't need throttling."""
        with self._lock:
            direction = "↑" if new > old else "↓"
            print(f"    [Resource] OCR target {old} {direction} {new} "
                  f"(cpu={cpu_pct:.1f}%)", flush=True)

    # ── Prefetch / audio download ────────────────────────────────────────

    def audio_prefetch_start(self, sub_id: str):
        with self._lock:
            print(f"    [Prefetch] audio for {sub_id} starting...",
                  flush=True)

    def audio_prefetch_done(self, sub_id: str, elapsed: float, size_mb: float):
        with self._lock:
            rate = size_mb / max(elapsed, 0.001)
            print(f"    [Prefetch] audio for {sub_id}: {size_mb:.1f} MB "
                  f"in {elapsed:.1f}s ({rate:.1f} MB/s)", flush=True)

    def audio_prefetch_failed(self, sub_id: str, exc: BaseException):
        with self._lock:
            print(f"    [Prefetch] audio for {sub_id} failed: "
                  f"{type(exc).__name__}: {exc}", flush=True)

    # ── Email / resummarize / generic ────────────────────────────────────

    def email_summary(self, n: int):
        with self._lock:
            print(f"\n[Email] Sending summary for {n} lecture(s)...",
                  flush=True)

    def email_failed(self):
        with self._lock:
            print("[Email] Send failed, lectures will be retried next run.",
                  flush=True)

    def email_recovered_unsent(self, n: int):
        with self._lock:
            print(f"[Email] Including {n} previously unsent lecture(s).",
                  flush=True)

    def resummarize_header(self, n: int):
        with self._lock:
            print(f"\n[Resummarize] {n} lecture(s) eligible for v2 upgrade.",
                  flush=True)

    def resummarize_one(self, course_title: str, sub_title: str):
        with self._lock:
            print(f"  -- Resummarize: [{course_title}] {sub_title}",
                  flush=True)

    def info(self, msg: str):
        """Generic info line — escape hatch for one-off messages."""
        with self._lock:
            print(msg, flush=True)

    # ── Semester course crawl ────────────────────────────────────────────

    def crawl_courses_start(self, term: str):
        with self._lock:
            print(f"\n[Crawl] Fetching semester {term} course catalog...",
                  flush=True)

    def crawl_courses_done(self, term: str, fetched: int,
                           deleted: int, upserted: int, elapsed: float):
        with self._lock:
            print(
                f"[Crawl] Term {term}: {fetched} courses fetched, "
                f"{upserted} upserted, {deleted} removed in {elapsed:.1f}s",
                flush=True,
            )

    def crawl_courses_failed(self, term: str, exc: BaseException):
        with self._lock:
            print(
                f"[Crawl] Term {term} failed: "
                f"{type(exc).__name__}: {exc}", flush=True,
            )
