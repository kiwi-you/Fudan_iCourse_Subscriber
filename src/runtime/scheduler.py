"""Concurrency primitives: thread pools, prefetch caches, resource monitor.

Owns three resource pools and coordinates work across them so the
LectureRunner can focus on per-lecture business logic.

  image_pool       20 workers   IO bound  (per-image HTTP)
  ocr_pool          8 workers   CPU bound (RapidOCR runs)
                                       └── gated by DynamicSemaphore whose
                                           target is steered by ResourceMonitor
  audio_downloader  2 slots     IO bound  (ffmpeg URL → audio.raw to disk)

The audio downloader is special: each "slot" hosts a running ffmpeg process
that writes f32le mono 16 kHz audio to a per-sub_id scratch file.  Transcriber
reads that file with tail-f semantics while ffmpeg is still writing — so the
network download isn't bottlenecked by ASR speed and the ASR isn't blocked
on download completion.  See ``AudioDownloader`` below.

ResourceMonitor polls ``psutil.cpu_percent()`` every second and nudges the
OCR semaphore target up/down to keep CPU "as close to 100 % as possible
without going over".  The pool size is the hard ceiling; the semaphore
target is the soft live concurrency.
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Optional

import psutil

from src.runtime import config
from src.api import icourse# ── DynamicSemaphore ────────────────────────────────────────────────────────

class DynamicSemaphore:
    """A semaphore whose target permit count can be retuned at runtime.

    Python's stdlib has no resize for ``ThreadPoolExecutor``, so we keep a
    fixed-size pool of MAX worker threads and gate "is now actually allowed
    to run" with this primitive.  Reducing the target makes new acquirers
    wait; in-flight workers complete naturally.  Increasing the target
    wakes waiters in order.

    Thread safety: every state change goes through the internal Condition.
    """

    def __init__(self, target: int):
        self._target = int(target)
        self._busy = 0
        self._cond = threading.Condition()

    @property
    def target(self) -> int:
        with self._cond:
            return self._target

    @property
    def busy(self) -> int:
        with self._cond:
            return self._busy

    def set_target(self, target: int) -> None:
        with self._cond:
            self._target = max(1, int(target))
            self._cond.notify_all()

    def acquire(self) -> None:
        with self._cond:
            while self._busy >= self._target:
                self._cond.wait()
            self._busy += 1

    def release(self) -> None:
        with self._cond:
            self._busy -= 1
            self._cond.notify()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *_a):
        self.release()


# ── ResourceMonitor ─────────────────────────────────────────────────────────

class ResourceMonitor:
    """Adjusts OCR concurrency from CPU pressure on a daemon thread.

    Started by ``Scheduler.start()``; stopped by ``Scheduler.shutdown()``.
    Independent of the OCR work itself — even when no OCR is happening, the
    monitor still ticks so its snapshots show useful CPU readings.

    Loop:
      every 1 s
        cpu = psutil.cpu_percent()
        if cpu < LOW  and target < MAX_TARGET: target += 1
        if cpu > HIGH and target > MIN_TARGET: target -= 1
        reporter.cpu_snapshot(...)   # throttled to 60 s internally
    """

    POLL_INTERVAL = 1.0

    def __init__(self, ocr_sem: DynamicSemaphore,
                 image_pool: ThreadPoolExecutor,
                 audio_downloader: "AudioDownloader",
                 reporter, *,
                 max_target: int = None, min_target: int = None,
                 cpu_high: int = None, cpu_low: int = None):
        self._sem = ocr_sem
        self._image_pool = image_pool
        self._audio_downloader = audio_downloader
        self._reporter = reporter
        self._max_target = max_target or config.OCR_MAX_TARGET
        self._min_target = min_target or config.OCR_MIN_TARGET
        self._cpu_high = cpu_high or config.RESOURCE_MONITOR_CPU_HIGH
        self._cpu_low = cpu_low or config.RESOURCE_MONITOR_CPU_LOW
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        # Prime psutil — the first call always returns 0.0.
        psutil.cpu_percent(interval=None)

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._loop, name="resource-monitor", daemon=True,
        )
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def _loop(self):
        while not self._stop.is_set():
            self._stop.wait(self.POLL_INTERVAL)
            if self._stop.is_set():
                return
            cpu = psutil.cpu_percent(interval=None)
            self._maybe_retune(cpu)
            self._reporter.cpu_snapshot(
                cpu_pct=cpu,
                ocr_busy=self._sem.busy,
                ocr_target=self._sem.target,
                ocr_max=self._max_target,
                image_busy=getattr(self._image_pool, "_work_queue", None) is not None
                            and self._image_pool._work_queue.qsize() or 0,
                image_max=self._image_pool._max_workers,
                audio_busy=self._audio_downloader.active_count,
                audio_max=self._audio_downloader.max_concurrent,
            )

    def _maybe_retune(self, cpu: float):
        old = self._sem.target
        if cpu < self._cpu_low and old < self._max_target:
            self._sem.set_target(old + 1)
            self._reporter.cpu_target_changed(old, old + 1, cpu)
        elif cpu > self._cpu_high and old > self._min_target:
            self._sem.set_target(old - 1)
            self._reporter.cpu_target_changed(old, old - 1, cpu)


# ── PrefetchCache (per-sub_id image bytes) ─────────────────────────────────

class PrefetchCache:
    """Per-lecture image-bytes pre-fetcher driven by the global image pool.

    schedule(client, course_id, sub_id) — fire all image downloads, return
                                          immediately.  Idempotent.
    wait(sub_id) -> (items, images)   — block until every download for sub_id
                                          resolves.
    discard(sub_id)                    — drop the cached entry (release bytes).
    in_flight(sub_id)                  — number of unfinished futures.

    The reporter (passed at __init__) is called for per-image ticks so
    progress logging is throttled in one place.  ``reporter`` may be ``None``
    in tests that don't care about output.
    """

    def __init__(self, image_pool: ThreadPoolExecutor, reporter=None):
        self._image_pool = image_pool
        self._reporter = reporter
        self._lock = threading.Lock()
        # sub_id -> {"items": list[dict]|None, "futures": dict[int, Future]}
        self._cache: dict[str, dict] = {}

    def schedule(self, client, course_id: str, sub_id: str):
        sub_id = str(sub_id)
        with self._lock:
            if sub_id in self._cache:
                return
            self._cache[sub_id] = {"items": None, "futures": {}}

        try:
            ppt_items = client.get_ppt_list(course_id, sub_id)
        except Exception as e:
            if self._reporter:
                self._reporter.ppt_list_failed(type(e).__name__, str(e))
            ppt_items = []
        for idx, item in enumerate(ppt_items, start=1):
            item["page_num"] = idx

        if self._reporter and ppt_items:
            self._reporter.image_progress_start(sub_id, len(ppt_items))

        futures: dict[int, Future] = {}
        for item in ppt_items:
            futures[item["page_num"]] = self._image_pool.submit(
                self._download_one, client, item, sub_id,
            )

        with self._lock:
            self._cache[sub_id]["items"] = ppt_items
            self._cache[sub_id]["futures"] = futures

    def _download_one(self, client, item: dict, sub_id: str) -> bytes | None:
        """Image-pool worker body. Goes through the module-level
        ``icourse.fetch_ppt_image`` so tests can monkey-patch it."""
        try:
            return icourse.fetch_ppt_image(client, item)
        finally:
            if self._reporter:
                self._reporter.image_progress_tick(sub_id)

    def wait(self, sub_id: str) -> tuple[list[dict], dict[int, bytes]]:
        sub_id = str(sub_id)
        with self._lock:
            entry = self._cache.get(sub_id)
        if entry is None:
            return [], {}
        items = entry.get("items") or []
        images: dict[int, bytes] = {}
        for page_num, fut in entry.get("futures", {}).items():
            try:
                img = fut.result()
            except Exception as e:
                print(
                    f"    [Prefetch {sub_id}] page {page_num} download "
                    f"failed: {type(e).__name__}: {e}"
                )
                img = None
            if img is not None:
                images[page_num] = img
        return items, images

    def discard(self, sub_id: str) -> None:
        sub_id = str(sub_id)
        with self._lock:
            self._cache.pop(sub_id, None)
        if self._reporter:
            self._reporter.image_progress_abort(sub_id)


# ── AudioDownloader (per-sub_id ffmpeg → disk audio file) ──────────────────

@dataclass
class AudioHandle:
    """Reference to an audio-extraction job."""

    sub_id: str
    path: str          # disk file ffmpeg writes f32le mono 16 kHz to
    process: subprocess.Popen
    stderr_chunks: list[bytes]


class AudioDownloader:
    """Spawn-and-track concurrent ``ffmpeg`` audio extractions.

    For each sub_id we spawn one ``ffmpeg -i <signed URL> -vn -ar 16000 -ac 1
    -f f32le <path>`` process.  ``ffmpeg`` writes the decoded mono float32
    audio straight to disk at network speed — no Python pipe in the loop, so
    download is NOT bottlenecked by ASR consumption.  Transcriber reads that
    file with tail-f semantics, processing chunks as they arrive.

    Concurrency is bounded by ``max_concurrent`` (default 2: current lecture
    being transcribed + one pre-decoded for the next lecture).  ``schedule()``
    returns immediately; if all slots are taken the background spawn waits.
    """

    SLOT_WAIT_TIMEOUT = 0  # 0 = wait forever

    def __init__(self, audio_dir: str, max_concurrent: int = None,
                 reporter=None):
        self._dir = audio_dir
        self.max_concurrent = max_concurrent or config.VIDEO_DOWNLOAD_CONCURRENCY
        self._sem = threading.BoundedSemaphore(self.max_concurrent)
        self._active: dict[str, AudioHandle | None] = {}  # None = pending spawn
        self._lock = threading.Lock()
        self._reporter = reporter
        os.makedirs(self._dir, exist_ok=True)

    @property
    def active_count(self) -> int:
        with self._lock:
            return sum(1 for h in self._active.values() if h is not None)

    def schedule(self, client, course_id: str, sub_id: str) -> None:
        """Reserve a slot for sub_id and spawn ffmpeg in the background.

        Returns immediately. If all slots are taken the spawn blocks in its
        background thread until a slot frees.  Idempotent — second call for
        the same sub_id is a no-op.
        """
        sub_id = str(sub_id)
        with self._lock:
            if sub_id in self._active:
                return
            self._active[sub_id] = None  # PENDING

        threading.Thread(
            target=self._spawn_when_ready,
            args=(client, course_id, sub_id),
            name=f"audio-spawn-{sub_id}",
            daemon=True,
        ).start()

    def _spawn_when_ready(self, client, course_id: str, sub_id: str):
        try:
            self._sem.acquire()
            try:
                url = client.get_video_url(course_id, sub_id)
                if not url:
                    with self._lock:
                        self._active.pop(sub_id, None)
                    self._sem.release()
                    return
                vpn_url, headers = client.get_stream_params(url)
                path = os.path.join(self._dir, f"{sub_id}.raw")
                if os.path.exists(path):
                    os.remove(path)

                cmd = [
                    "ffmpeg", "-y",
                    "-headers", headers,
                    "-reconnect", "1",
                    "-reconnect_streamed", "1",
                    "-reconnect_delay_max", "5",
                    "-i", vpn_url,
                    "-vn",
                    "-ar", "16000",
                    "-ac", "1",
                    "-f", "f32le",
                    path,
                ]
                proc = subprocess.Popen(
                    cmd, stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )

                # Drain stderr so the pipe never deadlocks.  Keep last few KB
                # for diagnostics if ffmpeg dies.
                stderr_chunks: list[bytes] = []

                def _drain():
                    try:
                        for chunk in proc.stderr:
                            stderr_chunks.append(chunk)
                            if len(stderr_chunks) > 2048:
                                # keep only the tail to bound memory
                                del stderr_chunks[: -1024]
                    except Exception:
                        pass

                threading.Thread(
                    target=_drain, name=f"audio-stderr-{sub_id}",
                    daemon=True,
                ).start()

                handle = AudioHandle(
                    sub_id=sub_id, path=path,
                    process=proc, stderr_chunks=stderr_chunks,
                )

                with self._lock:
                    self._active[sub_id] = handle

                if self._reporter:
                    self._reporter.audio_prefetch_start(sub_id)

                # Background monitor: release the semaphore slot when
                # ffmpeg exits.  We do NOT pop from _active here — that's
                # the caller's job (via release()).
                threading.Thread(
                    target=self._monitor, args=(handle,),
                    name=f"audio-monitor-{sub_id}", daemon=True,
                ).start()
            except Exception:
                with self._lock:
                    self._active.pop(sub_id, None)
                self._sem.release()
                raise
        except Exception as e:
            if self._reporter:
                self._reporter.audio_prefetch_failed(sub_id, e)

    def _monitor(self, handle: AudioHandle):
        handle.process.wait()
        self._sem.release()

    def get(self, sub_id: str, timeout: float = 120.0) -> AudioHandle | None:
        """Block until ffmpeg has been spawned for sub_id; return its handle.

        Returns None if sub_id was never scheduled (or already released).
        Raises TimeoutError if the spawn never happens within ``timeout``.
        """
        sub_id = str(sub_id)
        deadline = time.time() + timeout
        while True:
            with self._lock:
                entry = self._active.get(sub_id, "MISSING")
            if entry == "MISSING":
                return None
            if entry is not None:
                return entry
            if time.time() > deadline:
                raise TimeoutError(
                    f"audio download for {sub_id} did not start within "
                    f"{timeout}s — likely WebVPN session expired or "
                    f"get_video_url failed"
                )
            time.sleep(0.05)

    def release(self, sub_id: str) -> None:
        """Kill ffmpeg (if still alive) and delete the audio file.

        Called from LectureRunner Phase H once the lecture has been
        transcribed and summarized.
        """
        sub_id = str(sub_id)
        with self._lock:
            handle = self._active.pop(sub_id, None)
        if handle is None:
            return
        proc = handle.process
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        # The monitor thread releases the semaphore on its own.
        if os.path.exists(handle.path):
            try:
                os.remove(handle.path)
            except OSError:
                pass

    def shutdown(self) -> None:
        """Kill every in-flight ffmpeg and wipe the scratch directory."""
        with self._lock:
            sub_ids = list(self._active.keys())
        for sub_id in sub_ids:
            self.release(sub_id)


# ── Scheduler — single façade ──────────────────────────────────────────────

@dataclass
class ResourceSnapshot:
    cpu_pct: float
    ocr_busy: int
    ocr_target: int
    image_busy: int
    audio_busy: int


class Scheduler:
    """Single façade owning every concurrency primitive.

    LectureRunner gets one Scheduler.  Through it, every other layer talks
    to pools, prefetch caches, and the resource monitor by name — no module
    holds its own ThreadPoolExecutor.

    Lifecycle:
        scheduler = Scheduler(reporter=...)
        scheduler.start()           # spawns ResourceMonitor thread
        ... LectureRunner uses it ...
        scheduler.shutdown()        # drains pools, kills ffmpegs, stops monitor
    """

    def __init__(self, reporter):
        self._reporter = reporter
        self.image_pool = ThreadPoolExecutor(
            max_workers=config.IMAGE_WORKERS, thread_name_prefix="img",
        )
        self.ocr_pool = ThreadPoolExecutor(
            max_workers=config.OCR_MAX_WORKERS, thread_name_prefix="ocr",
        )
        self.ocr_semaphore = DynamicSemaphore(config.OCR_INITIAL_TARGET)
        self.image_cache = PrefetchCache(self.image_pool, reporter=reporter)
        self.audio_downloader = AudioDownloader(
            audio_dir=config.AUDIO_DIR,
            max_concurrent=config.VIDEO_DOWNLOAD_CONCURRENCY,
            reporter=reporter,
        )
        self._monitor = ResourceMonitor(
            self.ocr_semaphore, self.image_pool, self.audio_downloader,
            reporter,
        )

    def start(self):
        self._monitor.start()

    def prefetch_lecture(self, client, course_id: str, sub_id: str) -> None:
        """Schedule image + audio prefetch for a future lecture."""
        self.image_cache.schedule(client, course_id, sub_id)
        self.audio_downloader.schedule(client, course_id, sub_id)

    def submit_ocr(self, fn: Callable, *args, **kwargs) -> Future:
        """Submit an OCR job. The worker acquires the dynamic semaphore
        before doing real work, so live concurrency stays at the current
        target even though pool size is up to OCR_MAX_WORKERS."""
        def _wrapped():
            with self.ocr_semaphore:
                return fn(*args, **kwargs)
        return self.ocr_pool.submit(_wrapped)

    def snapshot(self) -> ResourceSnapshot:
        return ResourceSnapshot(
            cpu_pct=psutil.cpu_percent(interval=None),
            ocr_busy=self.ocr_semaphore.busy,
            ocr_target=self.ocr_semaphore.target,
            image_busy=self.image_pool._work_queue.qsize()
                       if hasattr(self.image_pool, "_work_queue") else 0,
            audio_busy=self.audio_downloader.active_count,
        )

    def shutdown(self) -> None:
        self._monitor.stop()
        self.audio_downloader.shutdown()
        self.image_pool.shutdown(wait=True)
        self.ocr_pool.shutdown(wait=True)
