"""PPT page filters: perceptual-hash dedup + multi-pattern invalid-page match.

Two stages, both run from main.py's _fetch_and_ocr_ppts:

1. ``dedup_dhash`` operates on dhashes computed from the raw image bytes
   (cheap, no OCR needed) and drops near-duplicate frames using a sliding
   window. The iCourse capture pipeline takes timed screenshots, so a
   classroom that stays on one slide for several minutes produces dozens
   of identical frames; collapsing them before OCR saves both time and
   prompt budget.

2. ``is_invalid_page`` runs after OCR and matches the recovered text
   against a list of feature substrings extracted from the two known
   classroom-noise screens (the desktop wallpaper and the e-learning
   resource portal). Patterns are deliberately long and topic-specific
   so they don't false-positive on real slides; punctuation and
   whitespace are stripped before matching to tolerate OCR noise.
"""

from __future__ import annotations

import io
import re
from typing import Iterable


def compute_dhash(image_bytes: bytes) -> str | None:
    """Perceptual hash for an image. Returns 16-hex string or None on error.

    Uses imagehash.dhash (8x8 difference hash). Identical/near-identical
    crops yield identical hashes; visually distinct frames almost always
    differ by more than 4 bits.  Caller must tolerate ``None`` (image
    decode failure, missing PIL, etc.) — those pages are excluded from
    the dedup pass and pass through to OCR untouched.
    """
    try:
        import imagehash
        from PIL import Image
    except ImportError:
        return None
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            return str(imagehash.dhash(img))
    except Exception:
        return None


def _hamming_hex(a: str, b: str) -> int:
    """Bit-count XOR of two equal-length hex strings."""
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def dedup_dhash(
    items: list[str | None],
    window: int = 5,
    threshold: int = 4,
) -> list[int]:
    """Sliding-window perceptual dedup. Returns sorted list of dropped indices.

    For each surviving anchor i, compare its dhash against the next
    ``window`` items; if Hamming distance ≤ threshold, mark the *later*
    index as dropped. Already-dropped images never become anchors —
    that prevents a chain of "near to last-kept" pages from cascading
    drops onto pages that aren't actually near the kept anchor.

    ``items`` may contain ``None`` (compute_dhash failure) — those
    indices are passed through (never dropped, never used as anchor).
    """
    n = len(items)
    dropped: set[int] = set()
    for i in range(n):
        if i in dropped:
            continue
        a = items[i]
        if a is None:
            continue
        for j in range(i + 1, min(i + 1 + window, n)):
            if j in dropped:
                continue
            b = items[j]
            if b is None:
                continue
            if _hamming_hex(a, b) <= threshold:
                dropped.add(j)
    return sorted(dropped)


# ── Invalid-page pattern matching ──────────────────────────────────────────
#
# Patterns are matched after _normalize_for_match strips whitespace and all
# non-alphanumeric/non-CJK characters and lowercases ASCII.  Pick patterns
# that are simultaneously:
#   - Specific enough that they don't appear in real lecture material
#     (avoid bare campus names, common headings).
#   - Long enough (≥6 normalized chars where possible) that incidental
#     OCR mis-recognition doesn't accidentally hit them.
#   - Drawn from features unique to the noise screens (URLs, the long
#     official policy titles, the EV recording pipeline references, the
#     classroom-equipment shutdown reminder).
INVALID_PAGE_PATTERNS: list[str] = [
    # ── Type 1: classroom desktop wallpaper ──
    "请不要关闭设备",
    "避免耽误第34节上课",
    "触控显示器无线话筒hdmi",
    "多媒体值班室",
    "本教室装有摄录及安全装置",
    # ── Type 2: e-learning resource portal screen ──
    "cfdfudaneducn",                       # the cfd.fudan.edu.cn URL
    "icoursefudaneducn",                   # the icourse.fudan.edu.cn URL
    "智慧教学资源平台使用规范",
    "教育部等九部门",
    "加快推进教育数字化",
    "本科课程评教提醒",
    "请于期末考试前完成评教",
    "微信搜索并关注复旦课评",
    "国务院关于深入实施",
    "板书效果展示",
    "双屏效果展示",
    "课程录制exe",
    "ev去噪",
    "录制完成桌面会生成",
    "推荐上传至elearning",
    "ppt演示者视图会影响录屏",
]

_NORMALIZE_RE = re.compile(r"[\W_]+", re.UNICODE)


def _normalize_for_match(text: str) -> str:
    """Lowercase + strip whitespace and punctuation. CJK chars are kept."""
    if not text:
        return ""
    return _NORMALIZE_RE.sub("", text).lower()


def is_invalid_page(text: str) -> bool:
    """True if any feature string matches the (normalized) OCR'd text."""
    norm = _normalize_for_match(text)
    if not norm:
        return False
    return any(p in norm for p in INVALID_PAGE_PATTERNS)


def normalize_for_match(text: str) -> str:  # noqa: D401  exported wrapper
    """Public alias for tests / debugging."""
    return _normalize_for_match(text)
