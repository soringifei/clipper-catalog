"""Jersey-number OCR on a player crop + temporal voting.

``read_jersey(crop_bgr)`` returns candidate readings ``[(text, conf), ...]``
sorted by confidence (may be empty). Backend: PaddleOCR when importable
(``REBELS_OCR_BACKEND=digits`` forces the built-in one), otherwise the
connected-components + template digit classifier in :mod:`.digits`.

``vote(readings)`` aggregates per-frame readings of one track. A single frame
never proves identity: the target needs ``min_votes`` consistent reads.
"""
from __future__ import annotations

import os
from collections import Counter
from typing import Iterable, Optional, Sequence

import cv2
import numpy as np

from . import digits as _digits

_paddle = None
_paddle_failed = False


def backend_name() -> str:
    return "paddleocr" if _get_paddle() is not None else "digits"


def _get_paddle():
    global _paddle, _paddle_failed
    if _paddle is not None or _paddle_failed:
        return _paddle
    if os.environ.get("REBELS_OCR_BACKEND", "") == "digits":
        _paddle_failed = True
        return None
    try:
        from paddleocr import PaddleOCR  # type: ignore
        _paddle = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
    except Exception:  # noqa: BLE001 - not installed / broken install
        _paddle_failed = True
    return _paddle


# --------------------------------------------------------------------------
def torso_crop(crop_bgr: np.ndarray, top: float = 0.18, bottom: float = 0.62) -> np.ndarray:
    """Upper-body part of a person box where the jersey number lives."""
    h, w = crop_bgr.shape[:2]
    if w > 1.2 * h:  # lying player: keep everything, orientation handled later
        return crop_bgr
    y1, y2 = int(h * top), max(int(h * bottom), int(h * top) + 2)
    x1, x2 = int(w * 0.08), max(int(w * 0.92), 2)
    return crop_bgr[y1:y2, x1:x2]


def _prep(gray: np.ndarray) -> np.ndarray:
    g = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))
    return clahe.apply(g)


def _binarizations(g: np.ndarray) -> list[np.ndarray]:
    h, w = g.shape
    bs = max(11, (min(h, w) // 3) | 1)
    outs = []
    for inv in (False, True):
        mode = cv2.THRESH_BINARY_INV if inv else cv2.THRESH_BINARY
        a = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, mode, bs, -4 if not inv else 4)
        _, o = cv2.threshold(g, 0, 255, mode | cv2.THRESH_OTSU)
        for b in (a, o):
            b = cv2.morphologyEx(b, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            outs.append(b)
    return outs


def _digit_blobs(binary: np.ndarray) -> list[tuple[tuple[int, int, int, int], np.ndarray]]:
    H, W = binary.shape
    n, lab, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if h < 0.22 * H or h > 0.95 * H or h < 12:
            continue
        ar = w / float(h)
        if ar < 0.12 or ar > 1.05:
            continue
        fill = area / float(w * h)
        if fill < 0.15 or fill > 0.9:
            continue
        # blobs that span the crop edge are usually background / body outline
        if (x <= 1 and x + w >= W - 1) or (y <= 1 and y + h >= H - 1):
            continue
        mask = (lab[y:y + h, x:x + w] == i).astype(np.uint8) * 255
        out.append(((int(x), int(y), int(w), int(h)), mask))
    return out


def _read_binary(binary: np.ndarray) -> list[tuple[str, float]]:
    blobs = _digit_blobs(binary)
    if not blobs:
        return []
    cls = []
    for (x, y, w, h), m in blobs:
        d, c = _digits.classify_blob(m)
        cls.append((x, y, w, h, d, c))
    res: list[tuple[str, float]] = []
    cls.sort(key=lambda r: r[0])
    H = binary.shape[0]
    for i, a in enumerate(cls):
        if a[4] is None:
            continue
        # single digit (numbers 0-9) - geometry: fairly large and alone
        res.append((str(a[4]), a[5] * 0.6))
        for b in cls[i + 1:]:
            if b[4] is None:
                continue
            ha, hb = a[3], b[3]
            hr = min(ha, hb) / max(ha, hb)
            gap = b[0] - (a[0] + a[2])
            dy = abs((a[1] + ha / 2) - (b[1] + hb / 2))
            if hr < 0.72 or dy > 0.25 * max(ha, hb) or gap < -0.1 * ha or gap > 0.7 * max(ha, hb):
                continue
            geo = hr * (1 - dy / max(ha, hb)) * min(1.0, max(ha, hb) / (0.3 * H))
            res.append((f"{a[4]}{b[4]}", float(np.sqrt(a[5] * b[5]) * geo)))
    return res


def _read_digits(crop_bgr: np.ndarray) -> list[tuple[str, float]]:
    if crop_bgr is None or crop_bgr.size == 0 or min(crop_bgr.shape[:2]) < 6:
        return []
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if crop_bgr.ndim == 3 else crop_bgr
    variants = [gray]
    h, w = gray.shape
    if w > 1.2 * h:  # orientation hint: the player is horizontal
        variants += [cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE),
                     cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE)]
    best: dict[str, float] = {}
    for v in variants:
        g = _prep(v)
        for b in _binarizations(g):
            for t, c in _read_binary(b):
                best[t] = max(best.get(t, 0.0), c)
    # multi-digit readings dominate their constituent single digits
    for t in [t for t in best if len(t) == 2]:
        for s in t:
            if s in best and best[s] <= best[t]:
                best[s] *= 0.5
    return sorted(best.items(), key=lambda kv: -kv[1])


def _read_paddle(crop_bgr: np.ndarray) -> list[tuple[str, float]]:
    ocr = _get_paddle()
    try:
        g = cv2.resize(crop_bgr, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        res = ocr.ocr(g, cls=True) or []
    except Exception:  # noqa: BLE001
        return []
    out: dict[str, float] = {}
    for page in res:
        for item in page or []:
            try:
                txt, conf = item[1]
            except Exception:  # noqa: BLE001
                continue
            txt = "".join(ch for ch in str(txt) if ch.isdigit())
            if 1 <= len(txt) <= 2:
                out[txt] = max(out.get(txt, 0.0), float(conf))
    return sorted(out.items(), key=lambda kv: -kv[1])


def read_jersey(crop_bgr: np.ndarray, is_torso: bool = False) -> list[tuple[str, float]]:
    """Read the jersey number from a person crop (or a torso crop if ``is_torso``).

    Returns ``[(text, conf), ...]`` best first; empty when nothing digit-like.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return []
    torso = crop_bgr if is_torso else torso_crop(crop_bgr)
    if _get_paddle() is not None:
        r = _read_paddle(torso)
        if r:
            return r
    return _read_digits(torso)


# --------------------------------------------------------------------------
def _top(reading) -> tuple[Optional[str], float]:
    if reading is None:
        return None, 0.0
    if isinstance(reading, tuple) and len(reading) == 2 and isinstance(reading[0], (str, type(None))):
        return reading[0], float(reading[1] or 0.0)
    if isinstance(reading, (list, tuple)):
        if not reading:
            return None, 0.0
        return _top(reading[0])
    return None, 0.0


def vote(readings: Iterable, target: str = "52", min_votes: int = 3,
         min_conf: float = 0.25) -> dict:
    """Temporal voting over one track's per-frame readings.

    ``readings``: per frame either ``[(text, conf), ...]`` (output of
    :func:`read_jersey`), a single ``(text, conf)`` or ``None``.
    Returns ``{"votes": {"52": n, "32": m, "unknown": k}, "best": str|None,
    "jersey_confidence": P(target), "n_frames": int}``.
    """
    votes: Counter = Counter()
    conf_sum: Counter = Counter()
    n = 0
    for r in readings:
        n += 1
        t, c = _top(r)
        if t is None or c < min_conf:
            votes["unknown"] += 1
            continue
        votes[t] += 1
        conf_sum[t] += c
    readable = {k: v for k, v in votes.items() if k != "unknown"}
    best = max(readable, key=lambda k: (readable[k], conf_sum[k])) if readable else None
    n_t = votes.get(target, 0)
    n_read = sum(readable.values())
    # partial reads that are consistent with the target (one digit dropped)
    partial = sum(v for k, v in readable.items() if len(k) == 1 and k in target)
    conflicting = n_read - n_t - partial
    if n_t == 0:
        conf = 0.0
    else:
        consistency = n_t / float(n_t + conflicting + 1e-9)
        mean_c = conf_sum[target] / n_t
        if n_t >= min_votes:
            strength = 0.6 + 0.4 * min(1.0, (n_t - min_votes + 1) / float(min_votes + 1))
            conf = consistency * strength * (0.75 + 0.25 * min(1.0, mean_c / 0.6))
        else:  # not enough independent frames: never enough on its own
            conf = 0.45 * consistency * n_t / float(min_votes)
    return {"votes": dict(votes), "best": best, "jersey_confidence": float(np.clip(conf, 0, 1)),
            "n_frames": n, "n_target": n_t, "n_conflicting": int(conflicting)}


def jersey_readable(v: dict) -> bool:
    return bool(v.get("votes")) and sum(x for k, x in v["votes"].items() if k != "unknown") > 0
