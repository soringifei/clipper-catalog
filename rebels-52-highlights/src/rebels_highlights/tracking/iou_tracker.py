"""Fallback multi-object tracker: IoU + centre distance + appearance, with
constant-velocity prediction and simple global motion compensation (phase
correlation of consecutive downscaled frames, i.e. camera pan).

Used when ultralytics trackers are unavailable or ``tracking.tracker: iou``.
Pure numpy/OpenCV; greedy association (no scipy on the box).
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / ua) if ua > 0 else 0.0


def global_motion(prev_gray: Optional[np.ndarray], gray: np.ndarray,
                  scale: float = 0.25) -> tuple[float, float]:
    """Camera translation (dx, dy) in full-res px between two frames."""
    if prev_gray is None or prev_gray.shape != gray.shape:
        return 0.0, 0.0
    a = cv2.resize(prev_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA).astype(np.float32)
    b = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA).astype(np.float32)
    win = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
    try:
        (dx, dy), resp = cv2.phaseCorrelate(a, b, win)
    except cv2.error:
        return 0.0, 0.0
    if resp < 0.05:
        return 0.0, 0.0
    return float(dx / scale), float(dy / scale)


class _T:
    __slots__ = ("id", "box", "vel", "emb", "lost", "hits")

    def __init__(self, tid: int, box: np.ndarray, emb: Optional[np.ndarray]):
        self.id = tid
        self.box = box.astype(np.float64)
        self.vel = np.zeros(4)
        self.emb = emb
        self.lost = 0
        self.hits = 1


class IouTracker:
    def __init__(self, track_buffer: int = 30, min_score: float = 0.25, iou_weight: float = 0.5,
                 dist_weight: float = 0.25, app_weight: float = 0.35, match_min: float = 0.3,
                 gmc: bool = True):
        self.track_buffer = track_buffer
        self.min_score = min_score
        self.iou_w, self.dist_w, self.app_w = iou_weight, dist_weight, app_weight
        self.match_min = match_min
        self.gmc = gmc
        self._next = 1
        self.tracks: list[_T] = []
        self._prev_gray: Optional[np.ndarray] = None
        self.last_shift = (0.0, 0.0)

    def reset(self) -> None:
        """Drop all tracks (hard cut); ids keep increasing so they never repeat."""
        self.tracks = []
        self._prev_gray = None

    def update(self, frame_bgr: Optional[np.ndarray], boxes: list, scores: list,
               embs: Optional[list] = None) -> list[Optional[int]]:
        """Return a track id (or None) per input detection."""
        embs = embs if embs is not None else [None] * len(boxes)
        dx = dy = 0.0
        if self.gmc and frame_bgr is not None:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if frame_bgr.ndim == 3 else frame_bgr
            dx, dy = global_motion(self._prev_gray, gray)
            self._prev_gray = gray
        self.last_shift = (dx, dy)
        preds = []
        for t in self.tracks:
            p = t.box + t.vel + np.array([dx, dy, dx, dy])
            preds.append(p)
        B = [np.asarray(b, np.float64) for b in boxes]
        pairs = []
        for i, (t, p) in enumerate(zip(self.tracks, preds)):
            ph = max(1.0, p[3] - p[1])
            pc = np.array([(p[0] + p[2]) / 2, (p[1] + p[3]) / 2])
            for j, b in enumerate(B):
                iou = _iou(p, b)
                bc = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
                bh = max(1.0, b[3] - b[1])
                dist = np.linalg.norm(pc - bc) / ph * (1.0 + 0.3 * t.lost)  # widen gate while lost
                size = min(ph, bh) / max(ph, bh)
                if iou <= 0.01 and dist > 1.0 + 0.15 * t.lost:
                    continue
                if size < 0.5:
                    continue
                app = 0.5
                if t.emb is not None and embs[j] is not None:
                    app = float(np.clip(np.dot(t.emb, embs[j]), 0, 1))
                s = self.iou_w * iou + self.dist_w * max(0.0, 1 - dist / (1.0 + 0.15 * t.lost)) \
                    + self.app_w * app
                s *= 0.7 + 0.3 * size
                if s >= self.match_min:
                    pairs.append((s, i, j))
        pairs.sort(key=lambda x: -x[0])
        used_t, used_d = set(), set()
        out: list[Optional[int]] = [None] * len(B)
        for s, i, j in pairs:
            if i in used_t or j in used_d:
                continue
            used_t.add(i)
            used_d.add(j)
            t = self.tracks[i]
            nb = B[j]
            old = t.box + np.array([dx, dy, dx, dy])
            if t.lost == 0:
                t.vel = 0.6 * t.vel + 0.4 * (nb - old)
            else:
                t.vel = (nb - old) / (t.lost + 1)
            t.box = nb
            if embs[j] is not None:
                t.emb = embs[j] if t.emb is None else _norm(0.85 * t.emb + 0.15 * embs[j])
            t.lost = 0
            t.hits += 1
            out[j] = t.id
        for i, t in enumerate(self.tracks):
            if i not in used_t:
                t.lost += 1
                t.box = preds[i]
                t.vel *= 0.8
        self.tracks = [t for t in self.tracks if t.lost <= self.track_buffer]
        for j, b in enumerate(B):
            if j in used_d or scores[j] < self.min_score:
                continue
            t = _T(self._next, b, embs[j])
            self._next += 1
            self.tracks.append(t)
            out[j] = t.id
        return out


def _norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v
