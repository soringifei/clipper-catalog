"""Runtime-built digit classifier (numpy + OpenCV only).

Templates are rendered at import-time-on-first-use from several bold fonts
(OpenCV Hershey fonts + TrueType bold fonts via PIL when present), augmented
with small rotations / shear / perspective / stroke-width changes / blur, and
described with HOG + a coarse pixel grid. Classification is cosine kNN.

Input to :func:`classify_blob` is a binary mask (uint8, foreground > 0) of a
single connected digit blob.
"""
from __future__ import annotations

import glob
import os
import threading
from typing import Optional

import cv2
import numpy as np

NORM_W, NORM_H = 32, 48
_INNER_W, _INNER_H = 26, 42

_FONT_GLOBS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSerifBold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
    "/usr/share/fonts/**/*Bold*.ttf",
    "/Library/Fonts/*Bold*.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
]
_CV_FONTS = [
    (cv2.FONT_HERSHEY_SIMPLEX, (7, 10, 13)),
    (cv2.FONT_HERSHEY_DUPLEX, (8, 12)),
    (cv2.FONT_HERSHEY_TRIPLEX, (8, 12)),
    (cv2.FONT_HERSHEY_COMPLEX, (9,)),
]

_lock = threading.Lock()
_model: Optional[tuple[np.ndarray, np.ndarray]] = None


# --------------------------------------------------------------------------
# normalisation + features
def normalize_mask(mask: np.ndarray) -> Optional[np.ndarray]:
    """Crop to foreground bbox, keep aspect, centre in a NORM_W x NORM_H canvas."""
    m = (mask > 0).astype(np.uint8) * 255
    ys, xs = np.nonzero(m)
    if len(xs) < 8:
        return None
    m = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = m.shape
    s = min(_INNER_W / w, _INNER_H / h)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    m = cv2.resize(m, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((NORM_H, NORM_W), np.uint8)
    x0, y0 = (NORM_W - nw) // 2, (NORM_H - nh) // 2
    out[y0:y0 + nh, x0:x0 + nw] = m
    return out


def hog(img: np.ndarray, cell: int = 8, nbins: int = 9) -> np.ndarray:
    """Plain numpy HOG (unsigned gradients, 2x2-cell L2-Hys blocks, stride 1 cell).

    OpenCV 5 dropped ``cv2.HOGDescriptor`` from the main module, so do it here.
    """
    f = img.astype(np.float32)
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=1)
    mag = np.sqrt(gx * gx + gy * gy)
    ang = (np.degrees(np.arctan2(gy, gx)) % 180.0) / (180.0 / nbins)
    b0 = np.floor(ang).astype(np.int32) % nbins
    b1 = (b0 + 1) % nbins
    w1 = ang - np.floor(ang)
    H, W = img.shape
    cy, cx = H // cell, W // cell
    hist = np.zeros((cy, cx, nbins), np.float32)
    ci = (np.arange(H) // cell)[:, None].clip(0, cy - 1)
    cj = (np.arange(W) // cell)[None, :].clip(0, cx - 1)
    idx = (ci * cx + cj).repeat(1, 0)
    idx = np.broadcast_to(idx, (H, W))
    flat = hist.reshape(-1)
    np.add.at(flat, (idx * nbins + b0).ravel(), (mag * (1 - w1)).ravel())
    np.add.at(flat, (idx * nbins + b1).ravel(), (mag * w1).ravel())
    blocks = []
    for i in range(cy - 1):
        for j in range(cx - 1):
            v = hist[i:i + 2, j:j + 2].reshape(-1)
            v = v / (np.linalg.norm(v) + 1e-6)
            v = np.minimum(v, 0.2)
            blocks.append(v / (np.linalg.norm(v) + 1e-6))
    return np.concatenate(blocks)


def features(norm: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(norm, (3, 3), 0)
    h = hog(blur).astype(np.float32)
    h /= np.linalg.norm(h) + 1e-6
    g = cv2.resize(blur, (8, 12), interpolation=cv2.INTER_AREA).astype(np.float32).reshape(-1)
    g -= g.mean()
    g /= np.linalg.norm(g) + 1e-6
    f = np.concatenate([h, 0.6 * g])
    return f / (np.linalg.norm(f) + 1e-6)


# --------------------------------------------------------------------------
# template rendering
def _font_files() -> list[str]:
    seen, out = set(), []
    for pat in _FONT_GLOBS:
        for f in sorted(glob.glob(pat, recursive=True)):
            b = os.path.basename(f).lower()
            if b in seen or "oblique" in b or "italic" in b:
                continue
            seen.add(b)
            out.append(f)
    return out[:10]


def _render_cv(d: str, font: int, thick: int) -> np.ndarray:
    img = np.zeros((140, 110), np.uint8)
    cv2.putText(img, d, (12, 118), font, 3.6, 255, thick, cv2.LINE_AA)
    return img


def _render_pil(d: str, path: str) -> Optional[np.ndarray]:
    try:
        from PIL import Image, ImageDraw, ImageFont
        font = ImageFont.truetype(path, 96)
    except Exception:  # noqa: BLE001
        return None
    im = Image.new("L", (120, 140), 0)
    ImageDraw.Draw(im).text((12, 4), d, fill=255, font=font)
    return np.array(im)


def _augment(img: np.ndarray, rng: np.random.Generator, n: int) -> list[np.ndarray]:
    out = [img]
    h, w = img.shape
    for _ in range(n):
        a = img.copy()
        ang = rng.uniform(-9, 9)
        sh = rng.uniform(-0.18, 0.18)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, rng.uniform(0.9, 1.05))
        M[0, 1] += sh
        M[0, 2] -= sh * h / 2
        a = cv2.warpAffine(a, M, (w, h))
        if rng.random() < 0.5:  # perspective (jersey curvature / camera angle)
            d = 0.08 * min(w, h)
            src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
            dst = src + rng.uniform(-d, d, src.shape).astype(np.float32)
            a = cv2.warpPerspective(a, cv2.getPerspectiveTransform(src, dst), (w, h))
        k = rng.integers(0, 3)
        if k == 1:
            a = cv2.dilate(a, np.ones((5, 5), np.uint8))
        elif k == 2:
            a = cv2.erode(a, np.ones((3, 3), np.uint8))
        if rng.random() < 0.5:
            a = cv2.GaussianBlur(a, (7, 7), rng.uniform(1.0, 2.5))
        if rng.random() < 0.3:  # low-res round trip
            s = rng.uniform(0.15, 0.3)
            a = cv2.resize(cv2.resize(a, None, fx=s, fy=s, interpolation=cv2.INTER_AREA),
                           (w, h), interpolation=cv2.INTER_LINEAR)
        out.append(a)
    return out


def _largest_blob_mask(img: np.ndarray) -> np.ndarray:
    _, b = cv2.threshold(img, 110, 255, cv2.THRESH_BINARY)
    return b  # digits may legitimately have one blob; keep all strokes


def build_templates(n_aug: int = 14, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    feats, labels = [], []
    renders: list[tuple[int, np.ndarray]] = []
    for d in range(10):
        for font, thicks in _CV_FONTS:
            for t in thicks:
                renders.append((d, _render_cv(str(d), font, t)))
        for f in _font_files():
            r = _render_pil(str(d), f)
            if r is not None and r.max() > 0:
                renders.append((d, r))
    for d, img in renders:
        for a in _augment(img, rng, n_aug):
            nm = normalize_mask(_largest_blob_mask(a))
            if nm is None:
                continue
            feats.append(features(nm))
            labels.append(d)
    return np.stack(feats).astype(np.float32), np.array(labels, np.int32)


def get_model() -> tuple[np.ndarray, np.ndarray]:
    global _model
    with _lock:
        if _model is None:
            _model = build_templates()
        return _model


# --------------------------------------------------------------------------
def classify_features(f: np.ndarray, k: int = 7) -> tuple[int, float, np.ndarray]:
    """Return (digit, confidence, per-class score vector)."""
    X, y = get_model()
    sims = X @ f
    scores = np.full(10, -1.0, np.float32)
    for d in range(10):
        s = sims[y == d]
        if len(s):
            top = np.sort(s)[-3:]
            scores[d] = float(top.mean())
    order = np.argsort(scores)[::-1]
    s1, s2 = float(scores[order[0]]), float(scores[order[1]])
    idx = np.argsort(sims)[-k:]
    agree = float(np.mean(y[idx] == order[0]))
    abs_q = np.clip((s1 - 0.55) / 0.3, 0, 1)
    margin_q = np.clip((s1 - s2) / 0.08, 0, 1)
    conf = float(abs_q * (0.4 + 0.3 * margin_q + 0.3 * agree))
    return int(order[0]), conf, scores


def classify_blob(mask: np.ndarray) -> tuple[Optional[int], float]:
    nm = normalize_mask(mask)
    if nm is None:
        return None, 0.0
    d, c, _ = classify_features(features(nm))
    return d, c
