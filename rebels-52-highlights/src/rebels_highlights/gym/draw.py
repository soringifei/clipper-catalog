"""Drawing primitives for the premium overlay look: anti-aliased PIL text sprites,
glowing skeleton, angle arcs, fading bar-path trail, panels, sparkline and cards.
All functions draw in place on a BGR uint8 frame."""
from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .pose import KP

WHITE = (255, 255, 255)
GREY = (190, 190, 190)
GREEN = (94, 214, 46)     # BGR
AMBER = (32, 176, 255)
RED = (48, 59, 255)

_FONT_PATHS = {"bold": "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
               "regular": "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"}


def set_fonts(bold: Optional[str], regular: Optional[str]) -> None:
    if bold:
        _FONT_PATHS["bold"] = bold
    if regular:
        _FONT_PATHS["regular"] = regular
    _font.cache_clear()
    text_sprite.cache_clear()


@lru_cache(maxsize=64)
def _font(weight: str, size: int):
    p = _FONT_PATHS.get(weight)
    if p and Path(p).exists():
        return ImageFont.truetype(p, size)
    for alt in _FONT_PATHS.values():
        if Path(alt).exists():
            return ImageFont.truetype(alt, size)
    return ImageFont.load_default()


@lru_cache(maxsize=4096)
def text_sprite(text: str, size: int, color: tuple = WHITE, weight: str = "bold",
                shadow: bool = True, tracking: int = 0) -> np.ndarray:
    """BGRA sprite of anti-aliased text (with a soft drop shadow)."""
    font = _font(weight, size)
    if tracking:
        widths = [font.getlength(c) + tracking for c in text]
        w = int(sum(widths)) + 1
    else:
        w = int(font.getlength(text)) + 1
    asc, desc = font.getmetrics()
    pad = max(4, size // 6)
    W, H = w + 2 * pad, asc + desc + 2 * pad
    rgb = (color[2], color[1], color[0])

    def put(img_, dx, dy, fill):
        d = ImageDraw.Draw(img_)
        if tracking:
            x = pad + dx
            for c, cw in zip(text, widths):
                d.text((x, pad + dy), c, font=font, fill=fill)
                x += cw
        else:
            d.text((pad + dx, pad + dy), text, font=font, fill=fill)

    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    if shadow:
        from PIL import ImageFilter
        sh = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        put(sh, 0, max(1, size // 18), (0, 0, 0, 160))
        img = sh.filter(ImageFilter.GaussianBlur(max(1, size // 14)))
    put(img, 0, 0, rgb + (255,))
    a = np.array(img)
    return np.ascontiguousarray(a[..., [2, 1, 0, 3]])


def blit(frame: np.ndarray, sprite: np.ndarray, x: float, y: float, alpha: float = 1.0,
         anchor: str = "lt") -> tuple[int, int, int, int]:
    """Alpha-blend a BGRA sprite. anchor: l/c/r + t/m/b. Returns the drawn box."""
    h, w = sprite.shape[:2]
    if anchor[0] == "c":
        x -= w / 2
    elif anchor[0] == "r":
        x -= w
    if anchor[1] == "m":
        y -= h / 2
    elif anchor[1] == "b":
        y -= h
    x, y = int(round(x)), int(round(y))
    H, W = frame.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0 or alpha <= 0:
        return (x, y, w, h)
    sp = sprite[y0 - y:y1 - y, x0 - x:x1 - x]
    a = sp[..., 3:4].astype(np.float32) * (alpha / 255.0)
    roi = frame[y0:y1, x0:x1].astype(np.float32)
    frame[y0:y1, x0:x1] = (roi * (1 - a) + sp[..., :3].astype(np.float32) * a).astype(np.uint8)
    return (x, y, w, h)


def text(frame, s, x, y, size=32, color=WHITE, weight="bold", alpha=1.0, anchor="lt",
         shadow=True, tracking=0):
    return blit(frame, text_sprite(str(s), int(size), tuple(color), weight, shadow, tracking),
                x, y, alpha, anchor)


@lru_cache(maxsize=256)
def _rr_mask(w: int, h: int, r: int) -> np.ndarray:
    mask = np.zeros((h, w), np.uint8)
    r = int(min(r, w // 2, h // 2))
    cv2.rectangle(mask, (r, 0), (w - r - 1, h - 1), 255, -1)
    cv2.rectangle(mask, (0, r), (w - 1, h - r - 1), 255, -1)
    for cx, cy in ((r, r), (w - r - 1, r), (r, h - r - 1), (w - r - 1, h - r - 1)):
        cv2.circle(mask, (cx, cy), r, 255, -1, cv2.LINE_AA)
    return mask.astype(np.float32)[..., None] / 255.0


def rounded_rect(frame, x0, y0, x1, y1, r, color=(0, 0, 0), alpha=0.55):
    x0, y0, x1, y1 = (int(round(v)) for v in (x0, y0, x1, y1))
    H, W = frame.shape[:2]
    x0c, y0c, x1c, y1c = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if x1c <= x0c or y1c <= y0c or alpha <= 0:
        return
    m = _rr_mask(x1 - x0, y1 - y0, int(r))[y0c - y0:y1c - y0, x0c - x0:x1c - x0] * alpha
    roi = frame[y0c:y1c, x0c:x1c].astype(np.float32)
    frame[y0c:y1c, x0c:x1c] = (roi * (1 - m) + np.array(color, np.float32) * m).astype(np.uint8)


def lerp_color(c1, c2, t):
    return tuple(int(round(a + (b - a) * t)) for a, b in zip(c1, c2))


def ramp_color(f: float) -> tuple:
    """0 -> green, 0.5 -> amber, 1 -> red (end range)."""
    f = float(np.clip(f, 0, 1))
    return lerp_color(GREEN, AMBER, f / 0.6) if f < 0.6 else lerp_color(AMBER, RED, (f - 0.6) / 0.4)


def _glow(frame, draw_fn, bbox, strength=0.55, blur=9):
    """Draw ``draw_fn(layer)`` into a black layer over bbox, blur it, add it to frame."""
    H, W = frame.shape[:2]
    x0, y0, x1, y1 = bbox
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(W, int(x1)), min(H, int(y1))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return
    layer = np.zeros((y1 - y0, x1 - x0, 3), np.uint8)
    draw_fn(layer, (x0, y0))
    small = cv2.resize(layer, ((x1 - x0) // 4 or 1, (y1 - y0) // 4 or 1), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), max(0.8, blur / 4))
    layer = cv2.resize(small, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
    roi = frame[y0:y1, x0:x1]
    cv2.addWeighted(roi, 1.0, layer, strength, 0, dst=roi)


def _pt(p):
    return (int(round(p[0] * 16)), int(round(p[1] * 16)))


def draw_skeleton(frame, kp: np.ndarray, near: str, scale: float, color=WHITE,
                  highlight: tuple = ()):
    """kp (17,2) output px with NaN for hidden joints. Far-side limbs drawn dimmer."""
    th = max(2, int(round(3 * scale)))
    far = "r" if near == "left" else "l"
    bones = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
             (11, 13), (13, 15), (12, 14), (14, 16)]
    far_idx = {KP[f"{far}_{n}"] for n in ("sho", "elb", "wri", "hip", "knee", "ank")}
    ok = ~np.isnan(kp[:, 0])
    pts = kp[ok]
    if len(pts) < 2:
        return
    pad = 30 * scale
    bbox = (pts[:, 0].min() - pad, pts[:, 1].min() - pad, pts[:, 0].max() + pad, pts[:, 1].max() + pad)

    def lines(img, off, thick, col_near, col_far):
        for a, b in bones:
            if ok[a] and ok[b]:
                pa = kp[a] - off
                pb = kp[b] - off
                col = col_far if (a in far_idx and b in far_idx) or (a in far_idx and b not in (5, 6, 11, 12)) else col_near
                cv2.line(img, _pt(pa), _pt(pb), col, thick, cv2.LINE_AA, shift=4)

    _glow(frame, lambda L, o: lines(L, np.array(o), th * 4, color, lerp_color(color, (0, 0, 0), 0.5)),
          bbox, strength=0.45, blur=int(10 * scale) + 2)
    lines(frame, np.zeros(2), th, color, lerp_color(color, (70, 70, 70), 0.45))
    r = max(3, int(round(5 * scale)))
    for i in range(5, 17):
        if ok[i]:
            c = color if i not in far_idx else lerp_color(color, (70, 70, 70), 0.45)
            cv2.circle(frame, _pt(kp[i]), r * 16, c, -1, cv2.LINE_AA, shift=4)
            if i in highlight:
                cv2.circle(frame, _pt(kp[i]), (r + 3) * 16, color, max(1, th // 2), cv2.LINE_AA, shift=4)


def draw_arc(frame, a, b, c, value: float, frac: float, radius: float, scale: float,
             emphasis: float = 0.0, label: Optional[str] = None, small: bool = False):
    """Angle arc at vertex b between rays b->a and b->c, value label on the bisector."""
    if any(np.isnan(v).any() for v in (a, b, c)):
        return
    col = ramp_color(frac)
    va, vc = np.asarray(a) - b, np.asarray(c) - b
    if np.linalg.norm(va) < 1 or np.linalg.norm(vc) < 1:
        return
    t1 = math.degrees(math.atan2(va[1], va[0]))
    t2 = math.degrees(math.atan2(vc[1], vc[0]))
    d = (t2 - t1 + 540) % 360 - 180          # signed shortest sweep
    r = radius * (1 + 0.35 * emphasis)
    th = max(2, int(round((3 + 3 * emphasis) * scale)))
    center = _pt(b)
    axes = (int(r * 16), int(r * 16))
    # translucent wedge
    x0, y0 = int(b[0] - r - 4), int(b[1] - r - 4)
    x1, y1 = int(b[0] + r + 4), int(b[1] + r + 4)
    H, W = frame.shape[:2]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if x1 <= x0 or y1 <= y0:
        return
    sub = frame[y0:y1, x0:x1]
    wedge = sub.copy()
    cv2.ellipse(wedge, (int((b[0] - x0) * 16), int((b[1] - y0) * 16)), axes, 0, t1, t1 + d,
                col, -1, cv2.LINE_AA, shift=4)
    cv2.addWeighted(wedge, 0.28 + 0.12 * emphasis, sub, 0.72 - 0.12 * emphasis, 0, dst=sub)
    if emphasis > 0:
        _glow(frame, lambda L, o: cv2.ellipse(L, (int((b[0] - o[0]) * 16), int((b[1] - o[1]) * 16)),
                                              axes, 0, t1, t1 + d, col, th * 3, cv2.LINE_AA, shift=4),
              (b[0] - r - 30, b[1] - r - 30, b[0] + r + 30, b[1] + r + 30), strength=0.8 * emphasis, blur=12)
    cv2.ellipse(frame, center, axes, 0, t1, t1 + d, col, th, cv2.LINE_AA, shift=4)
    # value on the bisector, outside the arc
    mid = math.radians(t1 + d / 2)
    off = r + (26 + 12 * emphasis) * scale * (0.8 if small else 1.0)
    tx, ty = b[0] + math.cos(mid) * off, b[1] + math.sin(mid) * off
    size = int(round((30 + 14 * emphasis) * scale * (0.72 if small else 1.0)))
    txt = f"{value:.0f}°" if label is None else label
    text(frame, txt, tx, ty, size, WHITE, anchor="cm")


def draw_vertical_ref(frame, b, length, scale, color=(200, 200, 200)):
    """Dashed vertical reference line up from b (for trunk lean)."""
    if np.isnan(b).any():
        return
    n = 6
    for i in range(n):
        if i % 2:
            continue
        y0 = b[1] - length * i / n
        y1 = b[1] - length * (i + 1) / n
        cv2.line(frame, _pt((b[0], y0)), _pt((b[0], y1)), color, max(1, int(2 * scale)), cv2.LINE_AA, shift=4)


def draw_trail(frame, pts: np.ndarray, color, scale: float):
    """Fading polyline for the bar path (oldest first)."""
    ok = ~np.isnan(pts[:, 0])
    n = len(pts)
    if ok.sum() < 2:
        return
    bbox = (np.nanmin(pts[:, 0]) - 20, np.nanmin(pts[:, 1]) - 20, np.nanmax(pts[:, 0]) + 20, np.nanmax(pts[:, 1]) + 20)

    def lines(img, off, glow):
        for i in range(1, n):
            if ok[i] and ok[i - 1]:
                f = i / n
                th = max(1, int(round((1.0 + 2.5 * f) * scale * (3 if glow else 1))))
                c = lerp_color((20, 20, 20), color, 0.25 + 0.75 * f)
                cv2.line(img, _pt(pts[i - 1] - off), _pt(pts[i] - off), c, th, cv2.LINE_AA, shift=4)

    _glow(frame, lambda L, o: lines(L, np.array(o), True), bbox, strength=0.4, blur=10)
    lines(frame, np.zeros(2), False)
    last = pts[ok][-1]
    cv2.circle(frame, _pt(last), int(9 * scale) * 16, color, -1, cv2.LINE_AA, shift=4)
    cv2.circle(frame, _pt(last), int(12 * scale) * 16, WHITE, max(1, int(2 * scale)), cv2.LINE_AA, shift=4)


def blur_background(img: np.ndarray, out_w: int, out_h: int, darken: float = 0.45) -> np.ndarray:
    """Cover-scale + heavy blur + darken, all done at 1/12 resolution (cheap)."""
    h, w = img.shape[:2]
    s = max(out_w / w, out_h / h)
    cw, ch = out_w / s, out_h / s
    x0, y0 = int((w - cw) / 2), int((h - ch) / 2)
    sub = img[y0:y0 + int(ch), x0:x0 + int(cw)]
    small = cv2.resize(sub, (max(1, out_w // 12), max(1, out_h // 12)), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 3)
    small = cv2.convertScaleAbs(small, alpha=1 - darken)
    return cv2.resize(small, (out_w, out_h), interpolation=cv2.INTER_LINEAR)


class Sparkline:
    """Pre-rendered angle-over-time graph; per frame only the cursor is drawn."""

    def __init__(self, values: np.ndarray, x0: int, y0: int, w: int, h: int, title: str,
                 accent, marks: list[int] = (), lo: Optional[float] = None,
                 hi: Optional[float] = None, unit: str = "°"):
        self.x0, self.y0, self.w, self.h = x0, y0, w, h
        v = np.asarray(values, float)
        self.n = len(v)
        vv = v[~np.isnan(v)]
        self.lo = lo if lo is not None else (float(vv.min()) if len(vv) else 0.0)
        self.hi = hi if hi is not None else (float(vv.max()) if len(vv) else 1.0)
        if self.hi - self.lo < 5:
            self.hi = self.lo + 5
        self.accent = accent
        self.v = v
        self.pad = 22
        base = np.zeros((h, w, 4), np.uint8)
        bgr = base[..., :3].copy()
        a = np.zeros((h, w), np.uint8)
        # panel
        rr = np.zeros((h, w), np.uint8)
        r = 22
        cv2.rectangle(rr, (r, 0), (w - r, h - 1), 255, -1)
        cv2.rectangle(rr, (0, r), (w - 1, h - r), 255, -1)
        for cx, cy in ((r, r), (w - r - 1, r), (r, h - r - 1), (w - r - 1, h - r - 1)):
            cv2.circle(rr, (cx, cy), r, 255, -1, cv2.LINE_AA)
        a = (rr.astype(np.float32) * 0.5).astype(np.uint8)
        pts = self._pts(np.arange(self.n))
        okp = ~np.isnan(pts[:, 1])
        line = np.zeros((h, w), np.uint8)
        for i in range(1, self.n):
            if okp[i] and okp[i - 1]:
                cv2.line(line, _pt(pts[i - 1]), _pt(pts[i]), 255, 3, cv2.LINE_AA, shift=4)
        for m in marks:
            if 0 <= m < self.n and okp[m]:
                cv2.circle(line, _pt(pts[m]), 6 * 16, 255, -1, cv2.LINE_AA, shift=4)
        grid = np.zeros((h, w), np.uint8)
        for gy in (0.0, 0.5, 1.0):
            y = int(self.pad + 30 + (h - self.pad * 2 - 30) * gy)
            cv2.line(grid, (self.pad, y), (w - self.pad, y), 60, 1, cv2.LINE_AA)
        bgr[:] = 0
        bgr[line > 0] = (235, 235, 235)
        bgr[(grid > 0) & (line == 0)] = (120, 120, 120)
        alpha = np.maximum(a, np.maximum(line, (grid.astype(np.float32) * 0.9).astype(np.uint8)))
        base[..., :3] = bgr
        base[..., 3] = alpha
        # line colour must be premultiplied properly for dark panel: panel colour black
        self.base = base
        self.title = title
        self.unit = unit

    def _pts(self, idx):
        idx = np.asarray(idx)
        x = self.pad + (self.w - 2 * self.pad) * idx / max(1, self.n - 1)
        top = self.pad + 30
        bot = self.h - self.pad
        y = bot - (bot - top) * (self.v[idx] - self.lo) / (self.hi - self.lo)
        return np.stack([x, y], -1)

    def draw(self, frame, i: int, alpha: float = 1.0):
        blit(frame, self.base, self.x0, self.y0, alpha)
        text(frame, self.title, self.x0 + self.pad, self.y0 + 10, 22, GREY, "bold", alpha, shadow=False,
             tracking=2)
        i = int(np.clip(i, 0, self.n - 1))
        p = self._pts([i])[0]
        x = self.x0 + p[0]
        cv2.line(frame, (int(x), self.y0 + self.pad + 26), (int(x), self.y0 + self.h - self.pad),
                 self.accent, 2, cv2.LINE_AA)
        if not np.isnan(p[1]):
            cv2.circle(frame, _pt((x, self.y0 + p[1])), 9 * 16, self.accent, -1, cv2.LINE_AA, shift=4)
            cv2.circle(frame, _pt((x, self.y0 + p[1])), 9 * 16, WHITE, 2, cv2.LINE_AA, shift=4)
            text(frame, f"{self.v[i]:.0f}{self.unit}", self.x0 + self.w - self.pad, self.y0 + 8, 24,
                 WHITE, anchor="rt", alpha=alpha, shadow=False)


def pill(frame, s, x, y, accent, size=26, anchor="rt", alpha=1.0):
    sp = text_sprite(s, size, WHITE, "bold", False, 1)
    h, w = sp.shape[:2]
    if anchor[0] == "r":
        x -= w + 24
    rounded_rect(frame, x, y, x + w + 24, y + h + 4, (h + 4) // 2, accent, 0.9 * alpha)
    blit(frame, sp, x + 12, y + 2, alpha)
