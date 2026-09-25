"""Cinematic look for the gym renders: grade (S-curve + split tone), vignette, film grain,
flash, condensed display type and kinetic (pop-in) text.

Self-contained on purpose: a shared engine may appear at ``rendering/style.py``; the
lead can swap these functions for it. Everything works on BGR uint8 frames in place or
returns new frames, and precomputes what it can so per-frame cost stays ~2-4 ms."""
from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from ..core.config import PROJECT_ROOT

DISPLAY_CANDIDATES = ["Anton-Regular.ttf", "Anton.ttf", "BebasNeue-Regular.ttf", "BebasNeue.ttf",
                      "Oswald-Bold.ttf", "Oswald-SemiBold.ttf"]
FALLBACK_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FALLBACK_REG = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


@lru_cache(maxsize=1)
def display_font_path() -> tuple[Optional[str], bool]:
    """(path, is_condensed). Looks in assets/fonts/ (vendored Anton/Bebas), else DejaVu."""
    for d in (PROJECT_ROOT / "assets/fonts", PROJECT_ROOT / "assets"):
        if d.is_dir():
            for name in DISPLAY_CANDIDATES:
                p = d / name
                if p.exists():
                    return str(p), True
            for p in sorted(d.rglob("*.[ot]tf")):
                n = p.name.lower()
                if any(k in n for k in ("anton", "bebas", "oswald", "condensed")):
                    return str(p), True
    return (FALLBACK_BOLD if Path(FALLBACK_BOLD).exists() else None), False


@lru_cache(maxsize=128)
def _font(path: Optional[str], size: int):
    if path and Path(path).exists():
        return ImageFont.truetype(path, size)
    return ImageFont.load_default()


@lru_cache(maxsize=4096)
def type_sprite(text: str, size: int, color: tuple = (255, 255, 255), face: str = "display",
                tracking: float = 0.0, glow: float = 0.0, glow_color: Optional[tuple] = None,
                shadow: float = 0.55) -> np.ndarray:
    """BGRA sprite. face: display (condensed) | bold | regular. The DejaVu fallback for
    'display' is squeezed horizontally so it still reads as a condensed sports face."""
    if face == "display":
        path, condensed = display_font_path()
    elif face == "bold":
        path, condensed = FALLBACK_BOLD, True
    else:
        path, condensed = FALLBACK_REG, True
    font = _font(path, size)
    widths = [font.getlength(c) + tracking * size for c in text]
    w = int(math.ceil(sum(widths))) + 2
    asc, desc = font.getmetrics()
    pad = int(size * 0.35) + int(glow * size * 0.3)
    W, H = w + 2 * pad, asc + desc + 2 * pad
    rgb = (color[2], color[1], color[0])

    def put(img, dx, dy, fill):
        d = ImageDraw.Draw(img)
        x = pad + dx
        for c, cw in zip(text, widths):
            d.text((x, pad + dy), c, font=font, fill=fill)
            x += cw

    base = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    if shadow > 0:
        sh = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        put(sh, 0, max(1, size // 20), (0, 0, 0, int(255 * shadow)))
        base = Image.alpha_composite(base, sh.filter(ImageFilter.GaussianBlur(max(1, size / 12))))
    if glow > 0:
        gc = glow_color or color
        gl = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        put(gl, 0, 0, (gc[2], gc[1], gc[0], int(255 * min(1.0, glow))))
        base = Image.alpha_composite(base, gl.filter(ImageFilter.GaussianBlur(max(2, size / 6))))
    txt = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    put(txt, 0, 0, rgb + (255,))
    img = Image.alpha_composite(base, txt)
    if face == "display" and not condensed:
        img = img.resize((max(1, int(W * 0.80)), H), Image.LANCZOS)
    a = np.array(img)
    return np.ascontiguousarray(a[..., [2, 1, 0, 3]])


def blit(frame: np.ndarray, sprite: np.ndarray, x: float, y: float, alpha: float = 1.0,
         anchor: str = "lt", scale: float = 1.0) -> tuple[int, int, int, int]:
    if scale != 1.0 and scale > 0:
        sprite = cv2.resize(sprite, (max(1, int(sprite.shape[1] * scale)), max(1, int(sprite.shape[0] * scale))),
                            interpolation=cv2.INTER_LINEAR)
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
    Hf, Wf = frame.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(Wf, x + w), min(Hf, y + h)
    if x1 <= x0 or y1 <= y0 or alpha <= 0.01:
        return (x, y, w, h)
    sp = sprite[y0 - y:y1 - y, x0 - x:x1 - x]
    a = sp[..., 3:4].astype(np.float32) * (min(1.0, alpha) / 255.0)
    roi = frame[y0:y1, x0:x1].astype(np.float32)
    frame[y0:y1, x0:x1] = (roi + (sp[..., :3].astype(np.float32) - roi) * a).astype(np.uint8)
    return (x, y, w, h)


def ease_out_back(t: float, s: float = 1.6) -> float:
    t = min(1.0, max(0.0, t)) - 1
    return t * t * ((s + 1) * t + s) + 1


def ease_out_cubic(t: float) -> float:
    t = min(1.0, max(0.0, t))
    return 1 - (1 - t) ** 3


def kinetic_alpha_scale(age_s: float, hold_s: float, pop_s: float = 0.22, out_s: float = 0.35
                        ) -> tuple[float, float, float]:
    """(alpha, scale, slide 0..1) for a pop-in / hold / fade-out text of total hold_s."""
    if age_s < 0 or age_s > hold_s:
        return 0.0, 1.0, 0.0
    if age_s < pop_s:
        t = age_s / pop_s
        return ease_out_cubic(t), 0.75 + 0.25 * ease_out_back(t), 1 - ease_out_cubic(t)
    if age_s > hold_s - out_s:
        t = (hold_s - age_s) / out_s
        return ease_out_cubic(t), 1.0 + 0.04 * (1 - t), 0.0
    return 1.0, 1.0, 0.0


class Grade:
    """Precomputed LUT grade + vignette + grain for one output size."""

    def __init__(self, w: int, h: int, contrast: float = 0.55, saturation: float = 0.88,
                 vignette: float = 0.42, grain: float = 5.0, warm: float = 0.04, seed: int = 7):
        self.w, self.h = w, h
        x = np.arange(256, dtype=np.float32) / 255
        # S-curve (smoothstep blend) + lifted blacks / soft highlight roll-off
        s = x * x * (3 - 2 * x)
        y = x + (s - x) * contrast
        y = 0.025 + y * 0.955
        luts = []
        for ch, tone in ((0, -warm), (1, 0.0), (2, warm)):     # B, G, R: teal shadows, warm highs
            t = y + tone * (x - 0.45)
            luts.append(np.clip(t * 255, 0, 255).astype(np.uint8))
        self.lut = np.stack(luts, -1).reshape(256, 1, 3)
        self.sat = saturation
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        r = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2) / math.sqrt(2)
        v = 1 - vignette * np.clip((r - 0.35) / 0.65, 0, 1) ** 1.6
        self.vig = cv2.merge([(v * 255).astype(np.uint8)] * 3)
        rng = np.random.default_rng(seed)
        self.grain = []
        if grain > 0:
            for _ in range(6):
                g = rng.normal(0, grain, (h // 2, w // 2)).astype(np.float32)
                g = cv2.resize(g, (w, h), interpolation=cv2.INTER_LINEAR)
                pos = np.clip(g, 0, 255).astype(np.uint8)
                neg = np.clip(-g, 0, 255).astype(np.uint8)
                self.grain.append((cv2.merge([pos] * 3), cv2.merge([neg] * 3)))
        self.k = 0

    def add_scrim(self, top: float = 0.0, bottom: float = 0.0) -> None:
        """Darken the top/bottom bands (text legibility without boxes)."""
        h = self.h
        y = np.linspace(0, 1, h, dtype=np.float32)[:, None]
        m = np.ones((h, 1), np.float32)
        if top > 0:
            m *= 1 - top * np.clip((0.22 - y) / 0.22, 0, 1) ** 1.5
        if bottom > 0:
            m *= 1 - bottom * np.clip((y - 0.70) / 0.30, 0, 1) ** 1.5
        v = self.vig.astype(np.float32) * m[..., None]
        self.vig = np.clip(v, 0, 255).astype(np.uint8)

    def apply(self, img: np.ndarray, grain: bool = True) -> np.ndarray:
        out = cv2.LUT(img, self.lut)
        if self.sat != 1.0:
            gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
            gray3 = cv2.merge([gray, gray, gray])
            out = cv2.addWeighted(out, self.sat, gray3, 1 - self.sat, 0)
        out = cv2.multiply(out, self.vig, scale=1 / 255)
        if grain and self.grain:
            self.k = (self.k + 1) % len(self.grain)
            pos, neg = self.grain[self.k]
            out = cv2.subtract(cv2.add(out, pos), neg)
        return out


def flash(img: np.ndarray, amount: float) -> np.ndarray:
    """Additive white flash (0..1)."""
    if amount <= 0.01:
        return img
    return cv2.addWeighted(img, 1.0, np.full_like(img, 255), 0.55 * amount, 0)


def punch_zoom(img: np.ndarray, z: float, cx: float, cy: float) -> np.ndarray:
    """Digital push-in by factor z (>1) around (cx, cy) in output px."""
    if z <= 1.001:
        return img
    h, w = img.shape[:2]
    M = np.float32([[z, 0, (1 - z) * cx], [0, z, (1 - z) * cy]])
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
