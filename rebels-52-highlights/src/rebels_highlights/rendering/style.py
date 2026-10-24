"""Shared visual style engine for vertical highlights (football + gym).

numpy / OpenCV / PIL only, built for ~30-60 ms per 1080x1920 frame on 4 CPUs.

Pieces (all stateless or cheap to construct, usable by any renderer):

* :func:`get_style` - ``render.style`` (``epic`` default, ``clean`` = legacy
  lower-third look) + ``render.epic`` tuning knobs -> :class:`Style`.
* fonts - vendored OFL fonts in ``assets/fonts`` (Anton display, Bebas Neue
  titles, Oswald text) with DejaVu fallback: :func:`font`.
* :class:`Grader` - cinematic grade: S-curve, teal shadows / warm highlights,
  saturation, vignette, animated film grain, light unsharp mask; ``mono``
  blends toward a red-black duotone (end card).
* :func:`spotlight` - feathered elliptical spotlight (outside desaturated and
  darkened).
* impact FX: :func:`impact_fx` (flash, decaying shake, zoom punch, RGB split,
  radial blur) keyed on frames since impact.
* kinetic type: :class:`TextSprite` (pre-rendered, premultiplied, soft
  shadow, optional accent underline) + :func:`blit` and :class:`Slam`
  (scale-overshoot slam-in with zoom echoes, hold, fast exit).
* :func:`light_streak` - additive tapered glow trail.
* :class:`RampMap` - smooth speed ramps (source<->output time maps).
* :func:`text_zone` - pick a text band that never covers the action.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FONT_DIR = PROJECT_ROOT / "assets" / "fonts"
FONT_FILES = {
    "display": ["anton/Anton-Regular.ttf"],
    "title": ["bebasneue/BebasNeue-Regular.ttf", "anton/Anton-Regular.ttf"],
    "text": ["oswald/Oswald-Variable.ttf", "bebasneue/BebasNeue-Regular.ttf"],
}
_FALLBACK_DIRS = ["/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/dejavu",
                  "/usr/local/share/fonts", "/Library/Fonts", "C:/Windows/Fonts"]
DEFAULT_ACCENT = "#E10600"
WHITE = (255, 255, 255)
STYLES = ("epic", "clean")


# ============================================================ colour / easing
def hex_to_rgb(color: str | Sequence[int]) -> tuple[int, int, int]:
    if isinstance(color, str):
        c = color.lstrip("#")
        return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]
    return tuple(int(v) for v in color[:3])  # type: ignore[return-value]


def hex_to_bgr(color: str | Sequence[int]) -> tuple[int, int, int]:
    r, g, b = hex_to_rgb(color)
    return b, g, r


def clamp01(x: float) -> float:
    return 0.0 if x <= 0 else 1.0 if x >= 1 else float(x)


def smoothstep(x: float) -> float:
    x = clamp01(x)
    return x * x * (3 - 2 * x)


def ease_out_cubic(x: float) -> float:
    x = clamp01(x)
    return 1 - (1 - x) ** 3


def ease_in_cubic(x: float) -> float:
    x = clamp01(x)
    return x ** 3


def ease_in_out(x: float) -> float:
    x = clamp01(x)
    return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2


# ======================================================================= style
@dataclass
class Style:
    name: str = "epic"
    accent: tuple[int, int, int] = (225, 6, 0)          # RGB
    grade: float = 1.0          # 0 = off, 1 = default strength
    contrast: float = 0.32      # S-curve mix
    saturation: float = 1.14
    teal: float = 12.0          # shadow tint (0-255 units at grade=1)
    warm: float = 10.0          # highlight tint
    vignette: float = 0.38
    grain: float = 0.055        # film grain amount (fraction of 255 std-ish)
    sharpen: float = 0.35
    fx: float = 1.0             # impact FX strength
    text: bool = True           # kinetic words
    ramp_speed: float = 0.30    # slow-mo floor for speed ramps
    replay_ramp_speed: float = 0.25
    coldopen_s: float = 0.9
    coldopen_speed: float = 0.35
    title_s: float = 0.7
    rewind_s: float = 0.45
    intro_s: float = 1.1
    end_s: float = 1.8
    fx_zoom_allowance: float = 1.15
    extra: dict = field(default_factory=dict)

    @property
    def epic(self) -> bool:
        return self.name == "epic"

    @property
    def accent_bgr(self) -> tuple[int, int, int]:
        r, g, b = self.accent
        return b, g, r


def get_style(cfg: Optional[dict], name: Optional[str] = None) -> Style:
    rc = (cfg or {}).get("render", {}) or {}
    nm = str(name or rc.get("style", "epic") or "epic").lower()
    if nm not in STYLES:
        nm = "epic"
    st = Style(name=nm, accent=hex_to_rgb(rc.get("accent_color", DEFAULT_ACCENT)))
    ec = rc.get("epic", {}) or {}
    for k, v in ec.items():
        if hasattr(st, k) and k not in ("name", "accent", "extra"):
            setattr(st, k, type(getattr(st, k))(v))
        else:
            st.extra[k] = v
    return st


# ======================================================================= fonts
@lru_cache(maxsize=16)
def font_path(role: str = "display") -> Optional[str]:
    for rel in FONT_FILES.get(role, []):
        p = FONT_DIR / rel
        if p.is_file():
            return str(p)
    return None


def _fallback_font(size: int, bold: bool = True) -> ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    for d in _FALLBACK_DIRS:
        p = Path(d) / name
        if p.exists():
            return ImageFont.truetype(str(p), size)
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # pragma: no cover
        return ImageFont.load_default()


@lru_cache(maxsize=128)
def font(role: str = "display", size: int = 100, weight: Optional[str] = None) -> ImageFont.ImageFont:
    """Vendored font for ``role`` (display|title|text), DejaVu Bold when missing."""
    p = font_path(role)
    if p is None:
        return _fallback_font(size, bold=role != "text")
    f = ImageFont.truetype(p, size)
    if weight:
        try:
            f.set_variation_by_name(weight)
        except Exception:  # noqa: BLE001 - static font or unknown instance
            pass
    return f


# ====================================================================== grading
class Grader:
    """Cinematic grade applied to every composed canvas frame (BGR uint8)."""

    def __init__(self, style: Style, size: tuple[int, int] = (1080, 1920), seed: int = 7,
                 n_grain: int = 6):
        import cv2
        self.cv2 = cv2
        self.st = style
        self.W, self.H = size
        s = float(style.grade)
        self.s = s
        x = np.arange(256, dtype=np.float32) / 255.0
        sc = x * x * (3 - 2 * x)
        y = x + (sc - x) * style.contrast * s * 1.6
        y = 0.018 * s + y * (1 - 0.045 * s)                  # lifted blacks, rolled highlights
        self.curve = np.clip(y * 255 + 0.5, 0, 255).astype(np.uint8)
        self._luts: dict[int, tuple] = {}
        # vignette (elliptical, soft)
        W, H = self.W, self.H
        yy, xx = np.mgrid[0:H // 4, 0:W // 4].astype(np.float32)
        nx = (xx - W / 8) / (W / 8)
        ny = (yy - H / 8) / (H / 8)
        r = np.sqrt(nx ** 2 * 1.0 + ny ** 2 * 0.85)
        v = 1 - style.vignette * s * np.clip((r - 0.45) / 0.85, 0, 1) ** 1.6
        v = cv2.resize(v, (W, H), interpolation=cv2.INTER_LINEAR)
        self.vig = cv2.merge([np.clip(v * 255, 0, 255).astype(np.uint8)] * 3)
        self.vig_f = v
        # animated grain: a few pre-baked fields, slightly clumped
        rng = np.random.default_rng(seed)
        self.grain = []
        if style.grain > 0 and s > 0:
            for _ in range(n_grain):
                g = rng.normal(0, 1, (H, W)).astype(np.float32)
                g = cv2.GaussianBlur(g, (0, 0), 0.7)
                g = g / (g.std() + 1e-6)
                g8 = np.clip(128 + g * 42, 0, 255).astype(np.uint8)
                self.grain.append(cv2.merge([g8, g8, g8]))

    def _tint_luts(self, mono: float) -> tuple:
        key = int(round(mono * 20))
        if key in self._luts:
            return self._luts[key]
        m = key / 20.0
        st, s = self.st, self.s
        sat = 1 + (st.saturation - 1) * s
        sat = sat * (1 - m) + 0.06 * m
        g = np.arange(256, dtype=np.float32) / 255.0
        ws = np.clip((0.55 - g) / 0.55, 0, 1) ** 1.4
        wh = np.clip((g - 0.45) / 0.55, 0, 1) ** 1.4
        teal, warm = st.teal * s * (1 - m), st.warm * s * (1 - m)
        tb = teal * ws - warm * wh * 0.55
        tg = teal * ws * 0.45 + warm * wh * 0.12
        tr = -teal * ws * 0.6 + warm * wh
        # duotone: crimson shadows/mids, clean whites
        wm = np.clip((0.35 - g) / 0.35, 0, 1) ** 1.5     # crimson only in the deep shadows
        tr = tr + m * 14 * wm
        tg = tg - m * 4 * wm
        tb = tb - m * 2 * wm
        base = (1 - sat) * g * 255
        K = 96.0
        luts = tuple(np.clip(base + t + K, 0, 255).astype(np.uint8) for t in (tb, tg, tr))
        self._luts[key] = (luts, sat, K)
        return self._luts[key]

    def __call__(self, frame: np.ndarray, k: int = 0, mono: float = 0.0,
                 grain: bool = True, sharpen: bool = True, vignette: float = 1.0) -> np.ndarray:
        cv2 = self.cv2
        if self.s <= 0 and mono <= 0:
            return frame
        f = frame
        if sharpen and self.st.sharpen > 0 and self.s > 0:
            a = self.st.sharpen * self.s
            bl = cv2.GaussianBlur(f, (0, 0), 1.2)
            f = cv2.addWeighted(f, 1 + a, bl, -a, 0)
        f = cv2.LUT(f, self.curve)
        (lb, lg, lr), sat, K = self._tint_luts(mono)
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        x = cv2.merge([cv2.LUT(gray, lb), cv2.LUT(gray, lg), cv2.LUT(gray, lr)])
        f = cv2.addWeighted(f, sat, x, 1.0, -K)
        if vignette > 0 and self.st.vignette > 0:
            if vignette >= 0.999:
                f = cv2.multiply(f, self.vig, scale=1 / 255.0)
            else:
                v = cv2.addWeighted(self.vig, vignette, np.full_like(self.vig, 255), 1 - vignette, 0)
                f = cv2.multiply(f, v, scale=1 / 255.0)
        if grain and self.grain:
            amt = self.st.grain * max(self.s, 0.5)
            g = self.grain[(k * 5 + (k // len(self.grain))) % len(self.grain)]
            f = cv2.addWeighted(f, 1.0, g, amt, -128 * amt)
        return f


# ====================================================================== spotlight
def spotlight(frame: np.ndarray, center: tuple[float, float], axes: tuple[float, float],
              amount: float = 1.0, feather: float = 0.55, dim: float = 0.28,
              desat: float = 0.92) -> np.ndarray:
    """Keep a feathered ellipse at full colour, desaturate + darken the rest."""
    import cv2
    if amount <= 0.001:
        return frame
    H, W = frame.shape[:2]
    q = 8
    mask = np.zeros((H // q + 1, W // q + 1), np.float32)
    cx, cy = center[0] / q, center[1] / q
    ax, ay = max(2.0, axes[0] / q), max(2.0, axes[1] / q)
    cv2.ellipse(mask, (int(round(cx)), int(round(cy))), (int(round(ax)), int(round(ay))),
                0, 0, 360, 1.0, -1, cv2.LINE_AA)
    sig = max(1.0, feather * (ax + ay) / 2)
    mask = cv2.GaussianBlur(mask, (0, 0), sig)
    mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_LINEAR)[:H, :W]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bg = cv2.addWeighted(frame, 1 - desat, cv2.merge([gray, gray, gray]), desat, 0)
    bg = cv2.convertScaleAbs(bg, alpha=dim)
    w1 = np.clip(1 - amount * (1 - mask), 0, 1).astype(np.float32)
    return cv2.blendLinear(frame, bg, w1, 1 - w1)


# ===================================================================== impact FX
def fx_shake(k: int, amp: float = 20.0, frames: int = 9, seed: int = 3) -> tuple[float, float]:
    """Decaying camera shake offset (px) ``k`` frames after impact."""
    if k < 0 or k >= frames:
        return 0.0, 0.0
    rng = np.random.default_rng(seed + 1000)
    dirs = rng.uniform(-1, 1, (frames, 2))
    dirs[:, 1] *= 1.25
    d = math.exp(-k / (frames / 3.2))
    sign = 1 if k % 2 == 0 else -1
    return float(dirs[k, 0] * amp * d * sign), float(dirs[k, 1] * amp * d * sign)


def fx_punch(k: int, amount: float = 0.08) -> float:
    """Zoom punch factor: +amount at impact easing back over 4 frames."""
    if k < 0 or k >= 5:
        return 1.0
    return 1.0 + amount * (1 - ease_out_cubic(k / 4.5))


def warp_canvas(frame: np.ndarray, zoom: float = 1.0, dx: float = 0.0, dy: float = 0.0,
                center: Optional[tuple[float, float]] = None) -> np.ndarray:
    import cv2
    if abs(zoom - 1) < 1e-4 and abs(dx) < 0.05 and abs(dy) < 0.05:
        return frame
    H, W = frame.shape[:2]
    cx, cy = center if center is not None else (W / 2, H / 2)
    M = np.float32([[zoom, 0, cx - zoom * cx + dx], [0, zoom, cy - zoom * cy + dy]])
    return cv2.warpAffine(frame, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def flash(frame: np.ndarray, amount: float, color: Sequence[int] = (255, 255, 255)) -> np.ndarray:
    import cv2
    if amount <= 0.001:
        return frame
    a = clamp01(amount)
    if tuple(color) == (255, 255, 255):
        return cv2.convertScaleAbs(frame, alpha=1 - a, beta=255 * a)
    col = np.empty_like(frame)
    col[:] = color
    return cv2.addWeighted(frame, 1 - a, col, a, 0)


def chromatic_aberration(frame: np.ndarray, px: float,
                         center: Optional[tuple[float, float]] = None) -> np.ndarray:
    """Radial RGB split: red scaled out, blue scaled in by ``px`` at the frame edge."""
    import cv2
    if px < 0.5:
        return frame
    H, W = frame.shape[:2]
    cx, cy = center if center is not None else (W / 2, H / 2)
    b, g, r = cv2.split(frame)
    out = []
    for ch, sgn in ((b, -1), (g, 0), (r, 1)):
        if sgn == 0:
            out.append(ch)
            continue
        z = 1 + sgn * px / (H / 2)
        M = np.float32([[z, 0, cx - z * cx], [0, z, cy - z * cy]])
        out.append(cv2.warpAffine(ch, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT))
    return cv2.merge(out)


def radial_blur(frame: np.ndarray, strength: float, center: Optional[tuple[float, float]] = None,
                n: int = 4) -> np.ndarray:
    """Cheap zoom blur: average of ``n`` slightly scaled copies."""
    import cv2
    if strength <= 0.002:
        return frame
    H, W = frame.shape[:2]
    cx, cy = center if center is not None else (W / 2, H / 2)
    acc = frame.astype(np.float32)
    for i in range(1, n):
        z = 1 + strength * i / (n - 1)
        M = np.float32([[z, 0, cx - z * cx], [0, z, cy - z * cy]])
        acc += cv2.warpAffine(frame, M, (W, H), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT).astype(np.float32)
    return (acc / n).astype(np.uint8)


def impact_fx(frame: np.ndarray, k: int, style: Style,
              center: Optional[tuple[float, float]] = None, seed: int = 3) -> np.ndarray:
    """All impact FX for the frame ``k`` frames after the contact (k<0: none)."""
    s = float(style.fx)
    if k < 0 or k > 10 or s <= 0:
        return frame
    H, W = frame.shape[:2]
    c = center if center is not None else (W / 2, H * 0.45)
    dx, dy = fx_shake(k, 20.0 * s, 9, seed)
    z = fx_punch(k, 0.085 * s)
    frame = warp_canvas(frame, z, dx, dy, c)
    if k < 3:
        frame = radial_blur(frame, (0.035, 0.022, 0.01)[k] * s, c, n=4)
    if k < 5:
        frame = chromatic_aberration(frame, (9, 7, 5, 3, 1.5)[k] * s, c)
    if k < 3:
        frame = flash(frame, (0.34, 0.2, 0.08)[k] * s)
    return frame


# ================================================================ kinetic type
class TextSprite:
    """Pre-rendered text (premultiplied BGR float32 + alpha) with soft shadow.

    ``lines``: one string or several (stacked, centred). ``tracking`` is extra
    letter spacing in em. ``underline``: RGB accent drawn as a thin bar under
    the last line (its width can be animated via ``blit(..., underline=f)``).
    """

    def __init__(self, lines: str | Sequence[str], role: str = "display", size: int = 160,
                 fill: Sequence[int] = WHITE, tracking: float = 0.0, shadow: float = 0.6,
                 underline: Optional[Sequence[int]] = None, weight: Optional[str] = None,
                 line_gap: float = 0.06, glyph_fills: Optional[dict[int, Sequence[int]]] = None):
        import cv2
        if isinstance(lines, str):
            lines = [lines]
        f = font(role, int(size), weight)
        self.size = int(size)
        adv = []
        for ln in lines:
            a = [f.getlength(ch) + tracking * size for ch in ln]
            if a:
                a[-1] -= tracking * size
            adv.append(a)
        asc, desc = f.getmetrics()
        # tight vertical metrics from real glyph boxes (display fonts have big leading)
        bbs = [f.getbbox(ln) if ln else (0, 0, 0, 0) for ln in lines]
        tops = [b[1] for b in bbs]
        bots = [b[3] for b in bbs]
        lh = [max(1, bo - to) for to, bo in zip(tops, bots)]
        gap = int(size * line_gap)
        w_txt = int(math.ceil(max(sum(a) for a in adv))) if adv else 1
        ul_h = max(4, int(size * 0.045)) if underline is not None else 0
        ul_gap = max(14, int(size * 0.12)) if underline is not None else 0
        h_txt = sum(lh) + gap * (len(lines) - 1) + ul_gap + ul_h
        pad = int(size * 0.28) + 8
        Wc, Hc = w_txt + 2 * pad, h_txt + 2 * pad
        mask = Image.new("L", (Wc, Hc), 0)
        col = Image.new("RGB", (Wc, Hc), tuple(fill[:3]))
        d = ImageDraw.Draw(mask)
        dc = ImageDraw.Draw(col)
        y = pad
        gi = 0
        self.line_boxes = []
        for ln, a, to, h in zip(lines, adv, tops, lh):
            x = pad + (w_txt - sum(a)) / 2
            self.line_boxes.append((x, y, x + sum(a), y + h))
            for ch, w in zip(ln, a):
                d.text((x, y - to), ch, font=f, fill=255)
                if glyph_fills and gi in glyph_fills:
                    gb = f.getbbox(ch)
                    dc.rectangle([x + gb[0] - 2, y - to + gb[1] - 2, x + gb[2] + 2, y - to + gb[3] + 2],
                                 fill=tuple(glyph_fills[gi][:3]))
                x += w
                gi += 1
            y += h + gap
        self.ul_rect = None
        if underline is not None:
            y += ul_gap - gap
            lb = self.line_boxes[-1]
            uw = min(lb[2] - lb[0], max(size * 1.2, (lb[2] - lb[0]) * 0.42))
            ux = (Wc - uw) / 2
            self.ul_rect = (ux, y, ux + uw, y + ul_h)
            d.rectangle(self.ul_rect, fill=255)
            dc.rectangle(self.ul_rect, fill=tuple(underline[:3]))
        a = np.asarray(mask, np.float32) / 255.0
        rgb = np.asarray(col, np.float32)[..., ::-1]
        # shadow: soft, slightly offset, dark
        if shadow > 0:
            sh = cv2.GaussianBlur(a, (0, 0), max(2.0, size * 0.06))
            off = int(size * 0.035)
            sh = np.roll(sh, (off, off // 2), axis=(0, 1)) * shadow
            sh2 = cv2.GaussianBlur(a, (0, 0), max(6.0, size * 0.22)) * shadow * 0.45
            sha = np.clip(np.maximum(sh, sh2), 0, 1)
        else:
            sha = np.zeros_like(a)
        alpha = a + sha * (1 - a)
        prem = rgb * a[..., None]            # shadow colour = black
        self.rgb = np.ascontiguousarray(prem, np.float32)
        self.a = np.ascontiguousarray(alpha, np.float32)
        self.w, self.h = Wc, Hc
        self.pad = pad
        self.text_h = h_txt
        self.text_w = w_txt


def blit(frame: np.ndarray, spr: TextSprite, cx: float, cy: float, scale: float = 1.0,
         alpha: float = 1.0, mblur: float = 0.0, underline: float = 1.0) -> np.ndarray:
    """Alpha-composite a sprite centred at (cx, cy) in place; returns frame."""
    import cv2
    if alpha <= 0.003 or scale <= 0.01:
        return frame
    rgb, a = spr.rgb, spr.a
    if underline < 0.999 and spr.ul_rect is not None:
        x1, y1, x2, y2 = (int(round(v)) for v in spr.ul_rect)
        cut = x1 + int((x2 - x1) * clamp01(underline))
        rgb = rgb.copy()
        a = a.copy()
        rgb[y1:y2 + 1, cut:x2 + 1] = 0
        a[y1:y2 + 1, cut:x2 + 1] = 0
    if abs(scale - 1) > 1e-3:
        nw, nh = max(1, int(spr.w * scale)), max(1, int(spr.h * scale))
        rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
        a = cv2.resize(a, (nw, nh), interpolation=cv2.INTER_LINEAR)
    if mblur >= 2:
        k = int(mblur) | 1
        ker = np.zeros((k, 1), np.float32)
        ker[:, 0] = 1.0 / k
        rgb = cv2.filter2D(rgb, -1, ker)
        a = cv2.filter2D(a, -1, ker)
    h, w = a.shape[:2]
    H, W = frame.shape[:2]
    x0, y0 = int(round(cx - w / 2)), int(round(cy - h / 2))
    fx1, fy1, fx2, fy2 = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
    if fx2 <= fx1 or fy2 <= fy1:
        return frame
    sx1, sy1 = fx1 - x0, fy1 - y0
    sa = a[sy1:sy1 + fy2 - fy1, sx1:sx1 + fx2 - fx1] * alpha
    sr = rgb[sy1:sy1 + fy2 - fy1, sx1:sx1 + fx2 - fx1] * alpha
    roi = frame[fy1:fy2, fx1:fx2].astype(np.float32)
    out = roi * (1 - sa[..., None]) + sr
    frame[fy1:fy2, fx1:fx2] = np.clip(out, 0, 255).astype(np.uint8)
    return frame


@dataclass
class Slam:
    """One kinetic text hit on the output timeline (seconds, output clock).

    kind ``slam``: scale 1.45 -> 0.96 -> 1.0 with zoom echoes and vertical
    motion blur on the first frames; ``rise``: short upward drift + fade;
    exit is a quick scale-up + fade (``exit_s``).
    """
    sprite: TextSprite
    t_in: float
    t_out: float
    x: float
    y: float
    kind: str = "slam"
    exit_s: float = 0.14
    underline_wipe: float = 0.0     # seconds for the accent underline to wipe in
    delay_ul: float = 0.1
    tag: str = ""
    drift: float = 0.0              # slow scale creep per second while held
    shake: bool = False             # small frame shake when the slam lands

    def state(self, t: float) -> Optional[tuple[float, float, float, float]]:
        """(scale, alpha, dy, mblur) or None when invisible."""
        if t < self.t_in or t > self.t_out + self.exit_s:
            return None
        u = t - self.t_in
        if self.kind == "slam":
            if u < 0.10:
                p = u / 0.10
                sc = 1.45 - 0.49 * ease_in_cubic(p) ** 0.7
                al = clamp01(0.35 + p * 1.3)
                mb = 26 * (1 - p)
            elif u < 0.22:
                p = (u - 0.10) / 0.12
                sc = 0.96 + 0.04 * ease_out_cubic(p)
                al, mb = 1.0, 0.0
            else:
                sc, al, mb = 1.0, 1.0, 0.0
            dy = 0.0
        else:  # rise
            p = ease_out_cubic(u / 0.3)
            sc, al, mb = 1.0, clamp01(u / 0.18), 0.0
            dy = 26 * (1 - p)
        sc *= 1 + self.drift * u
        if t > self.t_out:
            q = (t - self.t_out) / self.exit_s
            sc *= 1 + 0.10 * ease_in_cubic(q)
            al *= 1 - q
            dy -= 18 * q
        return sc, al, dy, mb

    def draw(self, frame: np.ndarray, t: float) -> np.ndarray:
        s = self.state(t)
        if s is None:
            return frame
        sc, al, dy, mb = s
        ul = 1.0
        if self.underline_wipe > 0:
            ul = ease_out_cubic((t - self.t_in - self.delay_ul) / self.underline_wipe)
        if self.kind == "slam" and t - self.t_in < 0.1:   # zoom echoes while slamming
            for e, ea in ((1.22, 0.16), (1.1, 0.26)):
                blit(frame, self.sprite, self.x, self.y + dy, sc * e, al * ea, mb * 1.4, ul)
        return blit(frame, self.sprite, self.x, self.y + dy, sc, al, mb, ul)

    def landed(self, t: float) -> Optional[int]:
        """Frames since the slam landed (for a small shake), None otherwise."""
        if self.kind != "slam":
            return None
        u = t - self.t_in - 0.10
        return int(u * 30) if 0 <= u < 0.3 else None


# ================================================================== light streak
def light_streak(frame: np.ndarray, pts: Sequence[tuple[float, float]], color_bgr: Sequence[int],
                 width: float = 26.0, alpha: float = 1.0) -> np.ndarray:
    """Additive tapered glow along ``pts`` (oldest first); head is brightest."""
    import cv2
    if alpha <= 0.01 or len(pts) < 3:
        return frame
    P = np.asarray(pts, np.float32)
    H, W = frame.shape[:2]
    m = int(width * 3)
    x1, y1 = int(max(0, P[:, 0].min() - m)), int(max(0, P[:, 1].min() - m))
    x2, y2 = int(min(W, P[:, 0].max() + m)), int(min(H, P[:, 1].max() + m))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return frame
    q = 2
    lw, lh = (x2 - x1) // q + 1, (y2 - y1) // q + 1
    glow = np.zeros((lh, lw, 3), np.float32)
    core = np.zeros((lh, lw, 3), np.float32)
    n = len(P)
    col = np.asarray(color_bgr, np.float32)
    for i in range(1, n):
        f = i / (n - 1)
        p0 = ((P[i - 1] - (x1, y1)) / q * 4).astype(np.int32)
        p1 = ((P[i] - (x1, y1)) / q * 4).astype(np.int32)
        th = max(1, int(width / q * (0.12 + 0.88 * f ** 1.3)))
        cv2.line(glow, tuple(int(v) for v in p0), tuple(int(v) for v in p1),
                 tuple(float(c) * f ** 1.6 for c in col), th, cv2.LINE_AA, shift=2)
        cth = max(1, int(th * 0.45))
        wv = 255.0 * f ** 2.2
        cv2.line(core, tuple(int(v) for v in p0), tuple(int(v) for v in p1),
                 (wv, wv, wv), cth, cv2.LINE_AA, shift=2)
    g = cv2.GaussianBlur(glow, (0, 0), max(1.5, width / q * 0.55)) * 1.7
    g2 = cv2.GaussianBlur(glow, (0, 0), max(3.0, width / q * 1.6)) * 0.9
    c = cv2.GaussianBlur(core, (0, 0), 0.8) * 1.4
    lay = (g + g2 + c) * alpha
    lay = cv2.resize(lay, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
    roi = frame[y1:y2, x1:x2].astype(np.float32)
    frame[y1:y2, x1:x2] = np.clip(roi + lay, 0, 255).astype(np.uint8)
    return frame


# ==================================================================== speed ramp
class RampMap:
    """Smooth speed ramp over source [s_a, s_b].

    Speed is 1x outside ``holds``, eases (smoothstep over ``ease`` source
    seconds) into ``vmin`` inside each hold window. Output time is the
    integral of 1/speed, so the map is strictly monotonic.
    """

    def __init__(self, s_a: float, s_b: float, holds: Iterable[tuple[float, float]] = (),
                 vmin: float = 0.3, ease: float = 0.35, fps: float = 30.0, v_base: float = 1.0):
        self.s_a, self.s_b = float(s_a), float(max(s_b, s_a + 1.0 / fps))
        self.holds = [(float(a), float(b)) for a, b in holds]
        self.vmin, self.ease, self.fps, self.v_base = float(vmin), float(ease), float(fps), float(v_base)
        m = max(64, int((self.s_b - self.s_a) * fps * 24))
        self.sg = np.linspace(self.s_a, self.s_b, m)
        v = self.speed_src(self.sg)
        inv = 1.0 / v
        T = np.concatenate([[0.0], np.cumsum((inv[1:] + inv[:-1]) / 2 * np.diff(self.sg))])
        self.T = T
        self.duration = float(T[-1])
        self.n = max(1, int(round(self.duration * fps)))

    def weight(self, s: np.ndarray) -> np.ndarray:
        s = np.asarray(s, float)
        w = np.zeros_like(s)
        e = max(self.ease, 1e-3)
        for h0, h1 in self.holds:
            up = np.clip((s - (h0 - e)) / e, 0, 1)
            dn = np.clip(((h1 + e) - s) / e, 0, 1)
            ww = np.minimum(up, dn)
            ww = ww * ww * (3 - 2 * ww)
            w = np.maximum(w, ww)
        return w

    def speed_src(self, s: np.ndarray) -> np.ndarray:
        return self.v_base - (self.v_base - self.vmin) * self.weight(s)

    def src(self, tau) -> np.ndarray:
        return np.interp(tau, self.T, self.sg)

    def out(self, s) -> np.ndarray:
        return np.interp(s, self.sg, self.T)

    def frames(self) -> np.ndarray:
        return self.src(np.arange(self.n) / self.fps)

    def slowness(self, tau) -> np.ndarray:
        """0 at full speed .. 1 at vmin, per output time."""
        return self.weight(self.src(tau))


# ==================================================================== layout
def text_zone(occupied: Sequence[tuple[float, float]], H: int = 1920, text_h: float = 240,
              zones: Sequence[float] = (0.22, 0.74, 0.30, 0.66, 0.18, 0.78), gap: float = 12.0) -> Optional[float]:
    """Centre y of a text band (height ``text_h``) not overlapping any y-range in
    ``occupied`` (canvas px). Candidates stay out of the platform UI areas
    (top ~13 %, bottom ~20 %). Returns None when nothing is free."""
    best, best_d = None, -1.0
    for z in zones:
        yc = z * H
        a, b = yc - text_h / 2, yc + text_h / 2
        if a < 0.12 * H or b > 0.82 * H:
            continue
        d = min((max(o0 - b, a - o1) for o0, o1 in occupied), default=1e9)
        if d >= gap and d > best_d:
            best, best_d = yc, d
    if best is None:  # no preset band is free: scan the whole safe area
        for yc in np.arange(0.12 * H + text_h / 2, 0.82 * H - text_h / 2 + 1, 8.0):
            a, b = yc - text_h / 2, yc + text_h / 2
            d = min((max(o0 - b, a - o1) for o0, o1 in occupied), default=1e9)
            if d >= gap and d > best_d:
                best, best_d = float(yc), d
    return best
