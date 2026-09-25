"""PIL overlay drawing for vertical highlights.

Clean social/recruiting style: white bold sans, subtle dark translucent bars,
one accent colour (Rebels red by default). All ``draw_*`` functions draw onto
an RGBA ``PIL.Image`` layer the size of the output canvas; :func:`composite`
blends such a layer onto a BGR numpy frame (only inside the layer's bbox, so
per-frame overlays stay cheap).
"""
from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

DEFAULT_ACCENT = "#E10600"  # Rebels red
WHITE = (255, 255, 255)

_FONT_DIRS = ["/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/dejavu",
              "/usr/local/share/fonts", "/Library/Fonts", "C:/Windows/Fonts"]


# --------------------------------------------------------------------------- fonts
@lru_cache(maxsize=64)
def load_font(size: int, bold: bool = True) -> ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    for d in _FONT_DIRS:
        p = Path(d) / name
        if p.exists():
            return ImageFont.truetype(str(p), size)
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        pass
    try:  # Pillow >= 10.1 can scale its bundled default font
        return ImageFont.load_default(size=size)
    except TypeError:  # pragma: no cover - very old Pillow
        return ImageFont.load_default()


def hex_to_rgb(color: str | Sequence[int]) -> tuple[int, int, int]:
    if isinstance(color, str):
        c = color.lstrip("#")
        return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]
    return tuple(int(v) for v in color[:3])  # type: ignore[return-value]


def accent_from_cfg(cfg: Optional[dict]) -> tuple[int, int, int]:
    rc = (cfg or {}).get("render", {}) if cfg else {}
    return hex_to_rgb(rc.get("accent_color", DEFAULT_ACCENT))


def text_size(font, text: str) -> tuple[int, int]:
    x1, y1, x2, y2 = font.getbbox(text)
    return x2 - x1, y2 - y1


def wrap_text(text: str, font, max_w: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if text_size(font, trial)[0] <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def new_layer(w: int, h: int) -> Image.Image:
    return Image.new("RGBA", (w, h), (0, 0, 0, 0))


def _a(alpha: float, base: int = 255) -> int:
    return int(max(0.0, min(1.0, alpha)) * base)


# --------------------------------------------------------------------- compositing
def composite(frame_bgr: np.ndarray, layer: Image.Image, opacity: float = 1.0) -> np.ndarray:
    """Alpha-blend an RGBA layer onto a BGR frame in place (bbox-limited)."""
    bbox = layer.getbbox()
    if not bbox or opacity <= 0:
        return frame_bgr
    x1, y1, x2, y2 = bbox
    rgba = np.asarray(layer.crop(bbox), dtype=np.float32)
    a = rgba[..., 3:4] / 255.0 * opacity
    src = rgba[..., 2::-1]  # RGB -> BGR
    roi = frame_bgr[y1:y2, x1:x2].astype(np.float32)
    frame_bgr[y1:y2, x1:x2] = (roi * (1 - a) + src * a).astype(np.uint8)
    return frame_bgr


# ------------------------------------------------------------------- player marks
def draw_halo(layer: Image.Image, cx: float, foot_y: float, box_w: float,
              color=DEFAULT_ACCENT, alpha: float = 1.0) -> None:
    """Flat ellipse ring at the player's feet (broadcast 'spotlight' ring)."""
    d = ImageDraw.Draw(layer)
    rgb = hex_to_rgb(color)
    rx = max(22.0, box_w * 0.75)
    ry = rx * 0.32
    w = max(4, int(rx * 0.10))
    d.ellipse([cx - rx, foot_y - ry, cx + rx, foot_y + ry], outline=rgb + (_a(alpha, 90),),
              width=w + 6)
    d.ellipse([cx - rx, foot_y - ry, cx + rx, foot_y + ry], outline=rgb + (_a(alpha),), width=w)


def draw_arrow(layer: Image.Image, cx: float, head_y: float, color=DEFAULT_ACCENT,
               alpha: float = 1.0, size: float = 46.0, label: Optional[str] = "52") -> None:
    """Downward-pointing marker above the player's head with a small number tag."""
    d = ImageDraw.Draw(layer)
    rgb = hex_to_rgb(color)
    gap = size * 0.35
    tip = (cx, head_y - gap)
    top = head_y - gap - size
    tri = [(cx - size * 0.6, top), (cx + size * 0.6, top), tip]
    shadow = [(x + 3, y + 3) for x, y in tri]
    d.polygon(shadow, fill=(0, 0, 0, _a(alpha, 110)))
    d.polygon(tri, fill=rgb + (_a(alpha),), outline=WHITE + (_a(alpha),))
    if label:
        f = load_font(int(size * 0.62))
        tw, th = text_size(f, label)
        bx1, by2 = cx - tw / 2 - 12, top - 8
        by1 = by2 - th - 16
        d.rounded_rectangle([bx1, by1, cx + tw / 2 + 12, by2], radius=8,
                            fill=(15, 15, 15, _a(alpha, 200)))
        d.text((cx - tw / 2, by1 + 6), label, font=f, fill=WHITE + (_a(alpha),),
               anchor=None)


def draw_trail(layer: Image.Image, points: Sequence[tuple[float, float]], color=DEFAULT_ACCENT,
               alpha: float = 1.0, width: int = 10) -> None:
    """Fading polyline: oldest points transparent, newest most opaque."""
    if len(points) < 2:
        return
    d = ImageDraw.Draw(layer)
    rgb = hex_to_rgb(color)
    n = len(points)
    for i in range(1, n):
        f = i / (n - 1)
        a = _a(alpha * (0.15 + 0.7 * f))
        d.line([points[i - 1], points[i]], fill=rgb + (a,), width=max(2, int(width * (0.5 + 0.5 * f))))


# ------------------------------------------------------------------------ text bars
def draw_ident_bar(layer: Image.Image, title: str, subtitle: Optional[str] = None,
                   color=DEFAULT_ACCENT, alpha: float = 1.0, y_frac: float = 0.70) -> None:
    """Lower-third ident: dark translucent bar, accent stripe, bold white text."""
    W, H = layer.size
    d = ImageDraw.Draw(layer)
    rgb = hex_to_rgb(color)
    ft = load_font(max(26, int(W * 0.044)))
    fs = load_font(max(22, int(W * 0.036)), bold=False)
    tw, th = text_size(ft, title)
    sw, sh = text_size(fs, subtitle) if subtitle else (0, 0)
    pad = int(W * 0.035)
    bar_w = min(W - 2 * pad, max(tw, sw) + 3 * pad)
    bar_h = th + (sh + pad // 2 if subtitle else 0) + 2 * pad
    x1 = (W - bar_w) // 2
    y1 = int(H * y_frac)
    d.rounded_rectangle([x1, y1, x1 + bar_w, y1 + bar_h], radius=14, fill=(10, 10, 12, _a(alpha, 185)))
    d.rectangle([x1, y1, x1 + 12, y1 + bar_h], fill=rgb + (_a(alpha),))
    tx = x1 + 12 + (bar_w - 12 - tw) // 2
    d.text((tx, y1 + pad - ft.getbbox(title)[1]), title, font=ft, fill=WHITE + (_a(alpha),))
    if subtitle:
        sx = x1 + 12 + (bar_w - 12 - sw) // 2
        d.text((sx, y1 + pad + th + pad // 2 - fs.getbbox(subtitle)[1]), subtitle, font=fs,
               fill=(225, 225, 225, _a(alpha)))


def draw_corner_tag(layer: Image.Image, text: str, color=DEFAULT_ACCENT, alpha: float = 1.0,
                    corner: str = "tl") -> None:
    """Small pill tag (e.g. 'REPLAY' or '#52 | MLB') in a top corner, below IG/TikTok UI."""
    W, H = layer.size
    d = ImageDraw.Draw(layer)
    f = load_font(max(20, int(W * 0.034)))
    tw, th = text_size(f, text)
    pad = 16
    x1 = int(W * 0.05) if corner == "tl" else int(W * 0.95) - tw - 2 * pad - 10
    y1 = int(H * 0.13)
    d.rounded_rectangle([x1, y1, x1 + tw + 2 * pad + 10, y1 + th + 2 * pad], radius=10,
                        fill=(10, 10, 12, _a(alpha, 180)))
    d.rectangle([x1, y1, x1 + 8, y1 + th + 2 * pad], fill=hex_to_rgb(color) + (_a(alpha),))
    d.text((x1 + 10 + pad, y1 + pad - f.getbbox(text)[1]), text, font=f, fill=WHITE + (_a(alpha),))


def render_end_card(background_bgr: np.ndarray, caption: str, ident: str,
                    info_lines: Iterable[str] = (), color=DEFAULT_ACCENT) -> np.ndarray:
    """Blurred/darkened freeze + caption card. Returns a new BGR frame.

    Only text passed in is drawn - callers must not synthesise play labels.
    """
    import cv2
    H, W = background_bgr.shape[:2]
    small = cv2.resize(background_bgr, (W // 6, H // 6), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 3)
    bg = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)
    bg = (bg.astype(np.float32) * 0.38).astype(np.uint8)
    layer = new_layer(W, H)
    d = ImageDraw.Draw(layer)
    rgb = hex_to_rgb(color)
    pad = int(W * 0.08)
    fi = load_font(int(W * 0.050))
    fc = load_font(int(W * 0.068))
    fs = load_font(int(W * 0.040), bold=False)
    y = int(H * 0.36)
    # ident line + accent rule
    iw, ih = text_size(fi, ident)
    d.text(((W - iw) // 2, y), ident, font=fi, fill=WHITE + (255,))
    y += ih + 34
    d.rectangle([(W - 160) // 2, y, (W + 160) // 2, y + 8], fill=rgb + (255,))
    y += 8 + 48
    # A neutral caption ("#52 | MLB | BUCHAREST REBELS") repeats the ident line:
    # show only the part after the ident (the play label), if any.
    norm = lambda s: " ".join(s.upper().replace("•", "|").split())  # noqa: E731
    if caption and norm(ident).endswith(norm(caption.split(" — ")[0])):
        caption = caption.split(" — ", 1)[1] if " — " in caption else ""
    for line in wrap_text(caption, fc, W - 2 * pad)[:4] if caption else []:
        lw, lh = text_size(fc, line)
        d.text(((W - lw) // 2, y - fc.getbbox(line)[1]), line, font=fc, fill=WHITE + (255,))
        y += lh + 26
    y += 20
    for line in info_lines:
        if not line:
            continue
        lw, lh = text_size(fs, line)
        d.text(((W - lw) // 2, y - fs.getbbox(line)[1]), line, font=fs, fill=(215, 215, 215, 255))
        y += lh + 18
    return composite(bg, layer)


# --------------------------------------------------------------------------- debug
def draw_debug_box(layer: Image.Image, box: Sequence[float], color, label: str = "",
                   width: int = 2) -> None:
    d = ImageDraw.Draw(layer)
    rgb = hex_to_rgb(color)
    d.rectangle(list(box), outline=rgb + (255,), width=width)
    if label:
        f = load_font(14)
        tw, th = text_size(f, label)
        d.rectangle([box[0], box[1] - th - 6, box[0] + tw + 6, box[1]], fill=(0, 0, 0, 170))
        d.text((box[0] + 3, box[1] - th - 5), label, font=f, fill=rgb + (255,))


def smoothstep(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3 - 2 * x)


def fade_alpha(t: float, start: float, end: float, fade_in: float = 0.15,
               fade_out: float = 0.3) -> float:
    """Opacity of an element visible in [start, end] with soft edges."""
    if t < start or t > end + fade_out:
        return 0.0
    a = min(1.0, (t - start) / fade_in) if fade_in > 0 else 1.0
    if t > end:
        a = min(a, 1.0 - (t - end) / fade_out) if fade_out > 0 else 0.0
    return max(0.0, min(1.0, a if not math.isnan(a) else 0.0))
