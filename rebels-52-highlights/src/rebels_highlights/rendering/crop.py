"""Context-aware 9:16 crop trajectory.

Per source time the crop centre is a weighted midpoint of #52's box centre,
the target (ball carrier/QB) centre when known and the contact point near
impact (``render.context_weights``), hard-constrained so #52 stays inside the
central ``render.safe_region`` of the crop width. The path is smoothed with a
zero-phase (forward+backward) EMA and a max-velocity clamp so the virtual
camera neither lags nor jitters.

If #52 and the target cannot both fit a 9:16 window at impact the path switches
to mode ``split``: a wider foreground crop shown as large as possible over a
blurred/darkened background (composition done in :func:`compose_frame`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

OUT_W, OUT_H = 1080, 1920


# --------------------------------------------------------------------- helpers
def _get(obj: Any, name: str, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def interp_boxes(times: Sequence[float], boxes: Sequence[Sequence[float]], grid: np.ndarray,
                 hold: float = np.inf) -> tuple[np.ndarray, np.ndarray]:
    """Linear box interpolation over ``grid`` (gaps filled).

    Returns (boxes[N,4], valid[N]); outside the observed span values are held
    at the nearest observation and ``valid`` is False beyond ``hold`` seconds.
    """
    n = len(grid)
    out = np.full((n, 4), np.nan)
    valid = np.zeros(n, bool)
    if not times or not boxes:
        return out, valid
    t = np.asarray(times, float)
    b = np.asarray(boxes, float).reshape(-1, 4)
    order = np.argsort(t)
    t, b = t[order], b[order]
    for k in range(4):
        out[:, k] = np.interp(grid, t, b[:, k])
    valid = (grid >= t[0] - hold) & (grid <= t[-1] + hold)
    return out, valid


def zero_phase_ema(x: np.ndarray, alpha: float) -> np.ndarray:
    if len(x) < 2 or alpha >= 1:
        return x.copy()
    f = x.copy()
    for i in range(1, len(x)):
        f[i] = alpha * x[i] + (1 - alpha) * f[i - 1]
    b = x.copy()
    for i in range(len(x) - 2, -1, -1):
        b[i] = alpha * x[i] + (1 - alpha) * b[i + 1]
    return 0.5 * (f + b)


def clamp_velocity(x: np.ndarray, max_step: float) -> np.ndarray:
    y = x.copy()
    for i in range(1, len(y)):  # forward
        y[i] = np.clip(y[i], y[i - 1] - max_step, y[i - 1] + max_step)
    z = x.copy()
    for i in range(len(z) - 2, -1, -1):  # backward (removes lag)
        z[i] = np.clip(z[i], z[i + 1] - max_step, z[i + 1] + max_step)
    return 0.5 * (y + z)


# ------------------------------------------------------------------------ path
@dataclass
class CropPath:
    times: np.ndarray             # source-time grid
    cx: np.ndarray                # smoothed crop centre x (source px)
    cy: np.ndarray                # smoothed crop centre y (source px)
    px: np.ndarray                # #52 centre x
    py: np.ndarray
    crop_w: float                 # crop (foreground crop in split mode) size, source px
    crop_h: float
    src_w: int
    src_h: int
    mode: str = "vertical"        # vertical | split
    safe_region: float = 0.6
    quality: float = 1.0
    safe_fraction: float = 1.0
    target_fraction: float = 1.0
    spread_at_impact: float = 0.0
    review_reasons: list[str] = field(default_factory=list)

    # display geometry of the crop on the 1080x1920 canvas
    @property
    def display_size(self) -> tuple[int, int]:
        if self.mode == "vertical":
            return OUT_W, OUT_H
        return OUT_W, int(round(OUT_W * self.crop_h / self.crop_w / 2) * 2)

    def _i(self, arr: np.ndarray, t: float) -> float:
        return float(np.interp(t, self.times, arr))

    def box(self, t: float, zoom: float = 1.0) -> tuple[float, float, float, float]:
        """Crop box (x1, y1, x2, y2) in source px at source time ``t``."""
        zoom = max(1.0, zoom)
        w, h = self.crop_w / zoom, self.crop_h / zoom
        cx, cy = self._i(self.cx, t), self._i(self.cy, t)
        if zoom > 1.0:  # tighter window: re-centre so #52 stays in the safe zone
            px, py = self._i(self.px, t), self._i(self.py, t)
            half = self.safe_region * w / 2
            cx = float(np.clip(cx, px - half, px + half))
            cy = float(np.clip(cy, py - h * 0.3, py + h * 0.3))
        cx = float(np.clip(cx, w / 2, self.src_w - w / 2)) if w < self.src_w else self.src_w / 2
        cy = float(np.clip(cy, h / 2, self.src_h - h / 2)) if h < self.src_h else self.src_h / 2
        return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2

    def summary(self) -> dict:
        return {"mode": self.mode, "quality": round(self.quality, 3),
                "safe_fraction": round(self.safe_fraction, 3),
                "target_fraction": round(self.target_fraction, 3),
                "crop_w": round(self.crop_w, 1), "crop_h": round(self.crop_h, 1),
                "spread_at_impact": round(self.spread_at_impact, 1),
                "review_reasons": list(self.review_reasons)}


def max_zoom(cfg: dict, path: CropPath) -> float:
    """Replay zoom capped so total digital upscale stays <= render.max_digital_zoom."""
    rc = cfg.get("render", {})
    want = float(rc.get("replay_zoom", 1.3))
    disp_w, _ = path.display_size
    base_scale = disp_w / path.crop_w          # upscale already applied by the crop
    cap = float(rc.get("max_digital_zoom", 2.2)) / max(base_scale, 1e-6)
    return float(max(1.0, min(want, cap)))


def compute_crop_path(trajectory: Any, event: Any, src_w: int, src_h: int,
                      t_start: float, t_end: float, cfg: dict, fps: float = 30.0,
                      out_size: tuple[int, int] = (OUT_W, OUT_H)) -> CropPath:
    rc = cfg.get("render", {})
    safe = float(rc.get("safe_region", 0.6))
    smooth = float(rc.get("crop_smoothing", 0.85))
    wts = rc.get("context_weights", {}) or {}
    wp, wt, wc = (float(wts.get("player", 0.5)), float(wts.get("target", 0.3)),
                  float(wts.get("contact", 0.2)))
    ow, oh = out_size
    aspect = ow / oh
    if src_w / src_h > aspect:
        crop_h, crop_w = float(src_h), float(src_h * aspect)
    else:  # portrait-ish source: full width, vertical window tracks too
        crop_w, crop_h = float(src_w), float(min(src_h, src_w / aspect))

    t_end = max(t_end, t_start + 1.0 / fps)
    grid = np.arange(t_start, t_end + 0.5 / fps, 1.0 / fps)
    reasons: list[str] = []

    pb, pvalid = interp_boxes(_get(trajectory, "times", []) or [],
                              _get(trajectory, "boxes", []) or [], grid, hold=1.0)
    has_player = bool(np.any(pvalid))
    if not has_player:
        pb[:] = [src_w * 0.45, src_h * 0.4, src_w * 0.55, src_h * 0.6]
        reasons.append("POOR_VERTICAL_CROP")
    px, py = (pb[:, 0] + pb[:, 2]) / 2, (pb[:, 1] + pb[:, 3]) / 2

    tb, tvalid = interp_boxes(_get(trajectory, "target_times", []) or [],
                              _get(trajectory, "target_boxes", []) or [], grid, hold=0.25)
    tx, ty = (tb[:, 0] + tb[:, 2]) / 2, (tb[:, 1] + tb[:, 3]) / 2

    impact = _get(event, "impact_s")
    contact = _get(event, "contact_point")
    if contact is not None and impact is not None:
        wct = wc * np.exp(-0.5 * ((grid - float(impact)) / 0.6) ** 2)
        ccx, ccy = float(contact[0]), float(contact[1])
    else:
        wct = np.zeros_like(grid)
        ccx = ccy = 0.0

    # --- split decision: can #52 and the target share one 9:16 window at impact?
    spread = 0.0
    if impact is not None and np.any(tvalid):
        win = (np.abs(grid - float(impact)) <= 0.5) & tvalid
        if np.any(win):
            lo = np.minimum(pb[win, 0], tb[win, 0])
            hi = np.maximum(pb[win, 2], tb[win, 2])
            spread = float(np.max(hi - lo))
    mode = "vertical"
    if spread > 0.9 * crop_w and crop_w < src_w:
        mode = "split"
        max_fg_aspect = (oh * 0.62) / ow          # fg height/width on canvas
        crop_w = float(min(src_w, max(crop_w, spread * 1.15)))
        crop_h = float(min(src_h, crop_w * max_fg_aspect))
        if spread > 0.95 * src_w:
            reasons.append("POOR_VERTICAL_CROP")

    tx0, ty0 = np.nan_to_num(tx), np.nan_to_num(ty)
    # target only pulls the frame when it can share it with #52 (full pull when
    # within ~0.35 crop widths, none beyond ~0.85): far targets keep #52 centred
    dist = np.abs(tx0 - px)
    prox = np.clip(1.0 - (dist - 0.35 * crop_w) / (0.5 * crop_w), 0.0, 1.0)
    tw = wt * tvalid.astype(float) * prox
    den = wp + tw + wct
    dx = (wp * px + tw * tx0 + wct * ccx) / den
    dy = (wp * py + tw * ty0 + wct * ccy) / den

    half_safe = safe * crop_w / 2

    def constrain(cx: np.ndarray) -> np.ndarray:
        cx = np.clip(cx, px - half_safe, px + half_safe)
        return np.clip(cx, crop_w / 2, src_w - crop_w / 2) if crop_w < src_w else np.full_like(cx, src_w / 2)

    def constrain_y(cy: np.ndarray) -> np.ndarray:
        if crop_h >= src_h:
            return np.full_like(cy, src_h / 2)
        cy = np.clip(cy, py - crop_h * 0.3, py + crop_h * 0.3)
        return np.clip(cy, crop_h / 2, src_h - crop_h / 2)

    alpha = max(0.02, 1.0 - smooth)
    max_step = crop_w * 1.4 / fps                   # max pan speed: 1.4 crop widths / s
    cx = constrain(dx)
    cx = clamp_velocity(zero_phase_ema(cx, alpha), max_step)
    cx = constrain(zero_phase_ema(constrain(cx), 0.35))
    cy = constrain_y(dy)
    cy = constrain_y(clamp_velocity(zero_phase_ema(cy, alpha), max_step))

    # --- quality
    inside = np.abs(px - cx) <= half_safe + 1e-6
    safe_fraction = float(np.mean(inside[pvalid])) if has_player else 0.0
    target_fraction = 1.0
    if impact is not None and np.any(tvalid):
        win = (np.abs(grid - float(impact)) <= 0.5) & tvalid
        if np.any(win):
            target_fraction = float(np.mean(np.abs(tx[win] - cx[win]) <= crop_w / 2))
    quality = 0.7 * safe_fraction + 0.3 * target_fraction
    if mode == "split":
        quality *= 0.8
    if quality < 0.6 and "POOR_VERTICAL_CROP" not in reasons:
        reasons.append("POOR_VERTICAL_CROP")
    return CropPath(times=grid, cx=cx, cy=cy, px=px, py=py, crop_w=crop_w, crop_h=crop_h,
                    src_w=int(src_w), src_h=int(src_h), mode=mode, safe_region=safe,
                    quality=float(quality), safe_fraction=safe_fraction,
                    target_fraction=target_fraction, spread_at_impact=spread,
                    review_reasons=reasons)


# ------------------------------------------------------------------ composition
def _warp(frame: np.ndarray, box: tuple[float, float, float, float], size: tuple[int, int]):
    import cv2
    x1, y1, x2, y2 = box
    sx, sy = size[0] / (x2 - x1), size[1] / (y2 - y1)
    M = np.float32([[sx, 0, -x1 * sx], [0, sy, -y1 * sy]])
    interp = cv2.INTER_AREA if sx < 1 else cv2.INTER_LINEAR
    return cv2.warpAffine(frame, M, size, flags=interp, borderMode=cv2.BORDER_REPLICATE), (sx, sy)


def blurred_background(frame: np.ndarray, center_x: Optional[float] = None,
                       size: tuple[int, int] = (OUT_W, OUT_H), darken: float = 0.42) -> np.ndarray:
    import cv2
    H, W = frame.shape[:2]
    cw = min(W, H * size[0] / size[1])
    cx = W / 2 if center_x is None else float(np.clip(center_x, cw / 2, W - cw / 2))
    x1 = int(round(cx - cw / 2))
    region = frame[:, max(0, x1):max(0, x1) + int(cw)]
    small = cv2.resize(region, (size[0] // 8, size[1] // 8), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 4)
    bg = cv2.resize(small, size, interpolation=cv2.INTER_LINEAR)
    return (bg.astype(np.float32) * darken).astype(np.uint8)


@dataclass
class Xform:
    """Maps source px -> canvas px for one composed frame."""
    x1: float
    y1: float
    sx: float
    sy: float
    ox: float = 0.0
    oy: float = 0.0

    def __call__(self, x: float, y: float) -> tuple[float, float]:
        return (x - self.x1) * self.sx + self.ox, (y - self.y1) * self.sy + self.oy

    def box(self, b: Sequence[float]) -> list[float]:
        a = self(b[0], b[1])
        c = self(b[2], b[3])
        return [a[0], a[1], c[0], c[1]]


def compose_frame(frame: np.ndarray, path: CropPath, t: float, zoom: float = 1.0,
                  accent_bgr: tuple[int, int, int] = (0, 6, 225)) -> tuple[np.ndarray, Xform]:
    """Build the 1080x1920 canvas for source frame at time ``t``."""
    box = path.box(t, zoom)
    if path.mode == "vertical":
        out, (sx, sy) = _warp(frame, box, (OUT_W, OUT_H))
        return out, Xform(box[0], box[1], sx, sy)
    dw, dh = path.display_size
    canvas = blurred_background(frame, (box[0] + box[2]) / 2)
    fg, (sx, sy) = _warp(frame, box, (dw, dh))
    oy = int((OUT_H - dh) * 0.42)
    canvas[oy:oy + dh] = fg
    canvas[max(0, oy - 6):oy] = accent_bgr
    canvas[oy + dh:oy + dh + 6] = accent_bgr
    return canvas, Xform(box[0], box[1], sx, sy, 0.0, float(oy))


def letterbox_frame(frame: np.ndarray) -> np.ndarray:
    """Whole landscape frame fitted on a blurred background (untracked angles)."""
    import cv2
    H, W = frame.shape[:2]
    canvas = blurred_background(frame)
    dh = int(round(OUT_W * H / W / 2) * 2)
    dh = min(dh, OUT_H)
    fg = cv2.resize(frame, (OUT_W, dh), interpolation=cv2.INTER_AREA if W > OUT_W else cv2.INTER_LINEAR)
    oy = (OUT_H - dh) // 2
    canvas[oy:oy + dh] = fg
    return canvas
