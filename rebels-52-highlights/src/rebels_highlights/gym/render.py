"""Cinematic 9:16 render of an analysed gym clip.

Look: graded footage (S-curve, split tone, vignette, grain), athlete-following crop
(or blurred background + scaled foreground for wide landscape shots), a thin glowing
skeleton on the exercise's key limbs only, sleek glowing angle arcs with condensed
display numbers, a tapered light streak for the bar path, and metrics as kinetic text
that pops in at each rep's peak. Every highlighted rep gets a smooth speed ramp into
its bottom/peak position (frame-blended slow motion + small push-in + flash) instead of
a freeze. Intro/outro are type-only cards (no boxes)."""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from . import draw as D
from . import style as S
from .analysis import ARC_JOINTS, JOINT_LABEL, Analysis
from .config import hex_to_bgr
from .pose import KP
from .video import FrameReader, FrameWriter, grab_frame, mux_audio

log = logging.getLogger(__name__)

# key limbs drawn per exercise ("n" = camera-facing side, "f" = far side)
_LEG_N = [("n_ank", "n_knee"), ("n_knee", "n_hip")]
_LEG_F = [("f_ank", "f_knee"), ("f_knee", "f_hip")]
_TRUNK = [("n_hip", "n_sho")]
_ARM_N = [("n_sho", "n_elb"), ("n_elb", "n_wri")]
KEY_LIMBS = {
    "squat": _LEG_N + _TRUNK,
    "deadlift": _LEG_N + _TRUNK + _ARM_N,
    "bench": _TRUNK + _ARM_N,
    "overhead_press": _TRUNK + _ARM_N,
    "olympic": _LEG_N + _TRUNK + _ARM_N,
    "jump": _LEG_N + _LEG_F + _TRUNK,
    "sprint": _LEG_N + _LEG_F + _TRUNK + _ARM_N,
    "agility": _LEG_N + _LEG_F + _TRUNK,
    "lunge": _LEG_N + _LEG_F + _TRUNK,
    "pushup": _LEG_N + _TRUNK + _ARM_N,
    "generic": _LEG_N + _TRUNK + _ARM_N,
}
PEAK_NAME = {"squat": "BOTTOM", "deadlift": "BOTTOM", "bench": "BAR AT CHEST",
             "overhead_press": "LOCKOUT", "olympic": "BAR PEAK", "jump": "PEAK HEIGHT",
             "lunge": "BOTTOM", "pushup": "BOTTOM"}


# ---------------------------------------------------------------------------
# crop planning
# ---------------------------------------------------------------------------
def smooth_centers(c: np.ndarray, factor: float, fps: float = 30.0) -> np.ndarray:
    """Zero-phase (forward+backward) exponential smoothing; NaN gaps are interpolated."""
    c = np.asarray(c, float)
    ok = ~np.isnan(c)
    if not ok.any():
        return np.zeros_like(c)
    t = np.arange(len(c))
    x = np.interp(t, t[ok], c[ok])
    a = 1 - factor ** (30.0 / fps)
    f = x.copy()
    for i in range(1, len(f)):
        f[i] = f[i - 1] + a * (x[i] - f[i - 1])
    for i in range(len(f) - 2, -1, -1):
        f[i] = f[i + 1] + a * (f[i] - f[i + 1])
    return f


def clamp_window(c: np.ndarray, size: float, limit: float) -> np.ndarray:
    """Keep a window of ``size`` centred at ``c`` inside [0, limit]."""
    if size >= limit:
        return np.full_like(c, limit / 2)
    return np.clip(c, size / 2, limit - size / 2)


@dataclass
class CropPlan:
    mode: str                  # crop | fit
    win_w: float
    win_h: float
    cx: np.ndarray
    cy: np.ndarray
    out_w: int
    out_h: int
    fg: tuple = (0, 0, 0, 0)  # x, y, w, h of the window in the output
    scale: float = 1.0         # source px -> output px

    def origin(self, f: int) -> tuple[float, float]:
        f = int(np.clip(f, 0, len(self.cx) - 1))
        return self.cx[f] - self.win_w / 2, self.cy[f] - self.win_h / 2

    def map(self, pts: np.ndarray, f: int) -> np.ndarray:
        ox, oy = self.origin(f)
        p = np.asarray(pts, float)
        out = np.empty_like(p)
        out[..., 0] = (p[..., 0] - ox) * self.scale + self.fg[0]
        out[..., 1] = (p[..., 1] - oy) * self.scale + self.fg[1]
        return out

    def to_dict(self) -> dict:
        return {"mode": self.mode, "window": [round(self.win_w, 1), round(self.win_h, 1)],
                "fg": list(self.fg), "scale": round(self.scale, 4)}


def plan_crop(an: Analysis, W: int, H: int, out_w: int, out_h: int, cfg: dict) -> CropPlan:
    seq = an.seq
    k = seq.kp
    T = seq.T
    x1 = np.full(T, np.nan); x2 = np.full(T, np.nan); y1 = np.full(T, np.nan); y2 = np.full(T, np.nan)
    for f in range(T):
        pts = k[f][~np.isnan(k[f, :, 0])]
        extra = []
        if an.profile.bar_path and not np.isnan(an.bar[f, 0]):
            extra.append(an.bar[f])
        if not np.isnan(seq.box[f, 0]):
            b = seq.box[f]
            extra += [b[:2], b[2:]]
        if extra:
            pts = np.vstack([pts, np.array(extra)]) if len(pts) else np.array(extra)
        if len(pts) >= 2:
            x1[f], y1[f] = pts[:, 0].min(), pts[:, 1].min()
            x2[f], y2[f] = pts[:, 0].max(), pts[:, 1].max()
    ok = ~np.isnan(x1)
    vc = cfg["video"]
    aspect = out_w / out_h
    if not ok.any():
        wh = min(H, W / aspect)
        return CropPlan("crop", wh * aspect, wh, np.full(T, W / 2), np.full(T, H / 2),
                        out_w, out_h, (0, 0, out_w, out_h), out_h / wh)
    bw, bh = (x2 - x1)[ok], (y2 - y1)[ok]
    need_w = float(np.percentile(bw, 92)) * 1.30 + 0.04 * W
    need_h = float(np.percentile(bh, 92)) * 1.25 + 0.04 * H
    max_h = min(H, W / aspect)                       # tallest 9:16 window that fits
    ccx = (x1 + x2) / 2
    ccy = (y1 + y2) / 2
    if need_w <= max_h * aspect * 1.02:
        win_h = float(np.clip(max(need_h, need_w / aspect), H / vc["max_zoom"], max_h))
        win_w = win_h * aspect
        mode, fg = "crop", (0, 0, out_w, out_h)
        scale = out_h / win_h
    else:
        # wide landscape shot where body/bar doesn't fit 9:16: blurred bg + foreground
        fg_aspect = 1.0 if need_w / max(need_h, 1) > 1.0 else 4 / 5
        win_w = float(np.clip(max(need_w, need_h * fg_aspect), 0, W))
        win_h = win_w / fg_aspect
        if win_h > H:
            win_h = float(H)
            win_w = win_h * fg_aspect
        fw = out_w
        fh = int(round(fw / fg_aspect))
        fy = int(round((out_h - fh) * 0.46))
        mode, fg = "fit", (0, fy, fw, fh)
        scale = fw / win_w
    fac = vc["crop_smoothing"]
    cx = smooth_centers(ccx, fac, seq.fps)
    cy = smooth_centers(ccy, fac, seq.fps)
    # hard constraint: keep the athlete inside the window, then re-smooth lightly
    m = 0.04
    lo = x2 - win_w / 2 + m * win_w
    hi = x1 + win_w / 2 - m * win_w
    fixable = ok & (lo <= hi)
    cx[fixable] = np.clip(cx[fixable], lo[fixable], hi[fixable])
    lo_y = y2 - win_h / 2 + m * win_h
    hi_y = y1 + win_h / 2 - m * win_h
    fy_ok = ok & (lo_y <= hi_y)
    cy[fy_ok] = np.clip(cy[fy_ok], lo_y[fy_ok], hi_y[fy_ok])
    cx = smooth_centers(cx, 0.6, seq.fps)
    cy = smooth_centers(cy, 0.6, seq.fps)
    cx = clamp_window(cx, win_w, W)
    cy = clamp_window(cy, win_h, H)
    return CropPlan(mode, win_w, win_h, cx, cy, out_w, out_h, fg, scale)


# ---------------------------------------------------------------------------
# edit plan: trim + speed ramps
# ---------------------------------------------------------------------------
@dataclass
class OutFrame:
    pos: float                # work-frame position (fractional -> frame blend)
    mode: str                 # intro | body | ramp | outro
    rep: Optional[int] = None
    phase: float = 0.0        # intro/outro progress 0..1; ramp amount 0..1 for 'ramp'


@dataclass
class EditPlan:
    body: tuple[int, int]
    peak_reps: list[int]
    frames: list[OutFrame] = field(default_factory=list)
    audio: list[tuple[float, float, float]] = field(default_factory=list)
    duration_s: float = 0.0
    note: str = ""
    turn_out: dict = field(default_factory=dict)   # rep idx -> output frame at its turn


def ramp_speed(d_s: np.ndarray, half_w: float, vmin: float) -> np.ndarray:
    """Speed factor for a ramp centred at 0 (seconds): 1 -> vmin -> 1 (raised cosine
    with a short plateau at the bottom)."""
    d = np.abs(np.asarray(d_s, float))
    core = 0.18 * half_w
    t = np.clip((d - core) / max(half_w - core, 1e-6), 0, 1)
    return vmin + (1 - vmin) * (0.5 - 0.5 * np.cos(np.pi * t))


def ramp_extra_s(half_w: float, vmin: float) -> float:
    x = np.linspace(-half_w, half_w, 400)
    y = 1 / ramp_speed(x, half_w, vmin) - 1
    return float(np.sum((y[1:] + y[:-1]) / 2 * np.diff(x)))


def _active_window(an: Analysis, max_frames: int) -> tuple[int, int]:
    k = an.seq.kp
    v = np.nan_to_num(np.nansum(np.linalg.norm(np.diff(k, axis=0), axis=-1), axis=1))
    if len(v) <= max_frames:
        return 0, an.seq.T
    c = np.convolve(v, np.ones(max_frames), mode="valid")
    s = int(np.argmax(c))
    return s, s + max_frames


def plan_edit(an: Analysis, cfg: dict, trim_reps: Optional[tuple[int, int]] = None,
              peaks_override: Optional[list[int]] = None, slowmo: bool = True) -> EditPlan:
    ec = cfg["edit"]
    fps = an.fps
    T = an.seq.T
    reps = an.reps
    scores = an.summary.get("rep_scores") or [0.5] * len(reps)
    hw, vmin = ec["ramp_half_s"], ec["ramp_min_speed"]
    per_peak = ramp_extra_s(hw, vmin)
    max_body = ec["max_clip_s"] - ec["intro_s"] - ec["outro_s"]
    pre, post = int(0.8 * fps), int(0.6 * fps)
    mode = ec["peak_moments"] if slowmo else "none"

    def choose(idx: list[int]) -> list[int]:
        if mode == "none" or not idx:
            return []
        if mode == "all" or (mode == "auto" and len(idx) <= 5):
            return list(idx)
        return sorted(sorted(idx, key=lambda i: scores[i], reverse=True)[:3])

    note = ""
    if trim_reps:
        i, j = trim_reps
        a, b = max(0, reps[i].start - pre), min(T, reps[j].end + post)
        peaks = choose(list(range(i, j + 1)))
    elif reps:
        a, b = 0, T
        peaks = choose(list(range(len(reps))))
        if (b - a) / fps + len(peaks) * per_peak > max_body:
            best, best_sc = None, -1.0
            for i in range(len(reps)):
                for j in range(i, len(reps)):
                    a2, b2 = max(0, reps[i].start - pre), min(T, reps[j].end + post)
                    pk = choose(list(range(i, j + 1)))
                    if len(pk) > 3:
                        pk = sorted(sorted(pk, key=lambda q: scores[q], reverse=True)[:3])
                    d = (b2 - a2) / fps + len(pk) * per_peak
                    if d > max_body and j > i:
                        break
                    sc = sum(scores[i:j + 1]) + 0.01 * (j - i)
                    if sc > best_sc:
                        best, best_sc = (i, j, a2, b2, pk), sc
            i, j, a, b, peaks = best
            if (b - a) / fps + len(peaks) * per_peak > max_body:
                b = a + int((max_body - len(peaks) * per_peak) * fps)
                peaks = [p for p in peaks if reps[p].turn < b - int(hw * fps)]
            note = f"trimmed to reps {i + 1}-{j + 1} of {len(reps)}"
    else:
        a, b = _active_window(an, int(max_body * fps))
        peaks = []
        if b - a < T:
            note = "trimmed to most active window"
    if peaks_override is not None:
        peaks = peaks_override
    plan = EditPlan((int(a), int(b)), list(peaks), note=note)
    _expand(plan, an, cfg)
    return plan


def _expand(plan: EditPlan, an: Analysis, cfg: dict) -> None:
    ofps = cfg["video"]["out_fps"]
    wfps = an.fps
    ec = cfg["edit"]
    hw, vmin = ec["ramp_half_s"], ec["ramp_min_speed"]
    a, b = plan.body
    fr: list[OutFrame] = []
    n_in = int(round(ec["intro_s"] * ofps))
    fr += [OutFrame(a, "intro", None, k / max(1, n_in - 1)) for k in range(n_in)]
    turns = [(i, an.reps[i].turn) for i in plan.peak_reps]
    step = wfps / ofps
    p = float(a)
    body_start = len(fr)
    while p < b - 1e-6:
        v, rep, amt = 1.0, None, 0.0
        for i, t in turns:
            d = (p - t) / wfps
            if abs(d) < hw:
                vv = float(ramp_speed(d, hw, vmin))
                if vv < v:
                    v, rep = vv, i
                    amt = (1 - vv) / (1 - vmin) if vmin < 1 else 0.0
        fr.append(OutFrame(min(p, an.seq.T - 1), "ramp" if rep is not None else "body", rep, amt))
        if rep is not None and rep not in plan.turn_out and p >= an.reps[rep].turn:
            plan.turn_out[rep] = len(fr) - 1
        p += v * step
    n_out = int(round(ec["outro_s"] * ofps))
    last = fr[-1].pos if len(fr) > body_start else a
    fr += [OutFrame(last, "outro", None, k / max(1, n_out - 1)) for k in range(n_out)]
    # audio: intro silence, body in 0.1 s chunks at their mean speed, outro silence
    au: list[tuple[float, float, float]] = [(0.0, n_in / ofps, 0.0)]
    body = fr[body_start:len(fr) - n_out]
    ch = max(1, int(round(0.1 * ofps)))
    for k in range(0, len(body), ch):
        grp = body[k:k + ch]
        p0 = grp[0].pos
        p1 = body[k + ch].pos if k + ch < len(body) else min(float(b), grp[-1].pos + step)
        src_d = max(1e-3, (p1 - p0) / wfps)
        out_d = len(grp) / ofps
        au.append((p0 / wfps, src_d, src_d / out_d))
    au.append((0.0, n_out / ofps, 0.0))
    plan.frames = fr
    plan.audio = _merge_audio(au)
    plan.duration_s = len(fr) / ofps


def _merge_audio(au: list[tuple[float, float, float]]) -> list[tuple[float, float, float]]:
    """Merge contiguous chunks with (almost) equal speed to keep the ffmpeg graph small."""
    out: list[list[float]] = []
    for s, d, sp in au:
        if out:
            ps, pd, psp = out[-1]
            if psp == 0 and sp == 0:
                out[-1][1] += d
                continue
            if psp > 0 and sp > 0 and abs(psp - sp) < 0.04 and abs(ps + pd - s) < 0.02:
                tot_out = pd / psp + d / sp
                out[-1][1] = pd + d
                out[-1][2] = (pd + d) / tot_out
                continue
        out.append([s, d, sp])
    return [tuple(x) for x in out]


# ---------------------------------------------------------------------------
# drawing helpers (cinematic)
# ---------------------------------------------------------------------------
def _p16(p):
    return (int(round(p[0] * 16)), int(round(p[1] * 16)))


def _glow_layer(frame, bbox, draw_fn, strength=0.9, sigma=6.0):
    """draw_fn(layer, offset) into a black layer, blur (at 1/4 res), add as light."""
    H, W = frame.shape[:2]
    x0, y0, x1, y1 = (int(v) for v in bbox)
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if x1 - x0 < 6 or y1 - y0 < 6:
        return
    L = np.zeros((y1 - y0, x1 - x0, 3), np.uint8)
    draw_fn(L, np.array([x0, y0], float))
    sm = cv2.resize(L, (max(1, (x1 - x0) // 4), max(1, (y1 - y0) // 4)), interpolation=cv2.INTER_AREA)
    sm = cv2.GaussianBlur(sm, (0, 0), max(0.8, sigma / 4))
    L = cv2.resize(sm, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
    roi = frame[y0:y1, x0:x1]
    cv2.addWeighted(roi, 1.0, L, strength, 0, dst=roi)


def _tint(c, t=0.55):
    """Blend a colour towards white (hot core of a light stroke)."""
    return tuple(int(v + (255 - v) * t) for v in c)


def draw_limbs(frame, pts: list, sc: float, color):
    """pts: [(p0, p1, is_far)] in output px. Thin glowing strokes + small joint dots."""
    segs = [(a, b, far) for a, b, far in pts if not (np.isnan(a).any() or np.isnan(b).any())]
    if not segs:
        return
    allp = np.array([p for s in segs for p in s[:2]])
    pad = 40 * sc
    bbox = (allp[:, 0].min() - pad, allp[:, 1].min() - pad, allp[:, 0].max() + pad, allp[:, 1].max() + pad)
    th = max(2, int(round(2.6 * sc)))

    def g(L, off):
        for a, b, far in segs:
            cv2.line(L, _p16(a - off), _p16(b - off), color if not far else tuple(int(v * 0.5) for v in color),
                     th * 4, cv2.LINE_AA, shift=4)

    _glow_layer(frame, bbox, g, 0.55, 14 * sc)
    for a, b, far in segs:
        c = (255, 255, 255) if not far else (150, 150, 150)
        cv2.line(frame, _p16(a), _p16(b), c, th, cv2.LINE_AA, shift=4)
    seen = set()
    for a, b, far in segs:
        for p in (a, b):
            key = (int(p[0]), int(p[1]))
            if key in seen:
                continue
            seen.add(key)
            cv2.circle(frame, _p16(p), int(max(3, 4.5 * sc) * 16), (255, 255, 255) if not far else (150, 150, 150),
                       -1, cv2.LINE_AA, shift=4)


def draw_arc_glow(frame, a, b, c, value: float, frac: float, radius: float, sc: float, emph: float,
                  label: Optional[str] = None, small: bool = False):
    """Glowing arc stroke between rays b->a and b->c, value in the display face."""
    if any(np.isnan(v).any() for v in (a, b, c)):
        return
    va, vc = np.asarray(a) - b, np.asarray(c) - b
    if np.linalg.norm(va) < 1 or np.linalg.norm(vc) < 1:
        return
    col = D.ramp_color(frac)
    t1 = math.degrees(math.atan2(va[1], va[0]))
    t2 = math.degrees(math.atan2(vc[1], vc[0]))
    d = (t2 - t1 + 540) % 360 - 180
    r = radius * (1 + 0.25 * emph)
    th = max(2, int(round((2.4 + 2.0 * emph) * sc)))
    axes = (int(r * 16), int(r * 16))

    def g(L, off):
        cv2.ellipse(L, _p16(np.asarray(b) - off), axes, 0, t1, t1 + d, col, th * 5, cv2.LINE_AA, shift=4)

    _glow_layer(frame, (b[0] - r - 40, b[1] - r - 40, b[0] + r + 40, b[1] + r + 40), g, 0.8 + 0.6 * emph, 16 * sc)
    core = _tint(col, 0.45)
    cv2.ellipse(frame, _p16(b), axes, 0, t1, t1 + d, core, th, cv2.LINE_AA, shift=4)
    for t in (t1, t1 + d):     # end ticks
        rad = math.radians(t)
        p0 = (b[0] + math.cos(rad) * (r - 6 * sc), b[1] + math.sin(rad) * (r - 6 * sc))
        p1 = (b[0] + math.cos(rad) * (r + 6 * sc), b[1] + math.sin(rad) * (r + 6 * sc))
        cv2.line(frame, _p16(p0), _p16(p1), core, max(1, th // 2 + 1), cv2.LINE_AA, shift=4)
    mid = math.radians(t1 + d / 2)
    off = r + (34 + 14 * emph) * sc * (0.8 if small else 1.0)
    tx, ty = b[0] + math.cos(mid) * off, b[1] + math.sin(mid) * off
    size = int(round((50 + 22 * emph) * sc * (0.7 if small else 1.0)))
    spr = S.type_sprite(label or f"{value:.0f}°", max(12, size), (255, 255, 255), "display",
                        0.02, round(0.35 + 0.4 * emph, 1), col)
    S.blit(frame, spr, tx, ty, 1.0, "cm")


def draw_streak(frame, pts: np.ndarray, color, sc: float):
    """Tapered light streak (bar path): thin dim tail -> hot bright head."""
    ok = ~np.isnan(pts[:, 0])
    n = len(pts)
    if ok.sum() < 2:
        return
    q = pts[ok]
    bbox = (q[:, 0].min() - 40, q[:, 1].min() - 40, q[:, 0].max() + 40, q[:, 1].max() + 40)

    def seg(L, off, glow):
        for i in range(1, n):
            if ok[i] and ok[i - 1]:
                f = i / n
                th = max(1, int(round((0.6 + 4.5 * f ** 1.5) * sc * (3.2 if glow else 1))))
                base = color if glow else _tint(color, 0.35 + 0.5 * f)
                cc = tuple(int(v * (0.15 + 0.85 * f)) for v in base)
                cv2.line(L, _p16(pts[i - 1] - off), _p16(pts[i] - off), cc, th, cv2.LINE_AA, shift=4)

    _glow_layer(frame, bbox, lambda L, o: seg(L, o, True), 1.0, 14 * sc)
    Hf, Wf = frame.shape[:2]
    x0, y0 = int(max(0, bbox[0])), int(max(0, bbox[1]))
    x1, y1 = int(min(Wf, bbox[2])), int(min(Hf, bbox[3]))
    if x1 > x0 and y1 > y0:
        sub = np.zeros((y1 - y0, x1 - x0, 3), np.uint8)
        seg(sub, np.array([x0, y0], float), False)
        roi = frame[y0:y1, x0:x1]
        cv2.add(roi, sub, dst=roi)                               # additive core
    head = q[-1]
    _glow_layer(frame, (head[0] - 50, head[1] - 50, head[0] + 50, head[1] + 50),
                lambda L, o: cv2.circle(L, _p16(head - o), int(12 * sc) * 16, color, -1, cv2.LINE_AA, shift=4),
                1.2, 18 * sc)
    cv2.circle(frame, _p16(head), int(max(3, 5 * sc)) * 16, (255, 255, 255), -1, cv2.LINE_AA, shift=4)


class Spark:
    """Boxless glowing angle-over-time line along the bottom (sits on the scrim)."""

    def __init__(self, v: np.ndarray, x0, y0, w, h, accent, marks, title: str, unit: str):
        self.v = np.asarray(v, float)
        self.n = len(self.v)
        vv = self.v[~np.isnan(self.v)]
        self.lo, self.hi = (float(vv.min()), float(vv.max())) if len(vv) else (0.0, 1.0)
        if self.hi - self.lo < 5:
            self.hi = self.lo + 5
        self.x0, self.y0, self.w, self.h = x0, y0, w, h
        self.accent, self.marks, self.title, self.unit = accent, marks, title, unit
        pts = self._pts(np.arange(self.n))
        L = np.zeros((h + 40, w + 40, 3), np.uint8)
        ok = ~np.isnan(pts[:, 1])
        off = np.array([x0 - 20, y0 - 20], float)
        for i in range(1, self.n):
            if ok[i] and ok[i - 1]:
                cv2.line(L, _p16(pts[i - 1] - off), _p16(pts[i] - off), (150, 150, 150), 2, cv2.LINE_AA, shift=4)
        self.layer = L

    def _pts(self, idx):
        idx = np.asarray(idx)
        x = self.x0 + self.w * idx / max(1, self.n - 1)
        y = self.y0 + self.h - self.h * (self.v[idx] - self.lo) / (self.hi - self.lo)
        return np.stack([x, y], -1)

    def draw(self, frame, i: int, u: float, alpha: float = 1.0):
        x0, y0 = self.x0 - 20, self.y0 - 20
        h, w = self.layer.shape[:2]
        Hf, Wf = frame.shape[:2]
        if y0 < 0 or x0 < 0 or y0 + h > Hf or x0 + w > Wf:
            return
        roi = frame[y0:y0 + h, x0:x0 + w]
        cv2.addWeighted(roi, 1.0, self.layer, 0.8 * alpha, 0, dst=roi)
        i = int(np.clip(i, 0, self.n - 1))
        pts = self._pts(np.arange(0, i + 1))
        ok = ~np.isnan(pts[:, 1])
        for k in range(1, i + 1):
            if ok[k] and ok[k - 1]:
                cv2.line(frame, _p16(pts[k - 1]), _p16(pts[k]), self.accent, 3, cv2.LINE_AA, shift=4)
        p = self._pts([i])[0]
        if not np.isnan(p[1]):
            _glow_layer(frame, (p[0] - 30, p[1] - 30, p[0] + 30, p[1] + 30),
                        lambda L, o: cv2.circle(L, _p16(p - o), 9 * 16, self.accent, -1, cv2.LINE_AA, shift=4),
                        1.2, 10)
            cv2.circle(frame, _p16(p), 5 * 16, (255, 255, 255), -1, cv2.LINE_AA, shift=4)
        S.blit(frame, S.type_sprite(self.title, int(26 * u), (190, 190, 190), "display", 0.08, 0, None, 0.4),
               self.x0, self.y0 - 50 * u, alpha * 0.9)


# ---------------------------------------------------------------------------
# kinetic text events
# ---------------------------------------------------------------------------
@dataclass
class Pop:
    start: int                 # output frame
    hold_s: float
    lines: list                # [(text, size, color, face, tracking)]
    zone: str = "upper"        # upper | lower
    side: Optional[str] = None  # left | right (None = opposite the athlete)


def build_pops(an: Analysis, plan: EditPlan, cfg: dict, accent) -> list[Pop]:
    frames = plan.frames
    unit = an.extras.get("speed_unit")
    pops: list[Pop] = []
    W = (255, 255, 255)
    G = (185, 185, 185)
    body_idx = [(k, of.pos) for k, of in enumerate(frames) if of.mode in ("body", "ramp")]

    def out_idx(work_f: float) -> Optional[int]:
        for k, pos in body_idx:
            if pos >= work_f:
                return k
        return None

    a, b = plan.body
    if an.exercise in ("sprint", "agility"):
        stats = []
        ex = an.extras
        if an.exercise == "sprint":
            if ex.get("cadence_steps_per_s"):
                stats.append((f"{ex['cadence_steps_per_s']:.1f}", "STEPS / SEC · CADENCE"))
            if ex.get("knee_drive_hip_deg"):
                stats.append((f"{ex['knee_drive_hip_deg']:.0f}°", "KNEE DRIVE · HIP ANGLE"))
            if ex.get("trunk_lean_deg") is not None:
                stats.append((f"{ex['trunk_lean_deg']:.0f}°", "TRUNK LEAN"))
        else:
            if ex.get("direction_changes") is not None:
                stats.append((str(ex["direction_changes"]), "DIRECTION CHANGES"))
            if ex.get("stance_knee_deg"):
                stats.append((f"{ex['stance_knee_deg']:.0f}°", "STANCE KNEE ANGLE"))
            if ex.get("lateral_speed_peak_mps_est"):
                stats.append((f"{ex['lateral_speed_peak_mps_est']:.1f}", "M/S LATERAL · EST."))
        for i, (big, lab) in enumerate(stats[:3]):
            k = out_idx(a + (b - a) * (0.15 + 0.28 * i))
            if k is not None:
                pops.append(Pop(k, 1.8, [(lab, 34, accent, "display", 0.1), (big, 170, W, "display", 0.0)],
                                "upper" if i % 2 == 0 else "lower"))
        return pops

    for i, r in enumerate(an.reps):
        if not (a <= r.turn < b):
            continue
        big = i in plan.peak_reps
        k = plan.turn_out.get(i) or out_idx(r.turn)
        if k is None:
            continue
        if an.exercise == "jump":
            h = r.extra.get("jump_height_m_est")
            if h is None:
                continue
            lines = [(f"JUMP {r.index} · HEIGHT EST.", 34, accent, "display", 0.1),
                     (f"{h * 100:.0f} CM", 170 if big else 110, W, "display", 0.0),
                     (f"{r.extra.get('flight_s', 0):.2f} S IN THE AIR", 40, G, "display", 0.06)]
            pops.append(Pop(max(0, k - 4), 2.2 if big else 1.4, lines, "upper"))
            continue
        first = an.profile.arcs[0] if an.profile.arcs else an.primary_name
        av = r.extra.get(f"{first}_at_turn_deg")
        jl = JOINT_LABEL.get(first, first.upper())
        lines = [(f"REP {r.index} · {PEAK_NAME.get(an.exercise, 'PEAK')}", 34, accent, "display", 0.1)]
        if av is not None and an.primary_name != "bar_y":
            lines.append((f"{av:.0f}°", 190 if big else 120, W, "display", 0.0))
            lines.append((f"{jl} ANGLE  ·  ROM {r.rom_deg:.0f}°", 40, G, "display", 0.06))
        else:
            lines.append((f"ROM {r.rom_deg:.0f}°", 150 if big else 100, W, "display", 0.0))
        if "below_parallel" in r.extra:
            lines.append(("BELOW PARALLEL" if r.extra["below_parallel"] else "ABOVE PARALLEL", 38,
                          accent if r.extra["below_parallel"] else G, "display", 0.1))
        pops.append(Pop(max(0, k - 3), 2.0 if big else 1.2, lines, "upper"))
        v = r.extra.get("bar_peak_up_speed")
        if big and v is not None and unit == "m/s":
            con = list(range(r.turn, r.end + 1)) if an.profile.polarity == "valley" else list(range(r.start, r.turn + 1))
            seg = an.bar_vy[con] if con else np.array([])
            if len(seg) and np.any(~np.isnan(seg)):
                kf = out_idx(con[int(np.nanargmax(seg))])
                if kf is not None:
                    pops.append(Pop(kf, 1.5, [(f"{v:.2f}", 150, W, "display", 0.0),
                                              ("M/S PEAK BAR SPEED · EST.", 34, accent, "display", 0.1),
                                              (f"DOWN {r.ecc_s:.1f}S  /  UP {r.con_s:.1f}S", 34, G, "display", 0.06)],
                                    "lower"))
        elif big:
            ke = out_idx(r.end)
            if ke is not None:
                pops.append(Pop(ke, 1.2, [(f"DOWN {r.ecc_s:.1f}S  /  UP {r.con_s:.1f}S", 46, W, "display", 0.06),
                                          ("TEMPO", 30, accent, "display", 0.1)], "lower"))
    return pops


# ---------------------------------------------------------------------------
# composer
# ---------------------------------------------------------------------------
class Composer:
    def __init__(self, an: Analysis, crop: CropPlan, cfg: dict, plan: EditPlan, decode_scale: float,
                 sparkline: bool = True, panel: bool = True, header: bool = True, grade: bool = True):
        self.an, self.crop, self.cfg, self.plan = an, crop, cfg, plan
        self.ds = decode_scale
        st = cfg["style"]
        D.set_fonts(st.get("font"), st.get("font_regular"))
        self.accent = hex_to_bgr(st["accent"])
        self.W, self.H = crop.out_w, crop.out_h
        self.u = self.W / 1080.0
        body_out = (an.body_px or an.seq.height * 0.6) * crop.scale
        self.draw_scale = float(np.clip(body_out / 900.0, 0.55, 1.35)) * max(self.u, 0.5)
        self.arc_r = float(np.clip(0.07 * body_out, 28, 85))
        self.trail_n = int(st["trail_s"] * an.fps)
        self.header = header
        self.panel = panel          # rep counter + kinetic pops
        self.grade = S.make_grade(self.W, self.H, st) if grade else None
        a, b = plan.body
        self.spark = None
        if sparkline and cfg["edit"]["sparkline"] and b - a > 5:
            vals = an.primary[a:b]
            if np.isnan(vals).mean() < 0.7:
                marks = [r.turn - a for r in an.reps if a <= r.turn < b]
                unit = "%" if an.primary_name in ("bar_y", "hip_y") else "°"
                title = JOINT_LABEL.get(an.primary_name, an.primary_name.upper()) + " · LIVE"
                u = self.u
                self.spark = Spark(vals, int(80 * u), int(self.H - 300 * u), int(920 * u), int(110 * u),
                                   self.accent, marks, title, unit)
        if self.grade is not None:
            self.grade.add_scrim(top=0.45 if header else 0.0, bottom=0.55 if self.spark else 0.25)
        self.pops = build_pops(an, plan, cfg, self.accent) if panel else []
        self.flash_at = sorted(plan.turn_out.values())
        self.near = an.side[0]
        self.far = "r" if self.near == "l" else "l"
        self._cur_zoom = (1.0, self.W / 2, self.H / 2)

    # -- base image --------------------------------------------------------------------
    def base(self, img: np.ndarray, f: int, of: Optional[OutFrame] = None) -> np.ndarray:
        c = self.crop
        ox, oy = c.origin(f)
        ds = self.ds
        if c.mode == "crop":
            out = self._crop_resize(img, ox * ds, oy * ds, c.win_w * ds, c.win_h * ds)
        else:
            x, y, w, h = c.fg
            s = c.scale / ds
            M = np.float32([[s, 0, -ox * ds * s], [0, s, -oy * ds * s]])
            out = D.blur_background(img, self.W, self.H, 0.35)
            fg = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            if not hasattr(self, "_fg_w"):
                feather = max(4, int(28 * self.u))
                ramp = np.ones(h, np.float32)
                e = np.linspace(0, 1, feather, dtype=np.float32)
                ramp[:feather] = e
                ramp[-feather:] = e[::-1]
                self._fg_w = np.ascontiguousarray(np.repeat(ramp[:, None], w, 1))
                self._bg_w = np.ascontiguousarray(1 - self._fg_w)
            roi = out[y:y + h, x:x + w]
            out[y:y + h, x:x + w] = cv2.blendLinear(fg, roi, self._fg_w, self._bg_w)
        z = 1.0
        if of is not None and of.mode == "ramp":
            z = 1.0 + self.cfg["edit"]["ramp_zoom"] * S.ease_out_cubic(of.phase)
        zc = (self.W / 2, self.H / 2)
        hip = self.an.seq.kp[f, [KP["l_hip"], KP["r_hip"]]]
        if not np.isnan(hip).all():
            m = c.map(np.nanmean(hip, 0), f)
            zc = (float(m[0]), float(m[1]))
        self._cur_zoom = (z, zc[0], zc[1])
        if z > 1.001:
            out = S.punch_zoom(out, z, zc[0], zc[1])
        if self.grade is not None:
            out = self.grade.apply(out)
        return out

    def _crop_resize(self, img, x, y, w, h):
        H, W = img.shape[:2]
        x0 = int(np.clip(round(x), 0, max(0, W - round(w))))
        y0 = int(np.clip(round(y), 0, max(0, H - round(h))))
        x1, y1 = min(W, x0 + int(round(w))), min(H, y0 + int(round(h)))
        sub = img[y0:y1, x0:x1]
        interp = cv2.INTER_AREA if (x1 - x0) > self.W else cv2.INTER_LINEAR
        return cv2.resize(sub, (self.W, self.H), interpolation=interp)

    def map(self, pts: np.ndarray, f: int) -> np.ndarray:
        p = self.crop.map(pts, f)
        z, zx, zy = self._cur_zoom
        if z > 1.001:
            p = p.copy()
            p[..., 0] = (p[..., 0] - zx) * z + zx
            p[..., 1] = (p[..., 1] - zy) * z + zy
        return p

    # -- overlays -----------------------------------------------------------------------
    def overlays(self, out: np.ndarray, f: int, of: OutFrame, k_out: Optional[int] = None) -> None:
        an = self.an
        kp = self.map(an.seq.kp[f], f)
        sc = self.draw_scale
        emph = of.phase if of.mode == "ramp" else 0.0
        n, fa = self.near, self.far
        limbs = []
        for p0, p1 in KEY_LIMBS.get(an.exercise, KEY_LIMBS["generic"]):
            s0, j0 = p0.split("_", 1)
            s1, j1 = p1.split("_", 1)
            i0 = KP[f"{n if s0 == 'n' else fa}_{j0}"]
            i1 = KP[f"{n if s1 == 'n' else fa}_{j1}"]
            limbs.append((kp[i0], kp[i1], s0 == "f" and s1 == "f"))
        draw_limbs(out, limbs, sc, self.accent)
        if an.profile.bar_path:
            s0 = max(self.plan.body[0], f - self.trail_n)
            draw_streak(out, self.map(an.bar[s0:f + 1], f), self.accent, sc)
        for j in an.profile.arcs:
            if j == "trunk":
                hip, sho = kp[KP[f"{n}_hip"]], kp[KP[f"{n}_sho"]]
                v = an.angles["trunk"][f]
                if np.isnan(v) or np.isnan(hip).any() or np.isnan(sho).any():
                    continue
                L = self.arc_r * 1.7
                D.draw_vertical_ref(out, hip, L, sc * 0.8, (170, 170, 170))
                lo, hi = an.arc_ranges.get("trunk", (0, 90))
                draw_arc_glow(out, hip + np.array([0, -L]), hip, sho, v, (v - lo) / max(hi - lo, 1),
                              self.arc_r * 1.7, sc, emph * 0.5, small=True)
                continue
            a_, b_, c_ = (kp[KP[f"{n}_{nm}"]] for nm in ARC_JOINTS[j])
            v = (an.angles.get("body_line", np.full(an.seq.T, np.nan))[f] if j == "body_line"
                 else an.angles[f"{j}_{n}"][f])
            if np.isnan(v):
                continue
            lo, hi = an.arc_ranges.get(j, (0, 180))
            prim = j == an.profile.arcs[0]
            draw_arc_glow(out, a_, b_, c_, v, (hi - v) / max(hi - lo, 1), self.arc_r * (1 if prim else 0.8), sc,
                          emph if prim else emph * 0.5, small=not prim)
        u = self.u
        if self.header:
            S.blit(out, S.type_sprite(self.cfg["athlete"]["tag"], int(34 * u), (255, 255, 255), "display", 0.08,
                                      0, None, 0.5), 56 * u, 96 * u)
            cv2.line(out, (int(60 * u), int(166 * u)), (int(120 * u), int(166 * u)), self.accent,
                     max(2, int(4 * u)), cv2.LINE_AA)
            if an.label_visible:
                S.blit(out, S.type_sprite(an.profile.label, int(30 * u), (190, 190, 190), "display", 0.12,
                                          0, None, 0.5), 134 * u, 166 * u, 1.0, "lm")
        if self.panel and an.reps and an.exercise not in ("sprint", "agility"):
            nrep, _ = an.rep_at(f)
            S.blit(out, S.type_sprite(f"{nrep:02d}", int(96 * u), (255, 255, 255), "display", 0.0, 0.3,
                                      self.accent), self.W - 128 * u, 70 * u, 1.0, "rt")
            S.blit(out, S.type_sprite(f"/{len(an.reps):02d}", int(40 * u), (170, 170, 170), "display"),
                   self.W - 56 * u, 104 * u, 1.0, "rt")
            S.blit(out, S.type_sprite("REPS" if an.exercise != "jump" else "JUMPS", int(24 * u), self.accent,
                                      "display", 0.2), self.W - 58 * u, 150 * u, 1.0, "rt")
        if self.spark is not None:
            self.spark.draw(out, f - self.plan.body[0], u)
        if k_out is not None and self.panel:
            self._draw_pops(out, k_out, kp)

    def _draw_pops(self, out, k: int, kp: np.ndarray) -> None:
        ofps = self.cfg["video"]["out_fps"]
        u = self.u
        hips = kp[[KP["l_hip"], KP["r_hip"], KP["l_sho"], KP["r_sho"]]]
        ax = float(np.nanmean(hips[:, 0])) if not np.isnan(hips[:, 0]).all() else self.W / 2
        side = "left" if ax > self.W / 2 else "right"
        for p in self.pops:
            age = (k - p.start) / ofps
            alpha, scl, slide = S.kinetic_alpha_scale(age, p.hold_s)
            if alpha <= 0.01:
                continue
            sd = p.side or side
            x = 64 * u if sd == "left" else self.W - 64 * u
            ax_ = "l" if sd == "left" else "r"
            y = (300 if p.zone == "upper" else 1240) * u + slide * 40 * u
            for text, size, col, face, trk in p.lines:
                spr = S.type_sprite(text, int(size * u), tuple(col), face, trk,
                                    0.25 if size >= 100 else 0.0, self.accent)
                bx = S.blit(out, spr, x, y, alpha, ax_ + "t", scl if size >= 100 else 1.0)
                y += bx[3] * 0.80

    # -- cards -----------------------------------------------------------------------------
    def _card_bg(self, bg: Optional[np.ndarray], darken: float) -> np.ndarray:
        key = (id(bg), darken)
        if getattr(self, "_bg_key", None) != key:
            self._bg_key = key
            b = (D.blur_background(bg, self.W, self.H, darken) if bg is not None
                 else np.zeros((self.H, self.W, 3), np.uint8))
            self._bg = self.grade.apply(b, grain=False) if self.grade is not None else b
        out = self._bg.copy()
        return self.grade.add_grain(out) if self.grade is not None else out

    def intro(self, bg: Optional[np.ndarray], of: OutFrame) -> np.ndarray:
        out = self._card_bg(bg, 0.55)
        u = self.u
        an = self.an
        ph = of.phase
        a1, s1, sl1 = S.kinetic_alpha_scale(ph * self.cfg["edit"]["intro_s"], 10)
        title = an.profile.label if an.label_visible else "TRAINING"
        cy = self.H * 0.40
        spr = S.type_sprite(title, int(170 * u), (255, 255, 255), "display", 0.01, 0.3, self.accent)
        if spr.shape[1] > self.W * 0.9:
            spr = cv2.resize(spr, (int(self.W * 0.9), int(spr.shape[0] * self.W * 0.9 / spr.shape[1])),
                             interpolation=cv2.INTER_AREA)
        S.blit(out, spr, self.W / 2, cy + sl1 * 50 * u, a1, "cm", s1)
        e2 = S.ease_out_cubic((ph - 0.15) * 2.5)
        bw = 300 * u * e2
        cv2.line(out, (int(self.W / 2 - bw / 2), int(cy + 115 * u)), (int(self.W / 2 + bw / 2), int(cy + 115 * u)),
                 self.accent, max(2, int(5 * u)), cv2.LINE_AA)
        S.blit(out, S.type_sprite(self.cfg["athlete"]["tag"], int(48 * u), (255, 255, 255), "display", 0.1),
               self.W / 2, cy + 150 * u, e2, "ct")
        S.blit(out, S.type_sprite("MOVEMENT BREAKDOWN", int(30 * u), (170, 170, 170), "display", 0.3),
               self.W / 2, cy - 150 * u, S.ease_out_cubic((ph - 0.3) * 3), "ct")
        name = self.cfg["athlete"].get("name")
        if name:
            S.blit(out, S.type_sprite(name.upper(), int(40 * u), (200, 200, 200), "display", 0.1),
                   self.W / 2, cy + 215 * u, e2, "ct")
        return S.flash(out, max(0.0, 1 - ph * 6) * 0.6)

    def outro(self, bg: Optional[np.ndarray], of: OutFrame) -> np.ndarray:
        out = self._card_bg(bg, 0.72)
        u = self.u
        rows = summary_rows(self.an)
        t = of.phase * self.cfg["edit"]["outro_s"]
        y = self.H * 0.12
        S.blit(out, S.type_sprite("SESSION", int(34 * u), self.accent, "display", 0.3), 90 * u, y,
               S.ease_out_cubic(t * 4))
        y += 60 * u
        for i, (lab, val) in enumerate(rows[:5]):
            a, s, sl = S.kinetic_alpha_scale(t - 0.12 - 0.13 * i, 100)
            if a > 0:
                vs = S.type_sprite(val, int(118 * u), (255, 255, 255), "display", 0.0, 0.2, self.accent)
                bx = S.blit(out, vs, 90 * u - sl * 60 * u, y, a, "lt", s)
                S.blit(out, S.type_sprite(lab, int(30 * u), (175, 175, 175), "display", 0.12), 96 * u,
                       y + bx[3] * 0.80, a)
            y += 222 * u
        e = S.ease_out_cubic(t * 2)
        S.blit(out, S.type_sprite("ESTIMATES FROM 2D PHONE VIDEO", int(24 * u), (140, 140, 140), "display", 0.2),
               90 * u, self.H - 330 * u, e)
        S.blit(out, S.type_sprite(self.cfg["athlete"]["tag"], int(46 * u), (255, 255, 255), "display", 0.08),
               90 * u, self.H - 290 * u, e)
        return out


def summary_rows(an: Analysis) -> list[tuple[str, str]]:
    s = an.summary
    rows = []
    if s.get("reps"):
        rows.append(("REPS" if an.exercise != "jump" else "JUMPS", str(s["reps"])))
    if s.get("best_jump_height_m_est"):
        rows.append(("BEST JUMP · EST.", f"{s['best_jump_height_m_est'] * 100:.0f} CM"))
    if s.get("best_rom_deg") and an.exercise != "jump":
        rows.append(("BEST RANGE OF MOTION", f"{s['best_rom_deg']:.0f}°"))
    if s.get("below_parallel_reps"):
        rows.append(("REPS BELOW PARALLEL", s["below_parallel_reps"]))
    if s.get("peak_bar_speed_mps_est"):
        rows.append(("PEAK BAR SPEED · M/S · EST.", f"{s['peak_bar_speed_mps_est']:.2f}"))
    if s.get("tempo_ecc_s") is not None and an.exercise != "jump":
        rows.append(("TEMPO · DOWN / UP", f"{s['tempo_ecc_s']:.1f}S / {s['tempo_con_s']:.1f}S"))
    if s.get("steps"):
        rows.append(("STEPS", str(s["steps"])))
    if s.get("cadence_steps_per_s"):
        rows.append(("CADENCE · STEPS / SEC", f"{s['cadence_steps_per_s']:.1f}"))
    if s.get("knee_drive_hip_deg"):
        rows.append(("KNEE DRIVE · HIP ANGLE", f"{s['knee_drive_hip_deg']:.0f}°"))
    if s.get("direction_changes") is not None:
        rows.append(("DIRECTION CHANGES", str(s["direction_changes"])))
    if s.get("stance_knee_deg"):
        rows.append(("STANCE KNEE ANGLE", f"{s['stance_knee_deg']:.0f}°"))
    sym = s.get("symmetry_pct_est")
    if sym is not None:
        rows.append(("LEFT / RIGHT SYMMETRY · EST.", f"{sym:.0f}%"))
    return rows


# ---------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------
def decode_scale_for(crop: CropPlan) -> float:
    """Decode at the resolution the output needs (+ headroom for the ramp push-in)."""
    return float(min(1.0, crop.scale * 1.25))


def render_clip(path: Path, info: dict, an: Analysis, cfg: dict, out_dir: Path, stem: str,
                plan: Optional[EditPlan] = None, slowmo: bool = True) -> dict:
    t0 = time.time()
    vc = cfg["video"]
    OW, OH = vc["width"], vc["height"]
    W, H = info["disp_w"], info["disp_h"]
    crop = plan_crop(an, W, H, OW, OH, cfg)
    plan = plan or plan_edit(an, cfg, slowmo=slowmo)
    ds = decode_scale_for(crop)
    dw, dh = int(round(W * ds / 2) * 2), int(round(H * ds / 2) * 2)
    ds = dw / W
    comp = Composer(an, crop, cfg, plan, ds)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / f".{stem}_video.mp4"
    final = out_dir / f"{stem}_gym.mp4"
    thumb = out_dir / f"{stem}_gym.jpg"
    a, b = plan.body
    wfps = an.fps
    first = grab_frame(path, a / wfps)
    first = cv2.resize(first, (dw, dh), interpolation=cv2.INTER_AREA) if first is not None else None
    bg_first = comp.base(first, a) if first is not None else None
    writer = FrameWriter(tmp, OW, OH, vc["out_fps"], vc["crf"], vc["preset"])
    reader = iter(FrameReader(path, wfps, (dw, dh), start_s=a / wfps, duration_s=(b - a + 2) / wfps))
    buf: dict[int, np.ndarray] = {}
    last_read = a - 1
    thumb_img, thumb_score = None, -1.0
    last_base = bg_first
    best_rep = None
    if an.reps and an.summary.get("rep_scores"):
        cands = list(plan.peak_reps) or list(range(len(an.reps)))
        best_rep = max(cands, key=lambda i: an.summary["rep_scores"][i])

    def get(i: int) -> Optional[np.ndarray]:
        nonlocal last_read
        while last_read < i:
            try:
                fr = next(reader)
            except StopIteration:
                return buf.get(last_read)
            last_read += 1
            buf[last_read] = fr
            for kk in [kk for kk in buf if kk < last_read - 2]:
                del buf[kk]
        return buf.get(i, buf.get(last_read))

    try:
        for k, of in enumerate(plan.frames):
            if of.mode == "intro":
                writer.write(comp.intro(bg_first, of))
                continue
            if of.mode == "outro":
                writer.write(comp.outro(last_base, of))
                continue
            i0 = int(math.floor(of.pos))
            frac = of.pos - i0
            img = get(i0)
            if img is None:
                continue
            if frac > 0.04:                       # frame blending = smooth slow motion
                nxt = get(i0 + 1)
                if nxt is not None and nxt is not img:
                    img = cv2.addWeighted(img, 1 - frac, nxt, frac, 0)
            f = int(np.clip(round(of.pos), 0, an.seq.T - 1))
            out = comp.base(img, f, of)
            last_base = out
            out = out.copy()
            comp.overlays(out, f, of, k)
            since = [k - t for t in comp.flash_at if 0 <= k - t < 5]
            if since:                             # flash on the peak position
                out = S.flash(out, 0.45 * (1 - min(since) / 5))
            writer.write(out)
            sc = 0.0
            if of.mode == "ramp" and of.rep is not None:
                sc = 1.0 + of.phase + (1.0 if of.rep == best_rep else 0.0)
                to = plan.turn_out.get(of.rep)
                if to is not None and to <= k < to + 8:
                    sc -= 0.6     # skip flash frames / pop-in start
            elif thumb_img is None:
                sc = 0.1
            if sc > thumb_score:
                thumb_img, thumb_score = out, sc
    finally:
        writer.close()
    if thumb_img is not None:
        cv2.imwrite(str(thumb), thumb_img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    mux_audio(tmp, path, plan.audio, final, vc["loudnorm"], bool(info.get("has_audio")))
    tmp.unlink(missing_ok=True)
    ofps = vc["out_fps"]
    hw = None
    if best_rep is not None:
        idx = [k for k, of in enumerate(plan.frames) if of.rep == best_rep]
        if idx:
            hw = [max(0.0, idx[0] / ofps - 1.2), min(plan.duration_s, idx[-1] / ofps + 1.2)]
    body_idx = [k for k, of in enumerate(plan.frames) if of.mode not in ("intro", "outro")]
    return {"output_file": str(final), "thumbnail_file": str(thumb) if thumb_img is not None else None,
            "duration_s": round(plan.duration_s, 2), "frames": len(plan.frames),
            "render_seconds": round(time.time() - t0, 1), "crop": crop.to_dict(),
            "decode_scale": round(ds, 3), "body_frames": list(plan.body), "note": plan.note,
            "peak_reps": [an.reps[i].index for i in plan.peak_reps],
            "display_font": S.display_font_path()[0],
            "body_window_s": [round(body_idx[0] / ofps, 2), round((body_idx[-1] + 1) / ofps, 2)] if body_idx else None,
            "highlight_window_s": hw}
