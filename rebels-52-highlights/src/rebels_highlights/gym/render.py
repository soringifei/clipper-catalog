"""9:16 render of an analysed gym clip: smooth athlete-following crop (or blurred
background + scaled foreground), skeleton, angle arcs, bar-path trail, live metric
panel, sparkline, per-rep slow-mo + freeze 'peak position' moments, intro/outro cards."""
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
from .analysis import ARC_JOINTS, JOINT_LABEL, Analysis
from .config import hex_to_bgr
from .pose import KP
from .video import FrameReader, FrameWriter, grab_frame, mux_audio

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# crop planning
# ---------------------------------------------------------------------------
def smooth_centers(c: np.ndarray, factor: float, fps: float = 30.0) -> np.ndarray:
    """Zero-phase (forward+backward) exponential smoothing; NaN gaps are held/interpolated."""
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
        # landscape clip where the body/bar doesn't fit a 9:16 window: blurred bg + fg
        fg_aspect = 1.0 if need_w / max(need_h, 1) > 1.0 else 4 / 5
        win_w = float(np.clip(max(need_w, need_h * fg_aspect), 0, W))
        win_h = win_w / fg_aspect
        if win_h > H:
            win_h = float(H)
            win_w = win_h * fg_aspect
        fw = out_w
        fh = int(round(fw / fg_aspect))
        fy = int(round((out_h - fh) * 0.42))
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
# edit plan (trim + slow-mo + freeze)
# ---------------------------------------------------------------------------
@dataclass
class Seg:
    s: float          # work frame start
    e: float          # work frame end (exclusive)
    speed: float      # 1, 0.5, or 0 for freeze
    rep: Optional[int] = None   # rep index (0-based) for peak moments
    dur: float = 0.0            # freeze seconds


@dataclass
class OutFrame:
    pos: float
    mode: str               # intro | body | slow | freeze | outro
    rep: Optional[int] = None
    phase: float = 0.0      # 0..1 within freeze/intro/outro


@dataclass
class EditPlan:
    body: tuple[int, int]
    segs: list[Seg]
    peak_reps: list[int]
    frames: list[OutFrame] = field(default_factory=list)
    audio: list[tuple[float, float, float]] = field(default_factory=list)
    duration_s: float = 0.0
    note: str = ""


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
    w = ec["slowmo_window_s"]
    sp = ec["slowmo_speed"]
    fz = ec["freeze_s"]
    per_peak_extra = (2 * w / sp - 2 * w) + fz          # extra seconds per peak moment
    fixed = ec["intro_s"] + ec["outro_s"]
    max_body = ec["max_clip_s"] - fixed
    pre, post = int(0.8 * fps), int(0.6 * fps)

    mode = ec["peak_moments"] if slowmo else "none"

    def choose_peaks(idx: list[int]) -> list[int]:
        if mode == "none" or not idx:
            return []
        if mode == "all" or (mode == "auto" and len(idx) <= 5):
            return list(idx)
        best = sorted(idx, key=lambda i: scores[i], reverse=True)[:3]
        return sorted(best)

    note = ""
    if trim_reps:
        i, j = trim_reps
        a, b = max(0, reps[i].start - pre), min(T, reps[j].end + post)
        peaks = choose_peaks(list(range(i, j + 1)))
    elif reps:
        a, b = 0, T
        peaks = choose_peaks(list(range(len(reps))))
        if (b - a) / fps + len(peaks) * per_peak_extra > max_body:
            peaks = choose_peaks(list(range(len(reps))))[:3] if mode != "none" else []
            best, best_sc = None, -1.0
            for i in range(len(reps)):
                for j in range(i, len(reps)):
                    a2, b2 = max(0, reps[i].start - pre), min(T, reps[j].end + post)
                    pk = choose_peaks(list(range(i, j + 1)))
                    d = (b2 - a2) / fps + len(pk) * per_peak_extra
                    if d > max_body and j > i:
                        break
                    sc = sum(scores[i:j + 1]) + 0.01 * (j - i)
                    if sc > best_sc:
                        best, best_sc = (i, j, a2, b2, pk), sc
            i, j, a, b, peaks = best
            if (b - a) / fps + len(peaks) * per_peak_extra > max_body:
                b = a + int((max_body - len(peaks) * per_peak_extra) * fps)
                peaks = [p for p in peaks if reps[p].turn < b - int(w * fps)]
            note = f"trimmed to reps {i + 1}-{j + 1} of {len(reps)}"
    else:
        a, b = _active_window(an, int(max_body * fps))
        peaks = []
        if b - a < T:
            note = "trimmed to most active window"
    if peaks_override is not None:
        peaks = peaks_override

    segs: list[Seg] = []
    cur = float(a)
    wf = w * fps
    for pi in peaks:
        p = float(reps[pi].turn)
        s0 = max(cur, p - wf)
        e1 = min(float(b), p + wf)
        if p < cur or p >= b:
            continue
        if s0 > cur:
            segs.append(Seg(cur, s0, 1.0))
        if p > s0:
            segs.append(Seg(s0, p, sp, pi))
        segs.append(Seg(p, p, 0.0, pi, fz))
        if e1 > p:
            segs.append(Seg(p, e1, sp, pi))
        cur = e1
    if cur < b:
        segs.append(Seg(cur, float(b), 1.0))
    plan = EditPlan((int(a), int(b)), segs, peaks, note=note)
    _expand(plan, an, cfg)
    return plan


def _expand(plan: EditPlan, an: Analysis, cfg: dict) -> None:
    ofps = cfg["video"]["out_fps"]
    wfps = an.fps
    ec = cfg["edit"]
    fr: list[OutFrame] = []
    au: list[tuple[float, float, float]] = []
    n_in = int(round(ec["intro_s"] * ofps))
    for k in range(n_in):
        fr.append(OutFrame(plan.body[0], "intro", None, k / max(1, n_in - 1)))
    au.append((0.0, n_in / ofps, 0.0))
    for sg in plan.segs:
        if sg.speed == 0:
            n = int(round(sg.dur * ofps))
            fr += [OutFrame(sg.s, "freeze", sg.rep, k / max(1, n - 1)) for k in range(n)]
            au.append((sg.s / wfps, n / ofps, 0.0))
            continue
        step = sg.speed * wfps / ofps
        n = int(round((sg.e - sg.s) / step))
        if n <= 0:
            continue
        mode = "body" if sg.speed == 1 else "slow"
        fr += [OutFrame(min(sg.s + k * step, an.seq.T - 1), mode, sg.rep) for k in range(n)]
        au.append((sg.s / wfps, n * step / wfps, sg.speed))
    n_out = int(round(ec["outro_s"] * ofps))
    last = fr[-1].pos if fr else 0
    for k in range(n_out):
        fr.append(OutFrame(last, "outro", None, k / max(1, n_out - 1)))
    au.append((0.0, n_out / ofps, 0.0))
    plan.frames = fr
    plan.audio = au
    plan.duration_s = len(fr) / ofps


# ---------------------------------------------------------------------------
# frame composition
# ---------------------------------------------------------------------------
class Composer:
    def __init__(self, an: Analysis, crop: CropPlan, cfg: dict, plan: EditPlan,
                 decode_scale: float, sparkline: bool = True, panel: bool = True,
                 header: bool = True):
        self.an, self.crop, self.cfg, self.plan = an, crop, cfg, plan
        self.ds = decode_scale
        st = cfg["style"]
        D.set_fonts(st.get("font"), st.get("font_regular"))
        self.accent = hex_to_bgr(st["accent"])
        self.skel = hex_to_bgr(st["skeleton_color"])
        self.W, self.H = crop.out_w, crop.out_h
        self.u = self.W / 1080.0     # UI unit scale
        body_out = (an.body_px or an.seq.height * 0.6) * crop.scale
        self.body_out = body_out
        self.draw_scale = float(np.clip(body_out / 900.0, 0.55, 1.4)) * max(self.u, 0.5)
        self.arc_r = float(np.clip(0.075 * body_out, 30, 90))
        self.trail_n = int(st["trail_s"] * an.fps)
        a, b = plan.body
        self.spark = None
        if sparkline and cfg["edit"]["sparkline"] and b - a > 5:
            vals = an.primary[a:b]
            if np.isnan(vals).mean() < 0.7:
                marks = [r.turn - a for r in an.reps if a <= r.turn < b]
                unit = "%" if an.primary_name in ("bar_y", "hip_y") else "°"
                title = JOINT_LABEL.get(an.primary_name, an.primary_name.upper())
                title = f"{title} {'(% STATURE) ' if unit == '%' else ''}OVER TIME"
                sh = int(150 * self.u)
                self.spark = D.Sparkline(vals, int(60 * self.u), int(self.H - 330 * self.u), int(960 * self.u),
                                         sh, title, self.accent, marks, unit=unit)
        self.panel = panel
        self.header = header
        self.near = an.side

    # -- base image (crop / fit) --------------------------------------------------
    def base(self, img: np.ndarray, f: int) -> np.ndarray:
        c = self.crop
        ox, oy = c.origin(f)
        ds = self.ds
        s = c.scale / ds
        M = np.float32([[s, 0, -ox * ds * s + c.fg[0]], [0, s, -oy * ds * s + c.fg[1]]])
        if c.mode == "crop":
            return self._crop_resize(img, ox * ds, oy * ds, c.win_w * ds, c.win_h * ds)
        out = D.blur_background(img, self.W, self.H, 0.55)
        fg = cv2.warpAffine(img, M, (self.W, self.H), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT)
        x, y, w, h = c.fg
        # soft shadow under the foreground card
        D.rounded_rect(out, x, y + 10, x + w, y + h + 18, 0, (0, 0, 0), 0.35)
        out[y:y + h, x:x + w] = fg[y:y + h, x:x + w]
        return out

    def _crop_resize(self, img, x, y, w, h):
        """Integer-aligned slice + resize (much faster than warpAffine). The overlay
        mapping uses the exact float origin, so keep the slice origin in sync."""
        H, W = img.shape[:2]
        x0 = int(np.clip(round(x), 0, max(0, W - round(w))))
        y0 = int(np.clip(round(y), 0, max(0, H - round(h))))
        x1, y1 = min(W, x0 + int(round(w))), min(H, y0 + int(round(h)))
        sub = img[y0:y1, x0:x1]
        interp = cv2.INTER_AREA if (x1 - x0) > self.W else cv2.INTER_LINEAR
        return cv2.resize(sub, (self.W, self.H), interpolation=interp)

    # -- overlays -----------------------------------------------------------------------
    def overlays(self, out: np.ndarray, f: int, of: OutFrame) -> None:
        an, c = self.an, self.crop
        kp = c.map(an.seq.kp[f], f)
        sc = self.draw_scale
        emph = 0.0
        if of.mode == "freeze":
            emph = 0.6 + 0.4 * math.sin(of.phase * math.pi)
        elif of.mode == "slow":
            emph = 0.3
        near = an.side[0]
        arcs = an.profile.arcs
        hl = tuple(KP[f"{near}_{ARC_JOINTS[j][1]}"] for j in arcs if j in ARC_JOINTS)
        if not np.isnan(kp[:, 0]).all():
            D.draw_skeleton(out, kp, an.side, sc, self.skel, hl)
        # bar path trail
        if an.profile.bar_path:
            s0 = max(self.plan.body[0], f - self.trail_n)
            pts = c.map(an.bar[s0:f + 1], f)
            D.draw_trail(out, pts, self.accent, sc)
        # arcs
        for j in arcs:
            if j == "trunk":
                hip = kp[KP[f"{near}_hip"]]
                sho = kp[KP[f"{near}_sho"]]
                v = an.angles["trunk"][f]
                if np.isnan(v) or np.isnan(hip).any() or np.isnan(sho).any():
                    continue
                L = self.arc_r * 1.6
                D.draw_vertical_ref(out, hip, L, sc)
                lo, hi = an.arc_ranges.get("trunk", (0, 90))
                frac = (v - lo) / max(hi - lo, 1)
                D.draw_arc(out, hip + np.array([0, -L]), hip, sho, v, frac, self.arc_r * 1.6, sc,
                           emph * 0.5, label=f"{v:.0f}°", small=True)
                continue
            names = ARC_JOINTS[j]
            a_, b_, c_ = (kp[KP[f"{near}_{n}"]] for n in names)
            if j == "body_line":
                v = an.angles.get("body_line", np.full(an.seq.T, np.nan))[f]
            else:
                v = an.angles[f"{j}_{near}"][f]
            if np.isnan(v):
                continue
            lo, hi = an.arc_ranges.get(j, (0, 180))
            frac = (hi - v) / max(hi - lo, 1)
            e = emph if j == arcs[0] else emph * 0.5
            D.draw_arc(out, a_, b_, c_, v, frac, self.arc_r, sc, e)
        u = self.u
        # header
        if self.header:
            x0 = 48 * u
            D.rounded_rect(out, x0, 118 * u, x0 + 8 * u, 150 * u, 2, self.accent, 1.0)
            D.text(out, self.cfg["athlete"]["tag"], x0 + 22 * u, 112 * u, int(30 * u), D.WHITE, tracking=1)
            if an.label_visible:
                D.text(out, an.profile.label, x0, 158 * u, int(52 * u), D.WHITE)
        if of.mode == "slow" or of.mode == "freeze":
            D.pill(out, f"{self.cfg['edit']['slowmo_speed']:g}×  SLOW-MO" if of.mode == "slow"
                   else "PEAK POSITION", self.W - 40 * u, 118 * u, self.accent, int(26 * u))
        if self.panel and of.mode != "freeze":
            self._panel(out, f)
        if self.spark is not None:
            self.spark.draw(out, f - self.plan.body[0])
        if of.mode == "freeze" and of.rep is not None:
            self._callout(out, of)

    def _rows(self, f: int) -> list[tuple[str, str]]:
        an = self.an
        rows = []
        n, rep = an.rep_at(f)
        done = [r for r in an.reps if r.end <= f]
        unit = an.extras.get("speed_unit")
        for key in an.profile.panel:
            if key == "rep" and an.reps:
                rows.append(("REP" if an.exercise != "jump" else "JUMP", f"{n} / {len(an.reps)}"))
            elif key == "angle":
                pn = an.primary_name
                if pn in ("bar_y", "hip_y"):
                    pn = "hip" if an.exercise == "olympic" else "knee"
                    v = an.angles[f"{pn}_{an.side[0]}"][f]
                elif pn == "knee_front":
                    v = an.primary[f]
                else:
                    v = an.primary[f]
                if not np.isnan(v):
                    rows.append((JOINT_LABEL.get(pn, pn.upper()), f"{v:.0f}°"))
            elif key == "speed" and unit == "m/s":
                v = an.bar_vy[f]
                if not np.isnan(v):
                    if abs(v) < 0.05:
                        rows.append(("BAR SPEED · EST", "0.00 m/s"))
                    else:
                        rows.append(("BAR SPEED · EST", f"{'↑' if v > 0 else '↓'}{abs(v):.2f} m/s"))
            elif key == "tempo" and done:
                r = done[-1]
                rows.append(("TEMPO  DOWN / UP", f"↓{r.ecc_s:.1f}s  ↑{r.con_s:.1f}s"))
            elif key == "height":
                js = [j for j in an.jumps if j.landing <= f]
                if js:
                    rows.append(("JUMP HEIGHT · EST", f"{js[-1].height_m_est * 100:.0f} cm"))
            elif key == "flight":
                js = [j for j in an.jumps if j.landing <= f]
                if js:
                    rows.append(("FLIGHT TIME", f"{js[-1].flight_s:.2f} s"))
            elif key == "hipext" and done:
                v = done[-1].extra.get("hip_ext_peak_dps")
                if v:
                    rows.append(("HIP EXTENSION PEAK", f"{v:.0f}°/s"))
            elif key == "strides":
                st = an.extras.get("step_frames") or []
                rows.append(("STEPS", f"{sum(1 for s in st if s <= f)}"))
            elif key == "cadence" and an.extras.get("cadence_steps_per_s"):
                rows.append(("CADENCE", f"{an.extras['cadence_steps_per_s']:.1f} /s"))
            elif key == "trunk":
                v = an.angles["trunk"][f]
                if not np.isnan(v):
                    rows.append(("TRUNK LEAN", f"{v:.0f}°"))
            elif key == "cod":
                cf = an.extras.get("cod_frames") or []
                rows.append(("DIRECTION CHANGES", f"{sum(1 for s in cf if s <= f)}"))
            elif key == "stance":
                v = np.nanmean([an.angles["knee_l"][f], an.angles["knee_r"][f]]) \
                    if not (np.isnan(an.angles["knee_l"][f]) and np.isnan(an.angles["knee_r"][f])) else np.nan
                if not np.isnan(v):
                    rows.append(("STANCE KNEE", f"{v:.0f}°"))
        return rows[:4]

    def _panel(self, out: np.ndarray, f: int) -> None:
        rows = self._rows(f)
        if not rows:
            return
        u = self.u
        x0, y0 = 48 * u, 250 * u
        rh = 84 * u
        w = 380 * u
        D.rounded_rect(out, x0, y0, x0 + w, y0 + rh * len(rows) + 24 * u, 22 * u, (12, 12, 12), 0.55)
        D.rounded_rect(out, x0, y0 + 22 * u, x0 + 6 * u, y0 + rh * len(rows) + 2 * u, 2, self.accent, 1.0)
        for i, (lab, val) in enumerate(rows):
            y = y0 + 14 * u + i * rh
            D.text(out, lab, x0 + 28 * u, y, int(19 * u), D.GREY, tracking=2, shadow=False)
            D.text(out, val, x0 + 26 * u, y + 22 * u, int(40 * u), D.WHITE)

    def _callout(self, out: np.ndarray, of: OutFrame) -> None:
        an = self.an
        r = an.reps[of.rep]
        u = self.u
        a = min(1.0, of.phase * 5)
        pos = {"squat": "BOTTOM", "deadlift": "BOTTOM", "bench": "BAR AT CHEST",
               "overhead_press": "LOCKOUT", "olympic": "BAR PEAK", "jump": "PEAK HEIGHT",
               "lunge": "BOTTOM", "pushup": "BOTTOM"}.get(an.exercise, "PEAK POSITION")
        l1 = f"REP {r.index} · {pos}" if an.exercise != "jump" else f"JUMP {r.index} · {pos}"
        if an.exercise == "jump":
            l2 = f"{r.extra.get('jump_height_m_est', 0) * 100:.0f} cm  ·  {r.extra.get('flight_s', 0):.2f} s air"
            l3 = "JUMP HEIGHT EST. FROM FLIGHT TIME (h = g·t²/8)"
        else:
            first = an.profile.arcs[0] if an.profile.arcs else an.primary_name
            av = r.extra.get(f"{first}_at_turn_deg")
            jl = JOINT_LABEL.get(first, first.upper())
            l2 = (f"{jl} {av:.0f}°   ROM {r.rom_deg:.0f}°" if av is not None else f"ROM {r.rom_deg:.0f}°")
            bits = []
            if "below_parallel" in r.extra:
                bits.append("BELOW PARALLEL" if r.extra["below_parallel"] else "ABOVE PARALLEL")
            if r.extra.get("bar_peak_up_speed") is not None and an.extras.get("speed_unit") == "m/s":
                bits.append(f"PEAK BAR {r.extra['bar_peak_up_speed']:.2f} m/s EST.")
            bits.append(f"↓{r.ecc_s:.1f}s ↑{r.con_s:.1f}s")
            l3 = "  ·  ".join(bits)
        cy = 250 * u
        w = 1000 * u
        x0 = (self.W - w) / 2
        D.rounded_rect(out, x0, cy, x0 + w, cy + 190 * u, 26 * u, (10, 10, 10), 0.62 * a)
        D.rounded_rect(out, x0, cy, x0 + w, cy + 6 * u, 3, self.accent, a)
        D.text(out, l1, self.W / 2, cy + 22 * u, int(26 * u), D.GREY, alpha=a, anchor="ct", tracking=3, shadow=False)
        D.text(out, l2, self.W / 2, cy + 58 * u, int(58 * u), D.WHITE, alpha=a, anchor="ct")
        col = D.GREEN if "BELOW PARALLEL" in l3 else D.WHITE
        D.text(out, l3, self.W / 2, cy + 138 * u, int(24 * u), col, alpha=a, anchor="ct", shadow=False)

    # -- cards ------------------------------------------------------------------------------
    def _card_bg(self, bg: Optional[np.ndarray], darken: float) -> np.ndarray:
        key = (id(bg), darken)
        if getattr(self, "_bg_key", None) != key:
            self._bg_key = key
            self._bg = (D.blur_background(bg, self.W, self.H, darken) if bg is not None
                        else np.zeros((self.H, self.W, 3), np.uint8))
        return self._bg.copy()

    def intro(self, bg: np.ndarray, of: OutFrame) -> np.ndarray:
        out = self._card_bg(bg, 0.62)
        u = self.u
        e = 1 - (1 - min(1.0, of.phase * 2.2)) ** 3
        dy = (1 - e) * 60 * u
        cy = self.H * 0.40
        an = self.an
        D.text(out, "TRAINING SESSION · MOVEMENT ANALYSIS", self.W / 2, cy - 120 * u + dy, int(26 * u), D.GREY,
               alpha=e, anchor="ct", tracking=3, shadow=False)
        title = an.profile.label if an.label_visible else "TRAINING"
        D.text(out, title, self.W / 2, cy - 70 * u + dy, int(104 * u), D.WHITE, alpha=e, anchor="ct")
        bw = 220 * u * e
        D.rounded_rect(out, self.W / 2 - bw / 2, cy + 70 * u, self.W / 2 + bw / 2, cy + 80 * u, 4, self.accent, 1.0)
        D.text(out, self.cfg["athlete"]["tag"], self.W / 2, cy + 110 * u + dy, int(40 * u), D.WHITE,
               alpha=e, anchor="ct", tracking=1)
        name = self.cfg["athlete"].get("name")
        if name:
            D.text(out, name.upper(), self.W / 2, cy + 170 * u + dy, int(34 * u), D.GREY, alpha=e, anchor="ct")
        return out

    def outro(self, bg: np.ndarray, of: OutFrame) -> np.ndarray:
        out = self._card_bg(bg, 0.70)
        u = self.u
        e = 1 - (1 - min(1.0, of.phase * 2.5)) ** 3
        an = self.an
        rows = summary_rows(an)
        top = self.H * 0.24
        D.text(out, "SESSION SUMMARY", self.W / 2, top, int(30 * u), D.GREY, alpha=e, anchor="ct", tracking=4,
               shadow=False)
        title = an.profile.label if an.label_visible else "TRAINING"
        D.text(out, title, self.W / 2, top + 44 * u, int(78 * u), D.WHITE, alpha=e, anchor="ct")
        D.rounded_rect(out, self.W / 2 - 110 * u, top + 150 * u, self.W / 2 + 110 * u, top + 158 * u, 4, self.accent, e)
        y = top + 210 * u
        for i, (lab, val) in enumerate(rows[:6]):
            ee = float(np.clip(of.phase * 3 - 0.15 * i, 0, 1))
            D.rounded_rect(out, 90 * u, y, self.W - 90 * u, y + 108 * u, 20 * u, (15, 15, 15), 0.5 * ee)
            D.text(out, lab, 130 * u, y + 36 * u, int(26 * u), D.GREY, alpha=ee, tracking=2, shadow=False)
            D.text(out, val, self.W - 130 * u, y + 22 * u, int(52 * u), D.WHITE, alpha=ee, anchor="rt")
            y += 124 * u
        D.text(out, "Estimates from 2D phone video · not lab-measured", self.W / 2, y + 30 * u, int(24 * u),
               D.GREY, alpha=e, anchor="ct", shadow=False, weight="regular")
        D.text(out, self.cfg["athlete"]["tag"], self.W / 2, self.H - 300 * u, int(36 * u), D.WHITE, alpha=e,
               anchor="ct", tracking=1)
        return out


def summary_rows(an: Analysis) -> list[tuple[str, str]]:
    s = an.summary
    rows = []
    if s.get("reps"):
        rows.append(("REPS" if an.exercise != "jump" else "JUMPS", str(s["reps"])))
    if s.get("best_jump_height_m_est"):
        rows.append(("BEST JUMP · EST", f"{s['best_jump_height_m_est'] * 100:.0f} cm"))
    if s.get("best_rom_deg") and an.exercise != "jump":
        rows.append(("BEST ROM", f"{s['best_rom_deg']:.0f}°"))
    if s.get("below_parallel_reps"):
        rows.append(("BELOW PARALLEL", s["below_parallel_reps"]))
    if s.get("peak_bar_speed_mps_est"):
        rows.append(("PEAK BAR SPEED · EST", f"{s['peak_bar_speed_mps_est']:.2f} m/s"))
    if s.get("tempo_ecc_s") is not None and an.exercise not in ("jump",):
        rows.append(("TEMPO DOWN / UP", f"{s['tempo_ecc_s']:.1f}s / {s['tempo_con_s']:.1f}s"))
    if s.get("steps"):
        rows.append(("STEPS", str(s["steps"])))
    if s.get("cadence_steps_per_s"):
        rows.append(("CADENCE", f"{s['cadence_steps_per_s']:.1f} steps/s"))
    if s.get("knee_drive_hip_deg"):
        rows.append(("KNEE DRIVE (HIP ANGLE)", f"{s['knee_drive_hip_deg']:.0f}°"))
    if s.get("direction_changes") is not None:
        rows.append(("DIRECTION CHANGES", str(s["direction_changes"])))
    if s.get("stance_knee_deg"):
        rows.append(("STANCE KNEE", f"{s['stance_knee_deg']:.0f}°"))
    if s.get("trunk_lean_deg") is not None and an.exercise == "sprint":
        rows.append(("TRUNK LEAN", f"{s['trunk_lean_deg']:.0f}°"))
    sym = s.get("symmetry_pct_est")
    rows.append(("L/R SYMMETRY · EST", f"{sym:.0f}%" if sym is not None else "n/a (side view)"))
    return rows


# ---------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------
def decode_scale_for(crop: CropPlan) -> float:
    """Decode at the resolution the output needs (x1.15) - big speed-up for 4K."""
    return float(min(1.0, crop.scale * 1.15))


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
    reader = iter(FrameReader(path, wfps, (dw, dh), start_s=a / wfps,
                              duration_s=(b - a + 2) / wfps))
    buf: dict[int, np.ndarray] = {}
    last_read = a - 1
    thumb_img, thumb_score = None, -1.0
    last_base = bg_first
    best_rep = None
    if an.reps and an.summary.get("rep_scores"):
        cands = [i for i in plan.peak_reps] or list(range(len(an.reps)))
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
            for k in [k for k in buf if k < last_read - 2]:
                del buf[k]
        return buf.get(i, buf.get(last_read))

    try:
        for of in plan.frames:
            if of.mode == "intro":
                writer.write(comp.intro(bg_first, of))
                continue
            if of.mode == "outro":
                writer.write(comp.outro(last_base, of))
                continue
            i0 = int(math.floor(of.pos))
            fr = of.pos - i0
            img = get(i0)
            if img is None:
                continue
            if fr > 0.05:
                nxt = get(i0 + 1)
                if nxt is not None and nxt is not img:
                    img = cv2.addWeighted(img, 1 - fr, nxt, fr, 0)
            f = int(np.clip(round(of.pos), 0, an.seq.T - 1))
            base = comp.base(img, f)
            last_base = base
            out = base.copy()
            comp.overlays(out, f, of)
            writer.write(out)
            sc = 0.0
            if of.mode == "freeze":
                sc = 2.0 + (1.0 if of.rep == best_rep else 0) - abs(of.phase - 0.6)
            elif thumb_img is None or thumb_score < 0.5:
                sc = 0.1 if abs(of.pos - (a + b) / 2) < 2 else 0.0
            if sc > thumb_score:
                thumb_img, thumb_score = out, sc
    finally:
        writer.close()
    if thumb_img is not None:
        cv2.imwrite(str(thumb), thumb_img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    mux_audio(tmp, path, plan.audio, final, vc["loudnorm"], bool(info.get("has_audio")))
    tmp.unlink(missing_ok=True)
    # output-time window around the best rep (used by the compilation)
    ofps = vc["out_fps"]
    hw = None
    if best_rep is not None:
        idx = [k for k, of in enumerate(plan.frames) if of.rep == best_rep]
        if idx:
            hw = [max(0.0, idx[0] / ofps - 1.2), min(plan.duration_s, idx[-1] / ofps + 0.8)]
    body_idx = [k for k, of in enumerate(plan.frames) if of.mode not in ("intro", "outro")]
    return {"output_file": str(final), "thumbnail_file": str(thumb) if thumb_img is not None else None,
            "duration_s": round(plan.duration_s, 2), "frames": len(plan.frames),
            "render_seconds": round(time.time() - t0, 1), "crop": crop.to_dict(),
            "decode_scale": round(ds, 3), "body_frames": list(plan.body), "note": plan.note,
            "peak_reps": [an.reps[i].index for i in plan.peak_reps],
            "body_window_s": [round(body_idx[0] / ofps, 2), round((body_idx[-1] + 1) / ofps, 2)] if body_idx else None,
            "highlight_window_s": hw}
