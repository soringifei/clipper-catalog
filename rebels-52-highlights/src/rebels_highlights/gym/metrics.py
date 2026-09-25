"""Kinematics from 2D keypoints: joint angles, rep segmentation, tempo, bar path,
velocity, jump flight time, pixel->metre scale and L/R symmetry.

Everything here is pure numpy and an *estimate* from a single 2D camera: angles are
projected angles (true only when the limb moves in the image plane, i.e. a side view)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np

from .pose import KP, PoseSeq

G = 9.81
# nose->ankle segment chain (shank+thigh+trunk+neck/head-to-nose) ~ 0.89 x stature
SEGMENT_CHAIN_FRAC = 0.89


# ---------------------------------------------------------------------------
# angle math
# ---------------------------------------------------------------------------
def angle_3pt(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Angle ABC in degrees (0..180) at vertex b; works on (...,2) arrays, NaN-safe."""
    a, b, c = (np.asarray(v, float) for v in (a, b, c))
    v1, v2 = a - b, c - b
    n = np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.sum(v1 * v2, axis=-1) / n
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def angle_vs_vertical(p_low: np.ndarray, p_high: np.ndarray) -> np.ndarray:
    """Angle (deg, 0..180) of segment low->high vs vertical-up in image coords (y down)."""
    d = np.asarray(p_high, float) - np.asarray(p_low, float)
    return np.degrees(np.arctan2(np.abs(d[..., 0]), -d[..., 1]))


def joint_angles(seq: PoseSeq) -> dict[str, np.ndarray]:
    k = seq.kp
    out: dict[str, np.ndarray] = {}
    for s in ("l", "r"):
        sho, elb, wri = k[:, KP[f"{s}_sho"]], k[:, KP[f"{s}_elb"]], k[:, KP[f"{s}_wri"]]
        hip, knee, ank = k[:, KP[f"{s}_hip"]], k[:, KP[f"{s}_knee"]], k[:, KP[f"{s}_ank"]]
        out[f"knee_{s}"] = angle_3pt(hip, knee, ank)
        out[f"hip_{s}"] = angle_3pt(sho, hip, knee)
        out[f"elbow_{s}"] = angle_3pt(sho, elb, wri)
        out[f"shoulder_{s}"] = angle_3pt(hip, sho, elb)
        # dorsiflexion proxy: forward shank tilt vs vertical (no toe keypoint in COCO)
        out[f"ankle_{s}"] = angle_vs_vertical(ank, knee)
    mh = mid(k, "l_hip", "r_hip")
    ms = mid(k, "l_sho", "r_sho")
    out["trunk"] = angle_vs_vertical(mh, ms)
    return out


def mid(k: np.ndarray, a: str, b: str) -> np.ndarray:
    """Midpoint of two keypoints; falls back to whichever one is visible."""
    pa, pb = k[:, KP[a]], k[:, KP[b]]
    m = (pa + pb) / 2
    m = np.where(np.isnan(m), np.where(np.isnan(pa), pb, pa), m)
    return m


def side_signal(ang: dict[str, np.ndarray], joint: str, side: str) -> np.ndarray:
    if joint == "trunk":
        return ang["trunk"]
    if side == "both":
        a, b = ang[f"{joint}_l"], ang[f"{joint}_r"]
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                return np.nanmean(np.stack([a, b]), 0)
    return ang[f"{joint}_{side[0]}"]


def near_side(seq: PoseSeq) -> str:
    """Side facing the camera = higher mean keypoint confidence."""
    L = [KP[n] for n in ("l_sho", "l_elb", "l_wri", "l_hip", "l_knee", "l_ank")]
    R = [KP[n] for n in ("r_sho", "r_elb", "r_wri", "r_hip", "r_knee", "r_ank")]
    return "left" if seq.conf[:, L].mean() >= seq.conf[:, R].mean() else "right"


# ---------------------------------------------------------------------------
# rep segmentation
# ---------------------------------------------------------------------------
def fill_nan_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float).copy()
    ok = ~np.isnan(x)
    if ok.sum() == 0:
        return np.zeros_like(x)
    t = np.arange(len(x))
    return np.interp(t, t[ok], x[ok])


def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x
    k = np.ones(w) / w
    return np.convolve(np.pad(x, w // 2, mode="edge"), k, mode="valid")[: len(x)]


def zigzag(x: np.ndarray, thr: float) -> list[tuple[int, str]]:
    """Alternating extrema whose successive differences are >= thr (a trailing,
    unconfirmed extreme is included when it is >= thr from the last pivot).
    Returns [(index, 'peak'|'valley'), ...]."""
    piv: list[tuple[int, str]] = []
    n = len(x)
    if n == 0:
        return piv
    trend = 0            # +1 looking for a peak, -1 looking for a valley
    hi_i = lo_i = 0
    for i in range(1, n):
        if trend == 0:
            if x[i] > x[hi_i]:
                hi_i = i
            if x[i] < x[lo_i]:
                lo_i = i
            if x[hi_i] - x[lo_i] >= thr:
                if hi_i > lo_i:
                    piv.append((lo_i, "valley"))
                    trend = 1
                else:
                    piv.append((hi_i, "peak"))
                    trend = -1
        elif trend == 1:
            if x[i] > x[hi_i]:
                hi_i = i
            if x[hi_i] - x[i] >= thr:
                piv.append((hi_i, "peak"))
                trend, lo_i = -1, i
        else:
            if x[i] < x[lo_i]:
                lo_i = i
            if x[i] - x[lo_i] >= thr:
                piv.append((lo_i, "valley"))
                trend, hi_i = 1, i
    if piv:
        last = x[piv[-1][0]]
        if trend == 1 and x[hi_i] - last >= thr and hi_i != piv[-1][0]:
            piv.append((hi_i, "peak"))
        elif trend == -1 and last - x[lo_i] >= thr and lo_i != piv[-1][0]:
            piv.append((lo_i, "valley"))
    return piv


@dataclass
class Rep:
    index: int
    start: int            # frame leaving the start position
    turn: int             # frame of the turning point (bottom for squats)
    end: int
    rom_deg: float
    min_deg: float
    max_deg: float
    ecc_s: float          # lowering / eccentric seconds
    con_s: float          # lifting / concentric seconds
    pause_s: float
    peak_ang_vel_dps: float
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        for k, v in list(d.items()):
            if isinstance(v, float):
                d[k] = round(v, 2)
        return d


def segment_reps(signal: np.ndarray, fps: float, polarity: str = "valley",
                 min_rom: float = 25.0, rel_rom: float = 0.4,
                 smooth_s: float = 0.12) -> list[Rep]:
    """Reps from a 1-D joint-angle signal.

    polarity 'valley': a rep is top -> bottom -> top (squat knee angle, bench elbow);
    eccentric = the descent. 'peak': bottom -> top -> bottom (OHP elbow at lockout, bar
    height of a clean); concentric = the rise. A rep counts when its ROM is >= min_rom
    and >= rel_rom x the clip's largest swing (ignores wobbles/unracking)."""
    x = fill_nan_1d(signal)
    if len(x) < 3:
        return []
    xs = moving_average(x, max(1, int(round(smooth_s * fps))))
    sgn = 1.0 if polarity == "valley" else -1.0
    y = sgn * xs   # turning points are always valleys of y
    swing = np.nanpercentile(y, 98) - np.nanpercentile(y, 2)
    thr = max(min_rom, rel_rom * swing) * 0.8
    piv = zigzag(y, thr)
    # treat the ends as "top" candidates when the signal starts/ends near the top
    valleys = [i for i, t in enumerate(piv) if t[1] == "valley"]
    reps: list[Rep] = []
    vel = np.gradient(xs) * fps
    for vi in valleys:
        b = piv[vi][0]
        a = piv[vi - 1][0] if vi > 0 else int(np.argmax(y[: b + 1]))
        c = piv[vi + 1][0] if vi + 1 < len(piv) else b + int(np.argmax(y[b:]))
        top = min(y[a], y[c])
        rom = max(y[a], y[c]) - y[b]
        if top - y[b] < max(min_rom, rel_rom * swing) or c <= b or a >= b:
            continue
        r = rom
        # leaving the top (10% of ROM) / arriving at the bottom (5%) / back at top (10%)
        seg1 = y[a:b + 1]
        s_i = a + int(np.argmax(seg1 < y[a] - 0.10 * r))
        near_b = np.flatnonzero(y[a:c + 1] <= y[b] + 0.05 * r) + a
        b_in, b_out = int(near_b[0]), int(near_b[-1])
        seg2 = y[b:c + 1]
        above = np.flatnonzero(seg2 >= y[c] - 0.10 * r)
        e_i = b + int(above[0]) if len(above) else c
        first, second = (b_in - s_i) / fps, (e_i - b_out) / fps
        ecc, con = (first, second) if polarity == "valley" else (second, first)
        reps.append(Rep(index=len(reps) + 1, start=int(s_i), turn=int(b), end=int(e_i),
                        rom_deg=float(abs(rom)), min_deg=float(np.min(xs[a:c + 1])),
                        max_deg=float(np.max(xs[a:c + 1])), ecc_s=float(max(0.0, ecc)),
                        con_s=float(max(0.0, con)), pause_s=float(max(0.0, (b_out - b_in) / fps)),
                        peak_ang_vel_dps=float(np.max(np.abs(vel[s_i:e_i + 1])))
                        if e_i > s_i else 0.0))
    return reps


# ---------------------------------------------------------------------------
# scale, bar path, velocity
# ---------------------------------------------------------------------------
def body_scale_px(seq: PoseSeq) -> Optional[float]:
    """Estimated stature in px from the 90th-percentile length of each segment in the
    ankle->knee->hip->shoulder->nose chain (robust to foreshortening and lying poses)."""
    k = seq.kp
    segs = [("ank", "knee"), ("knee", "hip"), ("hip", "sho")]
    vals = []
    for a, b in segs:
        L = []
        for s in ("l", "r"):
            d = np.linalg.norm(k[:, KP[f"{s}_{a}"]] - k[:, KP[f"{s}_{b}"]], axis=1)
            L.append(d)
        d = np.nanmax(np.stack(L), 0)
        if np.all(np.isnan(d)):
            return None
        vals.append(np.nanpercentile(d, 90))
    ms = mid(k, "l_sho", "r_sho")
    dn = np.linalg.norm(k[:, KP["nose"]] - ms, axis=1)
    head = np.nanpercentile(dn, 90) if not np.all(np.isnan(dn)) else 0.4 * vals[2]
    tot = sum(vals) + head
    return float(tot / SEGMENT_CHAIN_FRAC) if tot > 0 else None


def metres_per_px(seq: PoseSeq, height_m: float) -> Optional[float]:
    h = body_scale_px(seq)
    return height_m / h if h and h > 20 else None


def bar_proxy(seq: PoseSeq) -> np.ndarray:
    """Wrists midpoint (T,2); NaN when neither wrist is visible."""
    return mid(seq.kp, "l_wri", "r_wri")


def velocity(path: np.ndarray, fps: float, smooth_s: float = 0.1) -> np.ndarray:
    """(T,2) px -> (T,2) px/s using a centred difference on a lightly smoothed path."""
    out = np.full_like(path, np.nan, dtype=float)
    for c in range(path.shape[1]):
        v = path[:, c]
        ok = ~np.isnan(v)
        if ok.sum() < 3:
            continue
        f = moving_average(fill_nan_1d(v), max(1, int(round(smooth_s * fps))))
        g = np.gradient(f) * fps
        g[~ok] = np.nan
        out[:, c] = g
    return out


# ---------------------------------------------------------------------------
# jumps
# ---------------------------------------------------------------------------
def jump_height_from_flight(t_flight: float) -> float:
    """h = g t^2 / 8 (flight-time method; assumes take-off and landing in same posture)."""
    return G * t_flight ** 2 / 8.0


@dataclass
class Jump:
    takeoff: int
    landing: int
    flight_s: float
    height_m_est: float
    rise_px: float

    def to_dict(self) -> dict:
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def detect_jumps(seq: PoseSeq, body_px: Optional[float], min_flight_s: float = 0.15,
                 max_flight_s: float = 1.2) -> list[Jump]:
    """Flight = both ankles above the ground line by > 3% stature (ground line = rolling
    high percentile of ankle y), with the hips rising too."""
    k = seq.kp
    fps = seq.fps
    la, ra = k[:, KP["l_ank"], 1], k[:, KP["r_ank"], 1]
    ank = np.nanmax(np.stack([la, ra]), 0)   # lower ankle (larger y)
    if np.isnan(ank).mean() > 0.6 or not body_px:
        return []
    a = fill_nan_1d(ank)
    win = int(2.0 * fps)
    ground = np.array([np.percentile(a[max(0, i - win): i + win + 1], 85) for i in range(len(a))])
    air = (ground - a) > 0.03 * body_px
    air &= ~np.isnan(ank)
    hip = fill_nan_1d(mid(k, "l_hip", "r_hip")[:, 1])
    jumps = []
    idx = np.flatnonzero(np.diff(np.concatenate([[0], air.astype(int), [0]])))
    for s, e in zip(idx[::2], idx[1::2]):
        # refine edges: the 3% threshold is only for detection; contact is ~1% stature
        gap = ground - a
        while s > 0 and gap[s - 1] > 0.01 * body_px:
            s -= 1
        while e < len(a) and gap[e] > 0.01 * body_px:
            e += 1
        dur = (e - s) / fps
        if not (min_flight_s <= dur <= max_flight_s):
            continue
        hip_rise = np.median(hip[max(0, s - int(0.3 * fps)):s + 1]) - hip[s:e].min()
        if hip_rise < 0.04 * body_px:
            continue
        # flight time between the frames where ankles leave/reach ground (+1 frame)
        jumps.append(Jump(int(s), int(e), float(dur), float(jump_height_from_flight(dur)),
                          float(np.max(ground[s:e] - a[s:e]))))
    return jumps


# ---------------------------------------------------------------------------
# symmetry
# ---------------------------------------------------------------------------
def symmetry_pct(left: float, right: float) -> Optional[float]:
    if left is None or right is None or np.isnan(left) or np.isnan(right):
        return None
    m = (abs(left) + abs(right)) / 2
    if m <= 1e-6:
        return None
    return float(max(0.0, 100.0 - abs(left - right) / m * 100.0))


def rom_between(sig: np.ndarray, s: int, e: int) -> float:
    v = sig[s:e + 1]
    v = v[~np.isnan(v)]
    return float(v.max() - v.min()) if len(v) > 3 else float("nan")


def direction_changes(x: np.ndarray, fps: float, min_travel: float) -> int:
    """Count reversals of horizontal motion with >= min_travel px between reversals."""
    x = fill_nan_1d(x)
    if len(x) < 3:
        return 0
    xs = moving_average(x, max(1, int(0.2 * fps)))
    piv = zigzag(xs, min_travel)
    return max(0, len(piv) - 1)
