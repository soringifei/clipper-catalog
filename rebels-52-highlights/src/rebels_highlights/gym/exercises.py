"""Exercise profiles (what to measure/draw) and a kinematic heuristic classifier."""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import metrics as M
from .pose import KP, PoseSeq


@dataclass(frozen=True)
class Profile:
    key: str
    label: str                    # shown on screen when confident
    primary: str                  # joint for rep segmentation: knee|hip|elbow|bar_y|hip_y|none
    polarity: str = "valley"
    min_rom: float = 25.0
    arcs: tuple = ()              # joints drawn as angle arcs
    bar_path: bool = False
    lying: bool = False
    depth: bool = False           # hip-below-knee check at the bottom
    panel: tuple = ("rep", "angle", "speed", "tempo")
    notes: tuple = field(default_factory=tuple)


PROFILES: dict[str, Profile] = {
    "squat": Profile("squat", "SQUAT", "knee", "valley", 30, ("knee", "hip", "trunk"), True,
                     depth=True),
    "deadlift": Profile("deadlift", "DEADLIFT / HINGE", "hip", "valley", 30,
                        ("hip", "knee", "trunk"), True),
    "bench": Profile("bench", "BENCH PRESS", "elbow", "valley", 25, ("elbow", "shoulder"),
                     True, lying=True),
    "overhead_press": Profile("overhead_press", "OVERHEAD PRESS", "elbow", "peak", 30,
                              ("elbow", "shoulder"), True),
    "olympic": Profile("olympic", "OLYMPIC LIFT", "bar_y", "peak", 0, ("hip", "knee"), True,
                       panel=("rep", "angle", "speed", "hipext")),
    "jump": Profile("jump", "JUMP", "hip_y", "peak", 0, ("knee", "hip"), False,
                    panel=("rep", "angle", "height", "flight")),
    "sprint": Profile("sprint", "SPRINT", "none", "valley", 0, ("hip", "knee", "trunk"), False,
                      panel=("strides", "angle", "cadence", "trunk")),
    "agility": Profile("agility", "AGILITY / SHUFFLE", "none", "valley", 0, ("knee", "hip"),
                       False, panel=("cod", "angle", "stance", "trunk")),
    "lunge": Profile("lunge", "LUNGE", "knee_front", "valley", 30, ("knee",), False),
    "pushup": Profile("pushup", "PUSH-UP", "elbow", "valley", 25, ("elbow", "body_line"),
                      False, lying=True),
    "generic": Profile("generic", "TRAINING", "auto", "valley", 25, (), False),
}
ALIASES = {"bench_press": "bench", "press": "overhead_press", "ohp": "overhead_press",
           "clean": "olympic", "snatch": "olympic", "hinge": "deadlift", "rdl": "deadlift",
           "run": "sprint", "shuffle": "agility", "lateral": "agility", "push_up": "pushup",
           "box_jump": "jump", "broad_jump": "jump", "sled": "sprint", "linebacker": "agility"}


def normalize_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    n = name.strip().lower().replace("-", "_").replace(" ", "_")
    n = ALIASES.get(n, n)
    return n if n in PROFILES else None


def _ramp(x: float, lo: float, hi: float) -> float:
    if x is None or np.isnan(x):
        return 0.0
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


def _rng(v: np.ndarray, lo: float = 5, hi: float = 95) -> float:
    v = v[~np.isnan(v)]
    if len(v) < 5:
        return float("nan")
    return float(np.percentile(v, hi) - np.percentile(v, lo))


def _nanmean(*arrs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(np.stack(arrs), 0)


def features(seq: PoseSeq, ang: Optional[dict] = None) -> dict[str, float]:
    ang = ang or M.joint_angles(seq)
    k, fps = seq.kp, seq.fps
    H = M.body_scale_px(seq) or float(seq.height) * 0.6
    f: dict[str, float] = {"body_px": H}
    knee = _nanmean(ang["knee_l"], ang["knee_r"])
    hip = _nanmean(ang["hip_l"], ang["hip_r"])
    elbow = _nanmean(ang["elbow_l"], ang["elbow_r"])
    trunk = ang["trunk"]
    f["knee_rom"], f["hip_rom"], f["elbow_rom"] = _rng(knee), _rng(hip), _rng(elbow)
    f["knee_rom_indiv"] = np.nanmax([_rng(ang["knee_l"]), _rng(ang["knee_r"])])
    f["hip_rom_indiv"] = np.nanmax([_rng(ang["hip_l"]), _rng(ang["hip_r"])])
    tv = trunk[~np.isnan(trunk)]
    # lying = the whole body (shoulders -> ankles) is closer to horizontal than vertical
    body_ang = M.angle_vs_vertical(M.mid(k, "l_ank", "r_ank"), M.mid(k, "l_sho", "r_sho"))
    lying = np.where(np.isnan(body_ang), trunk > 72, body_ang > 60)
    okl = ~np.isnan(trunk)
    f["lying_frac"] = float(lying[okl].mean()) if okl.any() else 0.0
    f["trunk_max"] = float(np.percentile(tv, 95)) if len(tv) else float("nan")
    f["trunk_rom"] = _rng(trunk)
    mh = M.mid(k, "l_hip", "r_hip")
    ms = M.mid(k, "l_sho", "r_sho")
    mk = M.mid(k, "l_knee", "r_knee")
    wr = M.bar_proxy(seq)
    nose = k[:, KP["nose"]]
    f["hip_y_rng"] = _rng(mh[:, 1]) / H
    f["hip_x_rng"] = _rng(mh[:, 0], 2, 98) / H
    vx = M.velocity(mh, fps, 0.2)[:, 0]
    f["hip_speed"] = float(np.nanmedian(np.abs(vx)) / H) if np.any(~np.isnan(vx)) else 0.0
    f["dir_changes"] = float(M.direction_changes(mh[:, 0], fps, 0.25 * H))
    ok = ~np.isnan(wr[:, 1]) & ~np.isnan(ms[:, 1])
    f["wrist_above_sho"] = float((wr[ok, 1] < ms[ok, 1] - 0.03 * H).mean()) if ok.any() else 0.0
    okn = ~np.isnan(wr[:, 1]) & ~np.isnan(nose[:, 1])
    f["overhead_frac"] = float((wr[okn, 1] < nose[okn, 1]).mean()) if okn.any() else 0.0
    okh = ~np.isnan(wr[:, 1]) & ~np.isnan(mh[:, 1])
    f["wrist_below_hip"] = float((wr[okh, 1] > mh[okh, 1] - 0.02 * H).mean()) if okh.any() else 0.0
    f["bar_y_rng"] = _rng(wr[:, 1]) / H
    # olympic pattern: wrists from below knee to shoulder height within 1.2 s
    oly = 0
    okk = ~np.isnan(wr[:, 1]) & ~np.isnan(mk[:, 1]) & ~np.isnan(ms[:, 1])
    below = okk & (wr[:, 1] > mk[:, 1] - 0.02 * H)
    high = okk & (wr[:, 1] < ms[:, 1] + 0.06 * H)
    win = int(1.2 * fps)
    i = 0
    T = len(wr)
    while i < T:
        if below[i]:
            j = np.flatnonzero(high[i:i + win])
            if len(j):
                oly += 1
                i += int(j[0]) + int(0.5 * fps)
                continue
        i += 1
    f["oly_events"] = float(oly)
    # gait: anti-phase hips/knees
    a, b = M.fill_nan_1d(ang["hip_l"]), M.fill_nan_1d(ang["hip_r"])
    if np.std(a) > 1e-3 and np.std(b) > 1e-3 and len(a) > 10:
        f["hip_lr_corr"] = float(np.corrcoef(a, b)[0, 1])
    else:
        f["hip_lr_corr"] = 0.0
    ank_sep = np.abs(k[:, KP["l_ank"], 0] - k[:, KP["r_ank"], 0]) / H
    low_frames = mh[:, 1] > np.nanpercentile(mh[:, 1], 75) if np.any(~np.isnan(mh[:, 1])) else None
    f["ankle_sep_low"] = (float(np.nanmedian(ank_sep[low_frames])) if low_frames is not None
                          and np.any(~np.isnan(ank_sep[low_frames])) else 0.0)
    kl, kr = ang["knee_l"], ang["knee_r"]
    f["knee_lr_diff_low"] = (float(np.nanmedian(np.abs(kl - kr)[low_frames]))
                             if low_frames is not None and np.any(~np.isnan((kl - kr)[low_frames]))
                             else 0.0)
    sw = []
    for sd in ("l", "r"):
        rel = k[:, KP[f"{sd}_ank"], 0] - mh[:, 0]
        rel = rel[~np.isnan(rel)]
        if len(rel) > 5:
            sw.append(np.std(rel) / H)
    f["ankle_swing"] = float(max(sw)) if sw else 0.0
    f["jumps"] = float(len(M.detect_jumps(seq, H)))
    f["track_frac"] = seq.track_frac
    return {k2: (float(v) if v is not None else float("nan")) for k2, v in f.items()}


def classify(seq: PoseSeq, ang: Optional[dict] = None) -> tuple[str, float, dict]:
    """(exercise key, confidence 0..1, per-class scores). Heuristic, from kinematics."""
    f = features(seq, ang)
    r = _ramp
    lying = r(f["lying_frac"], 0.3, 0.7)
    stand = 1 - lying
    gait = (r(-f["hip_lr_corr"], 0.2, 0.6) * r(f["hip_rom_indiv"], 25, 50)
            * r(f["ankle_swing"], 0.06, 0.14))
    s: dict[str, float] = {}
    s["bench"] = lying * r(f["wrist_above_sho"], 0.4, 0.8) * r(f["elbow_rom"], 20, 45)
    s["pushup"] = lying * (1 - r(f["wrist_above_sho"], 0.2, 0.5)) * r(f["elbow_rom"], 20, 45)
    s["jump"] = stand * r(f["jumps"], 0.5, 1.0) * (1 - 0.7 * gait) * (1 - r(f["hip_speed"], 1.2, 2.5))
    s["sprint"] = stand * max(gait * r(f["knee_rom_indiv"], 25, 50),
                              r(f["hip_speed"], 1.2, 2.5) * r(f["knee_rom_indiv"], 30, 60))
    s["agility"] = stand * r(f["dir_changes"], 1.5, 3.5) * r(f["hip_x_rng"], 0.4, 1.0) * (1 - 0.5 * gait)
    s["olympic"] = stand * r(f["oly_events"], 0.5, 1.0) * r(f["hip_rom"], 25, 50) * r(f["bar_y_rng"], 0.3, 0.5)
    s["overhead_press"] = (stand * r(f["overhead_frac"], 0.15, 0.35) * r(f["elbow_rom"], 35, 60)
                           * (1 - r(f["knee_rom"], 25, 50)) * (1 - 0.8 * r(f["oly_events"], 0.5, 1)))
    lunge_shape = max(r(f["ankle_sep_low"], 0.18, 0.32), r(f["knee_lr_diff_low"], 20, 45))
    s["squat"] = (stand * r(f["knee_rom"], 35, 65) * r(f["hip_y_rng"], 0.10, 0.18)
                  * (1 - r(f["trunk_max"], 55, 75)) * (1 - 0.8 * lunge_shape)
                  * (1 - 0.8 * r(f["oly_events"], 0.5, 1)) * (1 - gait) * (1 - r(f["jumps"], 0.5, 1)))
    s["deadlift"] = (stand * r(f["hip_rom"], 30, 55) * r(f["trunk_max"], 35, 55)
                     * (1 - r(f["knee_rom"], 70, 100)) * r(f["wrist_below_hip"], 0.6, 0.9)
                     * (1 - 0.8 * r(f["oly_events"], 0.5, 1)) * (1 - gait))
    s["lunge"] = stand * r(f["knee_rom_indiv"], 35, 65) * lunge_shape * (1 - gait) * (1 - r(f["jumps"], 0.5, 1))
    s["generic"] = 0.3
    track_q = r(f["track_frac"], 0.4, 0.8)
    order = sorted(s, key=s.get, reverse=True)
    best, second = order[0], order[1]
    margin = s[best] - s[second]
    conf = s[best] * (0.5 + 0.5 * min(1.0, margin / 0.35)) * (0.5 + 0.5 * track_q)
    if best == "generic":
        conf = 0.0
    return best, float(round(conf, 3)), {k2: round(v, 3) for k2, v in s.items()} | {"_features": f}
