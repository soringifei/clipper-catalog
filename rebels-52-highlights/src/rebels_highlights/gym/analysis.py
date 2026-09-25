"""Turns a PoseSeq + exercise into per-frame signals, reps, bar path and a summary.

All numbers derived from a single 2D phone video are *estimates*; JSON keys carry the
``_est`` suffix where a physical unit (m, m/s) is involved."""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import metrics as M
from .exercises import PROFILES, Profile, classify, normalize_name
from .pose import KP, PoseSeq

ARC_JOINTS = {   # joint -> (proximal, vertex, distal) keypoint names without side prefix
    "knee": ("hip", "knee", "ank"),
    "hip": ("sho", "hip", "knee"),
    "elbow": ("sho", "elb", "wri"),
    "shoulder": ("hip", "sho", "elb"),
    "body_line": ("sho", "hip", "ank"),
}
JOINT_LABEL = {"knee": "KNEE", "hip": "HIP", "elbow": "ELBOW", "shoulder": "SHOULDER",
               "trunk": "TRUNK LEAN", "body_line": "BODY LINE", "bar_y": "BAR HEIGHT",
               "hip_y": "HIP HEIGHT", "knee_front": "FRONT KNEE"}


@dataclass
class Analysis:
    seq: PoseSeq
    exercise: str
    profile: Profile
    confidence: float
    exercise_source: str                      # auto | config | cli
    label_visible: bool
    side: str                                 # left | right (camera-facing side)
    angles: dict[str, np.ndarray]
    primary_name: str
    primary: np.ndarray                       # (T,) degrees (or % stature for *_y)
    reps: list[M.Rep]
    jumps: list[M.Jump]
    bar: np.ndarray                           # (T,2) px bar proxy (wrists / plate)
    bar_source: str
    bar_vy: np.ndarray                        # (T,) upward speed, m/s if scale else px/s
    m_per_px: Optional[float]
    body_px: Optional[float]
    arc_ranges: dict[str, tuple[float, float]]
    summary: dict = field(default_factory=dict)
    scores: dict = field(default_factory=dict)
    extras: dict = field(default_factory=dict)

    @property
    def fps(self) -> float:
        return self.seq.fps

    def arc_angle(self, joint: str, f: int) -> float:
        if joint == "trunk":
            return float(self.angles["trunk"][f])
        return float(self.angles[f"{joint}_{self.side[0]}"][f]) if f"{joint}_{self.side[0]}" in \
            self.angles else float("nan")

    def rep_at(self, f: int) -> tuple[int, Optional[M.Rep]]:
        """(number of reps started by frame f, the rep containing f or last completed)."""
        n, cur = 0, None
        for r in self.reps:
            if f >= r.start:
                n = r.index
                cur = r
        return n, cur

    def to_json(self) -> dict:
        r2 = lambda a: [None if np.isnan(v) else round(float(v), 1) for v in a]  # noqa: E731
        return {
            "exercise": {"key": self.exercise, "label": self.profile.label,
                         "confidence": round(self.confidence, 3), "source": self.exercise_source,
                         "label_shown": self.label_visible, "scores": self.scores},
            "camera_side": self.side,
            "fps": self.fps, "frames": self.seq.T, "track_fraction": round(self.seq.track_frac, 3),
            "scale": {"m_per_px_est": self.m_per_px, "body_px_est": self.body_px,
                      "method": "segment-chain stature (athlete height from config)"},
            "primary_signal": self.primary_name,
            "reps": [r.to_dict() for r in self.reps],
            "jumps": [j.to_dict() for j in self.jumps],
            "bar_source": self.bar_source,
            "summary": self.summary,
            "extras": self.extras,
            "series": {"angles": {k: r2(v) for k, v in self.angles.items()},
                       "bar_xy": [[None, None] if np.isnan(p[0]) else [round(float(p[0]), 1),
                                                                     round(float(p[1]), 1)]
                                  for p in self.bar],
                       "bar_vy": r2(self.bar_vy)},
        }


def _nanmean(*a):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(np.stack(a), 0)


def _pct_range(v: np.ndarray, lo=3, hi=97) -> tuple[float, float]:
    v = v[~np.isnan(v)]
    if len(v) < 3:
        return (0.0, 180.0)
    a, b = float(np.percentile(v, lo)), float(np.percentile(v, hi))
    return (a, b if b - a > 5 else a + 5)


def _side_or_mean(ang: dict, joint: str, side: str, seq: PoseSeq) -> np.ndarray:
    s = ang[f"{joint}_{side[0]}"].copy()
    other = ang[f"{joint}_{'r' if side == 'left' else 'l'}"]
    # fill gaps of the near side from the far side
    gap = np.isnan(s)
    s[gap] = other[gap]
    return s


def analyze(seq: PoseSeq, cfg: dict, exercise: Optional[str] = None, source: str = "auto",
            plate_path: Optional[np.ndarray] = None) -> Analysis:
    ang = M.joint_angles(seq)
    auto_key, auto_conf, scores = classify(seq, ang)
    feats = scores.pop("_features", {})
    key = normalize_name(exercise)
    auto_guess = None
    if key:
        conf = 1.0
    else:
        key, conf, source = auto_key, auto_conf, "auto"
        if conf < cfg["edit"]["label_min_confidence"]:
            # not sure what this is: neutral look, no exercise-specific claims
            auto_guess = {"key": key, "confidence": conf}
            key = "generic"
    prof = PROFILES[key]
    label_visible = conf >= cfg["edit"]["label_min_confidence"] and key != "generic"
    side = M.near_side(seq)
    fps = seq.fps
    body_px = M.body_scale_px(seq)
    m_per_px = M.metres_per_px(seq, cfg["athlete"]["height_m"])

    # --- primary signal -----------------------------------------------------
    pname = prof.primary
    if pname == "auto":
        best, best_rom = "knee", -1.0
        for j in ("knee", "hip", "elbow"):
            v = _side_or_mean(ang, j, side, seq)
            v = v[~np.isnan(v)]
            rom = float(np.percentile(v, 95) - np.percentile(v, 5)) if len(v) > 5 else 0
            if rom > best_rom:
                best, best_rom = j, rom
        pname = best
    bar = plate_path if plate_path is not None else M.bar_proxy(seq)
    bar_source = "plate_tracker" if plate_path is not None else "wrists_midpoint"
    if pname in ("knee", "hip", "elbow"):
        primary = _side_or_mean(ang, pname, side, seq)
    elif pname == "knee_front":
        primary = np.fmin(ang["knee_l"], ang["knee_r"])
    elif pname == "bar_y":
        primary = -bar[:, 1] / (body_px or seq.height) * 100.0
    elif pname == "hip_y":
        primary = -M.mid(seq.kp, "l_hip", "r_hip")[:, 1] / (body_px or seq.height) * 100.0
    else:  # sprint / agility: show the knee
        primary = _side_or_mean(ang, "knee", side, seq)
        pname = "knee"

    # --- reps / jumps ---------------------------------------------------------
    reps: list[M.Rep] = []
    jumps = M.detect_jumps(seq, body_px) if key in ("jump", "generic") else []
    if key == "jump":
        for i, j in enumerate(jumps):
            s = max(0, j.takeoff - int(0.8 * fps))
            e = min(seq.T - 1, j.landing + int(0.4 * fps))
            knee = _side_or_mean(ang, "knee", side, seq)
            seg = knee[s:j.takeoff + 1]
            low = s + int(np.nanargmin(seg)) if np.any(~np.isnan(seg)) else j.takeoff
            peak = j.takeoff + int(np.argmin(np.nan_to_num(-primary[j.takeoff:j.landing + 1],
                                                           nan=0.0))) if j.landing > j.takeoff else j.takeoff
            rom = M.rom_between(knee, s, j.takeoff)
            reps.append(M.Rep(index=i + 1, start=s, turn=int(peak), end=e,
                              rom_deg=0.0 if np.isnan(rom) else rom,
                              min_deg=float(np.nanmin(seg)) if np.any(~np.isnan(seg)) else float("nan"),
                              max_deg=float(np.nanmax(knee[s:e + 1])) if np.any(~np.isnan(knee[s:e + 1])) else float("nan"),
                              ecc_s=(low - s) / fps, con_s=(j.takeoff - low) / fps, pause_s=0.0,
                              peak_ang_vel_dps=0.0,
                              extra={"flight_s": round(j.flight_s, 3),
                                     "jump_height_m_est": round(j.height_m_est, 3),
                                     "countermovement_knee_deg": round(float(np.nanmin(seg)), 1)
                                     if np.any(~np.isnan(seg)) else None,
                                     "load_frame": int(low)}))
    elif pname == "bar_y" or prof.primary not in ("none",):
        min_rom = prof.min_rom if pname != "bar_y" else 25.0
        reps = M.segment_reps(primary, fps, prof.polarity, min_rom=min_rom)
        if pname == "bar_y":   # ROM of a pull = hip extension; bar travel kept separately
            hip_sig = _side_or_mean(ang, "hip", side, seq)
            hv = np.gradient(M.fill_nan_1d(hip_sig)) * fps
            for r in reps:
                r.extra["bar_travel_pct_stature"] = round(r.rom_deg, 1)
                if m_per_px and body_px:
                    r.extra["bar_travel_m_est"] = round(r.rom_deg / 100 * body_px * m_per_px, 2)
                seg = hip_sig[r.start:r.turn + 1]
                if np.any(~np.isnan(seg)):
                    r.min_deg, r.max_deg = float(np.nanmin(seg)), float(np.nanmax(seg))
                    r.rom_deg = r.max_deg - r.min_deg
                    r.peak_ang_vel_dps = float(np.max(np.abs(hv[r.start:r.turn + 1])))

    # --- bar velocity ----------------------------------------------------------
    v = M.velocity(bar, fps, smooth_s=0.1)
    vy = -v[:, 1]  # up positive
    if m_per_px:
        vy = vy * m_per_px
    speed_unit = "m/s" if m_per_px else "px/s"

    # --- per-rep extras -----------------------------------------------------------
    hip_near = _side_or_mean(ang, "hip", side, seq)
    hip_vel = np.gradient(M.fill_nan_1d(hip_near)) * fps
    for r in reps:
        s, t, e = r.start, r.turn, r.end
        if prof.bar_path:
            con = slice(t, e + 1) if prof.polarity == "valley" else slice(s, t + 1)
            seg = vy[con]
            if np.any(~np.isnan(seg)):
                r.extra["bar_peak_up_speed"] = round(float(np.nanmax(seg)), 2)
                disp = bar[con.start, 1] - bar[min(con.stop - 1, seq.T - 1), 1]
                dur = (con.stop - 1 - con.start) / fps
                if dur > 0.05 and not np.isnan(disp):
                    mv = disp / dur * (m_per_px or 1.0)
                    r.extra["bar_mean_con_speed"] = round(float(mv), 2)
            xs = bar[s:e + 1, 0]
            if np.any(~np.isnan(xs)) and body_px:
                r.extra["bar_drift_pct_stature"] = round(float((np.nanmax(xs) - np.nanmin(xs)) / body_px * 100), 1)
        if prof.depth:
            hy = seq.kp[t, KP[f"{side[0]}_hip"], 1]
            ky = seq.kp[t, KP[f"{side[0]}_knee"], 1]
            if not np.isnan(hy) and not np.isnan(ky):
                r.extra["below_parallel"] = bool(hy > ky)
                if body_px:
                    r.extra["hip_minus_knee_pct_stature"] = round(float((hy - ky) / body_px * 100), 1)
        if key == "olympic":
            r.extra["hip_ext_peak_dps"] = round(float(np.nanmax(hip_vel[s:t + 1])), 0) if t > s else None
        # L/R symmetry of the primary joint ROM (needs both sides confidently visible)
        if pname in ("knee", "hip", "elbow"):
            lc = seq.conf[s:e + 1, KP[f"l_{ {'knee': 'knee', 'hip': 'hip', 'elbow': 'elb'}[pname]}"]].mean()
            rc = seq.conf[s:e + 1, KP[f"r_{ {'knee': 'knee', 'hip': 'hip', 'elbow': 'elb'}[pname]}"]].mean()
            if min(lc, rc) >= 0.6:
                sym = M.symmetry_pct(M.rom_between(ang[f"{pname}_l"], s, e),
                                     M.rom_between(ang[f"{pname}_r"], s, e))
                if sym is not None:
                    r.extra["symmetry_pct_est"] = round(sym, 0)
        # arc joint extremes at the turning point
        for j in prof.arcs:
            a = float(ang["trunk"][t]) if j == "trunk" else float(ang.get(f"{j}_{side[0]}", np.full(seq.T, np.nan))[t]) if j in ARC_JOINTS and j != "body_line" else float("nan")
            if j == "body_line":
                k = seq.kp[t]
                a = float(M.angle_3pt(k[KP[f"{side[0]}_sho"]], k[KP[f"{side[0]}_hip"]], k[KP[f"{side[0]}_ank"]]))
            if not np.isnan(a):
                r.extra[f"{j}_at_turn_deg"] = round(a, 1)

    # --- arc colour ranges ------------------------------------------------------
    arc_ranges = {}
    for j in prof.arcs:
        if j == "trunk":
            arc_ranges[j] = _pct_range(ang["trunk"])
        elif j == "body_line":
            k = seq.kp
            bl = M.angle_3pt(k[:, KP[f"{side[0]}_sho"]], k[:, KP[f"{side[0]}_hip"]], k[:, KP[f"{side[0]}_ank"]])
            ang["body_line"] = bl
            arc_ranges[j] = _pct_range(bl)
        else:
            arc_ranges[j] = _pct_range(ang[f"{j}_{side[0]}"])

    an = Analysis(seq=seq, exercise=key, profile=prof, confidence=conf, exercise_source=source,
                  label_visible=label_visible, side=side, angles=ang, primary_name=pname,
                  primary=primary, reps=reps, jumps=jumps, bar=bar, bar_source=bar_source,
                  bar_vy=vy, m_per_px=m_per_px, body_px=body_px, arc_ranges=arc_ranges,
                  scores=scores)
    an.extras["speed_unit"] = speed_unit
    if auto_guess:
        an.extras["auto_guess_low_confidence"] = auto_guess
    an.extras["features"] = {k: (round(v, 3) if isinstance(v, float) and not np.isnan(v) else None)
                             for k, v in feats.items()}
    _sport_extras(an)
    an.summary = summarize(an)
    return an


def _sport_extras(an: Analysis) -> None:
    seq, fps, H = an.seq, an.fps, an.body_px
    if an.exercise == "sprint":
        steps = 0
        peaks = []
        for s in ("l", "r"):
            hip = M.fill_nan_1d(an.angles[f"hip_{s}"])
            piv = M.zigzag(-M.moving_average(hip, max(1, int(0.08 * fps))), 20)
            pk = [i for i, t in piv if t == "peak"]      # max hip flexion = knee drive
            steps += len(pk)
            an.extras.setdefault("step_frames", []).extend(int(i) for i in pk)
            peaks.append([float(an.angles[f"hip_{s}"][i]) for i in pk])
        an.extras["step_frames"] = sorted(an.extras.get("step_frames", []))
        dur = seq.T / fps
        an.extras["steps"] = steps
        an.extras["cadence_steps_per_s"] = round(steps / dur, 2) if dur > 0 else None
        allp = [p for pp in peaks for p in pp if not np.isnan(p)]
        an.extras["knee_drive_hip_deg"] = round(float(np.median(allp)), 1) if allp else None
        if len(peaks[0]) >= 2 and len(peaks[1]) >= 2:
            an.extras["symmetry_pct_est"] = M.symmetry_pct(180 - np.median(peaks[0]), 180 - np.median(peaks[1]))
        tr = an.angles["trunk"]
        an.extras["trunk_lean_deg"] = round(float(np.nanmedian(tr)), 1) if np.any(~np.isnan(tr)) else None
        if an.m_per_px:
            mh = M.mid(seq.kp, "l_hip", "r_hip")
            vx = np.abs(M.velocity(mh, fps, 0.3)[:, 0]) * an.m_per_px
            if np.any(~np.isnan(vx)):
                an.extras["hip_speed_peak_mps_est"] = round(float(np.nanpercentile(vx, 95)), 2)
                an.extras["hip_speed_note"] = "valid only for a static camera"
    if an.exercise == "agility" and H:
        mh = M.mid(seq.kp, "l_hip", "r_hip")
        xs = M.moving_average(M.fill_nan_1d(mh[:, 0]), max(1, int(0.2 * fps)))
        piv = M.zigzag(xs, 0.25 * H)
        an.extras["cod_frames"] = [int(i) for i, _ in piv[1:]]
        an.extras["direction_changes"] = max(0, len(piv) - 1)
        kn = _nanmean(an.angles["knee_l"], an.angles["knee_r"])
        an.extras["stance_knee_deg"] = round(float(np.nanmedian(kn)), 1) if np.any(~np.isnan(kn)) else None
        hy = mh[:, 1]
        if np.any(~np.isnan(hy)):
            an.extras["hip_height_var_pct_stature"] = round(float((np.nanpercentile(hy, 95) - np.nanpercentile(hy, 5)) / H * 100), 1)
        if an.m_per_px:
            vx = np.abs(M.velocity(mh, fps, 0.2)[:, 0]) * an.m_per_px
            if np.any(~np.isnan(vx)):
                an.extras["lateral_speed_peak_mps_est"] = round(float(np.nanpercentile(vx, 95)), 2)


def summarize(an: Analysis) -> dict:
    reps = an.reps
    s: dict = {"exercise": an.profile.label if an.label_visible else "TRAINING",
               "reps": len(reps) if an.exercise not in ("sprint", "agility") else None}
    unit = an.extras.get("speed_unit")
    if reps:
        roms = [r.rom_deg for r in reps]
        s["best_rom_deg"] = round(max(roms), 0)
        s["mean_rom_deg"] = round(float(np.mean(roms)), 0)
        s["rom_consistency_pct"] = round(max(0.0, 100 - float(np.std(roms) / (np.mean(roms) + 1e-6) * 100)), 0)
        s["tempo_ecc_s"] = round(float(np.median([r.ecc_s for r in reps])), 1)
        s["tempo_con_s"] = round(float(np.median([r.con_s for r in reps])), 1)
        s["peak_ang_vel_dps"] = round(max(r.peak_ang_vel_dps for r in reps), 0)
        sp = [r.extra.get("bar_peak_up_speed") for r in reps if r.extra.get("bar_peak_up_speed") is not None]
        if sp:
            s[f"peak_bar_speed_{'mps_est' if unit == 'm/s' else 'pxps'}"] = round(max(sp), 2)
        mcv = [r.extra.get("bar_mean_con_speed") for r in reps if r.extra.get("bar_mean_con_speed") is not None]
        if mcv:
            s[f"mean_con_speed_{'mps_est' if unit == 'm/s' else 'pxps'}"] = round(float(np.mean(mcv)), 2)
            if len(mcv) >= 3 and mcv[0] > 0:
                s["velocity_loss_pct_est"] = round(max(0.0, (mcv[0] - mcv[-1]) / mcv[0] * 100), 0)
        sym = [r.extra["symmetry_pct_est"] for r in reps if "symmetry_pct_est" in r.extra]
        s["symmetry_pct_est"] = round(float(np.mean(sym)), 0) if sym else an.extras.get("symmetry_pct_est")
        if an.profile.depth:
            bp = [r.extra.get("below_parallel") for r in reps if "below_parallel" in r.extra]
            s["below_parallel_reps"] = f"{sum(bool(b) for b in bp)}/{len(reps)}" if bp else None
        s["rep_scores"] = rep_scores(an)
    if an.jumps:
        s["jumps"] = len(an.jumps)
        s["best_jump_height_m_est"] = round(max(j.height_m_est for j in an.jumps), 2)
        s["best_flight_s"] = round(max(j.flight_s for j in an.jumps), 2)
    for k in ("steps", "cadence_steps_per_s", "knee_drive_hip_deg", "trunk_lean_deg",
              "hip_speed_peak_mps_est", "direction_changes", "stance_knee_deg",
              "lateral_speed_peak_mps_est"):
        if an.extras.get(k) is not None:
            s[k] = an.extras[k]
    if an.exercise == "sprint" and s.get("symmetry_pct_est") is None and an.extras.get("symmetry_pct_est"):
        s["symmetry_pct_est"] = round(an.extras["symmetry_pct_est"], 0)
    return s


def rep_scores(an: Analysis) -> list[float]:
    """0..1 quality per rep: ROM close to the set's best + speed (for trimming/highlights)."""
    reps = an.reps
    if not reps:
        return []
    roms = np.array([r.rom_deg for r in reps])
    med = np.median(roms) + 1e-6
    cons = 1 - np.clip(np.abs(roms - med) / med, 0, 1)
    depth = roms / (roms.max() + 1e-6)
    vel = np.array([r.extra.get("bar_peak_up_speed") or r.peak_ang_vel_dps for r in reps], float)
    vel = vel / (np.nanmax(vel) + 1e-6) if np.nanmax(vel) > 0 else np.zeros_like(vel)
    if an.exercise == "jump":
        h = np.array([r.extra.get("jump_height_m_est", 0) for r in reps])
        return [round(float(x), 3) for x in h / (h.max() + 1e-6)]
    return [round(float(x), 3) for x in 0.4 * cons + 0.3 * depth + 0.3 * np.nan_to_num(vel)]


def clip_score(an: Analysis) -> float:
    """For picking the best clips for a compilation."""
    q = np.mean(an.summary.get("rep_scores") or [0.3])
    n = min(1.0, len(an.reps) / 5)
    return float(0.35 * an.confidence + 0.25 * n + 0.25 * q + 0.15 * an.seq.track_frac)
