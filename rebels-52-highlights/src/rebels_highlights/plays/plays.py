"""Play segmentation from a camera-compensated motion profile.

A football play looks like: pre-snap lull (players set, little motion) ->
sudden jump in player motion at the snap -> motion decays when the whistle
blows. We measure motion as the mean dense optical-flow magnitude after
subtracting the median flow (camera pan/zoom approximated as a global shift).
"""
from __future__ import annotations

import math
from typing import Any, Optional

import cv2
import numpy as np

from ..core.models import Play, Scene, VideoInfo

DEFAULTS: dict[str, Any] = {
    "min_play_s": 3.0,
    "max_play_s": 15.0,
    "snap_motion_jump": 2.5,     # post-snap motion / pre-snap baseline
    "settle_motion_ratio": 0.35,  # end when motion < ratio * post-snap peak ...
    "settle_hold_s": 1.0,        # ... for this long
    "lead_s": 3.0,               # play start = snap - lead
    "tail_s": 1.0,               # extra time after motion settles
    "lull_window_s": 1.5,        # pre-snap baseline window
    "sustain_s": 0.8,            # post-snap motion must be sustained this long
    "motion_floor": 0.15,        # absolute noise floor (px/s at analysis width)
    "analysis_width": 256,
    "scene_edge_skip_s": 0.25,   # ignore flow right after a cut
    "replay_window_s": 60.0,     # replay must start within this after the live play
    "fallback_confidence": 0.25,  # scene starting mid-play (no visible lull)
}


def plays_cfg(cfg: dict) -> dict:
    """``cfg['plays']`` merged over module defaults."""
    return {**DEFAULTS, **(cfg.get("plays") or {})}


# --------------------------------------------------------------------- motion
def motion_profile(video_path: str, cfg: dict, start_s: Optional[float] = None,
                   end_s: Optional[float] = None) -> dict:
    """Camera-compensated motion per sample.

    Returns ``{"t": [...], "motion": [...], "camera_motion": [...], "fps": f}``;
    motion and camera_motion are in pixels/second at ``analysis_width``.
    """
    pc = plays_cfg(cfg)
    target_fps = float((cfg.get("proxy") or {}).get("fps", 10))
    aw = int(pc["analysis_width"])
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or target_fps
    step = max(1, int(round(fps / target_fps)))
    sample_fps = fps / step
    idx = 0
    if start_s:
        idx = int(round(start_s * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    end_idx = int(round(end_s * fps)) if end_s is not None else None
    ts: list[float] = []
    mot: list[float] = []
    cam: list[float] = []
    prev: Optional[np.ndarray] = None
    while end_idx is None or idx <= end_idx:
        if (idx % step) != 0:
            if not cap.grab():
                break
            idx += 1
            continue
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(cv2.resize(frame, (aw, max(8, int(h * aw / w))),
                                       interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        if prev is not None:
            m, c = _flow_motion(prev, gray)
            ts.append(round(idx / fps, 3))
            mot.append(round(m * sample_fps, 4))
            cam.append(round(c * sample_fps, 4))
        prev = gray
        idx += 1
    cap.release()
    return {"t": ts, "motion": mot, "camera_motion": cam, "fps": sample_fps}


def _flow_motion(prev: np.ndarray, cur: np.ndarray) -> tuple[float, float]:
    """(residual motion, camera motion) in px/frame between two gray frames."""
    flow = cv2.calcOpticalFlowFarneback(prev, cur, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    fx, fy = flow[..., 0], flow[..., 1]
    mx, my = float(np.median(fx)), float(np.median(fy))
    res = np.hypot(fx - mx, fy - my)
    return float(res.mean()), math.hypot(mx, my)


# ------------------------------------------------------------------ segmenting
def _smooth(x: np.ndarray, k: int = 3) -> np.ndarray:
    if len(x) < k or k <= 1:
        return x
    return np.convolve(np.pad(x, (k // 2, k - 1 - k // 2), mode="edge"),
                       np.ones(k) / k, mode="valid")


def _snap_conf(ratio: float, jump: float) -> float:
    """0.5 at exactly the jump threshold, 1.0 at 4x the threshold (log scale)."""
    if ratio <= 0:
        return 0.0
    v = 0.5 + 0.5 * (math.log(ratio) - math.log(jump)) / math.log(4.0)
    return float(min(1.0, max(0.0, v)))


def _find_plays_in_scene(t: np.ndarray, m: np.ndarray, sc: Scene, pc: dict,
                         allow_fallback: bool) -> list[dict]:
    """Return [{start,end,snap,conf}] for plays inside one scene."""
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.1
    lull_n = max(2, int(round(pc["lull_window_s"] / dt)))
    sus_n = max(2, int(round(pc["sustain_s"] / dt)))
    hold_n = max(1, int(round(pc["settle_hold_s"] / dt)))
    jump = float(pc["snap_motion_jump"])
    floor = max(float(pc["motion_floor"]), 0.2 * float(np.median(m)) if len(m) else 0.0)
    ms = _smooth(m, 3)
    out: list[dict] = []

    def _end_after(k0: int) -> tuple[int, bool]:
        """Index where motion settles after the snap at k0 (or last index)."""
        peak, low = 0.0, 0
        for k in range(k0, len(ms)):
            peak = max(peak, ms[k])
            if t[k] - t[k0] < pc["min_play_s"] * 0.5:
                continue
            low = low + 1 if ms[k] < pc["settle_motion_ratio"] * peak else 0
            if low >= hold_n:
                return k - hold_n + 1, True
        return len(ms) - 1, False

    i = lull_n
    while i < len(m) - sus_n + 1:
        base = max(float(np.mean(m[i - lull_n:i])), floor)
        after = float(np.mean(m[i:i + sus_n]))
        # pre-snap lull: the window is quiet (isolated blips, e.g. keyframes, allowed)
        lull_ok = float(np.percentile(m[i - lull_n:i], 80)) <= jump * base
        if m[i] >= jump * base and after >= jump * base and lull_ok:
            k_end, settled = _end_after(i)
            snap = float(t[i - 1] + t[i]) / 2.0 if i > 0 else float(t[i])
            end_t = min(sc.end_s, float(t[k_end]) + (pc["tail_s"] if settled else 0.0))
            conf = _snap_conf(after / base, jump)
            if not settled:
                conf *= 0.85  # cut before the play visibly ended
            out.append({"snap": snap, "end": end_t, "conf": conf})
            i = k_end + lull_n
        else:
            i += 1

    if not out and allow_fallback and len(m) >= sus_n:
        # scene begins mid-play (broadcast cut in after the snap)
        head = float(np.mean(m[:sus_n]))
        if head >= jump * floor:
            k_end, settled = _end_after(0)
            end_t = min(sc.end_s, float(t[k_end]) + (pc["tail_s"] if settled else 0.0))
            out.append({"snap": None, "end": end_t,
                        "conf": float(pc["fallback_confidence"])})
    return out


def _clamp_span(start: float, end: float, sc: Scene, pc: dict) -> tuple[float, float]:
    lo, hi = float(pc["min_play_s"]), float(pc["max_play_s"])
    if end - start > hi:
        end = start + hi
    if end - start < lo:
        end = min(sc.end_s, start + lo)
        start = max(sc.start_s, end - lo)
    return round(start, 3), round(end, 3)


def segment_plays(info: VideoInfo, scenes: list[Scene], motion: dict, cfg: dict) -> list[Play]:
    """Find plays inside live scenes (and one per replay scene)."""
    pc = plays_cfg(cfg)
    t_all = np.asarray(motion.get("t") or [], dtype=float)
    m_all = np.asarray(motion.get("motion") or [], dtype=float)
    if not scenes and len(t_all):
        scenes = [Scene(scene_id=0, start_s=0.0, end_s=float(info.duration_s or t_all[-1]))]
    plays: list[Play] = []
    for sc in sorted(scenes, key=lambda s: s.start_s):
        if sc.kind not in ("live", "replay"):
            continue
        if sc.end_s - sc.start_s < pc["min_play_s"] * 0.6:
            continue
        sel = (t_all >= sc.start_s + pc["scene_edge_skip_s"]) & (t_all <= sc.end_s)
        t, m = t_all[sel], m_all[sel]
        if len(t) < 4:
            continue
        found = _find_plays_in_scene(t, m, sc, pc, allow_fallback=True)
        if sc.kind == "replay":
            # a replay scene is one play; keep the strongest snap if any
            best = max(found, key=lambda d: d["conf"]) if found else None
            snap = best["snap"] if best else None
            start, end = _clamp_span(sc.start_s, sc.end_s, sc, pc)
            conf = round(0.5 * (best["conf"] if best else 0.3), 3)
            p = Play(play_id="", game_id=info.game_id, start_s=start, end_s=end,
                     estimated_snap_s=round(snap, 3) if snap is not None else None,
                     snap_confidence=round(best["conf"], 3) if best and snap is not None else 0.0,
                     scene_ids=[sc.scene_id], confidence=conf,
                     replay_segments=[{"start": start, "end": end,
                                       "quality": conf, "is_replay": 1.0}])
            plays.append(p)
            continue
        for d in found:
            snap = d["snap"]
            start = max(sc.start_s, snap - pc["lead_s"]) if snap is not None else sc.start_s
            start, end = _clamp_span(start, d["end"], sc, pc)
            plays.append(Play(play_id="", game_id=info.game_id, start_s=start, end_s=end,
                              estimated_snap_s=round(snap, 3) if snap is not None else None,
                              snap_confidence=round(d["conf"], 3) if snap is not None else 0.0,
                              scene_ids=[sc.scene_id], confidence=round(d["conf"], 3)))
    plays.sort(key=lambda p: p.start_s)
    for n, p in enumerate(plays, 1):
        p.play_id = f"{info.game_id}_p{n:03d}"
    return plays


def is_replay_play(play: Play) -> bool:
    """True when ``segment_plays`` built this play from a replay scene."""
    return any(seg.get("is_replay") for seg in play.replay_segments)


def associate_replays(plays: list[Play], cfg: dict) -> list[Play]:
    """Attach replay-scene plays to the preceding live play (within the window).

    The live play stays canonical and gains ``replay_segments``; the replay play
    gets ``canonical_play_id``. Unmatched replays stay canonical (only view).
    """
    pc = plays_cfg(cfg)
    window = float(pc["replay_window_s"])
    ordered = sorted(plays, key=lambda p: p.start_s)
    last_live: dict[str, Play] = {}
    for p in ordered:
        if not is_replay_play(p):
            last_live[p.game_id] = p
            continue
        live = last_live.get(p.game_id)
        if live is None or p.start_s - live.end_s > window or p.start_s < live.start_s:
            continue
        p.canonical_play_id = live.play_id
        q = next((s.get("quality", p.confidence) for s in p.replay_segments
                  if s.get("is_replay")), p.confidence)
        seg = {"start": p.start_s, "end": p.end_s, "quality": float(q)}
        if seg not in live.replay_segments:
            live.replay_segments.append(seg)
    return ordered
