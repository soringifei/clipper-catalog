"""Vertical (1080x1920, 30 fps) highlight rendering for #52.

Edit structure of one clip (output timeline)::

    [A ident freeze 0.5-1.0 s: pre-snap frame, halo+arrow on #52, ident bar]
    [B+C+D live, full speed: lead-in (alignment/read), arrow fades right after
           the snap, NO graphics over the contact, lead-out 1-2 s]
    -- 4-frame dip --
    [E slow-motion replay of the decisive portion, eased zoom, fading pursuit
       trail that disappears before the contact]
    (-- dip -- [second angle from cand.replay_segments | slower 2nd replay])
    -- dip --
    [F end card: cand.caption + game info (known fields only)]

Frames are decoded with OpenCV, composed in numpy/PIL and piped as raw BGR to
ffmpeg (libx264 yuv420p +faststart); game audio for each segment is cut from
the source in the same ffmpeg call (time-stretched, low-passed and quieter
under the replay) and normalised by :func:`audio.audio.audio_filter_graph`.
"""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..audio.audio import SAMPLE_RATE, audio_filter_graph
from ..core.media import ffmpeg_bin, probe
from ..core.models import UNKNOWN, Candidate, EventResult, PlayerTrajectory, VideoInfo
from . import overlays as ov
from .crop import OUT_H, OUT_W, CropPath, compose_frame, compute_crop_path, letterbox_frame, max_zoom

PLAY_SHORT = {
    "tackle_for_loss": "tfl",
    "special_teams_tackle": "tackle",
    "run_stop_near_los": "run-stop",
    "open_field_tackle": "open-field-tackle",
    "solo_tackle": "solo-tackle",
    "assisted_tackle": "assisted-tackle",
    "forced_fumble": "forced-fumble",
    "fumble_recovery": "fumble-recovery",
    "qb_pressure": "qb-pressure",
    "block_shed": "block-shed",
    "coverage_play": "coverage",
    "kickoff_coverage": "kickoff-coverage",
    "punt_coverage": "punt-coverage",
    "return_block": "return-block",
    "other_defense": "defense",
    "other_special_teams": "special-teams-play",
}
AUDIO_CODEC = ["-c:a", "aac", "-b:a", "160k", "-ar", str(SAMPLE_RATE), "-ac", "2"]


# =============================================================== naming
def _slug(s: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", str(s).lower()).strip("-")
    return s or UNKNOWN


def _known(v: Any) -> bool:
    return v is not None and str(v).strip() != "" and str(v).strip().lower() != UNKNOWN


def _hms(t: Optional[float]) -> str:
    t = max(0, int(t or 0))
    return f"{t // 3600:02d}h{t % 3600 // 60:02d}m{t % 60:02d}s"


def _game_num(game_id: str) -> str:
    m = re.findall(r"\d+", game_id or "")
    return m[-1] if m else _slug(game_id)


def clip_basename(cand: Candidate) -> str:
    """'{date}_rebels-vs-{opp}_{pNNN}_{unit}_{type}_52_{HHhMMmSSs}_c{conf3}'."""
    m = re.search(r"p(\d+)$", cand.play_id or "")
    pnum = f"p{int(m[1]):03d}" if m else _slug(cand.play_id)
    unit = _slug(cand.unit.replace("_", "-")) if cand.unit else UNKNOWN
    ptype = PLAY_SHORT.get(cand.play_type, _slug(cand.play_type.replace("_", "-")))
    t = cand.snap_time_s if cand.snap_time_s is not None else cand.source_start_s
    conf = f"c{int(round(max(0.0, min(1.0, cand.identity_confidence)) * 100)):03d}"
    date = _slug(cand.date) if _known(cand.date) else "unknown-date"
    who = (f"rebels-vs-{_slug(cand.opponent)}" if _known(cand.opponent)
           else f"game-{_game_num(cand.game_id)}")
    num = int(cand.player_number or 52)
    return f"{date}_{who}_{pnum}_{unit}_{ptype}_{num}_{_hms(t)}_{conf}"


# =========================================================== frame access
class FrameReader:
    """Random-ish access to source frames with a small cache (sequential reads)."""

    def __init__(self, path: str, cache: int = 24):
        import cv2
        self.cv2 = cv2
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open video {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.pos = -1
        self.cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self.max_cache = cache
        self.last: Optional[np.ndarray] = None

    def index(self, i: int) -> np.ndarray:
        i = max(0, i if self.n <= 0 else min(i, self.n - 1))
        if i in self.cache:
            return self.cache[i]
        if not (self.pos < i <= self.pos + 120):
            self.cap.set(self.cv2.CAP_PROP_POS_FRAMES, i)
            self.pos = i - 1
        while self.pos < i:
            ok, f = self.cap.read()
            if not ok:
                break
            self.pos += 1
            self.last = f
            self.cache[self.pos] = f
            while len(self.cache) > self.max_cache:
                self.cache.popitem(last=False)
        if i in self.cache:
            return self.cache[i]
        if self.last is None:
            return np.zeros((self.h or 720, self.w or 1280, 3), np.uint8)
        return self.last

    def at(self, t: float, blend: bool = False) -> np.ndarray:
        f = max(0.0, t) * self.fps
        if not blend:
            return self.index(int(round(f)))
        i0 = int(math.floor(f + 1e-6))
        w = f - i0
        if w < 0.1:
            return self.index(i0)
        if w > 0.9:
            return self.index(i0 + 1)
        a = self.index(i0)
        b = self.index(i0 + 1)
        return self.cv2.addWeighted(a, 1 - w, b, w, 0)

    def close(self) -> None:
        self.cap.release()


# ============================================================ timeline plan
def _g(obj: Any, name: str, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def plan_timeline(cand: Candidate, event: Any, cfg: dict, video_dur: float,
                  zoom: float = 1.0) -> dict:
    """Compute source windows + output segments (pure; used by render_clip and tests)."""
    rc = cfg.get("render", {})
    fps = float(rc.get("fps", 30))
    slow = float(np.clip(rc.get("slowmo_speed", 0.55), 0.45, 0.65))
    lead_in, lead_out = float(rc.get("lead_in_s", 2.5)), float(rc.get("lead_out_s", 1.5))
    lead_out = float(np.clip(lead_out, 1.0, 2.0))
    ident_s = float(np.clip(rc.get("ident_card_s", 0.8), 0.5, 1.0))
    end_s = float(rc.get("end_card_s", 2.5))
    min_s, max_s = float(rc.get("min_clip_s", 15)), float(rc.get("max_clip_s", 60))
    vd = video_dur if video_dur and video_dur > 0 else max(cand.source_end_s + 5, 1.0)

    snap = cand.snap_time_s
    impact = _g(event, "impact_s") if _g(event, "impact_s") is not None else cand.impact_time_s
    if impact is None:
        base0 = snap if snap is not None else cand.source_start_s
        impact = base0 + max(1.0, (cand.source_end_s - base0) * 0.6)
    impact = float(np.clip(impact, 0.0, vd))
    if snap is not None and snap < impact:
        s0 = min(snap - 1.0, impact - lead_in)
    else:
        snap = None
        s0 = impact - max(lead_in, 3.0)
    s0 = max(0.0, s0, impact - 8.0)
    s1 = min(vd - 1.0 / fps, impact + lead_out)
    s1 = max(s1, min(vd, impact + 0.5))
    r0 = max(s0, impact - 2.5, (snap - 0.3) if snap is not None else -1e9)
    r1 = min(s1, impact + 1.0)

    segs: list[dict] = []

    def add(kind, dur, **kw):
        n = max(1, int(round(dur * fps)))
        segs.append({"kind": kind, "n": n, **kw})

    add("ident", ident_s, t0=s0, speed=0.0,
        audio=(max(0.0, s0 - ident_s), s0) if s0 >= ident_s else None, gain=0.8)
    add("live", s1 - s0, t0=s0, speed=1.0, audio=(s0, s1), gain=1.0)
    add("replay", (r1 - r0) / slow, t0=r0, speed=slow, zoom=zoom, audio=(r0, r1), gain=0.45,
        lowpass=2600, dip_in=True, trail=True)
    total = sum(s["n"] for s in segs) / fps + end_s
    need = min_s + 0.25 - total
    for rs in (cand.replay_segments or [])[:2]:
        if need <= 0.5:
            break
        a, b = float(rs.get("start_s", 0)), float(rs.get("end_s", 0))
        b = min(b, vd)
        if b - a < 1.0:
            continue
        dur = min(b - a, 8.0, max_s - total - 0.5)
        if dur < 1.0:
            break
        # centre the angle on its middle (broadcast replays frame the action there)
        a = a + max(0.0, (b - a - dur) / 2)
        add("angle", dur, t0=a, speed=1.0, audio=(a, a + dur), gain=0.5, dip_in=True,
            letterbox=True)
        total += dur
        need -= dur
    if need > 1.5:
        q0, q1 = max(s0, impact - 1.5), min(s1, impact + 0.7)
        sp = max(0.3, slow * 0.75)
        add("replay2", (q1 - q0) / sp, t0=q0, speed=sp, zoom=zoom * 1.08, audio=(q0, q1),
            gain=0.4, lowpass=2200, dip_in=True, trail=False)
        total += (q1 - q0) / sp
        need -= (q1 - q0) / sp
    end_total = end_s + max(0.0, need)
    total = sum(s["n"] for s in segs) / fps + end_total
    if total > max_s:  # never exceed the platform maximum
        end_total = max(1.5, end_total - (total - max_s))
    add("endcard", end_total, t0=s1, speed=0.0, audio=None, gain=0.0, dip_in=True)
    t = 0.0
    for s in segs:
        s["out_start"] = round(t, 4)
        t += s["n"] / fps
        s["out_end"] = round(t, 4)
    return {"s0": s0, "s1": s1, "snap": snap, "impact": impact, "r0": r0, "r1": r1,
            "segments": segs, "duration": t, "fps": fps}


# ============================================================ audio graph
def _atempo_chain(speed: float) -> str:
    parts, s = [], speed
    while s < 0.5:
        parts.append("atempo=0.5")
        s /= 0.5
    while s > 2.0:
        parts.append("atempo=2.0")
        s /= 2.0
    parts.append(f"atempo={s:.5f}")
    return ",".join(parts)


def _segments_audio_graph(segs: list[dict], fps: float, a_lo: float, input_idx: int,
                          has_audio: bool, cfg: dict) -> str:
    fmt = f"aformat=sample_fmts=fltp:sample_rates={SAMPLE_RATE}:channel_layouts=stereo"
    total = sum(s["n"] for s in segs) / fps
    if not has_audio:
        return audio_filter_graph(False, cfg, has_game_audio=False, duration_s=total)
    used = [s for s in segs if s.get("audio")]
    chains = []
    if used:
        outs = "".join(f"[src{i}]" for i in range(len(used)))
        chains.append(f"[{input_idx}:a]{fmt},asplit={len(used)}{outs}" if len(used) > 1
                      else f"[{input_idx}:a]{fmt}[src0]")
    k = 0
    labels = []
    for i, s in enumerate(segs):
        d = s["n"] / fps
        lab = f"a{i}"
        if s.get("audio"):
            a, b = s["audio"]
            c = [f"[src{k}]atrim=start={max(0.0, a - a_lo):.4f}:end={max(0.01, b - a_lo):.4f}",
                 "asetpts=PTS-STARTPTS"]
            if s.get("speed", 1.0) not in (0.0, 1.0):
                c.append(_atempo_chain(s["speed"]))
            if s.get("lowpass"):
                c.append(f"lowpass=f={int(s['lowpass'])}")
            c.append(f"volume={s.get('gain', 1.0):.3f}")
            c += [fmt, "apad", f"atrim=duration={d:.4f}",
                  f"afade=t=in:d=0.04,afade=t=out:st={max(0.0, d - 0.06):.4f}:d=0.06"]
            chains.append(",".join(c) + f"[{lab}]")
            k += 1
        else:
            chains.append(f"anullsrc=r={SAMPLE_RATE}:cl=stereo,atrim=duration={d:.4f},{fmt}[{lab}]")
        labels.append(f"[{lab}]")
    chains.append(f"{''.join(labels)}concat=n={len(labels)}:v=0:a=1[game]")
    chains.append(audio_filter_graph(False, cfg))
    return ";".join(chains)


# =============================================================== helpers
class _BoxTrack:
    def __init__(self, times, boxes):
        self.ok = bool(times) and bool(boxes)
        if self.ok:
            t = np.asarray(times, float)
            o = np.argsort(t)
            self.t = t[o]
            self.b = np.asarray(boxes, float).reshape(-1, 4)[o]

    def at(self, t: float, hold: float = 0.4) -> Optional[np.ndarray]:
        if not self.ok or t < self.t[0] - hold or t > self.t[-1] + hold:
            return None
        return np.array([np.interp(t, self.t, self.b[:, k]) for k in range(4)])

    def observed_near(self, t: float, tol: float) -> bool:
        return self.ok and bool(np.min(np.abs(self.t - t)) <= tol)


def _player_cfg(cfg: dict) -> dict:
    return cfg.get("player", {}) or {}


def _ident_text(cfg: dict, cand: Candidate) -> tuple[str, Optional[str]]:
    p = _player_cfg(cfg)
    num = p.get("player_number", cand.player_number or 52)
    pos = p.get("position_short", cand.position or "MLB")
    team = str(p.get("team_name", "Bucharest Rebels")).upper()
    name = p.get("player_name", UNKNOWN)
    if not _known(name) and _known(cand.player_name):
        name = cand.player_name
    return f"#{num} | {pos} | {team}", (str(name).upper() if _known(name) else None)


def _game_info_lines(cand: Candidate, cfg: dict) -> list[str]:
    lines = []
    team = _player_cfg(cfg).get("team_name", "Bucharest Rebels")
    if _known(cand.opponent):
        lines.append(f"{team} vs {cand.opponent}")
    if _known(cand.date):
        lines.append(str(cand.date))
    return lines


def _dip_gain(k: int, n: int, dip_in: bool, dip_out: bool) -> float:
    g = 1.0
    if dip_in and k < 2:
        g = min(g, (0.15, 0.55)[k])
    if dip_out and k >= n - 2:
        g = min(g, (0.55, 0.15)[k - (n - 2)])
    return g


def _visibility_pick(pl: _BoxTrack, tg: _BoxTrack, lo: float, hi: float, path: CropPath) -> float:
    """Time in [lo, hi] where #52 is observed, un-occluded and inside the crop."""
    best_t, best = hi, -1e9
    ts = np.arange(lo, hi + 1e-6, 1 / 15)
    for t in ts:
        b = pl.at(t, hold=0.0)
        if b is None:
            continue
        area = max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
        ovl = 0.0
        tb = tg.at(t, hold=0.0) if tg.ok else None
        if tb is not None:
            iw = max(0.0, min(b[2], tb[2]) - max(b[0], tb[0]))
            ih = max(0.0, min(b[3], tb[3]) - max(b[1], tb[1]))
            ovl = iw * ih / area
        x1, _, x2, _ = path.box(t)
        inside = 1.0 if x1 <= b[0] and b[2] <= x2 else 0.5
        obs = 1.0 if pl.observed_near(t, 0.15) else 0.6
        score = obs * inside * (1.0 - ovl) + 0.15 * (t - lo) / max(hi - lo, 1e-6)
        if score > best:
            best, best_t = score, t
    return float(best_t)


def _open_writer(cmd: list[str]) -> tuple[subprocess.Popen, Any]:
    log = tempfile.TemporaryFile()
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log)
    return proc, log


def _close_writer(proc: subprocess.Popen, log) -> None:
    try:
        proc.stdin.close()
    except BrokenPipeError:
        pass
    rc = proc.wait()
    if rc != 0:
        log.seek(0)
        err = log.read().decode(errors="replace")[-3000:]
        log.close()
        raise RuntimeError(f"ffmpeg failed ({rc}): {err}")
    log.close()


def _video_codec(cfg: dict) -> list[str]:
    rc = cfg.get("render", {})
    fps = int(rc.get("fps", 30))
    return ["-c:v", "libx264", "-preset", str(rc.get("preset", "medium")),
            "-crf", str(rc.get("crf", 20)), "-pix_fmt", "yuv420p", "-profile:v", "high",
            "-r", str(fps), "-g", str(fps * 2), "-movflags", "+faststart"]


# ============================================================ render_clip
def render_clip(cand: Candidate, info: VideoInfo, trajectory: PlayerTrajectory,
                event: EventResult, cfg: dict, out_dir: str,
                music_file: Optional[str] = None) -> dict:
    rc = cfg.get("render", {})
    accent = rc.get("accent_color", ov.DEFAULT_ACCENT)
    accent_rgb = ov.hex_to_rgb(accent)
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    base = clip_basename(cand)
    out_mp4 = out_dir_p / f"{base}.mp4"
    out_jpg = out_dir_p / f"{base}.jpg"

    reader = FrameReader(info.path)
    src_w, src_h = reader.w or info.width, reader.h or info.height
    vdur = info.duration_s or (reader.n / reader.fps if reader.n else 0.0)
    pinfo = probe(info.path)
    has_audio = bool(pinfo.get("has_audio"))
    vdur = vdur or pinfo.get("duration", 0.0)

    plan0 = plan_timeline(cand, event, cfg, vdur)
    fps = plan0["fps"]
    t_lo = min(s["t0"] for s in plan0["segments"] if not s.get("letterbox")) - 0.5
    path = compute_crop_path(trajectory, event, src_w, src_h, max(0.0, t_lo),
                             plan0["s1"] + 0.5, cfg, fps=fps)
    zoom = max_zoom(cfg, path)
    plan = plan_timeline(cand, event, cfg, vdur, zoom=zoom)
    segs, impact, snap, s0 = plan["segments"], plan["impact"], plan["snap"], plan["s0"]

    pl = _BoxTrack(_g(trajectory, "times", []), _g(trajectory, "boxes", []))
    tg = _BoxTrack(_g(trajectory, "target_times", []), _g(trajectory, "target_boxes", []))
    ident, name = _ident_text(cfg, cand)
    arrow_dur = float(rc.get("arrow_duration_s", 1.0))
    arrow_end = (snap + 0.1) if snap is not None else s0 + arrow_dur
    arrow_end = min(arrow_end, impact - 1.1)
    trail_on = bool(rc.get("replay_trail", True))
    accent_bgr = tuple(int(c) for c in accent_rgb[::-1])

    # ---- ffmpeg process: raw BGR frames on stdin + source audio window
    audio_windows = [s["audio"] for s in segs if s.get("audio")]
    a_lo = max(0.0, min(a for a, _ in audio_windows) - 1.0) if audio_windows else 0.0
    a_hi = max(b for _, b in audio_windows) + 1.0 if audio_windows else 1.0
    cmd = [ffmpeg_bin(), "-y", "-hide_banner", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{OUT_W}x{OUT_H}", "-r", str(int(fps)),
           "-i", "pipe:0"]
    if has_audio:
        cmd += ["-ss", f"{a_lo:.3f}", "-t", f"{a_hi - a_lo:.3f}", "-i", info.path]
    graph = _segments_audio_graph(segs, fps, a_lo, 1, has_audio, cfg)
    tmp_mp4 = out_mp4.with_suffix(".tmp.mp4")
    cmd += ["-filter_complex", graph, "-map", "0:v", "-map", "[aout]"] + _video_codec(cfg) \
        + AUDIO_CODEC + ["-f", "mp4", str(tmp_mp4)]
    proc, log = _open_writer(cmd)

    def player_marks(layer, xf, t, alpha):
        b = pl.at(t)
        if b is None or alpha <= 0:
            return
        cb = xf.box(b)
        bw = cb[2] - cb[0]
        ov.draw_halo(layer, (cb[0] + cb[2]) / 2, cb[3], bw, accent, alpha)
        ov.draw_arrow(layer, (cb[0] + cb[2]) / 2, cb[1], accent, alpha,
                      size=float(np.clip(bw * 0.6, 34, 64)),
                      label=str(_player_cfg(cfg).get("player_number", 52)))

    last_live = None
    try:
        for si, seg in enumerate(segs):
            kind, n = seg["kind"], seg["n"]
            nxt = segs[si + 1] if si + 1 < len(segs) else None
            dip_in = bool(seg.get("dip_in"))
            dip_out = bool(nxt and nxt.get("dip_in"))
            end_card = None
            if kind == "endcard":
                bg = last_live if last_live is not None else compose_frame(
                    reader.at(plan["s1"]), path, plan["s1"])[0]
                end_card = ov.render_end_card(bg, cand.caption or "", ident if not name else f"{name}  {ident}",
                                              _game_info_lines(cand, cfg), accent)
            for k in range(n):
                tout = k / fps
                if kind == "endcard":
                    frame = end_card.copy() if (dip_in and k < 2) else end_card
                else:
                    t = seg["t0"] + tout * seg["speed"]
                    src = reader.at(t, blend=0 < seg["speed"] < 1)
                    if seg.get("letterbox"):
                        frame = letterbox_frame(src)
                    else:
                        z = 1.0
                        if seg.get("zoom", 1.0) > 1.0:
                            z = 1.0 + (seg["zoom"] - 1.0) * ov.smoothstep(tout / 0.8)
                        frame, xf = compose_frame(src, path, t, z, accent_bgr)
                        layer = None
                        if kind == "ident":
                            layer = ov.new_layer(OUT_W, OUT_H)
                            player_marks(layer, xf, t, ov.smoothstep(k / 4))
                            ov.draw_ident_bar(layer, ident, name, accent, 1.0)
                        elif kind == "live":
                            last_live = frame
                            a_mark = ov.fade_alpha(t, s0 - 1, arrow_end, 0.0, 0.3)
                            a_bar = ov.fade_alpha(t, s0 - 1, s0, 0.0, 0.4)
                            if a_mark > 0 or a_bar > 0:
                                layer = ov.new_layer(OUT_W, OUT_H)
                                player_marks(layer, xf, t, a_mark)
                                if a_bar > 0:
                                    ov.draw_ident_bar(layer, ident, name, accent, a_bar)
                        elif kind in ("replay", "replay2"):
                            layer = ov.new_layer(OUT_W, OUT_H)
                            a_tag = ov.fade_alpha(tout, 0.0, 1.0, 0.15, 0.3)
                            if a_tag > 0:
                                ov.draw_corner_tag(layer, "REPLAY", accent, a_tag)
                            cut = impact - 0.35  # trail never reaches the contact
                            if trail_on and seg.get("trail") and pl.ok and t < cut:
                                a_tr = float(np.clip((cut - t) / 0.4, 0, 1))
                                ts = np.arange(max(seg["t0"], t - 2.0), t + 1e-6, 1 / 15)
                                pts = []
                                for tt in ts:
                                    b = pl.at(tt, hold=0.0)
                                    if b is not None:
                                        pts.append(xf((b[0] + b[2]) / 2, b[3]))
                                ov.draw_trail(layer, pts, accent, 0.8 * a_tr)
                        if layer is not None:
                            ov.composite(frame, layer)
                g = _dip_gain(k, n, dip_in, dip_out)
                if g < 1.0:
                    frame = (frame.astype(np.float32) * g).astype(np.uint8)
                if kind == "endcard" and k < 6:
                    frame = (frame.astype(np.float32) * min(1.0, 0.3 + k / 6)).astype(np.uint8)
                proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        _close_writer(proc, log)
    except BaseException:
        try:
            proc.kill()
        finally:
            log.close()
        if tmp_mp4.exists():
            tmp_mp4.unlink()
        reader.close()
        raise
    os.replace(tmp_mp4, out_mp4)

    # ---- thumbnail: frame just before impact where #52 is most visible
    t_th = _visibility_pick(pl, tg, max(s0, impact - 1.2), max(s0, impact - 0.1), path)
    thumb, _ = compose_frame(reader.at(t_th), path, t_th, 1.0, accent_bgr)
    if rc.get("thumbnail_tag", True):
        layer = ov.new_layer(OUT_W, OUT_H)
        p = _player_cfg(cfg)
        ov.draw_corner_tag(layer, f"#{p.get('player_number', 52)} | {p.get('position_short', 'MLB')}",
                           accent)
        ov.composite(thumb, layer)
    import cv2
    cv2.imwrite(str(out_jpg), thumb, [cv2.IMWRITE_JPEG_QUALITY, 90])
    # ---- landscape pre-snap seed frame (review page: exact manual #52 seeding)
    seed_t = max(0.0, (snap - 0.2) if snap is not None else s0)
    seed_img = reader.at(seed_t)
    seed_jpg = out_dir_p / f"{base}_seed.jpg"
    cv2.imwrite(str(seed_jpg), seed_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    reader.close()

    # ---- crop sidecar for QA: one entry per output frame of the real-time section
    crop_frames = []
    for seg in segs:
        if seg["kind"] != "live":
            continue
        for k in range(seg["n"]):
            t = seg["t0"] + k / fps
            b = pl.at(t)
            crop_frames.append({"t": round(t, 4), "crop": [round(v, 1) for v in path.box(t)],
                                "player_x": None if b is None else round(float(b[0] + b[2]) / 2, 1)})
    crop_json = out_dir_p / f"{base}.crop.json"
    crop_json.write_text(json.dumps({"mode": path.mode, "safe_region": path.safe_region,
                                     "frame_size": [src_w, src_h], "frames": crop_frames}))

    duration = plan["duration"]
    # ---- optional licensed music version (base file stays clean)
    music_out = None
    if music_file and Path(music_file).is_file():
        music_out = out_dir_p / f"{base}_music.mp4"
        graph = "[0:a]anull[game];[1:a]anull[music];" + audio_filter_graph(True, cfg)
        mcmd = [ffmpeg_bin(), "-y", "-hide_banner", "-loglevel", "error", "-i", str(out_mp4),
                "-stream_loop", "-1", "-i", str(music_file), "-filter_complex", graph,
                "-map", "0:v", "-map", "[aout]", "-c:v", "copy"] + AUDIO_CODEC + \
            ["-t", f"{duration:.3f}", "-movflags", "+faststart", str(music_out)]
        subprocess.run(mcmd, check=True, capture_output=True)

    timeline = {"basename": base, "fps": fps, "duration": duration,
                "source": {k: plan[k] for k in ("s0", "s1", "snap", "impact", "r0", "r1")},
                "impact_out": round(plan["segments"][1]["out_start"] + (impact - s0), 4),
                "segments": [{"kind": s["kind"], "start": s["out_start"], "end": s["out_end"],
                              "src_t0": s.get("t0"), "speed": s.get("speed")} for s in segs],
                "crop": path.summary(), "zoom": zoom}
    (out_dir_p / f"{base}.timeline.json").write_text(json.dumps(timeline, indent=1))
    for r in path.review_reasons:
        if r not in cand.review_reasons:
            cand.review_reasons.append(r)
    cand.output_file, cand.thumbnail_file = str(out_mp4), str(out_jpg)
    cand.output_duration_s = round(duration, 3)
    return {"output_file": str(out_mp4), "thumbnail_file": str(out_jpg),
            "output_duration_s": round(duration, 3),
            "music_file": str(music_out) if music_out else None,
            "crop_mode": path.mode, "crop_quality": round(path.quality, 3),
            "review_reasons": list(path.review_reasons),
            "timeline_file": str(out_dir_p / f"{base}.timeline.json"),
            "crop_file": str(crop_json), "seed_frame": str(seed_jpg),
            "seed_frame_size": [int(src_w), int(src_h)], "seed_frame_t": round(seed_t, 3)}


# ============================================================ render_debug
def render_debug(video_path, play, tracking, identity, event, cfg, out_path) -> str:
    """Landscape debug video: all boxes/track ids/team prob/OCR/identity/LOS/snap/impact."""
    import cv2
    reader = FrameReader(str(video_path))
    fps = reader.fps
    start, end = float(_g(play, "start_s", 0.0)), float(_g(play, "end_s", 0.0))
    snap = _g(play, "estimated_snap_s")
    tracks = (tracking or {}).get("tracks", []) or []
    tr = [(_g(t, "track_id"), _BoxTrack(_g(t, "times", []), _g(t, "boxes", [])),
           float(_g(t, "team_prob", 0.5))) for t in tracks]
    identity = identity or {}
    best = identity.get("best_track_id")
    idc = float(identity.get("identity_confidence", 0.0) or 0.0)
    evid = {(_g(e, "track_id")): e for e in identity.get("evidence", []) or []}
    traj = identity.get("trajectory") or {}
    tgt = _BoxTrack(_g(traj, "target_times", []), _g(traj, "target_boxes", []))
    impact = _g(event, "impact_s")
    contact = _g(event, "contact_point")
    los = _g(event, "los_x")
    ptype = _g(event, "play_type", "unclear")
    scale = min(1.0, 1280 / max(reader.w, 1))
    ow, oh = int(reader.w * scale) // 2 * 2, int(reader.h * scale) // 2 * 2
    tol = 1.5 / float(cfg.get("detection", {}).get("fine_fps", 15))
    out_path = str(out_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg_bin(), "-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo",
           "-pix_fmt", "bgr24", "-s", f"{ow}x{oh}", "-r", f"{fps:.4f}", "-i", "pipe:0",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", out_path]
    proc, log = _open_writer(cmd)
    f_small = ov.load_font(16)
    try:
        for i in range(int(start * fps), int(end * fps) + 1):
            t = i / fps
            frame = reader.index(i).copy()
            if scale != 1.0:
                frame = cv2.resize(frame, (ow, oh), interpolation=cv2.INTER_AREA)
            layer = ov.new_layer(ow, oh)
            d = ov.ImageDraw.Draw(layer)
            for tid, bt, tp in tr:
                if not bt.ok or not bt.observed_near(t, tol):
                    continue
                b = bt.at(t, 0.0)
                if b is None:
                    continue
                b = b * scale
                is_best = best is not None and tid == best
                col = "#E10600" if is_best else ("#27C24C" if tp >= 0.5 else "#3D8BFF")
                lab = f"id{tid} tp{tp:.2f}"
                if tid in evid:
                    lab += f" id{float(_g(evid[tid], 'identity_confidence', 0)):.2f}"
                ov.draw_debug_box(layer, b.tolist(), col, lab, 4 if is_best else 2)
            tb = tgt.at(t, 0.0)
            if tb is not None:
                ov.draw_debug_box(layer, (tb * scale).tolist(), "#FFD400", "target", 2)
            if los is not None:
                x = float(los) * scale
                d.line([(x, 0), (x, oh)], fill=(0, 200, 255, 200), width=2)
            if contact is not None and impact is not None and abs(t - impact) < 0.5:
                cx, cy = contact[0] * scale, contact[1] * scale
                d.ellipse([cx - 12, cy - 12, cx + 12, cy + 12], outline=(255, 0, 255, 255), width=3)
            info = [f"{_g(play, 'play_id', '')}  t={t:.2f}s  unit={_g(play, 'unit', '?')}",
                    f"best_track={best} identity={idc:.2f}  event={ptype} "
                    f"({float(_g(event, 'event_confidence', 0) or 0):.2f})"]
            if best in evid:
                info.append(f"ocr_votes={_g(evid[best], 'ocr_votes', {})}")
            if snap is not None and abs(t - snap) < 0.3:
                info.append("SNAP")
            if impact is not None and abs(t - impact) < 0.3:
                info.append("IMPACT")
            d.rectangle([0, 0, ow, 22 * len(info) + 8], fill=(0, 0, 0, 150))
            for j, line in enumerate(info):
                d.text((8, 5 + 22 * j), line, font=f_small, fill=(255, 255, 255, 255))
            ov.composite(frame, layer)
            proc.stdin.write(frame.tobytes())
        _close_writer(proc, log)
    finally:
        reader.close()
    return out_path
