"""'epic' edit of one vertical #52 clip (default ``render.style``).

Output timeline (typical 15-25 s)::

    coldopen  0.9 s   the hit at ~0.35x, punch-in, impact FX, riser
    flash     2 fr    hard cut to black (whoosh into the title)
    title     0.7 s   '#52' kinetic slam on a near-black graded plate
    rewind    0.45 s  'run it back': fast eased reverse scrub, duotone, motion blur
    intro     1.1 s   pre-snap freeze, feathered spotlight on #52, 1.05x push-in,
                      '52' slam + 'MLB  •  BUCHAREST REBELS' (sub drop)
    live      ramp    spotlight releases as the play starts; 1x through the read/
                      pursuit, eased 0.30x around the contact (frame-blended),
                      light-streak trail during the ramp, impact FX + boom,
                      kinetic words (READ. / CLOSE. / FINISH.) and the play label
                      from cand.caption only, never over the contact
    replay    ramp    tighter punch-in (<= max_digital_zoom), own 0.25x ramp, FX
    [angle / replay2] only when needed to reach render.min_clip_s
    endcard   1.8 s   best post-contact freeze -> red/black duotone, '#52' +
                      name (if known) + 'BUCHAREST REBELS', fade to black

Audio is assembled in numpy: game audio is re-timed with the same time maps
as the picture (slow-mo pitches down like tape and is low-passed by its
slowness), synthesized SFX (``audio.sfx``) are placed on impacts / slams and
duck the game audio; the result goes through ``audio_filter_graph`` loudnorm.
With a licensed music file, cut points and slams snap to its beats (+-120 ms).
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..audio import audio as au
from ..core.media import ffmpeg_bin, probe
from ..core.models import UNKNOWN, Candidate, EventResult, PlayerTrajectory, VideoInfo
from . import style as S
from .crop import OUT_H, OUT_W, CropPath, Xform, compose_frame, compute_crop_path, letterbox_frame, max_zoom

BEAT_TOL = 0.12
FREEZE_BEAT_TOL = 0.26
CONTACT_TYPES = ("tackle", "sack", "stop", "fumble", "pressure", "shed")


# ============================================================== small helpers
def _g(obj: Any, name: str, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _known(v: Any) -> bool:
    return v is not None and str(v).strip() != "" and str(v).strip().lower() != UNKNOWN


def caption_label(caption: str) -> Optional[str]:
    """The play label part of ``cand.caption`` ('#52 | MLB | TEAM — LABEL'), if any."""
    if not caption or " — " not in caption:
        return None
    lab = caption.split(" — ", 1)[1].strip()
    return lab or None


def zoom_cap(cfg: dict, path: CropPath) -> float:
    """Largest total zoom allowed by render.max_digital_zoom (may be < 1)."""
    rc = cfg.get("render", {})
    disp_w, _ = path.display_size
    base = disp_w / path.crop_w
    return float(rc.get("max_digital_zoom", 2.2)) / max(base, 1e-6)


def xform_for(path: CropPath, t: float, z: float = 1.0) -> Xform:
    x1, y1, x2, y2 = path.box(t, z)
    if path.mode == "vertical":
        return Xform(x1, y1, OUT_W / (x2 - x1), OUT_H / (y2 - y1))
    dw, dh = path.display_size
    return Xform(x1, y1, dw / (x2 - x1), dh / (y2 - y1), 0.0, float(int((OUT_H - dh) * 0.42)))


# ================================================================ timeline plan
def plan_epic(cand: Candidate, event: Any, cfg: dict, video_dur: float, zoom: float = 1.0,
              beats: Optional[list[float]] = None, style: Optional[S.Style] = None) -> dict:
    """Pure timeline plan for the epic edit (used by render + tests)."""
    st = style or S.get_style(cfg, "epic")
    rc = cfg.get("render", {})
    fps = float(rc.get("fps", 30))
    min_s, max_s = float(rc.get("min_clip_s", 15)), float(rc.get("max_clip_s", 60))
    vd = video_dur if video_dur and video_dur > 0 else max(cand.source_end_s + 5, 1.0)

    snap = cand.snap_time_s
    imp_ev = _g(event, "impact_s")
    impact_known = imp_ev is not None or cand.impact_time_s is not None
    impact = imp_ev if imp_ev is not None else cand.impact_time_s
    if impact is None:
        base0 = snap if snap is not None else cand.source_start_s
        impact = base0 + max(1.0, (cand.source_end_s - base0) * 0.6)
    impact = float(np.clip(impact, 0.5, max(0.6, vd - 0.3)))
    if snap is not None and not snap < impact:
        snap = None
    s_i = (snap - 0.35) if snap is not None else impact - 2.6
    s_i = min(s_i, impact - 1.4)
    s_i = max(0.0, s_i, impact - 8.0)
    lead_out = float(np.clip(rc.get("lead_out_s", 1.5), 1.0, 2.0))
    sp = float(np.clip(st.coldopen_speed, 0.25, 0.5))

    p = {"co_n": int(round(st.coldopen_s * fps)), "fl_n": 2, "ti_n": int(round(st.title_s * fps)),
         "rw_n": int(round(st.rewind_s * fps)), "in_n": int(round(st.intro_s * fps)),
         "lead_out": lead_out, "rep_pre": 1.7, "rep_post": 0.9, "angles": [], "rep2": False,
         "end_n": int(round(st.end_s * fps))}

    def build(p: dict) -> list[dict]:
        segs: list[dict] = []
        c0 = max(0.0, impact - 0.30 * sp)
        segs.append({"kind": "coldopen", "n": p["co_n"], "t0": c0, "speed": sp,
                     "impact_k": int(round((impact - c0) / sp * fps))})
        segs.append({"kind": "flash", "n": p["fl_n"]})
        segs.append({"kind": "title", "n": p["ti_n"]})
        segs.append({"kind": "rewind", "n": p["rw_n"], "from": min(impact + 0.15, vd), "to": s_i})
        segs.append({"kind": "intro", "n": p["in_n"], "t0": s_i})
        s1 = max(min(vd - 1.0 / fps, impact + p["lead_out"]), impact + 0.4)
        live = S.RampMap(s_i, s1, [(impact - 0.12, impact + 0.18)], st.ramp_speed, 0.32, fps)
        segs.append({"kind": "live", "n": live.n, "ramp": live, "t0": s_i, "t1": s1})
        r0 = max(s_i, impact - p["rep_pre"], (snap - 0.3) if snap is not None else -1e9)
        r1 = min(s1, impact + p["rep_post"])
        rep = S.RampMap(r0, r1, [(impact - 0.15, impact + 0.22)], st.replay_ramp_speed, 0.4, fps)
        segs.append({"kind": "replay", "n": rep.n, "ramp": rep, "t0": r0, "t1": r1, "zoom": zoom})
        for a, d in p["angles"]:
            segs.append({"kind": "angle", "n": int(round(d * fps)), "t0": a, "speed": 1.0})
        if p["rep2"]:
            q0, q1 = max(s_i, impact - 1.3), min(s1, impact + 0.8)
            r2 = S.RampMap(q0, q1, [(impact - 0.1, impact + 0.25)], st.ramp_speed, 0.35, fps)
            segs.append({"kind": "replay2", "n": r2.n, "ramp": r2, "t0": q0, "t1": q1,
                         "zoom": 1.0 + (zoom - 1.0) * 0.5})
        segs.append({"kind": "endcard", "n": p["end_n"], "t0": min(s1, impact + 0.45)})
        t = 0.0
        for s in segs:
            s["out_start"] = round(t, 4)
            t += s["n"] / fps
            s["out_end"] = round(t, 4)
        return segs

    def total(p: dict) -> float:
        return sum(s["n"] for s in build(p)) / fps

    # ---- pad to the platform minimum with real footage first, cards last
    target = min_s + 0.25
    need = target - total(p)
    if need > 0:
        p["lead_out"] = min(2.2, p["lead_out"] + need)
        need = target - total(p)
    for rs in (cand.replay_segments or [])[:2]:
        if need <= 0.5:
            break
        a, b = float(rs.get("start_s", 0)), min(float(rs.get("end_s", 0)), vd)
        if b - a < 1.0:
            continue
        d = min(b - a, 6.0, need + 0.5)
        if d < 1.0:
            break
        p["angles"].append((a + max(0.0, (b - a - d) / 2), d))
        need = target - total(p)
    if need > 0:
        p["rep_pre"] = min(3.2, p["rep_pre"] + need)
        need = target - total(p)
    if need > 0.8:
        p["rep2"] = True
        need = target - total(p)
    if need > 0:
        add = min(int(np.ceil(need * fps)), int(0.8 * fps))
        p["end_n"] += add
        need = target - total(p)
    if need > 0:
        p["in_n"] += min(int(np.ceil(need * fps)), int(0.5 * fps))
        need = target - total(p)
    if need > 0:
        p["end_n"] += int(np.ceil(need * fps))

    # ---- snap cut points to the music grid (+-120 ms)
    snapped = []
    if beats:
        order = [("co_n", "coldopen", "f", 18, 36), ("fl_n", "flash", "f", 1, 5),
                 ("ti_n", "title", "f", 15, 30), ("in_n", "intro", "f", 24, 48),
                 ("lead_out", "live", "s", 0.8, 2.6), ("rep_post", "replay", "s", 0.5, 1.4),
                 ("end_n", "endcard", "f", 42, 90)]
        for key, kind, typ, lo, hi in order:
            segs = build(p)
            seg = next(s for s in segs if s["kind"] == kind)
            end = seg["out_end"]
            # action-tied cuts move <= 120 ms; freeze cards (no natural moment) may
            # stretch up to ~half a beat so the cut still lands on the grid
            tol = BEAT_TOL if kind in ("coldopen", "live", "replay") else FREEZE_BEAT_TOL
            tgt = au.snap_to_beat(end, beats, tol)
            d = tgt - end
            if abs(d) < 0.5 / fps:
                continue
            if typ == "f":
                p[key] = int(np.clip(p[key] + int(round(d * fps)), lo, hi))
            else:  # 1x tail of a ramp: source seconds == output seconds
                p[key] = float(np.clip(p[key] + d, lo, hi))
            snapped.append(kind)
        if total(p) < min_s:
            p["end_n"] += int(np.ceil((min_s + 0.05 - total(p)) * fps))

    segs = build(p)
    dur = sum(s["n"] for s in segs) / fps
    if dur > max_s:  # never exceed the platform maximum
        cut = int(np.ceil((dur - max_s) * fps))
        segs[-1]["n"] = max(int(1.2 * fps), segs[-1]["n"] - cut)
        dur = sum(s["n"] for s in segs) / fps
    live = next(s for s in segs if s["kind"] == "live")
    rep = next(s for s in segs if s["kind"] == "replay")
    return {"segments": segs, "fps": fps, "duration": dur, "impact": impact,
            "impact_known": impact_known, "snap": snap, "s0": s_i, "s1": live["t1"],
            "r0": rep["t0"], "r1": rep["t1"], "zoom": zoom, "params": p,
            "beat_snapped": snapped}


def seg_source_times(seg: dict, fps: float) -> Optional[np.ndarray]:
    """Source time of every output frame of a moving segment (None for cards)."""
    n = seg["n"]
    k = np.arange(n)
    kind = seg["kind"]
    if "ramp" in seg:
        return seg["ramp"].frames()
    if kind in ("coldopen", "angle"):
        return seg["t0"] + k / fps * seg["speed"]
    if kind == "rewind":
        u = np.array([S.ease_in_out(x) for x in (k / max(1, n - 1))])
        return seg["from"] + (seg["to"] - seg["from"]) * u
    if kind in ("intro", "endcard"):
        return np.full(n, seg["t0"])
    return None


# =================================================================== renderer
class EpicClip:
    def __init__(self, cand: Candidate, info: VideoInfo, trajectory: Any, event: Any,
                 cfg: dict, st: S.Style, music_file: Optional[str]):
        from .render import FrameReader, _BoxTrack
        self.cand, self.info, self.event, self.cfg, self.st = cand, info, event, cfg, st
        self.rc = cfg.get("render", {})
        self.reader = FrameReader(info.path)
        r = self.reader
        self.src_w, self.src_h = r.w or info.width, r.h or info.height
        pinfo = probe(info.path)
        self.has_audio = bool(pinfo.get("has_audio"))
        self.vdur = info.duration_s or (r.n / r.fps if r.n else 0.0) or pinfo.get("duration", 0.0)
        self.pl = _BoxTrack(_g(trajectory, "times", []), _g(trajectory, "boxes", []))
        self.tg = _BoxTrack(_g(trajectory, "target_times", []), _g(trajectory, "target_boxes", []))
        self.trajectory = trajectory
        self.beats = None
        if music_file and Path(music_file).is_file():
            try:
                b, _ = au.detect_beats(music_file)
                mdur = float(probe(music_file).get("duration") or 0.0)
                self.beats = au.tile_beats(b, mdur, 65.0) if mdur > 0 else b
            except Exception:  # noqa: BLE001 - beat grid is a nicety
                self.beats = None
        plan0 = plan_epic(cand, event, cfg, self.vdur, 1.0, None, st)
        fps = plan0["fps"]
        ts = [seg_source_times(s, fps) for s in plan0["segments"] if s["kind"] != "angle"]
        ts = np.concatenate([t for t in ts if t is not None])
        self.path = compute_crop_path(trajectory, event, self.src_w, self.src_h,
                                      max(0.0, float(ts.min()) - 1.0), float(ts.max()) + 0.5,
                                      cfg, fps=fps)
        zoom = max_zoom(cfg, self.path)
        cap = zoom_cap(cfg, self.path)
        self.fx_zoom = max(1.0, cap) * st.fx_zoom_allowance      # momentary FX / freeze push-ins
        self.plan = plan_epic(cand, event, cfg, self.vdur, zoom, self.beats, st)
        self.fps = fps
        self.impact = self.plan["impact"]
        self.contact = _g(event, "contact_point")
        pt = str(_g(event, "play_type", cand.play_type) or "")
        self.has_contact = self.plan["impact_known"] and (
            self.contact is not None or any(w in pt for w in CONTACT_TYPES))
        self.grader = S.Grader(st, (OUT_W, OUT_H))
        self.accent = st.accent
        self.accent_bgr = st.accent_bgr
        p = cfg.get("player", {}) or {}
        self.num = str(p.get("player_number", cand.player_number or 52))
        self.pos = str(p.get("position_short", cand.position or "MLB")).upper()
        self.team = str(p.get("team_name", "Bucharest Rebels")).upper()
        name = p.get("player_name", UNKNOWN)
        if not _known(name) and _known(cand.player_name):
            name = cand.player_name
        self.name = str(name).upper() if _known(name) else None
        self.label = caption_label(cand.caption or "")
        self.k_global = 0
        self.sfx_events: list[tuple[str, float, float, bool]] = []   # (name, t, gain, align_end)
        self.impacts_out: list[float] = []
        self.crop_frames: list[dict] = []

    # ------------------------------------------------------------ geometry
    def player_canvas(self, xf: Xform, t: float) -> Optional[list[float]]:
        b = self.pl.at(t)
        return None if b is None else xf.box(b)

    def occupied(self, ts: np.ndarray, zoom: float = 1.0, contact: bool = True) -> list[tuple[float, float]]:
        occ = []
        for t in ts:
            xf = xform_for(self.path, float(t), zoom)
            for tr in (self.pl, self.tg):
                b = tr.at(float(t), hold=0.2)
                if b is not None:
                    cb = xf.box(b)
                    m = 0.25 * (cb[3] - cb[1])
                    occ.append((cb[1] - m, cb[3] + m))
            if contact and self.contact is not None and abs(t - self.impact) < 0.8:
                cy = xf(self.contact[0], self.contact[1])[1]
                occ.append((cy - 170, cy + 170))
        return occ

    def fx_center(self, xf: Xform, t: float) -> tuple[float, float]:
        if self.contact is not None:
            c = xf(self.contact[0], self.contact[1])
        else:
            b = self.pl.at(t)
            c = ((xf.box(b)[0] + xf.box(b)[2]) / 2, (xf.box(b)[1] + xf.box(b)[3]) / 2) if b is not None \
                else (OUT_W / 2, OUT_H * 0.45)
        return float(np.clip(c[0], 0, OUT_W)), float(np.clip(c[1], 0, OUT_H))

    # ------------------------------------------------------------ text plan
    def build_text(self) -> list[S.Slam]:
        st, fps, plan = self.st, self.fps, self.plan
        segs = {s["kind"]: s for s in plan["segments"]}
        slams: list[S.Slam] = []
        snap_b = (lambda t: au.snap_to_beat(t, self.beats, BEAT_TOL)) if self.beats else (lambda t: t)
        W = OUT_W
        # title: '#52' over the dark plate
        ti = segs["title"]
        t_in = snap_b(ti["out_start"] + 1 / fps)
        big = S.TextSprite(f"#{self.num}", "display", 470, underline=self.accent)
        slams.append(S.Slam(big, t_in, ti["out_end"] - 0.12, W / 2, OUT_H * 0.46, "slam",
                            exit_s=0.1, underline_wipe=0.2, delay_ul=0.12, drift=0.05, shake=True,
                            tag="title"))
        # intro: '52' + position/team (+ name) away from the spotlight
        it = segs["intro"]
        occ = self.occupied(np.array([it["t0"]]), 1.05, contact=False)
        num = S.TextSprite(self.num, "display", 400)
        sub = S.TextSprite(f"{self.pos}  •  {self.team}", "text", 46, tracking=0.28, weight="Medium",
                           fill=(235, 235, 235))
        nm = S.TextSprite(self.name, "title", 96) if self.name else None
        stack_h = num.text_h + 30 + sub.text_h + (nm.text_h + 18 if nm else 0)
        yc = S.text_zone(occ, OUT_H, stack_h + 40, (0.25, 0.72, 0.3, 0.68)) or OUT_H * 0.25
        y = yc - stack_h / 2
        t0 = snap_b(it["out_start"] + 0.1)
        t_exit = it["out_end"] + 0.06
        slams.append(S.Slam(num, t0, t_exit, W / 2, y + num.text_h / 2, "slam", exit_s=0.16,
                            drift=0.03, shake=True, tag="intro"))
        y += num.text_h + 30
        if nm:
            slams.append(S.Slam(nm, t0 + 0.2, t_exit, W / 2, y + nm.text_h / 2, "rise", exit_s=0.16))
            y += nm.text_h + 18
        slams.append(S.Slam(sub, t0 + 0.3, t_exit, W / 2, y + sub.text_h / 2, "rise", exit_s=0.16))
        # live: action words (descriptive verbs) + play label from the caption only
        if st.text:
            lv = segs["live"]
            ramp: S.RampMap = lv["ramp"]
            o0 = lv["out_start"]
            I, snap = self.impact, plan["snap"]
            words: list[tuple[str, float, float]] = []    # (text, out_in, out_out)
            close = None
            if snap is not None and I - snap >= 1.5 and self._moved(snap, I):
                a = o0 + float(ramp.out(max(snap + 0.75, I - 1.6)))
                b = min(a + 1.1, o0 + float(ramp.out(I - 0.3)))
                if b - a >= 0.35:
                    close = ("CLOSE.", a, b)
            if snap is not None and I - snap >= 0.9:
                a = o0 + float(ramp.out(snap)) + 0.02
                b = min(a + 0.75, o0 + float(ramp.out(I - 0.55)))
                if close:
                    b = min(b, close[1] - 0.16)
                words.append(("READ.", a, b))
            if close:
                words.append(close)
            post_end = lv["out_end"] - 0.05
            if self.has_contact:
                a = o0 + float(ramp.out(I + 0.03))
                b = post_end if not self.label else min(post_end, a + 0.85)
                words.append(("FINISH.", a, b))
            for txt, a, b in words:
                a = snap_b(a)
                if b - a < 0.3:
                    continue
                spr = S.TextSprite(txt, "display", 190)
                win = ramp.src(np.linspace(a - o0, b - o0, 6))
                yc = S.text_zone(self.occupied(win), OUT_H, spr.text_h + 30)
                if yc is None:
                    continue
                slams.append(S.Slam(spr, a, b, W / 2, yc, "slam", exit_s=0.12, drift=0.04, tag="word"))
            if self.label:
                a = o0 + float(ramp.out(I + 0.03))
                if any(s.tag == "word" and abs(s.t_in - a) < 0.05 for s in slams):
                    a = a + 0.85
                a = snap_b(a)
                b = lv["out_end"] - 0.02
                if b - a >= 0.5:
                    spr = self._label_sprite(self.label)
                    win = ramp.src(np.linspace(a - o0, b - o0, 6))
                    yc = S.text_zone(self.occupied(win), OUT_H, spr.text_h + 30)
                    if yc is not None:
                        slams.append(S.Slam(spr, a, b, W / 2, yc, "slam", exit_s=0.12,
                                            underline_wipe=0.22, delay_ul=0.12, drift=0.03, tag="label"))
        # end card stack
        ec = segs["endcard"]
        occ = self.occupied(np.array([ec["t0"]]), 1.0, contact=False)
        hs = S.TextSprite(f"#{self.num}", "display", 360, glyph_fills={0: self.accent})
        tm = S.TextSprite(self.team, "text", 44, tracking=0.34, weight="Medium", underline=self.accent)
        nm = S.TextSprite(self.name, "title", 120) if self.name else None
        stack_h = hs.text_h + 26 + tm.text_h + (nm.text_h + 16 if nm else 0)
        yc = S.text_zone(occ, OUT_H, stack_h + 40, (0.3, 0.7, 0.25, 0.74)) or OUT_H * 0.3
        y = yc - stack_h / 2
        t0 = snap_b(ec["out_start"] + 4 / fps)
        t_end = ec["out_end"] + 1.0         # held through the fade to black
        slams.append(S.Slam(hs, t0, t_end, W / 2, y + hs.text_h / 2, "slam", drift=0.02,
                            shake=True, tag="end"))
        y += hs.text_h + 16
        if nm:
            slams.append(S.Slam(nm, t0 + 0.18, t_end, W / 2, y + nm.text_h / 2, "rise"))
            y += nm.text_h + 26
        else:
            y += 10
        slams.append(S.Slam(tm, t0 + 0.3, t_end, W / 2, y + tm.text_h / 2, "rise",
                            underline_wipe=0.3, delay_ul=0.15))
        return slams

    def _moved(self, a: float, b: float) -> bool:
        pa, pb = self.pl.at(a), self.pl.at(b)
        if pa is None or pb is None:
            return False
        w = max(1.0, pa[2] - pa[0])
        return float(np.hypot((pb[0] + pb[2] - pa[0] - pa[2]) / 2, (pb[1] + pb[3] - pa[1] - pa[3]) / 2)) > 1.5 * w

    def _label_sprite(self, label: str) -> S.TextSprite:
        f_max = 0.86 * OUT_W
        for size in (150, 132, 116, 100, 88):
            f = S.font("display", size)
            if f.getlength(label) <= f_max:
                return S.TextSprite(label, "display", size, underline=self.accent)
        # two lines, split at the most balanced space / arrow
        words = label.split(" ")
        best = None
        for i in range(1, len(words)):
            l1, l2 = " ".join(words[:i]), " ".join(words[i:])
            d = abs(len(l1) - len(l2))
            if best is None or d < best[0]:
                best = (d, [l1, l2])
        lines = best[1] if best else [label]
        for size in (130, 112, 96, 84, 72):
            f = S.font("display", size)
            if max(f.getlength(ln) for ln in lines) <= f_max:
                return S.TextSprite(lines, "display", size, underline=self.accent)
        return S.TextSprite(lines, "display", 64, underline=self.accent)

    # ------------------------------------------------------------ frames
    def grade(self, frame: np.ndarray, mono: float = 0.0) -> np.ndarray:
        out = self.grader(frame, self.k_global, mono=mono)
        return out

    def compose(self, t: float, z: float = 1.0, blend: bool = False) -> tuple[np.ndarray, Xform]:
        src = self.reader.at(t, blend=blend)
        return compose_frame(src, self.path, t, z, self.accent_bgr)

    def streak(self, frame: np.ndarray, xf: Xform, t: float, alpha: float) -> None:
        if alpha <= 0.02 or not self.pl.ok:
            return
        ts = np.arange(t - 1.0, t - 0.04, 1 / 30)
        pts = []
        for tt in ts:
            b = self.pl.at(float(tt), hold=0.0)
            if b is not None:
                pts.append(xf((b[0] + b[2]) / 2, b[1] + 0.62 * (b[3] - b[1])))
        if len(pts) >= 4:
            b = self.pl.at(t)
            bw = (xf.box(b)[2] - xf.box(b)[0]) if b is not None else 90
            S.light_streak(frame, pts, self.accent_bgr, float(np.clip(bw * 0.45, 18, 56)), alpha)

    def render_frames(self, writer) -> None:
        plan, fps, st = self.plan, self.fps, self.st
        segs = plan["segments"]
        slams = getattr(self, "slams", None) or self.build_text()
        self.slams = slams
        I = self.impact
        byk = {s["kind"]: s for s in segs}
        # pre-render the rewind (it reads the source backwards)
        rw = byk["rewind"]
        rts = seg_source_times(rw, fps)
        rew: dict[int, np.ndarray] = {}
        prev = float(rw["from"])
        steps = [(k, float(rts[k]), float(rts[k - 1]) if k else prev) for k in range(rw["n"])]
        for k, t, tp in sorted(steps, key=lambda x: x[1]):
            samples = np.linspace(t, tp, 4) if abs(tp - t) > 1.5 / fps else [t]
            acc = None
            for tt in samples:
                f = self.reader.at(float(tt)).astype(np.float32)
                acc = f if acc is None else acc + f
            src = (acc / len(samples)).astype(np.uint8)
            fr, _ = compose_frame(src, self.path, t, 1.0 + 0.04 * (1 - k / max(1, rw["n"] - 1)),
                                  self.accent_bgr)
            rew[k] = fr
        # title plate: near-black blurred impact frame
        import cv2
        plate, _ = self.compose(I, 1.0)
        small = cv2.resize(plate, (OUT_W // 8, OUT_H // 8), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), 3)
        plate = (cv2.resize(small, (OUT_W, OUT_H)).astype(np.float32) * 0.16).astype(np.uint8)

        def overlay_text(frame: np.ndarray, tg: float) -> np.ndarray:
            shake = None
            for sl in slams:
                if sl.state(tg) is not None:
                    sl.draw(frame, tg)
                    lk = sl.landed(tg)
                    if sl.shake and lk is not None:
                        shake = lk
            if shake is not None:
                dx, dy = S.fx_shake(shake, 12.0 * st.fx, 7, seed=11)
                frame = S.warp_canvas(frame, 1.0, dx, dy)
            return frame

        for seg in segs:
            kind, n = seg["kind"], seg["n"]
            o0 = seg["out_start"]
            times = seg_source_times(seg, fps)
            for k in range(n):
                tg = o0 + k / fps
                frame = None
                if kind == "coldopen":
                    t = float(times[k])
                    zc = min(1.2, self.fx_zoom)
                    z = 1.0 + (zc - 1.0) * S.ease_out_cubic(k / max(1, n - 1))
                    frame, xf = self.compose(t, z, blend=True)
                    frame = self.grade(frame)
                    ki = k - seg["impact_k"]
                    if ki == 0:
                        self.impacts_out.append(tg)
                    if self.plan["impact_known"]:
                        frame = S.impact_fx(frame, ki, st, self.fx_center(xf, t))
                    if k >= n - 3:   # crush into the cut
                        frame = cv2.convertScaleAbs(frame, alpha=1.0 - 0.18 * (k - n + 4))
                elif kind == "flash":
                    frame = self.grade(np.zeros((OUT_H, OUT_W, 3), np.uint8))
                elif kind == "title":
                    frame = self.grade(plate.copy(), mono=0.6)
                elif kind == "rewind":
                    frame = self.grade(rew[k].copy(), mono=0.55)
                elif kind == "intro":
                    t = seg["t0"]
                    z = 1.0 + (min(1.05, self.fx_zoom) - 1.0) * S.ease_in_out(k / max(1, n - 1))
                    frame, xf = self.compose(t, z)
                    frame = self.grade(frame)
                    frame = self._spot(frame, xf, t, S.ease_out_cubic((k + 1) / 5))
                elif kind in ("live", "replay", "replay2"):
                    ramp: S.RampMap = seg["ramp"]
                    t = float(times[k])
                    tau = k / fps
                    zb = seg.get("zoom", 1.0)
                    if kind == "live":
                        rel = S.ease_in_out(tau / 0.4)
                        z = 1.0 + (min(1.05, self.fx_zoom) - 1.0) * (1 - rel)
                    else:
                        z = zb * (1.0 + 0.03 * tau / max(1e-3, n / fps))
                    frame, xf = self.compose(t, z, blend=float(ramp.speed_src(np.array([t]))[0]) < 0.97)
                    if kind == "live":
                        b = self.pl.at(t)
                        self.crop_frames.append({
                            "t": round(t, 4), "crop": [round(v, 1) for v in self.path.box(t, z)],
                            "player_x": None if b is None else round(float(b[0] + b[2]) / 2, 1)})
                    frame = self.grade(frame)
                    if kind == "live" and tau < 0.4:
                        frame = self._spot(frame, xf, t, 1 - S.ease_in_out(tau / 0.4))
                    w = float(ramp.weight(np.array([t]))[0])
                    if w > 0.02 and t < I - 0.05:
                        self.streak(frame, xf, t, min(1.0, w * 1.6) * S.clamp01((I - 0.05 - t) / 0.12))
                    ki = int(round((tau - float(ramp.out(I))) * fps))
                    if ki == 0:
                        self.impacts_out.append(tg)
                    if self.plan["impact_known"] and 0 <= ki <= 10:
                        frame = S.impact_fx(frame, ki, st, self.fx_center(xf, t),
                                            seed=3 + len(self.impacts_out))
                    if kind != "live" and k < 2:   # hard cut with a 2-frame flash
                        frame = S.flash(frame, (0.28, 0.1)[k])
                elif kind == "angle":
                    t = float(times[k])
                    frame = self.grade(letterbox_frame(self.reader.at(t)))
                elif kind == "endcard":
                    t = seg["t0"]
                    z = 1.0 + 0.045 * S.ease_out_cubic(k / max(1, n - 1))
                    frame, xf = self.compose(t, z)
                    mono = S.ease_in_out(k / 7)
                    frame = self.grade(frame, mono=mono)
                    frame = cv2.convertScaleAbs(frame, alpha=1.0 - 0.22 * mono)
                    if k < 2:
                        frame = S.flash(frame, (0.3, 0.12)[k])
                if frame is None:
                    frame = np.zeros((OUT_H, OUT_W, 3), np.uint8)
                frame = overlay_text(frame, tg)
                if kind == "endcard":
                    fade = 0.3
                    rem = (n - 1 - k) / fps
                    if rem < fade:
                        frame = cv2.convertScaleAbs(frame, alpha=S.clamp01(rem / fade) ** 1.5)
                writer(frame)
                self.k_global += 1

    def _spot(self, frame: np.ndarray, xf: Xform, t: float, amount: float) -> np.ndarray:
        b = self.pl.at(t)
        if b is None or amount <= 0.01:
            return frame
        cb = xf.box(b)
        bw, bh = cb[2] - cb[0], cb[3] - cb[1]
        c = ((cb[0] + cb[2]) / 2, (cb[1] + cb[3]) / 2 - 0.04 * bh)
        ax = (max(bw * 1.25, 120.0), max(bh * 0.95, 170.0))
        return S.spotlight(frame, c, ax, amount)

    # ------------------------------------------------------------ audio
    def build_audio(self) -> np.ndarray:
        sr, fps, plan = au.SAMPLE_RATE, self.fps, self.plan
        segs = plan["segments"]
        total = sum(s["n"] for s in segs) / fps
        N = int(round(total * sr))
        st_t = np.full(N, np.nan)
        gain = np.zeros(N, np.float32)
        slow = np.zeros(N, np.float32)
        I = self.impact
        for s in segs:
            i0, i1 = int(round(s["out_start"] * sr)), min(N, int(round(s["out_end"] * sr)))
            if i1 <= i0:
                continue
            tau = np.arange(i1 - i0) / sr
            kind = s["kind"]
            g, w, tt = 0.0, 0.0, None
            if kind == "coldopen":
                tt, g, w = s["t0"] + tau * s["speed"], 0.95, 0.75
            elif kind == "rewind":
                d = s["n"] / fps
                u = np.clip(tau / d, 0, 1)
                u = np.where(u < 0.5, 4 * u ** 3, 1 - (-2 * u + 2) ** 3 / 2)
                tt, g, w = s["from"] + (s["to"] - s["from"]) * u, 0.3, 0.3
            elif kind == "intro":
                d = s["n"] / fps
                tt, g, w = s["t0"] - d + tau, 0.5, 0.6
            elif kind in ("live", "replay", "replay2"):
                ramp = s["ramp"]
                tt = ramp.src(tau)
                g = 1.0 if kind == "live" else 0.6
                w = ramp.weight(tt)
                if kind != "live":
                    w = np.maximum(w, 0.3)
            elif kind == "angle":
                tt, g, w = s["t0"] + tau, 0.5, 0.2
            elif kind == "endcard":
                d = s["n"] / fps
                tt, g, w = s["t0"] + tau, 0.35 * np.clip(1 - tau / d, 0, 1) ** 1.5, 0.75
            if tt is None:
                continue
            ge = np.full(i1 - i0, 1.0, np.float32) * g
            fl = min(len(ge) // 2, int(0.006 * sr))
            if fl > 0:
                ge[:fl] *= np.linspace(0, 1, fl)
                ge[-fl:] *= np.linspace(1, 0, fl)
            st_t[i0:i1] = tt
            gain[i0:i1] = ge
            slow[i0:i1] = w
        game = np.zeros((N, 2), np.float32)
        if self.has_audio:
            ok = np.isfinite(st_t)
            if np.any(ok):
                a_lo = max(0.0, float(np.nanmin(st_t)) - 0.5)
                a_hi = float(np.nanmax(st_t)) + 0.5
                src = au.decode_audio(self.info.path, a_lo, a_hi - a_lo, sr, 2)
                game = au.varispeed(src, a_lo, st_t, sr)
        lp = au.lowpass_fft(game, sr, 650.0) if len(game) else game
        slow = au.moving_average(slow, int(0.03 * sr)).astype(np.float32)
        game = (game * (1 - slow[:, None]) + lp * slow[:, None] * 1.25) * gain[:, None]
        # ---- SFX
        fx = np.zeros_like(game)
        byk = {s["kind"]: s for s in segs}

        def put(name, t, g, end=False):
            au.place(fx, au.sfx(name, self.cfg, sr), t, g, sr, align_end=end)
            self.sfx_events.append((name, round(float(t), 3), g, end))

        co = byk["coldopen"]
        put("riser", co["out_end"], 0.45, True)
        if plan["impact_known"]:
            put("boom", co["out_start"] + co["impact_k"] / fps, 0.85)
        put("whoosh", byk["title"]["out_start"] + 1 / fps, 0.5, True)
        put("boom", byk["title"]["out_start"] + 1 / fps + 0.08, 0.45)
        put("rewind", byk["rewind"]["out_start"], 0.45)
        put("subdrop", byk["intro"]["out_start"] + 0.1, 0.8)
        for s in segs:
            if s["kind"] not in ("live", "replay", "replay2"):
                continue
            ramp = s["ramp"]
            h0 = ramp.holds[0][0] - ramp.ease * 0.5
            put("whoosh", s["out_start"] + float(ramp.out(h0)), 0.45, True)
            if plan["impact_known"]:
                put("boom", s["out_start"] + float(ramp.out(I)), 0.9 if s["kind"] == "live" else 0.7)
        for sl in getattr(self, "slams", []):
            if sl.tag in ("word", "label"):
                put("whoosh_short", sl.t_in + 0.06, 0.22, True)
            elif sl.tag == "end":
                put("boom", sl.t_in + 0.08, 0.55)
        fx = fx[:N]
        duck = au.duck_gain(fx, sr, 0.45)
        out = game * duck[:, None] + fx * 0.8
        return np.clip(out, -1.0, 1.0)


class _ThreadedPipe:
    """Writes frames to a pipe from a worker thread (bounded queue)."""

    def __init__(self, stream, depth: int = 6):
        import queue
        import threading
        self.q: "queue.Queue" = queue.Queue(maxsize=depth)
        self.stream, self.err = stream, None
        self.th = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while True:
            b = self.q.get()
            if b is None:
                return
            if self.err is None:
                try:
                    self.stream.write(b)
                except BaseException as e:  # noqa: BLE001
                    self.err = e

    def put(self, frame: np.ndarray) -> None:
        if self.err is not None:
            raise self.err
        self.q.put(np.ascontiguousarray(frame).tobytes())

    def __enter__(self):
        self.th.start()
        return self

    def __exit__(self, *exc):
        self.q.put(None)
        self.th.join()
        if self.err is not None and exc[0] is None:
            raise self.err
        return False


# ================================================================== render
def render_epic(cand: Candidate, info: VideoInfo, trajectory: PlayerTrajectory, event: EventResult,
                cfg: dict, out_dir: str, music_file: Optional[str] = None,
                style: Optional[S.Style] = None) -> dict:
    import cv2
    from .render import (AUDIO_CODEC, _close_writer, _open_writer, _video_codec, _visibility_pick,
                         clip_basename, write_music_version, write_seed_frame)
    st = style or S.get_style(cfg, "epic")
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    base = clip_basename(cand)
    out_mp4 = out_dir_p / f"{base}.mp4"
    out_jpg = out_dir_p / f"{base}.jpg"

    clip = EpicClip(cand, info, trajectory, event, cfg, st, music_file)
    plan, fps, path = clip.plan, clip.fps, clip.path
    segs = plan["segments"]
    # end card freeze: best post-contact frame
    ec = next(s for s in segs if s["kind"] == "endcard")
    I = plan["impact"]
    lo, hi = min(I + 0.12, plan["s1"]), min(plan["s1"], I + 0.95)
    ec["t0"] = _visibility_pick(clip.pl, clip.tg, lo, max(lo, hi), path) if clip.pl.ok else ec["t0"]

    tmp_mp4 = out_mp4.with_suffix(".tmp.mp4")
    wav = Path(tempfile.mkstemp(suffix=".wav", prefix="epic_")[1])
    cmd = [ffmpeg_bin(), "-y", "-hide_banner", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{OUT_W}x{OUT_H}", "-r", str(int(fps)),
           "-i", "pipe:0", "-i", str(wav),
           "-filter_complex", "[1:a]anull[game];" + au.audio_filter_graph(False, cfg),
           "-map", "0:v", "-map", "[aout]"] + _video_codec(cfg) + AUDIO_CODEC + \
        ["-shortest", "-f", "mp4", str(tmp_mp4)]
    try:
        # audio needs the slam times -> build text first (cheap), render audio, then frames
        clip.slams = clip.build_text()
        au.write_wav(wav, clip.build_audio(), au.SAMPLE_RATE)
        proc, log = _open_writer(cmd)
        try:
            with _ThreadedPipe(proc.stdin) as pipe:   # overlap composition with x264
                clip.render_frames(pipe.put)
            _close_writer(proc, log)
        except BaseException:
            try:
                proc.kill()
            finally:
                log.close()
            raise
    except BaseException:
        if tmp_mp4.exists():
            tmp_mp4.unlink()
        clip.reader.close()
        raise
    finally:
        if wav.exists():
            wav.unlink()
    os.replace(tmp_mp4, out_mp4)

    # ---- poster thumbnail: graded impact frame + '52' in the display font
    t_th = _visibility_pick(clip.pl, clip.tg, max(plan["s0"], I - 0.45), max(plan["s0"], I + 0.05), path) \
        if clip.pl.ok else I
    thumb = poster(clip, t_th)
    cv2.imwrite(str(out_jpg), thumb, [cv2.IMWRITE_JPEG_QUALITY, 92])
    seed_jpg, seed_t = write_seed_frame(clip.reader, out_dir_p, base, plan["snap"], plan["s0"])
    clip.reader.close()

    crop_json = out_dir_p / f"{base}.crop.json"
    crop_json.write_text(json.dumps({"mode": path.mode, "safe_region": path.safe_region,
                                     "frame_size": [clip.src_w, clip.src_h], "frames": clip.crop_frames}))
    duration = sum(s["n"] for s in segs) / fps
    music_out = write_music_version(out_mp4, music_file, duration, cfg, out_dir_p, base)
    live = next(s for s in segs if s["kind"] == "live")
    impact_out = live["out_start"] + float(live["ramp"].out(I))
    timeline = {
        "basename": base, "style": "epic", "fps": fps, "duration": duration,
        "source": {k: plan[k] for k in ("s0", "s1", "snap", "impact", "r0", "r1")},
        "impact_known": plan["impact_known"],
        "impact_out": round(impact_out, 4),
        "impacts_out": [round(t, 4) for t in clip.impacts_out],
        "segments": [{"kind": s["kind"], "start": s["out_start"], "end": s["out_end"],
                      "src_t0": s.get("t0", s.get("from")), "src_t1": s.get("t1", s.get("to")),
                      "speed": s.get("speed", (s["ramp"].vmin if "ramp" in s else 0.0))}
                     for s in segs],
        "text": [{"text": getattr(sl, "tag", ""), "in": round(sl.t_in, 3), "out": round(sl.t_out, 3)}
                 for sl in clip.slams],
        "sfx": [{"name": n, "t": t, "gain": g} for n, t, g, _ in clip.sfx_events],
        "beat_snapped": plan["beat_snapped"],
        "crop": path.summary(), "zoom": plan["zoom"]}
    tl_path = out_dir_p / f"{base}.timeline.json"
    tl_path.write_text(json.dumps(timeline, indent=1))
    for r in path.review_reasons:
        if r not in cand.review_reasons:
            cand.review_reasons.append(r)
    cand.output_file, cand.thumbnail_file = str(out_mp4), str(out_jpg)
    cand.output_duration_s = round(duration, 3)
    return {"output_file": str(out_mp4), "thumbnail_file": str(out_jpg),
            "output_duration_s": round(duration, 3),
            "music_file": str(music_out) if music_out else None,
            "crop_mode": path.mode, "crop_quality": round(path.quality, 3),
            "review_reasons": list(path.review_reasons), "style": "epic",
            "timeline_file": str(tl_path), "crop_file": str(crop_json), "seed_frame": str(seed_jpg),
            "seed_frame_size": [int(clip.src_w), int(clip.src_h)], "seed_frame_t": round(seed_t, 3)}


def poster(clip: EpicClip, t: float) -> np.ndarray:
    """Poster-style thumbnail: graded frame, heavy vignette, huge '52'."""
    import cv2
    st = clip.st
    frame, xf = clip.compose(t, 1.0)
    frame = clip.grader(frame, 0, grain=True)
    frame = cv2.multiply(frame, clip.grader.vig, scale=1 / 255.0)
    occ = clip.occupied(np.array([t]), 1.0, contact=False)
    num = S.TextSprite(clip.num, "display", 540)
    tm = S.TextSprite(clip.team, "text", 46, tracking=0.34, weight="Medium", underline=st.accent)
    h = num.text_h + 24 + tm.text_h
    yc = (S.text_zone(occ, OUT_H, h + 40, (0.7, 0.26, 0.66, 0.3, 0.74, 0.22), gap=60)
          or S.text_zone(occ, OUT_H, h + 40, (0.7, 0.26, 0.66, 0.3, 0.74, 0.22)) or OUT_H * 0.7)
    # soft dark gradient behind the type for legibility
    g = np.clip(1 - np.abs(np.arange(OUT_H) - yc) / (h * 0.9), 0, 1) ** 1.5 * 0.45
    frame = (frame.astype(np.float32) * (1 - g)[:, None, None]).astype(np.uint8)
    y = yc - h / 2
    S.blit(frame, num, OUT_W / 2, y + num.text_h / 2)
    S.blit(frame, tm, OUT_W / 2, y + num.text_h + 24 + tm.text_h / 2)
    return frame
