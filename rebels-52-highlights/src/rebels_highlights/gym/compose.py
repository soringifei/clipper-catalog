"""Multi-clip outputs: the compilation mix of the best clips, and the split-screen
side-by-side comparison (rep 1 vs last rep of one clip, or best reps of two clips)."""
from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np

from ..core.media import run_ffmpeg
from . import draw as D
from . import style as S
from .analysis import JOINT_LABEL, Analysis
from .config import hex_to_bgr, resolve_path
from .pipeline import analyse_clip, next_version, safe_stem
from .render import Composer, EditPlan, OutFrame, plan_crop
from .video import FrameReader, FrameWriter, mux_audio

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------
class SeqFrames:
    """Sequential random-ish access (non-decreasing indices) over a decoded window."""

    def __init__(self, path: Path, fps: float, size: tuple[int, int], start: int, n: int):
        self.start = start
        self.it: Iterator[np.ndarray] = iter(FrameReader(path, fps, size, start / fps, (n + 2) / fps))
        self.last = start - 1
        self.buf: dict[int, np.ndarray] = {}

    def get(self, i: int) -> Optional[np.ndarray]:
        while self.last < i:
            try:
                fr = next(self.it)
            except StopIteration:
                break
            self.last += 1
            self.buf[self.last] = fr
            for k in [k for k in self.buf if k < self.last - 2]:
                del self.buf[k]
        return self.buf.get(i, self.buf.get(self.last))


def T(frame, text, x, y, size, color=(255, 255, 255), alpha=1.0, anchor="lt", tracking=0.06, glow=0.0,
      accent=None):
    return S.blit(frame, S.type_sprite(str(text), int(size), tuple(color), "display", tracking, glow, accent),
                  x, y, alpha, anchor)


def _dark_canvas(W: int, H: int) -> np.ndarray:
    y = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
    top, bot = np.array([22, 20, 18], np.float32), np.array([10, 10, 12], np.float32)
    return np.repeat((top * (1 - y) + bot * y).astype(np.uint8), W, axis=1)


def _card_video(frames_fn, n: int, out: Path, cfg: dict) -> Path:
    vc = cfg["video"]
    tmp = out.with_suffix(".v.mp4")
    w = FrameWriter(tmp, vc["width"], vc["height"], vc["out_fps"], vc["crf"], vc["preset"])
    try:
        for k in range(n):
            w.write(frames_fn(k / max(1, n - 1)))
    finally:
        w.close()
    mux_audio(tmp, None, [(0, n / vc["out_fps"], 0)], out, vc["loudnorm"], False)
    tmp.unlink(missing_ok=True)
    return out


# ---------------------------------------------------------------------------
# compilation mix
# ---------------------------------------------------------------------------
def build_mix(results: list[dict], cfg: dict, duration_s: int, max_clips: Optional[int] = None) -> Optional[dict]:
    vc, mc = cfg["video"], cfg["mix"]
    cands = [r for r in results if r.get("render", {}).get("output_file")
             and Path(r["render"]["output_file"]).exists()]
    if not cands:
        log.warning("mix: no rendered clips")
        return None
    cands.sort(key=lambda r: r.get("clip_score", 0), reverse=True)
    n = min(len(cands), max_clips or mc["max_clips"])
    intro_s, outro_s = 1.6, 3.0
    per = max(3.0, min(mc["per_clip_max_s"], (duration_s - intro_s - outro_s) / n))
    n = max(1, min(n, int((duration_s - intro_s - outro_s) // 3)))
    chosen = cands[:n]
    out_dir = resolve_path(cfg, "outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"gym_mix_{duration_s}s"
    ver = next_version(out_dir, prefix)
    final = out_dir / f"{prefix}_v{ver:02d}.mp4"
    tmpdir = out_dir / f".{prefix}_v{ver:02d}_tmp"
    tmpdir.mkdir(exist_ok=True)
    accent = hex_to_bgr(cfg["style"]["accent"])
    D.set_fonts(cfg["style"].get("font"), cfg["style"].get("font_regular"))
    W, H = vc["width"], vc["height"]
    u = W / 1080
    labels = []
    for r in chosen:
        ex = r["exercise"]
        labels.append(ex["label"] if ex.get("label_shown") else "TRAINING")

    grade = S.make_grade(W, H, cfg["style"])

    def intro(ph):
        f = _dark_canvas(W, H)
        a, sc, sl = S.kinetic_alpha_scale(ph * intro_s, 10)
        spr = S.type_sprite("TRAINING MIX", int(160 * u), (255, 255, 255), "display", 0.01, 0.3, accent)
        S.blit(f, spr, W / 2, H * 0.40 + sl * 50 * u, a, "cm", sc)
        e = S.ease_out_cubic((ph - 0.15) * 2.5)
        cv2.line(f, (int(W / 2 - 150 * u * e), int(H * 0.40 + 110 * u)), (int(W / 2 + 150 * u * e), int(H * 0.40 + 110 * u)),
                 accent, max(2, int(5 * u)), cv2.LINE_AA)
        T(f, cfg["athlete"]["tag"], W / 2, H * 0.40 + 145 * u, 48 * u, alpha=e, anchor="ct")
        uniq = list(dict.fromkeys(labels))
        T(f, "  ·  ".join(uniq[:4]), W / 2, H * 0.40 + 215 * u, 30 * u, (170, 170, 170), alpha=e, anchor="ct",
          tracking=0.15)
        return S.flash(grade.apply(f), max(0.0, 1 - ph * 6) * 0.6)

    tot_reps = sum((r.get("summary", {}).get("reps") or 0) for r in chosen)
    best_jump = max([r.get("summary", {}).get("best_jump_height_m_est") or 0 for r in chosen])
    best_speed = max([r.get("summary", {}).get("peak_bar_speed_mps_est") or 0 for r in chosen])

    def outro(ph):
        f = _dark_canvas(W, H)
        t = ph * outro_s
        T(f, "SESSION TOTALS", 90 * u, H * 0.18, 34 * u, accent, alpha=S.ease_out_cubic(t * 4), tracking=0.3)
        rows = [("CLIPS", str(len(chosen)))]
        if tot_reps:
            rows.append(("REPS ANALYSED", str(tot_reps)))
        if best_speed:
            rows.append(("PEAK BAR SPEED · M/S · EST.", f"{best_speed:.2f}"))
        if best_jump:
            rows.append(("BEST JUMP · EST.", f"{best_jump * 100:.0f} CM"))
        y = H * 0.18 + 60 * u
        for i, (lab, val) in enumerate(rows):
            a, sc, sl = S.kinetic_alpha_scale(t - 0.12 - 0.13 * i, 100)
            if a > 0:
                bx = S.blit(f, S.type_sprite(val, int(118 * u), (255, 255, 255), "display", 0.0, 0.2, accent),
                            90 * u - sl * 60 * u, y, a, "lt", sc)
                T(f, lab, 96 * u, y + bx[3] * 0.80, 30 * u, (175, 175, 175), alpha=a, tracking=0.12)
            y += 222 * u
        e = S.ease_out_cubic(t * 2)
        T(f, "ESTIMATES FROM 2D PHONE VIDEO", 90 * u, H - 330 * u, 24 * u, (140, 140, 140), alpha=e, tracking=0.2)
        T(f, cfg["athlete"]["tag"], 90 * u, H - 290 * u, 46 * u, alpha=e, tracking=0.08)
        return grade.apply(f)

    ofps = vc["out_fps"]
    iv = _card_video(intro, int(intro_s * ofps), tmpdir / "intro.mp4", cfg)
    ov = _card_video(outro, int(outro_s * ofps), tmpdir / "outro.mp4", cfg)
    args: list[str] = ["-i", str(iv)]
    segs = []
    for r in chosen:
        rd = r["render"]
        bw = rd.get("body_window_s") or [0, rd["duration_s"]]
        hw = rd.get("highlight_window_s") or bw
        c = (hw[0] + hw[1]) / 2
        L = min(per, bw[1] - bw[0])
        s = float(np.clip(c - L / 2, bw[0], max(bw[0], bw[1] - L)))
        segs.append((rd["output_file"], s, L))
        args += ["-ss", f"{s:.3f}", "-t", f"{L:.3f}", "-i", rd["output_file"]]
    args += ["-i", str(ov)]
    k = len(segs) + 2
    fc = []
    for i in range(k):
        dur = intro_s if i == 0 else (outro_s if i == k - 1 else segs[i - 1][2])
        fo = max(0.0, dur - 0.15)
        fc.append(f"[{i}:v]fps={ofps},scale={W}:{H},setsar=1,format=yuv420p,"
                  f"fade=t=in:st=0:d=0.15,fade=t=out:st={fo:.3f}:d=0.15[v{i}]")
        fc.append(f"[{i}:a]aresample=48000,aformat=channel_layouts=stereo,"
                  f"afade=t=in:st=0:d=0.1,afade=t=out:st={fo:.3f}:d=0.15[a{i}]")
    fc.append("".join(f"[v{i}][a{i}]" for i in range(k)) + f"concat=n={k}:v=1:a=1[v][a0]")
    fc.append(f"[a0]loudnorm={vc['loudnorm']},aresample=48000[a]")
    args += ["-filter_complex", ";".join(fc), "-map", "[v]", "-map", "[a]", "-c:v", "libx264",
             "-preset", vc["preset"], "-crf", str(vc["crf"]), "-pix_fmt", "yuv420p", "-c:a", "aac",
             "-b:a", "160k", "-movflags", "+faststart", str(final)]
    run_ffmpeg(args)
    for p in tmpdir.iterdir():
        p.unlink()
    tmpdir.rmdir()
    total = intro_s + outro_s + sum(s[2] for s in segs)
    return {"output_file": str(final), "duration_s": round(total, 2),
            "clips": [{"file": r["file"], "segment": [round(s[1], 2), round(s[2], 2)]}
                      for r, s in zip(chosen, segs)]}


# ---------------------------------------------------------------------------
# side by side
# ---------------------------------------------------------------------------
def _pick_rep(an: Analysis, which: str) -> Optional[int]:
    if not an.reps:
        return None
    if which == "first":
        return 0
    if which == "last":
        return len(an.reps) - 1
    sc = an.summary.get("rep_scores") or [0] * len(an.reps)
    return int(np.argmax(sc))


def side_by_side(a_path: Path, b_path: Optional[Path], cfg: dict, exercise: Optional[str] = None,
                 provider=None, provider_b=None, use_cache: bool = True) -> dict:
    t0 = time.time()
    vc = cfg["video"]
    W, H = vc["width"], vc["height"]
    u = W / 1080
    ofps = vc["out_fps"]
    info_a, an_a, _ = analyse_clip(a_path, cfg, exercise, provider, use_cache)
    if b_path is None:
        info_b, an_b, pb = info_a, an_a, a_path
        ra, rb = _pick_rep(an_a, "first"), _pick_rep(an_a, "last")
        if ra is None or ra == rb:
            raise RuntimeError("side-by-side of one clip needs >= 2 detected reps")
        lab_a, lab_b = f"REP {ra + 1}", f"REP {rb + 1}"
        title = f"REP {ra + 1} vs REP {rb + 1}"
    else:
        info_b, an_b, _ = analyse_clip(b_path, cfg, exercise or an_a.exercise, provider_b, use_cache)
        pb = b_path
        ra, rb = _pick_rep(an_a, "best"), _pick_rep(an_b, "best")
        lab_a, lab_b = "A", "B"
        title = "SESSION A vs B"
    pw, ph = W // 2, int(round(W / 2 * 16 / 9))          # 540 x 960 panels
    sides = []
    for an, info, path, r in ((an_a, info_a, a_path, ra), (an_b, info_b, pb, rb)):
        fps = an.fps
        if r is not None:
            rep = an.reps[r]
            s = max(0, rep.start - int(0.5 * fps))
            e = min(an.seq.T, rep.end + int(0.5 * fps))
        else:
            s, e = 0, min(an.seq.T, int(6 * fps))
        crop = plan_crop(an, info["disp_w"], info["disp_h"], pw, ph, cfg)
        ds = min(1.0, crop.scale * 1.15)
        dw, dh = int(round(info["disp_w"] * ds / 2) * 2), int(round(info["disp_h"] * ds / 2) * 2)
        ds = dw / info["disp_w"]
        plan = EditPlan((s, e), [], [])
        comp = Composer(an, crop, cfg, plan, ds, sparkline=False, panel=False, header=False)
        sides.append({"an": an, "path": path, "rep": r, "s": s, "e": e, "comp": comp, "size": (dw, dh),
                      "fps": fps})
    dur = max((sd["e"] - sd["s"]) / sd["fps"] for sd in sides)
    n1 = int(dur * ofps)
    n2 = int(dur * ofps / cfg["edit"]["slowmo_speed"])
    accent = hex_to_bgr(cfg["style"]["accent"])
    # comparison chart of the primary signal (time since window start)
    cx0, cy0, cw, chh = int(60 * u), int(270 * u + ph + 30 * u), int(960 * u), int(230 * u)
    curves = []
    for sd in sides:
        v = sd["an"].primary[sd["s"]:sd["e"]]
        t = np.arange(len(v)) / sd["fps"]
        curves.append((t, v))
    allv = np.concatenate([c[1][~np.isnan(c[1])] for c in curves]) if curves else np.array([0, 1])
    lo, hi = (float(allv.min()), float(allv.max())) if len(allv) else (0.0, 1.0)
    hi = hi if hi - lo > 5 else lo + 5
    base = _dark_canvas(W, H)
    pad = 24 * u

    def cpt(t, v):
        x = cx0 + pad + (cw - 2 * pad) * t / max(dur, 1e-3)
        y = cy0 + chh - pad - (chh - 2 * pad - 30 * u) * (v - lo) / (hi - lo)
        return x, y

    for (t, v), col in zip(curves, (D.WHITE, accent)):
        for i in range(1, len(t)):
            if not (np.isnan(v[i]) or np.isnan(v[i - 1])):
                p0, p1 = cpt(t[i - 1], v[i - 1]), cpt(t[i], v[i])
                cv2.line(base, (int(p0[0] * 16), int(p0[1] * 16)), (int(p1[0] * 16), int(p1[1] * 16)), col, 3,
                         cv2.LINE_AA, shift=4)
    jl = JOINT_LABEL.get(an_a.primary_name, an_a.primary_name.upper())
    T(base, f"{jl} THROUGH THE REP", cx0 + pad, cy0, 28 * u, (170, 170, 170), tracking=0.15)
    T(base, lab_a, cx0 + cw - pad - 110 * u, cy0, 30 * u, anchor="rt")
    T(base, lab_b, cx0 + cw - pad, cy0, 30 * u, accent, anchor="rt")
    # header + table
    T(base, cfg["athlete"]["tag"], 56 * u, 70 * u, 34 * u, tracking=0.08)
    cv2.line(base, (int(60 * u), int(128 * u)), (int(120 * u), int(128 * u)), accent, max(2, int(4 * u)), cv2.LINE_AA)
    T(base, title, 56 * u, 150 * u, 60 * u, glow=0.25, accent=accent, tracking=0.02)
    ty = cy0 + chh + 24 * u

    def rep_val(sd, key):
        if sd["rep"] is None:
            return "—"
        r = sd["an"].reps[sd["rep"]]
        if key == "rom":
            return f"{r.rom_deg:.0f}°"
        if key == "tempo":
            return f"{r.ecc_s:.1f}S / {r.con_s:.1f}S"
        if key == "speed":
            v = r.extra.get("bar_peak_up_speed")
            return f"{v:.2f} m/s" if v is not None and sd["an"].extras.get("speed_unit") == "m/s" else "—"
        if key == "turn":
            j = sd["an"].profile.arcs[0] if sd["an"].profile.arcs else None
            v = r.extra.get(f"{j}_at_turn_deg") if j else None
            return f"{v:.0f}°" if v is not None else "—"
        return "—"

    first_arc = an_a.profile.arcs[0] if an_a.profile.arcs else "joint"
    for key, lab in (("rom", "ROM"), ("turn", f"{JOINT_LABEL.get(first_arc, first_arc.upper())} AT TURN"),
                     ("tempo", "TEMPO DOWN / UP"), ("speed", "PEAK BAR SPEED EST.")):
        T(base, rep_val(sides[0], key), 80 * u, ty, 50 * u)
        T(base, lab, W / 2, ty + 12 * u, 26 * u, (160, 160, 160), anchor="ct", tracking=0.15)
        T(base, rep_val(sides[1], key), W - 80 * u, ty, 50 * u, accent, anchor="rt")
        ty += 66 * u
    T(base, "ESTIMATES FROM 2D VIDEO", W / 2, H - 80 * u, 22 * u, (130, 130, 130), anchor="ct", tracking=0.2)

    out_dir = resolve_path(cfg, "outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"sbs_{safe_stem(a_path)}_vs_{safe_stem(pb) if b_path else 'reps'}"
    ver = next_version(out_dir, stem)
    final = out_dir / f"{stem}_v{ver:02d}.mp4"
    tmp = out_dir / f".{stem}_tmp.mp4"
    wr = FrameWriter(tmp, W, H, ofps, vc["crf"], vc["preset"])
    thumb = None
    try:
        for pass_i, (n, speed) in enumerate(((n1, 1.0), (n2, cfg["edit"]["slowmo_speed"]))):
            readers = [SeqFrames(sd["path"], sd["fps"], sd["size"], sd["s"], sd["e"] - sd["s"]) for sd in sides]
            for k in range(n):
                t = k * speed / ofps
                fr = base.copy()
                for j, (sd, rd) in enumerate(zip(sides, readers)):
                    pos = min(sd["s"] + t * sd["fps"], sd["e"] - 1)
                    f = int(round(pos))
                    img = rd.get(int(math.floor(pos)))
                    if img is None:
                        continue
                    comp = sd["comp"]
                    of = OutFrame(pos, "slow" if speed < 1 else "body", sd["rep"])
                    panel = comp.base(img, f)
                    comp.overlays(panel, f, of)
                    x0 = j * pw
                    fr[int(270 * u):int(270 * u) + ph, x0:x0 + pw] = panel
                    T(fr, lab_a if j == 0 else lab_b, x0 + 24 * u, 282 * u, 54 * u,
                      (255, 255, 255) if j == 0 else accent, glow=0.3, accent=accent)
                    px, py = cpt(t, sd["an"].primary[f]) if f < sd["an"].seq.T else (None, None)
                    if px is not None and not np.isnan(py):
                        cv2.circle(fr, (int(px), int(py)), int(8 * u), D.WHITE if j == 0 else accent, -1, cv2.LINE_AA)
                xx = cpt(t, lo)[0]
                cv2.line(fr, (int(xx), int(cy0 + 34 * u)), (int(xx), int(cy0 + chh - pad)), (150, 150, 150), 1)
                cv2.line(fr, (pw, int(270 * u)), (pw, int(270 * u) + ph), accent, max(2, int(4 * u)))
                if speed < 1:
                    T(fr, f"{speed:g}X SLOW", W - 56 * u, 80 * u, 40 * u, accent, anchor="rt", tracking=0.1)
                wr.write(fr)
                if pass_i == 0 and k == n // 2:
                    thumb = fr.copy()
    finally:
        wr.close()
    total = (n1 + n2) / ofps
    mux_audio(tmp, None, [(0, total, 0)], final, vc["loudnorm"], False)
    tmp.unlink(missing_ok=True)
    tpath = final.with_suffix(".jpg")
    if thumb is not None:
        cv2.imwrite(str(tpath), thumb, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return {"output_file": str(final), "thumbnail_file": str(tpath), "duration_s": round(total, 2),
            "a": {"file": str(a_path), "rep": None if ra is None else ra + 1},
            "b": {"file": str(pb), "rep": None if rb is None else rb + 1},
            "render_seconds": round(time.time() - t0, 1)}
