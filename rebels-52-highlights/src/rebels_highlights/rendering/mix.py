"""Multi-play mix ("52_MLB_Bucharest-Rebels_mix_...") from rendered single clips.

* only AUTO_APPROVED / approved clips with identity >= mix.min_identity and an
  existing rendered file;
* opener = strongest (overall score among tier S/A), closer = next strongest,
  middle ordered for variety of play_type/unit, at most
  mix.max_same_game_consecutive clips of one game in a row, no identical
  adjacent play types when avoidable;
* each clip is trimmed to its core (ident card only on the first clip,
  full-speed play + short replay, no end card) using the clip's
  ``<base>.timeline.json`` written by render_clip; trimming is progressively
  tightened to fit the target - weak filler is never added to pad;
* hard cuts with a 4-frame dip (2 out + 2 in); all segments are normalised
  (1080x1920, 30 fps, yuv420p, 48k stereo) and joined with ffmpeg's concat
  filter in one encode for frame-exact A/V sync.

Clips rendered in the ``epic`` style (timeline ``style: epic``) get the epic
compilation structure instead: the opener keeps its cold open (its single best
hit), '#52' title slam, run-it-back and spotlight intro; every other clip enters
on a hard cut into its live play and leaves right after its replay impact; every
second cut gets a 3-frame white flash; the final clip runs into its end card. No
dips. With a licensed ``player.music_file`` clip boundaries are nudged onto the
music's beats (+-120 ms) and a ``_music`` version is written next to the mix.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Optional

from ..audio.audio import SAMPLE_RATE, audio_filter_graph, detect_beats, snap_to_beat, tile_beats
from ..core.media import ffmpeg_bin, probe
from ..core.models import Candidate
from .crop import OUT_H, OUT_W

APPROVED = {"AUTO_APPROVED", "APPROVED"}
# (max seconds kept before impact or None = full lead-in, replay seconds kept)
TRIM_LEVELS = [(None, 3.0), (None, 2.0), (2.5, 2.0), (2.0, 1.5), (1.8, 0.0)]
MIN_MIX_S = 15.0


def _eligible(c: Candidate, cfg: dict) -> bool:
    min_id = float(cfg.get("mix", {}).get("min_identity", 0.8))
    status = str(c.review_status or "").upper()
    return (status in APPROVED and c.identity_confidence >= min_id
            and bool(c.output_file) and Path(c.output_file).is_file())


def _timeline(c: Candidate) -> Optional[dict]:
    p = Path(str(c.output_file))
    tl = p.with_name(p.name[:-len(p.suffix)] + ".timeline.json") if p.suffix else None
    if tl and tl.is_file():
        try:
            return json.loads(tl.read_text())
        except (OSError, ValueError):
            return None
    return None


def _is_epic(tl: Optional[dict]) -> bool:
    return bool(tl) and tl.get("style") == "epic"


def epic_pieces(first: bool, last: bool, pre: Optional[float], rep: float,
                tl: dict) -> list[tuple[float, float]]:
    """Output-time pieces of an epic clip used in the mix.

    Opener from 0 (cold open + title + run-it-back + intro), others from the
    live play (``pre`` s before impact when set); end right after the replay
    impact (``rep`` = TRIM_LEVELS replay s: >=3 full replay, else replay impact
    + rep/2.5 s, 0 = no replay). The last clip runs into its end card (a hard
    cut to it when the replay was trimmed)."""
    segs = {s["kind"]: s for s in tl["segments"]}
    live, replay, end = segs.get("live"), segs.get("replay"), segs.get("endcard")
    if live is None:
        return [(0.0, float(tl.get("duration", 0.0)))]
    impact_out = float(tl.get("impact_out", live["end"]))
    if first:
        start = 0.0
    elif pre is not None:
        start = max(live["start"] + 0.4, impact_out - pre)
    else:
        start = live["start"] + 0.4          # skip the spotlight release (no intro here)
    stop = float(live["end"])
    if replay is not None and rep > 0:
        rimp = [t for t in tl.get("impacts_out", []) if replay["start"] <= t <= replay["end"]]
        stop = float(replay["end"]) if (rep >= 3.0 or not rimp) else \
            float(min(replay["end"], rimp[0] + rep / 2.5))
    pieces = [(float(start), stop)]
    if last and end is not None:
        if abs(stop - end["start"]) < 1e-3:
            pieces = [(float(start), float(end["end"]))]
        else:
            pieces.append((float(end["start"]), float(end["end"])))
    return pieces


def clip_pieces(c: Candidate, first: bool, pre: Optional[float], rep: float,
                tl: Optional[dict], last: bool = False) -> list[tuple[float, float]]:
    if _is_epic(tl):
        return epic_pieces(first, last, pre, rep, tl)  # type: ignore[arg-type]
    return [core_range(c, first, pre, rep, tl)]


def core_range(c: Candidate, first: bool, pre: Optional[float], rep: float,
               tl: Optional[dict]) -> tuple[float, float]:
    """(start, end) in the rendered clip's output time for this trim level."""
    if _is_epic(tl):
        p = epic_pieces(first, False, pre, rep, tl)  # type: ignore[arg-type]
        return p[0][0], p[-1][1]
    if not tl:  # no timeline: drop a ~2.5 s end card, keep the rest
        dur = c.output_duration_s or probe(c.output_file)["duration"]
        return 0.0, max(1.0, dur - 2.5)
    segs = {s["kind"]: s for s in tl["segments"]}
    live, ident, replay = segs.get("live"), segs.get("ident"), segs.get("replay")
    if live is None:
        return 0.0, float(tl.get("duration", 0.0))
    impact_out = float(tl.get("impact_out", live["end"]))
    start = live["start"]
    if first and ident is not None:
        start = ident["start"]
    elif pre is not None:
        start = max(live["start"], impact_out - pre)
    end = live["end"]
    if replay is not None and rep > 0:
        end = min(replay["end"], replay["start"] + rep)
    return float(start), float(end)


def _score(c: Candidate) -> float:
    return float(c.overall_highlight_score or 0.0)


def _pick_opener(cands: list[Candidate]) -> Candidate:
    strong = [c for c in cands if c.tier in ("S", "A")] or cands
    return max(strong, key=_score)


def order_clips(sel: list[Candidate], cfg: dict, opener: Optional[Candidate] = None) -> list[Candidate]:
    if len(sel) <= 1:
        return list(sel)
    max_run = int(cfg.get("mix", {}).get("max_same_game_consecutive", 2))
    opener = opener if opener in sel else _pick_opener(sel)
    rest = [c for c in sel if c is not opener]
    closer = _pick_opener(rest)
    middle = [c for c in rest if c is not closer]
    seq = [opener]

    def run_len(s: list[Candidate], game: str) -> int:
        n = 0
        for c in reversed(s):
            if c.game_id != game:
                break
            n += 1
        return n

    while middle:
        last_slot = len(middle) == 1
        seen = {c.play_type for c in seq}

        def pen(c: Candidate) -> float:
            p = -5.0 * _score(c)
            if c.play_type == seq[-1].play_type:
                p += 10
            if c.unit == seq[-1].unit:
                p += 1.5
            if c.play_type not in seen:
                p -= 3
            if run_len(seq, c.game_id) + 1 > max_run:
                p += 100
            if last_slot:
                if c.play_type == closer.play_type:
                    p += 10
                if run_len(seq + [c], closer.game_id) + 1 > max_run:
                    p += 100
            return p

        nxt = min(middle, key=pen)
        seq.append(nxt)
        middle.remove(nxt)
    seq.append(closer)
    return seq


def _endcard_len(tl: Optional[dict]) -> float:
    if not _is_epic(tl):
        return 0.0
    e = next((s for s in tl["segments"] if s["kind"] == "endcard"), None)
    return float(e["end"] - e["start"]) if e else 0.0


def _select(elig: list[Candidate], tls: dict, target: float) -> tuple[list[Candidate], tuple, Candidate]:
    opener = _pick_opener(elig)
    budget = target - 0.15 - max((_endcard_len(tls.get(c.clip_id)) for c in elig), default=0.0)
    best: tuple[list[Candidate], tuple] = ([], TRIM_LEVELS[0])
    for level in TRIM_LEVELS:
        pre, rep = level

        def dur(c: Candidate, first: bool) -> float:
            return sum(b - a for a, b in clip_pieces(c, first, pre, rep, tls.get(c.clip_id)))

        total = dur(opener, True)
        chosen = [opener] if total <= budget else []
        if not chosen:
            continue
        for c in sorted((c for c in elig if c is not opener), key=_score, reverse=True):
            d = dur(c, False)
            if total + d <= budget:
                chosen.append(c)
                total += d
        if len(chosen) > len(best[0]):
            best = (chosen, level)
        if len(chosen) == len(elig):
            break
    return best[0], best[1], opener


def _music_file(cfg: dict) -> Optional[str]:
    m = (cfg.get("player", {}) or {}).get("music_file") or ""
    if not m:
        return None
    p = Path(m)
    if not p.is_absolute() and cfg.get("root"):
        p = Path(cfg["root"]) / p
    return str(p) if p.is_file() else None


def align_to_beats(pieces: list[tuple[float, float, int]], beats: list[float], fps: int,
                   tol: float = 0.12, protect: int = 1) -> list[tuple[float, float, int]]:
    """Nudge each clip boundary (end of the last piece of a clip) onto the nearest
    beat within ``tol`` by lengthening/shortening that piece. ``protect``: pieces
    shorter than this many seconds are never shortened below it."""
    if not beats:
        return pieces
    out, t = [], 0.0
    for i, (a, b, ci) in enumerate(pieces):
        end_of_clip = i + 1 < len(pieces) and pieces[i + 1][2] != ci
        if end_of_clip:
            tgt = snap_to_beat(t + (b - a), beats, tol)
            nb = a + round((tgt - t) * fps) / fps
            if nb - a >= max(protect, 0.5):
                b = nb
        out.append((a, b, ci))
        t += b - a
    return out


def build_mix(clips: list[Candidate], cfg: dict, out_dir: str, duration_s: int,
              version: int = 1) -> Optional[dict]:
    elig = [c for c in clips if _eligible(c, cfg)]
    if not elig:
        return None
    tls = {c.clip_id: _timeline(c) for c in elig}
    sel, (pre, rep), opener = _select(elig, tls, float(duration_s))
    if not sel:
        return None
    seq = order_clips(sel, cfg, opener)
    fps = int(cfg.get("render", {}).get("fps", 30))
    epic = all(_is_epic(tls.get(c.clip_id)) for c in seq)
    pieces: list[tuple[float, float, int]] = []
    for i, c in enumerate(seq):
        for a, b in clip_pieces(c, i == 0, pre, rep, tls.get(c.clip_id), last=i == len(seq) - 1):
            pieces.append((a, a + round((b - a) * fps) / fps, i))
    music = _music_file(cfg) if epic else None
    beats: list[float] = []
    if music:
        try:
            bt, _ = detect_beats(music)
            beats = tile_beats(bt, float(probe(music).get("duration") or 0.0), duration_s + 5)
        except Exception:  # noqa: BLE001
            beats = []
        pieces = align_to_beats(pieces, beats, fps)
    total = sum(b - a for a, b, _ in pieces)
    if total > duration_s + 0.05:   # beat nudges may not push past the target
        a, b, ci = pieces[-1]
        pieces[-1] = (a, b - round((total - duration_s) * fps) / fps, ci)
        total = sum(b - a for a, b, _ in pieces)
    if total < MIN_MIX_S:
        return None

    nums = [re.findall(r"\d+", c.game_id or "") for c in seq]
    nums = [n[-1] for n in nums if n]
    if nums:
        lo, hi = min(nums, key=int), max(nums, key=int)
    else:
        lo = hi = "000"
    name = f"52_MLB_Bucharest-Rebels_mix_games-{lo}-{hi}_{int(round(total))}s_v{int(version):02d}.mp4"
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    out = out_dir_p / name

    rc = cfg.get("render", {})
    dip = 2.0 / fps
    flash = 3.0 / fps
    cmd = [ffmpeg_bin(), "-y", "-hide_banner", "-loglevel", "error"]
    chains, labels = [], []
    has_audio = {}
    fmt = f"aformat=sample_fmts=fltp:sample_rates={SAMPLE_RATE}:channel_layouts=stereo"
    for j, (a, b, ci) in enumerate(pieces):
        c = seq[ci]
        if ci not in has_audio:
            has_audio[ci] = probe(c.output_file)["has_audio"]
        d = b - a
        cmd += ["-ss", f"{a:.4f}", "-t", f"{d:.4f}", "-i", str(c.output_file)]
        base = (f"[{j}:v]scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=decrease,"
                f"pad={OUT_W}:{OUT_H}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps},format=yuv420p,"
                f"trim=duration={d:.4f},setpts=PTS-STARTPTS")
        new_clip = j > 0 and pieces[j - 1][2] != ci
        if not epic:
            base += f",fade=t=in:st=0:d={dip:.4f},fade=t=out:st={max(0.0, d - dip):.4f}:d={dip:.4f}"
            afade = (f",afade=t=in:d={dip:.4f},afade=t=out:st={max(0.0, d - dip):.4f}:d={dip:.4f}")
        else:
            # hard cuts; every second clip boundary (and the jump into the end card) flashes
            if j > 0 and ((new_clip and ci % 2 == 0) or not new_clip):
                base += f",fade=t=in:st=0:d={flash:.4f}:color=white"
            afade = f",afade=t=in:d=0.008,afade=t=out:st={max(0.0, d - 0.008):.4f}:d=0.008"
        chains.append(base + f"[v{j}]")
        if has_audio[ci]:
            chains.append(f"[{j}:a]{fmt},asetpts=PTS-STARTPTS,apad,atrim=duration={d:.4f}{afade}[a{j}]")
        else:
            chains.append(f"anullsrc=r={SAMPLE_RATE}:cl=stereo,atrim=duration={d:.4f},{fmt}[a{j}]")
        labels.append(f"[v{j}][a{j}]")
    chains.append(f"{''.join(labels)}concat=n={len(pieces)}:v=1:a=1[vout][aout]")
    cmd += ["-filter_complex", ";".join(chains), "-map", "[vout]", "-map", "[aout]",
            "-c:v", "libx264", "-preset", str(rc.get("preset", "medium")),
            "-crf", str(rc.get("crf", 20)), "-pix_fmt", "yuv420p", "-profile:v", "high",
            "-r", str(fps), "-g", str(fps * 2),
            "-c:a", "aac", "-b:a", "160k", "-ar", str(SAMPLE_RATE), "-ac", "2",
            "-t", f"{total:.4f}", "-movflags", "+faststart", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"mix ffmpeg failed: {r.stderr[-3000:]}")
    actual = probe(out)["duration"] or total
    res = {"output_file": str(out), "clip_ids": [c.clip_id for c in seq],
           "duration_s": round(float(actual), 2), "target_s": int(duration_s),
           "trim_level": {"pre_impact_s": pre, "replay_s": rep},
           "style": "epic" if epic else "clean",
           "short_of_target": actual < 0.7 * float(duration_s)}
    if music:
        mout = out.with_name(out.stem + "_music.mp4")
        graph = "[0:a]anull[game];[1:a]anull[music];" + audio_filter_graph(True, cfg)
        mc = [ffmpeg_bin(), "-y", "-hide_banner", "-loglevel", "error", "-i", str(out),
              "-stream_loop", "-1", "-i", music, "-filter_complex", graph, "-map", "0:v",
              "-map", "[aout]", "-c:v", "copy", "-c:a", "aac", "-b:a", "160k",
              "-ar", str(SAMPLE_RATE), "-ac", "2", "-t", f"{actual:.3f}", "-movflags", "+faststart",
              str(mout)]
        rr = subprocess.run(mc, capture_output=True, text=True)
        if rr.returncode == 0:
            res["music_file"] = str(mout)
            res["beats_aligned"] = bool(beats)
    return res
