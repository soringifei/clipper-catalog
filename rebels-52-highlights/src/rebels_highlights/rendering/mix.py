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
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Optional

from ..audio.audio import SAMPLE_RATE
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


def core_range(c: Candidate, first: bool, pre: Optional[float], rep: float,
               tl: Optional[dict]) -> tuple[float, float]:
    """(start, end) in the rendered clip's output time for this trim level."""
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


def _select(elig: list[Candidate], tls: dict, target: float) -> tuple[list[Candidate], tuple, Candidate]:
    opener = _pick_opener(elig)
    budget = target - 0.15
    best: tuple[list[Candidate], tuple] = ([], TRIM_LEVELS[0])
    for level in TRIM_LEVELS:
        pre, rep = level

        def dur(c: Candidate, first: bool) -> float:
            a, b = core_range(c, first, pre, rep, tls.get(c.clip_id))
            return b - a

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
    ranges = [core_range(c, i == 0, pre, rep, tls.get(c.clip_id)) for i, c in enumerate(seq)]
    fps = int(cfg.get("render", {}).get("fps", 30))
    ranges = [(a, a + round((b - a) * fps) / fps) for a, b in ranges]
    total = sum(b - a for a, b in ranges)
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
    cmd = [ffmpeg_bin(), "-y", "-hide_banner", "-loglevel", "error"]
    chains, labels = [], []
    has_audio = [probe(c.output_file)["has_audio"] for c in seq]
    for i, (c, (a, b)) in enumerate(zip(seq, ranges)):
        d = b - a
        cmd += ["-ss", f"{a:.4f}", "-t", f"{d:.4f}", "-i", str(c.output_file)]
        chains.append(
            f"[{i}:v]scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=decrease,"
            f"pad={OUT_W}:{OUT_H}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps},format=yuv420p,"
            f"trim=duration={d:.4f},setpts=PTS-STARTPTS,"
            f"fade=t=in:st=0:d={dip:.4f},fade=t=out:st={max(0.0, d - dip):.4f}:d={dip:.4f}[v{i}]")
        fmt = f"aformat=sample_fmts=fltp:sample_rates={SAMPLE_RATE}:channel_layouts=stereo"
        if has_audio[i]:
            chains.append(
                f"[{i}:a]{fmt},asetpts=PTS-STARTPTS,apad,atrim=duration={d:.4f},"
                f"afade=t=in:d={dip:.4f},afade=t=out:st={max(0.0, d - dip):.4f}:d={dip:.4f}[a{i}]")
        else:
            chains.append(f"anullsrc=r={SAMPLE_RATE}:cl=stereo,atrim=duration={d:.4f},{fmt}[a{i}]")
        labels.append(f"[v{i}][a{i}]")
    chains.append(f"{''.join(labels)}concat=n={len(seq)}:v=1:a=1[vout][aout]")
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
    return {"output_file": str(out), "clip_ids": [c.clip_id for c in seq],
            "duration_s": round(float(actual), 2), "target_s": int(duration_s),
            "trim_level": {"pre_impact_s": pre, "replay_s": rep},
            "short_of_target": actual < 0.7 * float(duration_s)}
