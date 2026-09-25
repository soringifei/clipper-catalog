"""Per-clip orchestration: probe -> pose (cached) -> analysis -> render -> json.
Failures are logged to reports/errors.jsonl (stage 'gym') and never stop the batch."""
from __future__ import annotations

import json
import logging
import time
import traceback
from pathlib import Path
from typing import Optional

import numpy as np

from ..core.store import _json_default
from .analysis import Analysis, analyze, clip_score
from .config import exercise_override, resolve_path
from .pose import PoseProvider, build_sequence, load_or_run_pose
from .video import discover_videos, probe_video

log = logging.getLogger(__name__)


def _atomic_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1, default=_json_default))
    tmp.replace(path)


def log_error(cfg: dict, clip: str, stage: str, err: BaseException) -> None:
    rep = Path(cfg["root"]) / "reports"
    rep.mkdir(parents=True, exist_ok=True)
    rec = {"ts": time.time(), "game_id": None, "clip": clip, "stage": f"gym/{stage}",
           "error": str(err), "traceback": "".join(traceback.format_exception(err))[-4000:]}
    with (rep / "errors.jsonl").open("a") as fh:
        fh.write(json.dumps(rec) + "\n")


def safe_stem(path: Path) -> str:
    s = "".join(c if c.isalnum() or c in "-_" else "_" for c in path.stem)
    return s or "clip"


def analyse_clip(path: Path, cfg: dict, exercise: Optional[str] = None,
                 provider: Optional[PoseProvider] = None, use_cache: bool = True,
                 debug: bool = False, out_dir: Optional[Path] = None) -> tuple[dict, Analysis, dict]:
    """Returns (probe info, Analysis, timing dict)."""
    t0 = time.time()
    info = probe_video(path, cfg["video"]["max_work_fps"])
    raw, cached = load_or_run_pose(path, info, cfg, resolve_path(cfg, "cache"), provider, use_cache)
    t_pose = time.time() - t0
    info["n_frames"] = raw.n_frames
    seq = build_sequence(raw, cfg)
    source = "auto"
    ex = exercise
    if ex:
        source = "cli"
    else:
        ex = exercise_override(cfg, path.name)
        if ex:
            source = "config"
    plate = None
    if cfg["pose"].get("plate_tracker"):
        from .barpath import track_plate
        try:
            plate = track_plate(path, info, seq, cfg)
        except Exception as e:  # noqa: BLE001
            log.warning("plate tracker failed (%s), using wrists", e)
    an = analyze(seq, cfg, ex, source, plate)
    if debug and out_dir is not None:
        from .debug import render_debug
        try:
            render_debug(path, info, raw, seq, an, out_dir / f"debug_{safe_stem(path)}.mp4")
        except Exception as e:  # noqa: BLE001
            log.warning("debug render failed: %s", e)
    return info, an, {"pose_s": round(t_pose, 2), "pose_cached": cached,
                      "pose_model_s": raw.meta.get("seconds"),
                      "analysis_s": round(time.time() - t0 - t_pose, 2)}


def process_clip(path: Path, cfg: dict, exercise: Optional[str] = None,
                 provider: Optional[PoseProvider] = None, resume: bool = False,
                 use_cache: bool = True, debug: bool = False, slowmo: bool = True) -> Optional[dict]:
    from .render import render_clip
    stem = safe_stem(path)
    out_dir = resolve_path(cfg, "outputs") / stem
    aj = out_dir / "analysis.json"
    if resume and aj.exists():
        prev = json.loads(aj.read_text())
        outp = Path(prev.get("render", {}).get("output_file", ""))
        if outp.exists() and aj.stat().st_mtime >= path.stat().st_mtime and \
                (not exercise or prev.get("exercise", {}).get("key") == exercise):
            log.info("resume: %s already rendered", path.name)
            prev["_resumed"] = True
            return prev
    t0 = time.time()
    info, an, timing = analyse_clip(path, cfg, exercise, provider, use_cache, debug, out_dir)
    if an.seq.track_frac < 0.05:
        raise RuntimeError(f"no athlete tracked in {path.name} (track fraction {an.seq.track_frac:.2f})")
    rend = render_clip(path, info, an, cfg, out_dir, stem, slowmo=slowmo)
    rend["total_seconds"] = round(time.time() - t0, 1)
    data = {"file": str(path), "stem": stem,
            "source": {k: info.get(k) for k in ("duration", "fps", "width", "height", "disp_w", "disp_h",
                                                 "rotation", "vcodec", "has_audio", "work_fps")},
            **an.to_json(), "timing": timing, "render": rend,
            "clip_score": round(clip_score(an), 3),
            "disclaimer": "All metrics are estimates from 2D video keypoints (projected angles; "
                          "speeds/heights scaled from athlete height)."}
    _atomic_json(aj, data)
    return data


def run_batch(inputs: list[str], cfg: dict, exercise: Optional[str] = None,
              max_clips: Optional[int] = None, resume: bool = False, debug: bool = False,
              provider: Optional[PoseProvider] = None, use_cache: bool = True,
              slowmo: bool = True) -> list[dict]:
    files = discover_videos(inputs)
    if max_clips:
        files = files[:max_clips]
    results = []
    for f in files:
        try:
            log.info("gym: %s", f)
            r = process_clip(f, cfg, exercise, provider, resume, use_cache, debug, slowmo)
            if r:
                results.append(r)
        except Exception as e:  # noqa: BLE001 - one clip must not kill the batch
            log.error("gym: %s failed: %s", f.name, e)
            log_error(cfg, str(f), "clip", e)
    return results


def next_version(out_dir: Path, prefix: str) -> int:
    v = 1
    while (out_dir / f"{prefix}_v{v:02d}.mp4").exists():
        v += 1
    return v
