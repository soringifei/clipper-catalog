"""Automated QA of rendered vertical clips.

Checks are ``True`` (pass), ``False`` (fail) or ``None`` (could not be evaluated -
e.g. no crop metadata and no source frame size). Only ``False`` fails a clip.

Crop metadata (optional, written by the renderer next to the clip as
``<base>.crop.json``) is accepted in either form (source px):

* ``{"frames": [{"t": s, "crop": [x1, y1, x2, y2], "player_x": x | null}, ...]}``
* ``{"crop_boxes": [[x1,y1,x2,y2], ...], "player_x": [x | null, ...]}``
  (``player_boxes`` may be given instead of ``player_x``).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

from ..core.media import probe

REQUIRED_FIELDS = ["clip_id", "game_id", "play_id", "play_type", "review_status",
                   "source_start_s", "source_end_s", "identity_confidence",
                   "event_confidence", "overall_highlight_score"]


def _d(x: Any) -> dict:
    if x is None:
        return {}
    return dict(x) if isinstance(x, dict) else x.to_dict()


def _not_black(path: str, n: int = 5) -> Optional[bool]:
    try:
        import cv2
        import numpy as np
    except ImportError:  # pragma: no cover
        return None
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    idxs = [int(total * (i + 0.5) / n) for i in range(n)] if total > 0 else list(range(n))
    ok_frames, good = 0, 0
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = cap.read()
        if not ok or fr is None:
            continue
        ok_frames += 1
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if g.mean() > 8 and g.std() > 3:
            good += 1
    cap.release()
    if ok_frames == 0:
        return None
    return good == ok_frames


def _load_crop_meta(clip: Path) -> Optional[list[tuple[list[float], Optional[float]]]]:
    p = clip.with_suffix(".crop.json")
    if not p.exists():
        return None
    try:
        j = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    pairs: list[tuple[list[float], Optional[float]]] = []
    if isinstance(j, dict) and isinstance(j.get("frames"), list):
        for f in j["frames"]:
            if isinstance(f, dict) and f.get("crop"):
                pairs.append((f["crop"], f.get("player_x")))
    elif isinstance(j, dict) and isinstance(j.get("crop_boxes"), list):
        px = j.get("player_x")
        if px is None and isinstance(j.get("player_boxes"), list):
            px = [(b[0] + b[2]) / 2 if b else None for b in j["player_boxes"]]
        px = px or []
        for i, c in enumerate(j["crop_boxes"]):
            pairs.append((c, px[i] if i < len(px) else None))
    return pairs or None


def _safe_fraction(pairs, safe: float) -> Optional[float]:
    inside = total = 0
    for crop, x in pairs:
        if x is None or not crop:
            continue
        x1, _, x2, _ = crop
        w = x2 - x1
        if w <= 0:
            continue
        m = (1 - safe) / 2 * w
        total += 1
        inside += int(x1 + m <= x <= x2 - m)
    return inside / total if total else None


def _simulated_pairs(traj: dict, cand: dict, frame_size, cfg: dict):
    """Re-create an EMA-smoothed 9:16 crop following #52 when no crop metadata exists."""
    fw, fh = frame_size
    r = cfg.get("render", {})
    alpha = float(r.get("crop_smoothing", 0.85))
    cw = min(fw, fh * r.get("width", 1080) / r.get("height", 1920))
    t0, t1 = float(cand.get("source_start_s") or 0), float(cand.get("source_end_s") or 1e9)
    pts = [((b[0] + b[2]) / 2) for t, b in zip(traj.get("times", []), traj.get("boxes", []))
           if b and t0 <= t <= t1]
    if not pts:
        return None
    cx, pairs = pts[0], []
    for x in pts:
        cx = alpha * cx + (1 - alpha) * x
        c = min(max(cx, cw / 2), fw - cw / 2)
        pairs.append(([c - cw / 2, 0, c + cw / 2, fh], x))
    return pairs


def qa_clip(path: str, cand: Any, trajectory: Any, cfg: dict) -> dict:
    c = _d(cand)
    traj = _d(trajectory)
    r = cfg.get("render", {})
    p = Path(path)
    checks: dict[str, Optional[bool]] = {k: None for k in (
        "exists", "readable", "container_mp4", "duration_ok", "aspect_9x16", "dims_1080x1920",
        "vcodec_h264", "audio_present", "not_black", "safe_region", "thumbnail_exists",
        "manifest_fields")}
    info: dict = {}
    checks["exists"] = p.exists() and p.stat().st_size > 0
    if checks["exists"]:
        try:
            info = probe(str(p))
            checks["readable"] = bool(info.get("width")) and float(info.get("duration") or 0) > 0
        except Exception:  # noqa: BLE001
            checks["readable"] = False
    else:
        checks["readable"] = False
    if checks["readable"]:
        dur = float(info["duration"])
        w, h = int(info["width"]), int(info["height"])
        checks["container_mp4"] = p.suffix.lower() == ".mp4" and "mp4" in str(info.get("format", ""))
        lo, hi = float(r.get("min_clip_s", 15)), float(r.get("max_clip_s", 60))
        checks["duration_ok"] = lo - 0.1 <= dur <= hi + 0.1
        checks["aspect_9x16"] = abs(w / h - 9 / 16) < 0.01 if h else False
        checks["dims_1080x1920"] = (w, h) == (int(r.get("width", 1080)), int(r.get("height", 1920)))
        checks["vcodec_h264"] = str(info.get("vcodec", "")).lower() in ("h264", "avc1")
        checks["audio_present"] = bool(info.get("has_audio"))
        checks["not_black"] = _not_black(str(p))
    # safe region
    safe = float(r.get("safe_region", 0.6))
    pairs = _load_crop_meta(p)
    safe_src = "crop_meta" if pairs else None
    if pairs is None and traj.get("boxes"):
        fs = c.get("frame_size") or c.get("seed_frame_size") or traj.get("frame_size")
        if fs and fs[0]:
            pairs = _simulated_pairs(traj, c, fs, cfg)
            safe_src = "simulated" if pairs else None
    frac = _safe_fraction(pairs, safe) if pairs else None
    checks["safe_region"] = None if frac is None else frac >= 0.9
    # thumbnail
    thumb = c.get("thumbnail_file")
    tp = Path(thumb) if thumb else p.with_suffix(".jpg")
    if thumb and not tp.is_absolute() and cfg.get("root"):
        tp = Path(cfg["root"]) / tp
    checks["thumbnail_exists"] = tp.exists()
    missing = [k for k in REQUIRED_FIELDS if c.get(k) in (None, "")]
    checks["manifest_fields"] = not missing
    failures = [k for k, v in checks.items() if v is False]
    return {"clip_id": c.get("clip_id"), "path": str(p), "passed": not failures,
            "checks": checks, "failures": failures,
            "details": {"duration_s": info.get("duration"), "size": [info.get("width"), info.get("height")],
                        "vcodec": info.get("vcodec"), "safe_fraction": None if frac is None else round(frac, 3),
                        "safe_region_source": safe_src, "missing_fields": missing}}


def write_qa_report(results: list[dict], store) -> str:
    from collections import Counter
    fails = Counter(f for q in results for f in q.get("failures", []))
    skipped = Counter(k for q in results for k, v in (q.get("checks") or {}).items() if v is None)
    rep = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "summary": {"total": len(results), "passed": sum(1 for q in results if q.get("passed")),
                       "failed": sum(1 for q in results if not q.get("passed")),
                       "failures_by_check": dict(fails), "not_evaluated_by_check": dict(skipped)},
           "results": results}
    out = Path(store.reports) / "qa.json"
    out.write_text(json.dumps(rep, indent=1))
    return str(out)
