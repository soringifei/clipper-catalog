"""Gym config: built-in defaults <- config/gym.yaml <- CLI overrides."""
from __future__ import annotations

import copy
import fnmatch
from pathlib import Path
from typing import Any, Optional

import yaml

from ..core.config import PROJECT_ROOT, _deep_merge

DEFAULTS: dict[str, Any] = {
    "paths": {"outputs": "outputs/gym", "cache": "cache/gym", "models": "cache/models"},
    "athlete": {"tag": "#52 | MLB | BUCHAREST REBELS", "name": "", "height_m": 1.85},
    "pose": {"model": "yolo11n-pose.pt", "device": "cpu", "imgsz": 640, "conf": 0.25,
             "tracker": "bytetrack.yaml", "max_pose_hz": 30, "analysis_long_side": 960,
             "kp_min_conf": 0.45, "max_gap_s": 0.25, "smoothing": "one_euro",
             "one_euro": {"min_cutoff": 1.2, "beta": 0.02, "d_cutoff": 1.0},
             "plate_tracker": False},
    "video": {"max_work_fps": 60, "out_fps": 30, "width": 1080, "height": 1920, "crf": 20,
              "preset": "veryfast", "loudnorm": "I=-14:TP=-1.5:LRA=11", "max_zoom": 1.6,
              "crop_smoothing": 0.90},
    "edit": {"min_clip_s": 15, "max_clip_s": 60, "intro_s": 1.6, "outro_s": 3.0,
             "ramp_half_s": 0.6, "ramp_min_speed": 0.3, "ramp_zoom": 0.07,
             "slowmo_speed": 0.5, "peak_moments": "auto", "sparkline": True,
             "label_min_confidence": 0.55},
    "style": {"accent": "#E10600",
              "font": "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "font_regular": "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "skeleton_color": "#FFFFFF", "trail_s": 1.6,
              "contrast": 0.55, "saturation": 0.88, "vignette": 0.42, "grain": 5.0},
    "mix": {"max_clips": 6, "per_clip_max_s": 12},
    "exercise_overrides": {},
}


def load_gym_config(path: Optional[str] = None, overrides: Optional[dict] = None,
                    root: Optional[Path] = None) -> dict[str, Any]:
    root = Path(root or PROJECT_ROOT)
    cfg = copy.deepcopy(DEFAULTS)
    p = Path(path) if path else root / "config/gym.yaml"
    if not p.is_absolute():
        p = root / p
    if p.exists():
        cfg = _deep_merge(cfg, yaml.safe_load(p.read_text()) or {})
    cfg["exercise_overrides"] = cfg.get("exercise_overrides") or {}
    cfg["root"] = str(root)
    return _deep_merge(cfg, overrides or {})


def resolve_path(cfg: dict, key: str) -> Path:
    p = Path(cfg["paths"][key])
    return p if p.is_absolute() else Path(cfg["root"]) / p


def exercise_override(cfg: dict, filename: str) -> Optional[str]:
    name = Path(filename).name
    for pat, ex in (cfg.get("exercise_overrides") or {}).items():
        if fnmatch.fnmatch(name.lower(), str(pat).lower()):
            return str(ex)
    return None


def hex_to_bgr(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)
