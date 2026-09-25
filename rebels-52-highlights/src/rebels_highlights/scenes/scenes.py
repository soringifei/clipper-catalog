"""Hard-cut scene detection (PySceneDetect) + live / replay / non_play labelling.

Heuristics (all tunable under ``cfg['scenes']``):

* ``non_play``: mean fraction of grass-green pixels < ``green_min`` (scoreboard,
  crowd, sideline, studio graphics, black frames).
* ``replay``: a green (field) scene shorter than ``replay_max_s`` that either
  shows a slow-motion signature (many near-duplicate consecutive frames while
  the picture still changes over longer spans) or looks like the preceding
  live scene (HSV-histogram correlation >= ``replay_hist_sim``) while being
  bracketed by a transition (followed by a very short / non-field scene).
* otherwise ``live``.
"""
from __future__ import annotations

from typing import Any, Optional

import cv2
import numpy as np

from ..core.models import Scene

DEFAULTS: dict[str, Any] = {
    "detector": "adaptive",      # adaptive | content
    "threshold": 3.0,            # adaptive_threshold (content: 27.0 used if <= 10)
    "min_scene_len_s": 0.6,
    "min_content_val": 15.0,
    "sample_every_s": 0.5,       # frame sampling for classification
    "analysis_width": 320,
    "green_min": 0.25,           # < => non_play
    "green_hsv_lo": [32, 40, 40],
    "green_hsv_hi": [90, 255, 255],
    "replay_max_s": 12.0,
    "replay_hist_sim": 0.80,
    "transition_max_s": 1.2,     # a scene this short counts as a wipe/transition
    "slowmo_dup_ratio": 0.35,    # fraction of near-duplicate consecutive frames
    "dup_frame_eps": 0.6,        # mean abs diff (0-255) below which frames are "duplicates"
    "min_change": 2.0,           # mean abs diff over sample_every_s that proves motion
}


def scene_cfg(cfg: dict) -> dict:
    """``cfg['scenes']`` merged over module defaults."""
    return {**DEFAULTS, **(cfg.get("scenes") or {})}


def green_ratio(frame_bgr: np.ndarray, sc: Optional[dict] = None) -> float:
    """Fraction of pixels inside the grass-green HSV range."""
    sc = sc or DEFAULTS
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(sc["green_hsv_lo"], np.uint8),
                       np.array(sc["green_hsv_hi"], np.uint8))
    return float(np.count_nonzero(mask)) / mask.size


def hsv_hist(frame_bgr: np.ndarray) -> np.ndarray:
    """Normalised 2-D H/S histogram (for scene similarity)."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [30, 16], [0, 180, 0, 256])
    return cv2.normalize(h, h).flatten()


def _cut_times(video_path: str, sc: dict) -> tuple[list[tuple[float, float]], float, float]:
    """Return ([(start_s, end_s)...], fps, duration_s) using PySceneDetect."""
    from scenedetect import AdaptiveDetector, ContentDetector, SceneManager, open_video

    video = open_video(video_path)
    fps = float(video.frame_rate) or 25.0
    min_len = max(1, int(round(sc["min_scene_len_s"] * fps)))
    sm = SceneManager()
    if sc["detector"] == "content":
        thr = float(sc["threshold"]) if float(sc["threshold"]) > 10 else 27.0
        sm.add_detector(ContentDetector(threshold=thr, min_scene_len=min_len))
    else:
        sm.add_detector(AdaptiveDetector(adaptive_threshold=float(sc["threshold"]),
                                         min_scene_len=min_len,
                                         min_content_val=float(sc["min_content_val"])))
    sm.detect_scenes(video=video)
    raw = sm.get_scene_list(start_in_scene=True)
    duration = float(video.duration.get_seconds()) if video.duration is not None else 0.0
    spans = [(s.get_seconds(), e.get_seconds()) for s, e in raw]
    if not spans and duration > 0:
        spans = [(0.0, duration)]
    return spans, fps, duration


def _scene_stats(video_path: str, spans: list[tuple[float, float]], sc: dict) -> list[dict]:
    """One sequential decode pass: per-scene green ratio, histogram, slow-mo stats."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(sc["sample_every_s"] * fps)))
    aw = int(sc["analysis_width"])
    stats = [{"green": [], "hist": [], "dup": 0, "pairs": 0, "change": []} for _ in spans]
    si, idx, prev_sample = 0, 0, None
    prev_small: Optional[tuple[int, np.ndarray]] = None  # (frame idx, gray)
    while True:
        ok = cap.grab()
        if not ok:
            break
        t = idx / fps
        while si < len(spans) - 1 and t >= spans[si][1]:
            si, prev_sample, prev_small = si + 1, None, None
        # sample a frame + the next one (for the duplicate-frame test)
        pos = idx - int(round(spans[si][0] * fps))
        if pos % step in (0, 1):
            ok, frame = cap.retrieve()
            if ok:
                h, w = frame.shape[:2]
                small = cv2.resize(frame, (aw, max(1, int(h * aw / w))),
                                   interpolation=cv2.INTER_AREA)
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                st = stats[si]
                if pos % step == 0:
                    st["green"].append(green_ratio(small, sc))
                    st["hist"].append(hsv_hist(small))
                    if prev_sample is not None:
                        st["change"].append(float(cv2.absdiff(gray, prev_sample).mean()))
                    prev_sample = gray
                if prev_small is not None and prev_small[0] == idx - 1:
                    st["pairs"] += 1
                    if float(cv2.absdiff(gray, prev_small[1]).mean()) < sc["dup_frame_eps"]:
                        st["dup"] += 1
                prev_small = (idx, gray)
        idx += 1
    cap.release()
    return stats


def _similarity(a: list[np.ndarray], b: list[np.ndarray]) -> float:
    if not a or not b:
        return 0.0
    ha = np.mean(a, axis=0).astype(np.float32)
    hb = np.mean(b, axis=0).astype(np.float32)
    return float(cv2.compareHist(ha, hb, cv2.HISTCMP_CORREL))


def classify_scenes(spans: list[tuple[float, float]], stats: list[dict], sc: dict) -> list[str]:
    """Label each scene live / replay / non_play from its stats."""
    green = [float(np.mean(s["green"])) if s["green"] else 0.0 for s in stats]
    kinds = ["non_play" if g < sc["green_min"] else "live" for g in green]
    for i, (s, e) in enumerate(spans):
        if kinds[i] != "live" or (e - s) >= sc["replay_max_s"]:
            continue
        st = stats[i]
        moving = bool(st["change"]) and float(np.median(st["change"])) >= sc["min_change"]
        dup_ratio = st["dup"] / st["pairs"] if st["pairs"] else 0.0
        slowmo = moving and st["pairs"] >= 3 and dup_ratio >= sc["slowmo_dup_ratio"]
        prev_live = next((j for j in range(i - 1, -1, -1) if kinds[j] == "live"), None)
        # nearest previous live scene must be adjacent or separated only by a transition
        sim = 0.0
        if prev_live is not None and all(
                kinds[j] == "non_play" or spans[j][1] - spans[j][0] <= sc["transition_max_s"]
                for j in range(prev_live + 1, i)):
            sim = _similarity(stats[prev_live]["hist"], st["hist"])
        nxt = i + 1
        bracketed = (i > 0 and (kinds[i - 1] == "non_play" or
                                spans[i - 1][1] - spans[i - 1][0] <= sc["transition_max_s"])) \
            or (nxt < len(spans) and (green[nxt] < sc["green_min"] or
                                      spans[nxt][1] - spans[nxt][0] <= sc["transition_max_s"]))
        if slowmo or (sim >= sc["replay_hist_sim"] and bracketed and prev_live is not None):
            kinds[i] = "replay"
    return kinds


def detect_scenes(video_path: str, cfg: dict) -> list[Scene]:
    """Detect hard cuts and classify each scene as live / replay / non_play."""
    sc = scene_cfg(cfg)
    spans, _fps, _dur = _cut_times(video_path, sc)
    if not spans:
        return []
    stats = _scene_stats(video_path, spans, sc)
    kinds = classify_scenes(spans, stats, sc)
    return [Scene(scene_id=i, start_s=round(s, 3), end_s=round(e, 3), kind=k)
            for i, ((s, e), k) in enumerate(zip(spans, kinds))]
