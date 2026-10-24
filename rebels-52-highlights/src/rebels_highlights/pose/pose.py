"""Movement / pose features for #52 in one play (never used for identity).

``pose_features(frames_crops, boxes, times)`` returns floats in [0, 1] unless
the key says otherwise:

  stance            crouched at the start (low box aspect / bent torso)
  lean              forward torso lean (keypoints only; 0 without them)
  acceleration      peak speed-up, normalised
  deceleration      peak slow-down, normalised (tackles / contact)
  direction_change  largest heading change while moving (0..1 = 0..180 deg)
  fall              went from upright to on the ground (box aspect + keypoints)
  fall_t            time of the fall (seconds, -1 if none)
  max_speed         body-heights per second (raw, not normalised)
  keypoints         1.0 when a keypoint backend contributed, else 0.0

Keypoint backends (lazy): ultralytics ``yolo11n-pose.pt`` or mediapipe;
fallback is pure box kinematics.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

_pose_model = None
_pose_failed = False


def _get_pose_model(cfg: Optional[dict] = None):
    global _pose_model, _pose_failed
    if _pose_model is not None or _pose_failed:
        return _pose_model
    if os.environ.get("REBELS_POSE_BACKEND", "") == "box":
        _pose_failed = True
        return None
    try:
        from ultralytics import YOLO
        name = ((cfg or {}).get("pose") or {}).get("model", "yolo11n-pose.pt")
        root = (cfg or {}).get("root")
        if root and os.path.sep not in name:
            d = Path(root) / (cfg.get("paths", {}).get("cache", "cache")) / "models"
            d.mkdir(parents=True, exist_ok=True)
            name = str(d / name)
        _pose_model = ("ultralytics", YOLO(name))
    except Exception:  # noqa: BLE001
        try:
            import mediapipe as mp  # type: ignore
            _pose_model = ("mediapipe", mp.solutions.pose.Pose(static_image_mode=True))
        except Exception:  # noqa: BLE001
            _pose_failed = True
    return _pose_model


def _keypoints(crop) -> Optional[dict]:
    """{'sh': (x,y), 'hip': (x,y), 'ank': (x,y)} in crop px, or None."""
    m = _pose_model
    if m is None or crop is None or getattr(crop, "size", 0) == 0:
        return None
    kind, model = m
    try:
        if kind == "ultralytics":
            r = model.predict(crop, imgsz=256, conf=0.2, device="cpu", verbose=False)[0]
            if r.keypoints is None or len(r.keypoints) == 0:
                return None
            k = r.keypoints.xy.cpu().numpy()
            c = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else np.ones(k.shape[:2])
            i = int(np.argmax(c.mean(1)))
            k, c = k[i], c[i]
            pts = {}
            for name, (a, b) in {"sh": (5, 6), "hip": (11, 12), "ank": (15, 16)}.items():
                if c[a] > 0.3 and c[b] > 0.3:
                    pts[name] = ((k[a] + k[b]) / 2).tolist()
            return pts or None
        import cv2
        res = model.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        if not res.pose_landmarks:
            return None
        L = res.pose_landmarks.landmark
        H, W = crop.shape[:2]
        mid = lambda a, b: [(L[a].x + L[b].x) / 2 * W, (L[a].y + L[b].y) / 2 * H]  # noqa: E731
        return {"sh": mid(11, 12), "hip": mid(23, 24), "ank": mid(27, 28)}
    except Exception:  # noqa: BLE001
        return None


def _smooth(a: np.ndarray, k: int = 3) -> np.ndarray:
    if len(a) < k:
        return a
    ker = np.ones(k) / k
    pad = np.pad(a, ((k // 2, k // 2), (0, 0)), mode="edge") if a.ndim == 2 else np.pad(a, k // 2, mode="edge")
    if a.ndim == 2:
        return np.stack([np.convolve(pad[:, j], ker, mode="valid") for j in range(a.shape[1])], 1)
    return np.convolve(pad, ker, mode="valid")


def box_kinematics(boxes: Sequence[Sequence[float]], times: Sequence[float]) -> dict:
    out = {"stance": 0.0, "lean": 0.0, "acceleration": 0.0, "deceleration": 0.0,
           "direction_change": 0.0, "fall": 0.0, "fall_t": -1.0, "max_speed": 0.0}
    n = len(boxes)
    if n < 2:
        return out
    B = np.asarray(boxes, np.float64)
    T = np.asarray(times, np.float64)
    h = np.maximum(B[:, 3] - B[:, 1], 1.0)
    w = np.maximum(B[:, 2] - B[:, 0], 1.0)
    aspect = h / w
    scale = float(np.median(h))
    # ground-contact point (bottom centre) is steadier than the box centre
    c = np.stack([(B[:, 0] + B[:, 2]) / 2, B[:, 3] - 0.3 * h], 1)
    c = _smooth(c, 3)
    dt = np.maximum(np.diff(T), 1e-3)
    v = np.diff(c, axis=0) / dt[:, None] / scale           # body heights / s
    sp = np.linalg.norm(v, axis=1)
    out["max_speed"] = float(sp.max()) if len(sp) else 0.0
    if len(sp) >= 2:
        acc = np.diff(_smooth(sp, 3)) / np.maximum(dt[1:], 1e-3)
        out["acceleration"] = float(np.clip(acc.max() / 10.0, 0, 1))
        out["deceleration"] = float(np.clip(-acc.min() / 10.0, 0, 1))
    best = 0.0
    for i in range(len(v)):
        if sp[i] < 0.6:
            continue
        for j in range(i + 1, len(v)):
            if T[j + 1] - T[i + 1] > 0.6:
                break
            if sp[j] < 0.6:
                continue
            cosang = float(np.dot(v[i], v[j]) / (sp[i] * sp[j]))
            best = max(best, math.acos(max(-1.0, min(1.0, cosang))) / math.pi)
    out["direction_change"] = float(best)
    k0 = max(1, min(n, 3))
    out["stance"] = float(np.clip((2.3 - np.median(aspect[:k0])) / 1.0, 0, 1))
    # fall: upright earlier (aspect >= 1.5) then low (aspect <= 1.1 or height collapse)
    up = aspect >= 1.5
    fall, fall_t = 0.0, -1.0
    for i in range(1, n):
        if not up[:i].any():
            continue
        h_up = float(np.median(h[:i][up[:i]]))
        low_aspect = np.clip((1.3 - aspect[i]) / 0.5, 0, 1)
        collapse = np.clip((0.75 - h[i] / h_up) / 0.3, 0, 1)
        s = float(max(low_aspect, collapse * 0.8))
        # must persist (two consecutive samples) to count
        if i + 1 < n:
            s = min(s, float(max(np.clip((1.3 - aspect[i + 1]) / 0.5, 0, 1),
                                 np.clip((0.75 - h[i + 1] / h_up) / 0.3, 0, 1) * 0.8)))
        if s > fall:
            fall, fall_t = s, float(T[i])
    out["fall"], out["fall_t"] = fall, fall_t
    return out


def pose_features(frames_crops: Optional[Sequence] , boxes: Sequence[Sequence[float]],
                  times: Sequence[float], cfg: Optional[dict] = None,
                  max_crops: int = 16, crop_times: Optional[Sequence[float]] = None
                  ) -> dict[str, float]:
    """Box kinematics + (when available) keypoint lean / torso-angle features.

    ``frames_crops`` are person crops aligned with ``boxes``/``times`` or, when
    ``crop_times`` is given, a subset taken at those times.
    """
    out = box_kinematics(boxes, times)
    out["keypoints"] = 0.0
    crops = list(frames_crops or [])
    if not crops or not any(getattr(c, "size", 0) for c in crops):
        return {k: float(v) for k, v in out.items()}
    if _get_pose_model(cfg) is None:
        return {k: float(v) for k, v in out.items()}
    if crop_times is None:
        crop_times = list(times) if len(times) == len(crops) else \
            list(np.linspace(times[0], times[-1], len(crops))) if len(times) else [0.0] * len(crops)
    idx = np.linspace(0, len(crops) - 1, min(max_crops, len(crops))).astype(int)
    angles, tt = [], []
    for i in idx:
        kp = _keypoints(crops[i])
        if not kp or "sh" not in kp or "hip" not in kp:
            continue
        dx = kp["sh"][0] - kp["hip"][0]
        dy = kp["hip"][1] - kp["sh"][1]  # up is positive
        angles.append(math.degrees(math.atan2(abs(dx), max(dy, 1e-3))))  # 0 = upright
        tt.append(float(crop_times[i]))
    if angles:
        out["keypoints"] = 1.0
        a = np.array(angles)
        early = a[: max(1, len(a) // 3)]
        out["lean"] = float(np.clip((np.median(early) - 10) / 40.0, 0, 1))
        out["stance"] = float(max(out["stance"], np.clip((np.median(early) - 15) / 35.0, 0, 1)))
        late = a[len(a) // 2:]
        if len(late) and np.max(late) > 60 and a[0] < 45:
            kf = float(np.clip((np.max(late) - 55) / 30.0, 0, 1))
            if kf > out["fall"]:
                out["fall"] = kf
                out["fall_t"] = float(tt[len(a) // 2 + int(np.argmax(late))])
    return {k: float(v) for k, v in out.items()}
