"""Person detection.

``PersonDetector`` wraps ultralytics YOLO (COCO class 0 = person), imported
lazily. Without ultralytics (or with ``backend="opencv"``) it falls back to
OpenCV: the HOG people detector when the OpenCV build has it (OpenCV 5 dropped
it from the main module) plus a colour-blob detector for non-grass regions on
the field, which is what makes synthetic/test footage work.

Every detection is a dict ``{"box": [x1,y1,x2,y2], "score": float,
"referee": bool, "referee_score": float, "sideline": bool}``. Referees
(black/white vertical stripes) and sideline people (feet outside the grass
region) are flagged, not silently dropped; callers decide.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np


# --------------------------------------------------------------------------
# field / grass helpers
def grass_mask(frame_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return (((h >= 33) & (h <= 90) & (s >= 50) & (v >= 40)).astype(np.uint8)) * 255


def field_region(frame_bgr: np.ndarray, grass: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """Convex hull of the main grass area (uint8 mask) or None if no field visible."""
    g = grass if grass is not None else grass_mask(frame_bgr)
    H, W = g.shape
    if g.mean() / 255.0 < 0.15:
        return None
    s = min(1.0, 320.0 / W)  # work at low resolution, scale the hull back up
    gs = cv2.resize(g, (max(1, int(W * s)), max(1, int(H * s))), interpolation=cv2.INTER_NEAREST) if s < 1 else g
    k = max(5, int(min(gs.shape) * 0.03)) | 1
    closed = cv2.morphologyEx(gs, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(closed)
    if n <= 1:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    cnts, _ = cv2.findContours((lab == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    hull = cv2.convexHull(np.vstack(cnts)).astype(np.float32) / s
    m = np.zeros_like(g)
    cv2.fillConvexPoly(m, np.round(hull).astype(np.int32), 255)
    return m


def referee_score(crop_bgr: np.ndarray) -> float:
    """~1 for a black/white vertically striped torso, ~0 otherwise."""
    if crop_bgr is None or crop_bgr.size == 0:
        return 0.0
    h, w = crop_bgr.shape[:2]
    t = crop_bgr[int(0.18 * h):max(int(0.5 * h), int(0.18 * h) + 1), int(0.1 * w):max(int(0.9 * w), 1)]
    if t.size == 0 or t.shape[1] < 6:
        return 0.0
    hsv = cv2.cvtColor(t, cv2.COLOR_BGR2HSV)
    achro = float((hsv[..., 1] < 60).mean())
    if achro < 0.5:
        return 0.0
    v = hsv[..., 2].astype(np.float32)
    dark = v < 90
    bright = v > 150
    if dark.mean() < 0.2 or bright.mean() < 0.2:
        return 0.0
    # vertical stripes: columns are consistently dark or bright, rows alternate
    cols = (v.mean(0) > (v.mean() if v.std() > 1 else 128)).astype(np.int8)
    trans = int(np.abs(np.diff(cols)).sum())
    col_purity = float(np.mean(np.maximum(dark.mean(0), bright.mean(0))))
    row_prof = (v.mean(1) > v.mean()).astype(np.int8)
    row_trans = int(np.abs(np.diff(row_prof)).sum())
    if trans < 3:
        return 0.0
    s = min(1.0, trans / 6.0) * np.clip((col_purity - 0.5) / 0.4, 0, 1) * achro
    if row_trans > trans:  # horizontal stripes / numbers, not a ref shirt
        s *= 0.3
    return float(np.clip(s, 0, 1))


def annotate(frame_bgr: np.ndarray, dets: list[dict], field: Optional[np.ndarray] = None) -> list[dict]:
    """Add referee / sideline flags to detections in place."""
    if field is None:
        field = field_region(frame_bgr)
    H, W = frame_bgr.shape[:2]
    for d in dets:
        x1, y1, x2, y2 = [int(round(v)) for v in d["box"]]
        crop = frame_bgr[max(0, y1):min(H, y2), max(0, x1):min(W, x2)]
        rs = referee_score(crop)
        d["referee_score"] = rs
        d["referee"] = rs >= 0.5
        side = False
        # feet cut off by the frame edge or a close-up: we cannot tell, keep the player
        if field is not None and y2 < H - 3 and (y2 - y1) < 0.5 * H and field.mean() / 255.0 > 0.3:
            fx, fy = int(np.clip((x1 + x2) / 2, 0, W - 1)), int(np.clip(y2 - 1, 0, H - 1))
            # tolerate a margin: feet may be just past the painted hull
            m = max(3, int(0.02 * H))
            ys, xs = slice(max(0, fy - m), min(H, fy + m + 1)), slice(max(0, fx - m), min(W, fx + m + 1))
            side = not bool(field[ys, xs].any())
        d["sideline"] = side
    return dets


def nms(dets: list[dict], iou_thr: float = 0.5) -> list[dict]:
    dets = sorted(dets, key=lambda d: -d["score"])
    keep: list[dict] = []
    for d in dets:
        if all(box_iou(d["box"], k["box"]) < iou_thr for k in keep):
            keep.append(d)
    return keep


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / ua) if ua > 0 else 0.0


# --------------------------------------------------------------------------
def blob_detect(frame_bgr: np.ndarray, min_h: int = 18, field: Optional[np.ndarray] = None,
                grass: Optional[np.ndarray] = None) -> list[dict]:
    """Non-grass blobs on the field with person-like shape (synthetic/test footage)."""
    H, W = frame_bgr.shape[:2]
    g = grass if grass is not None else grass_mask(frame_bgr)
    if g.mean() / 255.0 < 0.15:
        return []
    cand = cv2.bitwise_not(g)
    if field is not None:
        cand = cv2.bitwise_and(cand, field)
    # remove thin painted lines / noise, then glue body parts together
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 9)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(cand)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if h < min_h or w < 6 or h > 0.9 * H or w > 0.5 * W:
            continue
        ar = h / float(w)
        if ar < 0.9 or ar > 5.0:
            continue
        fill = area / float(w * h)
        if fill < 0.35:
            continue
        # a blob touching the frame border (stands, graphics) is not a player
        if x <= 0 or y <= 0 or x + w >= W or y + h >= H:
            if fill < 0.6:
                continue
        score = float(np.clip(0.3 + 0.5 * fill, 0, 0.8))
        out.append({"box": [float(x), float(y), float(x + w), float(y + h)], "score": score})
    return out


def hog_detect(frame_bgr: np.ndarray) -> list[dict]:
    if not hasattr(cv2, "HOGDescriptor"):
        return []
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    H, W = frame_bgr.shape[:2]
    s = 1.0 if H >= 480 else 480.0 / H
    img = cv2.resize(frame_bgr, None, fx=s, fy=s) if s != 1.0 else frame_bgr
    rects, weights = hog.detectMultiScale(img, winStride=(8, 8), padding=(8, 8), scale=1.05)
    out = []
    for (x, y, w, h), wt in zip(rects, np.asarray(weights).reshape(-1)):
        if wt < 0.5:
            continue
        out.append({"box": [x / s, y / s, (x + w) / s, (y + h) / s],
                    "score": float(np.clip(0.3 + 0.2 * wt, 0, 0.9))})
    return out


# --------------------------------------------------------------------------
def resolve_weights(model: str, cfg: Optional[dict]) -> str:
    """Put bare model names (yolo11n.pt) in <root>/cache/models so downloads land there."""
    if os.path.sep in model or os.path.isabs(model) or not model.endswith(".pt"):
        return model
    root = (cfg or {}).get("root")
    cache = ((cfg or {}).get("paths") or {}).get("cache", "cache")
    if not root:
        return model
    d = Path(root) / cache / "models"
    d.mkdir(parents=True, exist_ok=True)
    return str(d / model)


def ultralytics_available() -> bool:
    try:
        import ultralytics  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


class PersonDetector:
    def __init__(self, model: str = "yolo11n.pt", device: str = "cpu", conf: float = 0.25,
                 imgsz: int = 960, cfg: Optional[dict] = None, backend: str = "auto"):
        self.model_name = model
        self.device = device
        self.conf = conf
        self.imgsz = imgsz
        self.cfg = cfg or {}
        self.yolo = None
        self.backend = "opencv"
        want = backend if backend != "auto" else (self.cfg.get("detection", {}).get("backend") or "auto")
        if want in ("auto", "ultralytics"):
            try:
                from ultralytics import YOLO
                self.yolo = YOLO(resolve_weights(model, self.cfg))
                self.backend = "ultralytics"
            except Exception:  # noqa: BLE001 - not installed / weights unavailable
                if want == "ultralytics":
                    raise
                self.yolo = None

    def detect(self, frame_bgr: np.ndarray, flag: bool = True) -> list[dict]:
        grass = grass_mask(frame_bgr)
        field = field_region(frame_bgr, grass)
        if self.yolo is not None:
            dets = self._detect_yolo(frame_bgr)
        else:
            dets = nms(blob_detect(frame_bgr, field=field, grass=grass) + hog_detect(frame_bgr), 0.45)
            dets = [d for d in dets if d["score"] >= min(self.conf, 0.3)]
        return annotate(frame_bgr, dets, field) if flag else dets

    def _detect_yolo(self, frame_bgr: np.ndarray) -> list[dict]:
        r = self.yolo.predict(frame_bgr, classes=[0], conf=self.conf, imgsz=self.imgsz,
                              device=self.device, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return []
        xyxy = r.boxes.xyxy.cpu().numpy()
        sc = r.boxes.conf.cpu().numpy()
        return [{"box": [float(v) for v in b], "score": float(s)} for b, s in zip(xyxy, sc)]

    def __call__(self, frame_bgr: np.ndarray) -> list[dict]:
        return self.detect(frame_bgr)


def is_player(det: dict) -> bool:
    """Players only: drop flagged referees and sideline people."""
    return not det.get("referee") and not det.get("sideline")
