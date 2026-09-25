"""Pose providers (YOLO pose via ultralytics, or a fake one for tests), main-athlete
selection, gap filling and keypoint smoothing.

Coordinates are always in *display* pixels (after rotation) of the source video; frame
indices are on the constant ``work_fps`` timeline produced by ``video.FrameReader``.
"""
from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol

import numpy as np

log = logging.getLogger(__name__)

KP = {"nose": 0, "l_eye": 1, "r_eye": 2, "l_ear": 3, "r_ear": 4, "l_sho": 5, "r_sho": 6,
      "l_elb": 7, "r_elb": 8, "l_wri": 9, "r_wri": 10, "l_hip": 11, "r_hip": 12,
      "l_knee": 13, "r_knee": 14, "l_ank": 15, "r_ank": 16}
SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]


@dataclass
class RawPose:
    """All person detections of the pose pass (cached as npz)."""
    n_frames: int
    fps: float
    width: int
    height: int
    frames: np.ndarray            # (N,) int  work-frame index of each detection
    ids: np.ndarray               # (N,) int  tracker id or -1
    boxes: np.ndarray             # (N,4)
    scores: np.ndarray            # (N,)
    kps: np.ndarray               # (N,17,3) x, y, conf
    processed: np.ndarray         # (M,) int  frames the model actually ran on
    meta: dict = field(default_factory=dict)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.stem + ".tmp.npz")
        np.savez_compressed(tmp, frames=self.frames, ids=self.ids, boxes=self.boxes,
                            scores=self.scores, kps=self.kps, processed=self.processed,
                            hdr=np.array([self.n_frames, self.fps, self.width, self.height]))
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "RawPose":
        z = np.load(path)
        n, fps, w, h = z["hdr"].tolist()
        return cls(int(n), float(fps), int(w), int(h), z["frames"], z["ids"], z["boxes"],
                   z["scores"], z["kps"], z["processed"])


@dataclass
class PoseSeq:
    """Main athlete's smoothed keypoints, one row per work frame."""
    kp: np.ndarray        # (T,17,2) NaN where unknown/low confidence
    conf: np.ndarray      # (T,17)
    box: np.ndarray       # (T,4) NaN where not tracked
    fps: float
    width: int
    height: int
    track_frac: float = 0.0   # fraction of frames with the athlete tracked

    @property
    def T(self) -> int:
        return self.kp.shape[0]

    @property
    def valid(self) -> np.ndarray:
        return ~np.isnan(self.box[:, 0])


class PoseProvider(Protocol):
    name: str

    def run(self, path: Path, info: dict, cfg: dict) -> RawPose: ...


class FakePoseProvider:
    """Injects a known keypoint sequence (T,17,3) in display px - used by tests/demos."""
    name = "fake"

    def __init__(self, kps: np.ndarray, track_id: int = 1):
        self.kps = np.asarray(kps, float)
        self.track_id = track_id

    def run(self, path: Path, info: dict, cfg: dict) -> RawPose:
        T = min(len(self.kps), info["n_frames"]) if info.get("n_frames") else len(self.kps)
        k = self.kps[:T]
        xy = k[:, :, :2]
        boxes = np.stack([np.nanmin(xy[..., 0], 1), np.nanmin(xy[..., 1], 1),
                          np.nanmax(xy[..., 0], 1), np.nanmax(xy[..., 1], 1)], 1)
        pad = 0.08 * (boxes[:, 3] - boxes[:, 1])[:, None]
        boxes = boxes + np.hstack([-pad, -pad, pad, pad])
        fr = np.arange(T)
        return RawPose(T, info["work_fps"], info["disp_w"], info["disp_h"], fr,
                       np.full(T, self.track_id), boxes, np.full(T, 0.9), k, fr)


class YoloPoseProvider:
    name = "yolo"

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._model = None

    def model(self):
        if self._model is None:
            from ultralytics import YOLO
            from .config import resolve_path
            w = self.cfg["pose"]["model"]
            p = Path(w)
            if not p.is_absolute() and len(p.parts) == 1:
                d = resolve_path(self.cfg, "models")
                d.mkdir(parents=True, exist_ok=True)
                p = d / w
                if not p.exists():   # ultralytics downloads into cwd -> move it
                    import os
                    cwd = os.getcwd()
                    try:
                        os.chdir(d)
                        YOLO(w)
                    finally:
                        os.chdir(cwd)
            self._model = YOLO(str(p))
        return self._model

    def run(self, path: Path, info: dict, cfg: dict) -> RawPose:
        from .video import FrameReader
        pc = cfg["pose"]
        W, H, fps = info["disp_w"], info["disp_h"], info["work_fps"]
        s = min(1.0, pc["analysis_long_side"] / max(W, H))
        aw, ah = int(round(W * s / 2) * 2), int(round(H * s / 2) * 2)
        stride = max(1, int(round(fps / pc["max_pose_hz"])))
        model = self.model()
        frames, ids, boxes, scores, kps, processed = [], [], [], [], [], []
        t0 = time.time()
        n = 0
        for i, img in enumerate(FrameReader(path, fps, (aw, ah))):
            n = i + 1
            if i % stride:
                continue
            res = model.track(img, persist=True, tracker=pc["tracker"], verbose=False,
                              device=pc["device"], imgsz=pc["imgsz"], conf=pc["conf"],
                              classes=[0])[0]
            processed.append(i)
            if res.boxes is None or len(res.boxes) == 0 or res.keypoints is None:
                continue
            b = res.boxes.xyxy.cpu().numpy() / s
            sc = res.boxes.conf.cpu().numpy()
            tid = (res.boxes.id.cpu().numpy().astype(int) if res.boxes.id is not None
                   else np.full(len(b), -1))
            kd = res.keypoints.data.cpu().numpy().copy()
            if kd.shape[-1] == 2:
                kd = np.concatenate([kd, np.ones(kd.shape[:2] + (1,))], -1)
            kd[..., :2] /= s
            for j in range(len(b)):
                frames.append(i); ids.append(tid[j]); boxes.append(b[j])
                scores.append(sc[j]); kps.append(kd[j])
        # reset tracker state for the next clip
        try:
            for p in getattr(model.predictor, "trackers", []) or []:
                p.reset()
        except Exception:  # noqa: BLE001
            pass
        if hasattr(model, "predictor") and model.predictor is not None:
            model.predictor = None
        log.info("pose: %d frames (%d inferred) in %.1fs", n, len(processed), time.time() - t0)
        return RawPose(n, fps, W, H, np.array(frames, int), np.array(ids, int),
                       np.array(boxes, float).reshape(-1, 4), np.array(scores, float),
                       np.array(kps, float).reshape(-1, 17, 3), np.array(processed, int),
                       meta={"stride": stride, "seconds": round(time.time() - t0, 2)})


def pose_cache_key(path: Path, cfg: dict, info: dict, provider: str) -> str:
    st = path.stat()
    pc = cfg["pose"]
    s = "|".join(map(str, [path.resolve(), st.st_size, int(st.st_mtime), provider, pc["model"],
                           pc["imgsz"], pc["conf"], pc["max_pose_hz"], pc["analysis_long_side"],
                           info["work_fps"]]))
    return hashlib.sha1(s.encode()).hexdigest()[:10]


# ---------------------------------------------------------------------------
# main athlete selection
# ---------------------------------------------------------------------------
def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def select_main_track(raw: RawPose) -> np.ndarray:
    """Index into raw detections for each work frame (-1 = none). Main athlete =
    track with the largest sum of sqrt(area) * centrality * kp-confidence (i.e. big,
    central and persistent); tracker id switches are bridged by box IoU."""
    if len(raw.frames) == 0:
        return np.full(raw.n_frames, -1)
    W, H = raw.width, raw.height
    area = np.sqrt(np.clip((raw.boxes[:, 2] - raw.boxes[:, 0]) * (raw.boxes[:, 3] - raw.boxes[:, 1]),
                           0, None) / (W * H))
    cx = (raw.boxes[:, 0] + raw.boxes[:, 2]) / 2
    cent = 1 - 0.6 * np.clip(np.abs(cx - W / 2) / (W / 2), 0, 1)
    kc = np.nanmean(raw.kps[:, :, 2], 1)
    w = area * cent * (0.3 + kc)
    ids = raw.ids.copy()
    # untracked detections get a pseudo id per frame so they can't win on persistence
    ids[ids < 0] = -1 - np.arange((ids < 0).sum())
    uniq = np.unique(ids)
    tot = {u: w[ids == u].sum() for u in uniq}
    main = max(tot, key=tot.get)
    by_frame: dict[int, list[int]] = {}
    for j, f in enumerate(raw.frames):
        by_frame.setdefault(int(f), []).append(j)
    out = np.full(raw.n_frames, -1)
    last_box, last_f = None, -10 ** 9
    gap_switch = max(1, int(0.5 * raw.fps))
    for f in sorted(by_frame):
        dets = by_frame[f]
        hit = [j for j in dets if ids[j] == main]
        if hit:
            j = hit[0]
        elif last_box is not None:
            ious = [(_iou(raw.boxes[j], last_box), j) for j in dets]
            best = max(ious)
            if best[0] < 0.3 or f - last_f > 2 * raw.fps:
                continue
            j = best[1]
            if ids[j] >= 0 and f - last_f >= 0 and tot.get(ids[j], 0) > 0:
                # adopt the new id if the old one has vanished
                if f - last_f >= gap_switch or not any(ids[k] == main for k in dets):
                    main = ids[j]
        else:
            continue
        out[f] = j
        last_box, last_f = raw.boxes[j], f
    return out


# ---------------------------------------------------------------------------
# filtering
# ---------------------------------------------------------------------------
def one_euro(x: np.ndarray, fps: float, min_cutoff: float = 1.2, beta: float = 0.02,
             d_cutoff: float = 1.0) -> np.ndarray:
    """One-Euro filter along axis 0 for a (T, ...) array; NaN resets the filter."""
    x = np.asarray(x, float)
    out = np.full_like(x, np.nan)
    te = 1.0 / fps

    def alpha(c):
        r = 2 * np.pi * c * te
        return r / (r + 1)

    prev = None
    dprev = None
    for t in range(len(x)):
        xt = x[t]
        if prev is None:
            prev = xt.copy()
            dprev = np.zeros_like(xt)
            out[t] = xt
            continue
        nan = np.isnan(xt) | np.isnan(prev)
        dx = np.where(nan, 0.0, (xt - np.nan_to_num(prev)) / te)
        ad = alpha(d_cutoff)
        dh = ad * dx + (1 - ad) * dprev
        cutoff = min_cutoff + beta * np.abs(dh)
        a = alpha(cutoff)
        xh = a * xt + (1 - a) * prev
        xh = np.where(np.isnan(prev), xt, xh)
        out[t] = xh
        prev = np.where(np.isnan(xt), np.nan, xh)
        dprev = np.where(nan, 0.0, dh)
    return out


def savgol(x: np.ndarray, window: int = 9, order: int = 2) -> np.ndarray:
    """Savitzky-Golay smoothing along axis 0 over NaN-free runs (runs shorter than the
    window are left unchanged)."""
    x = np.asarray(x, float)
    half = window // 2
    A = np.vander(np.arange(-half, half + 1), order + 1, increasing=True)
    coef = np.linalg.pinv(A)[0]
    out = x.copy()
    flat = out.reshape(len(x), -1)
    src = x.reshape(len(x), -1)
    for c in range(flat.shape[1]):
        v = src[:, c]
        ok = ~np.isnan(v)
        idx = np.flatnonzero(np.diff(np.concatenate([[0], ok.astype(int), [0]])))
        for s, e in zip(idx[::2], idx[1::2]):
            if e - s >= window:
                seg = np.pad(v[s:e], half, mode="edge")
                flat[s:e, c] = np.convolve(seg, coef[::-1], mode="valid")
    return out


def fill_gaps(x: np.ndarray, max_gap: int) -> np.ndarray:
    """Linear interpolation along axis 0 of NaN runs <= max_gap (interior only)."""
    x = np.asarray(x, float).copy()
    flat = x.reshape(len(x), -1)
    t = np.arange(len(x))
    for c in range(flat.shape[1]):
        v = flat[:, c]
        ok = ~np.isnan(v)
        if ok.sum() < 2:
            continue
        interp = np.interp(t, t[ok], v[ok])
        # only fill gaps short enough and bounded on both sides
        nan_idx = np.flatnonzero(~ok)
        if len(nan_idx) == 0:
            continue
        runs = np.split(nan_idx, np.flatnonzero(np.diff(nan_idx) > 1) + 1)
        for r in runs:
            if r[0] > 0 and r[-1] < len(v) - 1 and len(r) <= max_gap and ok[r[0] - 1] and ok[r[-1] + 1]:
                v[r] = interp[r]
    return x


def build_sequence(raw: RawPose, cfg: dict) -> PoseSeq:
    pc = cfg["pose"]
    T, fps = raw.n_frames, raw.fps
    pick = select_main_track(raw)
    kp = np.full((T, 17, 2), np.nan)
    conf = np.zeros((T, 17))
    box = np.full((T, 4), np.nan)
    for f in range(T):
        j = pick[f]
        if j >= 0:
            kp[f] = raw.kps[j, :, :2]
            conf[f] = raw.kps[j, :, 2]
            box[f] = raw.boxes[j]
    tracked = pick >= 0
    stride = int(np.median(np.diff(raw.processed))) if len(raw.processed) > 1 else 1
    low = conf < pc["kp_min_conf"]
    kp[low] = np.nan
    max_gap = max(stride, int(round(pc["max_gap_s"] * fps)))
    kp = fill_gaps(kp, max_gap)
    box = fill_gaps(box, max_gap)
    conf = fill_gaps(np.where(tracked[:, None], conf, np.nan), max_gap)
    conf = np.nan_to_num(conf)
    if pc["smoothing"] == "one_euro":
        oe = pc["one_euro"]
        # scale-aware: beta is per px/s -> normalise by frame height
        scale = raw.height / 1000.0
        kp = one_euro(kp / scale, fps, oe["min_cutoff"], oe["beta"], oe["d_cutoff"]) * scale
    elif pc["smoothing"] == "savgol":
        kp = savgol(kp, window=max(5, int(fps * 0.3) | 1), order=2)
    box = savgol(box, window=max(5, int(fps * 0.3) | 1), order=1)
    return PoseSeq(kp=kp, conf=conf, box=box, fps=fps, width=raw.width, height=raw.height,
                   track_frac=float((~np.isnan(box[:, 0])).mean()) if T else 0.0)


def load_or_run_pose(path: Path, info: dict, cfg: dict, cache_dir: Path,
                     provider: Optional[PoseProvider] = None,
                     use_cache: bool = True) -> tuple[RawPose, bool]:
    """Returns (raw pose, from_cache)."""
    provider = provider or YoloPoseProvider(cfg)
    key = pose_cache_key(path, cfg, info, provider.name)
    cp = cache_dir / f"{path.stem}-{key}" / "raw_pose.npz"
    if use_cache and provider.name != "fake" and cp.exists():
        return RawPose.load(cp), True
    raw = provider.run(path, info, cfg)
    if provider.name != "fake":
        raw.save(cp)
    return raw, False
