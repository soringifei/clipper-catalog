"""Detection + multi-object tracking inside one play window.

``track_play(video_path, play, cfg, device)`` decodes only
``[play.start_s, play.end_s]`` at ``detection.fine_fps``, splits the window at
hard cuts (track ids never cross a cut), detects people, drops referees and
sideline people, tracks, and annotates each track with:

* ``team_prob``   P(Rebels) when ``player.team_kit_hsv`` is set, else 0.5
                  (identity.py resolves the Rebels cluster from OCR votes)
* ``team_cluster`` / ``team_cluster_conf`` 2-means on torso colour of all tracks
* ``torso_hsv``    median torso colour [h, s, v]
* ``embedding``    mean appearance embedding (reid.embed_crop)
* ``id_switch_score`` / ``id_switch_t``  suspected identity switch inside the track
* ``segment``      index of the cut-free segment inside the play

Backends: ultralytics ``model.track`` with a custom tracker yaml (botsort by
default; tracktrack / deepocsort / bytetrack / ocsort when the installed
ultralytics ships them), else the IoU + appearance + GMC tracker in
``iou_tracker.py``.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Iterator, Optional, Sequence

import cv2
import numpy as np

from ..core.models import Play, Track
from ..detection.detector import PersonDetector, annotate, blob_detect
from ..reid import reid as _reid
from .iou_tracker import IouTracker

TRACKER_ALIASES = {
    "botsort": "botsort", "bot-sort": "botsort", "bot_sort": "botsort",
    "tracktrack": "tracktrack", "track-track": "tracktrack",
    "deepocsort": "deepocsort", "deep_oc_sort": "deepocsort", "deep-oc-sort": "deepocsort",
    "bytetrack": "bytetrack", "byte": "bytetrack", "ocsort": "ocsort", "oc-sort": "ocsort",
    "iou": "iou", "fallback": "iou",
}


# --------------------------------------------------------------------------
# decoding
def iter_frames(video_path: str, start_s: float, end_s: float, fps: float
                ) -> Iterator[tuple[float, int, np.ndarray]]:
    """Yield (t, frame_idx, frame) sampled at ``fps`` inside [start_s, end_s]."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {video_path}")
    try:
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if src_fps <= 0 or src_fps > 1000:
            src_fps = 30.0
        start_f = max(0, int(math.floor(start_s * src_fps)))
        if start_f > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
        idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or start_f)
        step = 1.0 / max(fps, 1e-3)
        next_t = start_s
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = idx / src_fps
            idx += 1
            if t > end_s + 1e-6:
                break
            if t + 0.5 / src_fps >= next_t:
                yield t, idx - 1, frame
                next_t += step
                while next_t <= t:
                    next_t += step
    finally:
        cap.release()


def read_frames_at(video_path: str, times: Sequence[float]) -> dict[float, np.ndarray]:
    """Decode the frames nearest to each requested time (sorted, seek on big gaps)."""
    out: dict[float, np.ndarray] = {}
    if not times:
        return out
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return out
    try:
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        pos = -10**9
        frame = None
        for t in sorted(set(float(x) for x in times)):
            target = int(round(t * src_fps))
            if target < pos or target - pos > src_fps * 2:
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, target))
                pos = target - 1
                frame = None
            while pos < target:
                ok, f = cap.read()
                if not ok:
                    break
                pos += 1
                frame = f
            if frame is not None:
                out[t] = frame
    finally:
        cap.release()
    return out


# --------------------------------------------------------------------------
# cut detection
def _frame_hist(frame: np.ndarray) -> np.ndarray:
    small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [30, 16], [0, 180, 0, 256])
    return cv2.normalize(h, h).astype(np.float32)


def is_hard_cut(prev_hist: Optional[np.ndarray], hist: np.ndarray, thr: float = 0.5) -> bool:
    if prev_hist is None:
        return False
    return cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA) > thr


# --------------------------------------------------------------------------
# colour helpers
def torso_hsv(crop_bgr: np.ndarray) -> Optional[list[float]]:
    if crop_bgr is None or crop_bgr.size == 0:
        return None
    h, w = crop_bgr.shape[:2]
    t = crop_bgr[int(0.18 * h):max(int(0.5 * h), int(0.18 * h) + 1), int(0.2 * w):max(int(0.8 * w), 1)]
    if t.size == 0:
        return None
    hsv = cv2.cvtColor(t, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
    grass = (hsv[:, 0] >= 33) & (hsv[:, 0] <= 90) & (hsv[:, 1] >= 50)
    px = hsv[~grass] if (~grass).sum() >= 0.2 * len(hsv) else hsv
    # jersey colour = dominant chroma; numbers are small and get outvoted by the median
    hue = px[:, 0] * (2 * np.pi / 180.0)
    c, s = np.median(np.cos(hue) * px[:, 1]), np.median(np.sin(hue) * px[:, 1])
    hmed = (math.degrees(math.atan2(s, c)) % 360) / 2.0
    return [float(hmed), float(np.median(px[:, 1])), float(np.median(px[:, 2]))]


def hsv_feature(hsv: Sequence[float]) -> np.ndarray:
    h, s, v = float(hsv[0]), float(hsv[1]), float(hsv[2])
    a = h * 2 * np.pi / 180.0
    r = s / 255.0
    return np.array([r * np.cos(a), r * np.sin(a), 0.8 * v / 255.0], np.float32)


def cluster_teams(tracks: list[dict], kit_hsv: Optional[Sequence[float]] = None) -> None:
    """2-means on per-track torso colour; writes team_cluster(_conf) and team_prob in place."""
    feats, idx = [], []
    for i, t in enumerate(tracks):
        if t.get("torso_hsv") is not None:
            feats.append(hsv_feature(t["torso_hsv"]))
            idx.append(i)
    for t in tracks:
        t["team_cluster"], t["team_cluster_conf"] = None, 0.0
        t.setdefault("team_prob", 0.5)
    if len(feats) < 2:
        return
    X = np.stack(feats).astype(np.float32)
    w = np.array([max(1, len(tracks[i]["times"])) for i in idx], np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)
    # weight long tracks by repeating them (cv2.kmeans has no sample weights)
    rep = np.repeat(np.arange(len(X)), np.clip((w / w.max() * 5).astype(int), 1, 5))
    _, _, centers = cv2.kmeans(X[rep], 2, None, crit, 5, cv2.KMEANS_PP_CENTERS)
    sep = float(np.linalg.norm(centers[0] - centers[1]))
    for k, i in enumerate(idx):
        d = np.linalg.norm(centers - X[k], axis=1)
        c = int(np.argmin(d))
        conf = float(d[1 - c] / (d.sum() + 1e-6))  # 0.5 .. 1
        conf = 0.5 + (conf - 0.5) * min(1.0, sep / 0.25)  # poorly separated -> uncertain
        tracks[i]["team_cluster"], tracks[i]["team_cluster_conf"] = c, conf
    if kit_hsv is not None:
        kf = hsv_feature(kit_hsv)
        dk = np.linalg.norm(centers - kf, axis=1)
        rebels = int(np.argmin(dk))
        for i in idx:
            t = tracks[i]
            p = t["team_cluster_conf"]
            t["team_prob"] = p if t["team_cluster"] == rebels else 1.0 - p
        for t in tracks:
            t["rebels_cluster"] = rebels
    tracks_centers = centers.tolist()
    for t in tracks:
        t["team_centers"] = tracks_centers


# --------------------------------------------------------------------------
def _tracker_yaml(name: str, cfg: dict) -> Optional[str]:
    """Write <cache>/trackers/<name>_rebels.yaml from the ultralytics default."""
    import yaml
    try:
        import ultralytics
    except Exception:  # noqa: BLE001
        return None
    base = Path(ultralytics.__file__).parent / "cfg" / "trackers" / f"{name}.yaml"
    if not base.exists():
        return None
    d = yaml.safe_load(base.read_text()) or {}
    tc = cfg.get("tracking", {})
    d["track_buffer"] = int(tc.get("track_buffer", d.get("track_buffer", 30)))
    if "with_reid" in d:
        d["with_reid"] = bool(tc.get("with_reid", False))
    if "gmc_method" in d and d.get("gmc_method") in (None, "none"):
        d["gmc_method"] = "sparseOptFlow"  # broadcast cameras pan constantly
    for k, v in (tc.get("overrides") or {}).items():
        d[k] = v
    root = Path(cfg.get("root", "."))
    out = root / cfg.get("paths", {}).get("cache", "cache") / "trackers" / f"{name}_rebels.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(d, sort_keys=False))
    return str(out)


def _resolve_tracker(cfg: dict, det: PersonDetector) -> tuple[str, Optional[str]]:
    want = TRACKER_ALIASES.get(str(cfg.get("tracking", {}).get("tracker", "botsort")).lower(), "botsort")
    if want == "iou" or det.yolo is None:
        return "iou", None
    for name in (want, "botsort"):
        y = _tracker_yaml(name, cfg)
        if y:
            return name, y
    return "iou", None


def _maybe_fallback(det: PersonDetector, video_path: str, play: Play) -> None:
    """YOLO sees no people but the field has player-like blobs (synthetic/cartoon
    footage, tiny far-away players): switch this play to the OpenCV detector."""
    d = max(0.0, play.end_s - play.start_s)
    probes = read_frames_at(video_path, [play.start_s + f * d for f in (0.2, 0.5, 0.8)])
    if not probes:
        return
    try:
        n_yolo = sum(len(det._detect_yolo(f)) for f in probes.values())
    except Exception:  # noqa: BLE001
        n_yolo = 0
    n_blob = sum(len(blob_detect(f)) for f in probes.values())
    if n_yolo == 0 and n_blob >= 2 * len(probes):
        det.yolo = None
        det.backend = "opencv"


def track_play(video_path: str, play, cfg: dict, device: str = "cpu") -> dict:
    if isinstance(play, dict):
        play = Play.from_dict(play)
    dc = cfg.get("detection", {})
    fps = float(dc.get("fine_fps", 15))
    det = PersonDetector(model=dc.get("model", "yolo11n.pt"), device=device,
                         conf=float(dc.get("conf", 0.25)), imgsz=int(dc.get("imgsz", 960)), cfg=cfg)
    if det.yolo is not None and str(dc.get("backend", "auto")) == "auto":
        _maybe_fallback(det, video_path, play)
    tracker_name, tracker_yaml = _resolve_tracker(cfg, det)
    emb_backend = _reid.backend_from_cfg(cfg)
    tb = int(cfg.get("tracking", {}).get("track_buffer", 30))
    iou_trk = IouTracker(track_buffer=max(3, int(tb * fps / 30.0)))
    cut_thr = float(cfg.get("tracking", {}).get("cut_threshold", 0.5))

    raw: dict[int, dict] = {}
    scene_ids = list(play.scene_ids or [])
    segment = 0
    prev_hist = None
    cuts: list[float] = []
    frame_size = [0, 0]
    local_map: dict[tuple[int, int], int] = {}
    next_gid = 1
    first_in_segment = True
    filtered = {"referee": 0, "sideline": 0}
    n_frames = 0

    for t, fidx, frame in iter_frames(video_path, play.start_s, play.end_s, fps):
        n_frames += 1
        frame_size = [int(frame.shape[1]), int(frame.shape[0])]
        hist = _frame_hist(frame)
        if is_hard_cut(prev_hist, hist, cut_thr):
            cuts.append(round(t, 3))
            segment += 1
            first_in_segment = True
            iou_trk.reset()
        prev_hist = hist

        dets_ids: list[tuple[dict, Optional[int]]] = []
        if tracker_name != "iou":
            try:
                r = det.yolo.track(frame, persist=not first_in_segment, tracker=tracker_yaml,
                                   classes=[0], conf=det.conf, imgsz=det.imgsz, device=device,
                                   verbose=False)[0]
                dl = []
                if r.boxes is not None and len(r.boxes):
                    xyxy = r.boxes.xyxy.cpu().numpy()
                    sc = r.boxes.conf.cpu().numpy()
                    ids = r.boxes.id.cpu().numpy().astype(int) if r.boxes.id is not None else [None] * len(sc)
                    dl = [({"box": [float(v) for v in b], "score": float(s)}, i)
                          for b, s, i in zip(xyxy, sc, ids)]

                annotate(frame, [d for d, _ in dl])
                for d, lid in dl:
                    if lid is None:
                        continue
                    key = (segment, int(lid))
                    if key not in local_map:
                        local_map[key] = next_gid
                        next_gid += 1
                    dets_ids.append((d, local_map[key]))
            except Exception:  # noqa: BLE001 - tracker failure -> fall back for the rest
                tracker_name = "iou"
                iou_trk.reset()
        first_in_segment = False
        crops_cache: dict[int, np.ndarray] = {}
        if tracker_name == "iou":
            dl = det.detect(frame)
            embs = []
            for k, d in enumerate(dl):
                c = _reid.crop_box(frame, d["box"])
                crops_cache[k] = c
                embs.append(_reid.embed_crop(c, emb_backend) if c.size else None)
            ids = iou_trk.update(frame, [d["box"] for d in dl], [d["score"] for d in dl], embs)
            # iou ids are unique across resets already; keep a separate range from yolo ids
            dets_ids = [(d, i) for d, i in zip(dl, ids) if i is not None]
            for d, e in zip(dl, embs):
                d["_emb"] = e

        for d, gid in dets_ids:
            if d.get("referee"):
                filtered["referee"] += 1
                continue
            if d.get("sideline"):
                filtered["sideline"] += 1
                continue
            crop = _reid.crop_box(frame, d["box"])
            if crop.size == 0:
                continue
            e = d.get("_emb")
            if e is None:
                e = _reid.embed_crop(crop, emb_backend)
            tr = raw.setdefault(gid, {"track_id": gid, "segment": segment, "times": [], "boxes": [],
                                      "scores": [], "_embs": [], "_torso": []})
            if tr["segment"] != segment:  # safety: never let an id cross a cut
                continue
            tr["times"].append(round(float(t), 3))
            tr["boxes"].append([round(float(v), 1) for v in d["box"]])
            tr["scores"].append(round(float(d["score"]), 3))
            tr["_embs"].append(e)
            tr["_torso"].append(torso_hsv(crop))

    tracks = []
    min_len = int(cfg.get("tracking", {}).get("min_track_len", 3))
    for gid, tr in sorted(raw.items()):
        if len(tr["times"]) < min_len:
            continue
        seg = tr["segment"]
        scene_id = scene_ids[min(seg, len(scene_ids) - 1)] if scene_ids else seg
        embs = [e for e in tr.pop("_embs") if e is not None]
        torso = [x for x in tr.pop("_torso") if x is not None]
        tr["scene_id"] = int(scene_id)
        m = _reid.mean_embedding(embs)
        tr["embedding"] = [round(float(v), 5) for v in m] if m is not None else None
        tr["embedding_backend"] = emb_backend
        if torso:
            T = np.array(torso, np.float32)
            a = T[:, 0] * 2 * np.pi / 180.0
            hmed = (math.degrees(math.atan2(np.median(np.sin(a) * T[:, 1]),
                                            np.median(np.cos(a) * T[:, 1]))) % 360) / 2.0
            tr["torso_hsv"] = [round(hmed, 1), float(np.median(T[:, 1])), float(np.median(T[:, 2]))]
        else:
            tr["torso_hsv"] = None
        sw, swt = switch_score(tr["times"], tr["boxes"], embs)
        tr["id_switch_score"], tr["id_switch_t"] = sw, swt
        tr["team_prob"] = 0.5
        tracks.append(tr)

    kit = (cfg.get("player") or {}).get("team_kit_hsv")
    cluster_teams(tracks, kit)
    # validate against the shared contract
    for tr in tracks:
        Track.from_dict(tr)
    return {"tracks": tracks, "fps": fps, "frame_size": frame_size, "scene_cuts": cuts,
            "tracker": tracker_name, "detector": det.backend, "embedding_backend": emb_backend,
            "n_frames": n_frames, "filtered": filtered,
            "play_id": play.play_id, "start_s": play.start_s, "end_s": play.end_s}


def switch_score(times: Sequence[float], boxes: Sequence[Sequence[float]],
                 embs: Sequence[np.ndarray], win: int = 3) -> tuple[float, Optional[float]]:
    """Suspected ID switch inside a track: appearance jump and/or sudden box-size jump."""
    best, best_t = 0.0, None
    n = len(times)
    if n >= 2 * win and len(embs) == n:
        E = np.stack([_reid.normalize(e) for e in embs])
        for i in range(win, n - win + 1):
            a = _reid.normalize(E[i - win:i].mean(0))
            b = _reid.normalize(E[i:i + win].mean(0))
            if a is None or b is None:
                continue
            d = 1.0 - float(a @ b)
            s = float(np.clip((d - 0.08) / 0.25, 0, 1))
            if s > best:
                best, best_t = s, times[i]
    for i in range(1, n):
        if times[i] - times[i - 1] > 0.25:
            continue
        h0 = boxes[i - 1][3] - boxes[i - 1][1]
        h1 = boxes[i][3] - boxes[i][1]
        if min(h0, h1) <= 0:
            continue
        r = max(h0, h1) / min(h0, h1)
        c0 = ((boxes[i - 1][0] + boxes[i - 1][2]) / 2, (boxes[i - 1][1] + boxes[i - 1][3]) / 2)
        c1 = ((boxes[i][0] + boxes[i][2]) / 2, (boxes[i][1] + boxes[i][3]) / 2)
        jump = math.hypot(c1[0] - c0[0], c1[1] - c0[1]) / max(h0, h1)
        s = float(np.clip((r - 1.35) / 0.5, 0, 1) * 0.7 + np.clip((jump - 0.6) / 0.6, 0, 1) * 0.5)
        s = min(1.0, s)
        if s > best:
            best, best_t = s, times[i]
    return round(best, 3), best_t
