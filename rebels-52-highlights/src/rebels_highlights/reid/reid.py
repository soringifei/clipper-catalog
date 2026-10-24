"""Appearance embeddings for re-identification.

Default backend ("hist") is a normalised HSV colour histogram of the torso and
lower body - cheap, CPU only, and consistent between game tracks and the
reference images. An optional torchvision ResNet18 global-pool embedding
("resnet18") is used only when explicitly requested via cfg
``reid.backend: resnet18`` and its weights are available; all embeddings in a
run must come from the same backend, so the backend name is stored alongside.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Iterable, Optional, Sequence

import cv2
import numpy as np

H_BINS, S_BINS, V_BINS = 12, 4, 4
HIST_DIM = 2 * (H_BINS * S_BINS + V_BINS)  # torso + lower body

_resnet = None
_resnet_failed = False


def backend_from_cfg(cfg: Optional[dict]) -> str:
    b = ((cfg or {}).get("reid") or {}).get("backend", "hist")
    if b == "resnet18" and _get_resnet() is None:
        return "hist"
    return b if b in ("hist", "resnet18") else "hist"


def _grass_mask(hsv: np.ndarray) -> np.ndarray:
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return ((h >= 33) & (h <= 90) & (s >= 50) & (v >= 40))


def _part_hist(hsv: np.ndarray) -> np.ndarray:
    if hsv.size == 0:
        return np.zeros(H_BINS * S_BINS + V_BINS, np.float32)
    keep = ~_grass_mask(hsv)
    if keep.sum() < 0.1 * keep.size:
        keep = np.ones_like(keep)
    px = hsv[keep]
    chroma = px[px[:, 1] >= 50]
    hs = np.zeros((H_BINS, S_BINS), np.float32)
    if len(chroma):
        hi = np.minimum((chroma[:, 0].astype(np.int32) * H_BINS) // 180, H_BINS - 1)
        si = np.minimum(((chroma[:, 1].astype(np.int32) - 50) * S_BINS) // 206, S_BINS - 1)
        np.add.at(hs, (hi, si), 1.0)
    achro = px[px[:, 1] < 50]
    vh = np.zeros(V_BINS, np.float32)
    if len(achro):
        vi = np.minimum(achro[:, 2].astype(np.int32) * V_BINS // 256, V_BINS - 1)
        np.add.at(vh, vi, 1.0)
    f = np.concatenate([hs.reshape(-1), vh * 1.5])
    return f / (f.sum() + 1e-6)


def hist_embedding(crop_bgr: np.ndarray) -> np.ndarray:
    """HSV histogram of torso (15-55% of height) and lower body (55-90%)."""
    if crop_bgr is None or crop_bgr.size == 0:
        return np.zeros(HIST_DIM, np.float32)
    h, w = crop_bgr.shape[:2]
    x1, x2 = int(0.15 * w), max(int(0.85 * w), int(0.15 * w) + 1)
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    torso = hsv[int(0.15 * h):max(int(0.55 * h), int(0.15 * h) + 1), x1:x2]
    legs = hsv[int(0.55 * h):max(int(0.9 * h), int(0.55 * h) + 1), x1:x2]
    f = np.concatenate([_part_hist(torso), 0.6 * _part_hist(legs)])
    f = np.sqrt(f)  # Hellinger mapping -> cosine behaves like Bhattacharyya
    return (f / (np.linalg.norm(f) + 1e-6)).astype(np.float32)


def _get_resnet():
    global _resnet, _resnet_failed
    if _resnet is not None or _resnet_failed:
        return _resnet
    try:
        import torch
        import torchvision
        m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        m.fc = torch.nn.Identity()
        m.eval()
        _resnet = m
    except Exception:  # noqa: BLE001 - no torch / no weights / offline
        _resnet_failed = True
    return _resnet


def resnet_embedding(crop_bgr: np.ndarray) -> Optional[np.ndarray]:
    m = _get_resnet()
    if m is None or crop_bgr is None or crop_bgr.size == 0:
        return None
    import torch
    x = cv2.resize(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB), (128, 256)).astype(np.float32) / 255.0
    x = (x - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    t = torch.from_numpy(x.transpose(2, 0, 1)[None].astype(np.float32))
    with torch.no_grad():
        f = m(t).numpy().reshape(-1)
    return (f / (np.linalg.norm(f) + 1e-6)).astype(np.float32)


def embed_crop(crop_bgr: np.ndarray, backend: str = "hist") -> np.ndarray:
    if backend == "resnet18":
        e = resnet_embedding(crop_bgr)
        if e is not None:
            return e
    return hist_embedding(crop_bgr)


def crop_box(frame: np.ndarray, box: Sequence[float], pad: float = 0.0) -> np.ndarray:
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    pw, ph = (x2 - x1) * pad, (y2 - y1) * pad
    x1, y1 = int(max(0, x1 - pw)), int(max(0, y1 - ph))
    x2, y2 = int(min(W, x2 + pw)), int(min(H, y2 + ph))
    if x2 <= x1 or y2 <= y1:
        return np.zeros((0, 0, 3), np.uint8)
    return frame[y1:y2, x1:x2]


# --------------------------------------------------------------------------
def normalize(v) -> Optional[np.ndarray]:
    if v is None:
        return None
    a = np.asarray(v, np.float32).reshape(-1)
    n = np.linalg.norm(a)
    return a / n if n > 0 else None


def mean_embedding(embs: Iterable) -> Optional[np.ndarray]:
    vs = [normalize(e) for e in embs if e is not None]
    vs = [v for v in vs if v is not None]
    if not vs:
        return None
    return normalize(np.mean(np.stack(vs), 0))


def cosine(a, b) -> float:
    a, b = normalize(a), normalize(b)
    if a is None or b is None or a.shape != b.shape:
        return 0.0
    return float(np.clip(a @ b, -1.0, 1.0))


def max_similarity(v, prototypes) -> float:
    if prototypes is None or v is None:
        return 0.0
    P = np.asarray(prototypes, np.float32)
    if P.ndim == 1:
        P = P[None]
    return max((cosine(v, p) for p in P), default=0.0)


def sim_to_conf(sim: float, lo: float = 0.55, hi: float = 0.95) -> float:
    """Map cosine similarity to a [0,1] confidence (hist embeddings are all positive)."""
    return float(np.clip((sim - lo) / (hi - lo), 0.0, 1.0))


# --------------------------------------------------------------------------
_IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _person_crops(img: np.ndarray, cfg: Optional[dict]) -> list[np.ndarray]:
    """Largest detected person (if a detector works), else the whole image."""
    try:
        from ..detection.detector import PersonDetector
        det = (cfg or {}).get("detection", {})
        d = PersonDetector(model=det.get("model", "yolo11n.pt"), device="cpu",
                           conf=0.35, imgsz=640, cfg=cfg, backend="ultralytics")
        dets = [x for x in d.detect(img) if not x.get("referee")]
        if dets:
            b = max(dets, key=lambda x: (x["box"][2] - x["box"][0]) * (x["box"][3] - x["box"][1]))
            c = crop_box(img, b["box"])
            if c.size:
                return [c]
    except Exception:  # noqa: BLE001
        pass
    return [img]


def load_reference_prototypes(ref_dir: str, cfg: Optional[dict] = None,
                              backend: Optional[str] = None) -> Optional[dict]:
    """Embeddings of reference_player_52/* images; None when the dir is empty/missing."""
    if not ref_dir:
        return None
    p = Path(ref_dir)
    if not p.is_absolute() and cfg and cfg.get("root"):
        p = Path(cfg["root"]) / p
    if not p.is_dir():
        return None
    files = sorted(f for f in glob.glob(str(p / "**" / "*"), recursive=True)
                   if f.lower().endswith(_IMG_EXT))
    backend = backend or backend_from_cfg(cfg)
    protos, used = [], []
    for f in files:
        img = cv2.imread(f)
        if img is None:
            continue
        for c in _person_crops(img, cfg):
            protos.append(embed_crop(c, backend))
            used.append(os.path.basename(f))
    if not protos:
        return None
    P = np.stack(protos)
    return {"prototypes": P.tolist(), "mean": normalize(P.mean(0)).tolist(),
            "files": used, "backend": backend, "n_images": len(set(used))}
