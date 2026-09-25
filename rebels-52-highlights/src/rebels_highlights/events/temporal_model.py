"""Stage-2 event model: tiny numpy softmax regression over feature vectors.

Train from reviewed labels (``review/labels.csv``: ``play_id,label``) plus the
per-play feature dicts already computed by stage 1. The model is stored as JSON
in ``cache/event_model.json``; when present, ``events.classify_event`` blends
its probabilities with the rule scores (weight 0.5).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Optional

import numpy as np

from ..core.models import PLAY_TYPES
from .features import FEATURE_KEYS, feature_vector

BLEND_WEIGHT = 0.5
_CACHE: dict[str, tuple[float, dict]] = {}


def model_path(cfg: dict) -> Path:
    root = Path(cfg.get("root", "."))
    return root / cfg.get("paths", {}).get("cache", "cache") / "event_model.json"


def read_labels(csv_path: str) -> dict[str, str]:
    out: dict[str, str] = {}
    p = Path(csv_path)
    if not p.exists():
        return out
    with p.open(newline="") as fh:
        for row in csv.DictReader(fh):
            pid, lab = (row.get("play_id") or "").strip(), (row.get("label") or "").strip()
            if pid and lab in PLAY_TYPES:
                out[pid] = lab
    return out


def train(labels: dict[str, str], features_by_play: dict[str, dict],
          epochs: int = 500, lr: float = 0.5, l2: float = 1e-3) -> Optional[dict]:
    ids = [p for p in labels if p in features_by_play]
    if len(ids) < 2:
        return None
    classes = sorted({labels[p] for p in ids})
    if len(classes) < 2:
        return None
    X = np.stack([feature_vector(features_by_play[p]) for p in ids])
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xn = (X - mu) / sd
    y = np.array([classes.index(labels[p]) for p in ids])
    Y = np.eye(len(classes))[y]
    W = np.zeros((X.shape[1], len(classes)))
    bias = np.zeros(len(classes))
    for _ in range(epochs):
        P = _softmax(Xn @ W + bias)
        G = (P - Y) / len(ids)
        W -= lr * (Xn.T @ G + l2 * W)
        bias -= lr * G.sum(0)
    return {"classes": classes, "keys": FEATURE_KEYS, "mu": mu.tolist(), "sd": sd.tolist(),
            "W": W.tolist(), "b": bias.tolist(), "n": len(ids)}


def train_from_files(labels_csv: str, features_by_play: dict[str, dict], out_path: str) -> Optional[dict]:
    model = train(read_labels(labels_csv), features_by_play)
    if model:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(model))
    return model


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def load_model(path) -> Optional[dict]:
    p = Path(path)
    if not p.exists():
        return None
    mt = p.stat().st_mtime
    hit = _CACHE.get(str(p))
    if hit and hit[0] == mt:
        return hit[1]
    try:
        m = json.loads(p.read_text())
    except Exception:
        return None
    _CACHE[str(p)] = (mt, m)
    return m


def predict_proba(model: dict, features: dict[str, float]) -> dict[str, float]:
    keys = model.get("keys", FEATURE_KEYS)
    x = np.array([float(features.get(k, 0.0) or 0.0) for k in keys])
    x[~np.isfinite(x)] = 0.0
    xn = (x - np.asarray(model["mu"])) / np.asarray(model["sd"])
    p = _softmax(xn @ np.asarray(model["W"]) + np.asarray(model["b"]))
    return {c: float(v) for c, v in zip(model["classes"], p)}


def blend(rule_scores: dict[str, float], model_probs: dict[str, float],
          weight: float = BLEND_WEIGHT) -> dict[str, float]:
    return {k: (1 - weight) * v + weight * model_probs.get(k, 0.0) for k, v in rule_scores.items()}


def maybe_blend(rule_scores: dict[str, float], features: dict[str, float], cfg: dict) -> tuple[dict[str, float], bool]:
    m = load_model(model_path(cfg))
    if not m:
        return rule_scores, False
    try:
        return blend(rule_scores, predict_proba(m, features)), True
    except Exception:
        return rule_scores, False
