"""Optional bar-end / plate tracker (config pose.plate_tracker: true).

opencv-python-headless here ships no CSRT/KCF (only MIL/Nano/Vit), so we use pyramidal
Lucas-Kanade optical flow on corner features inside an ROI seeded at the hands (in a
side view the plate/bar end sits at the hands). Forward-backward checked, re-seeded when
features are lost, and re-anchored to the wrists when it drifts > 15% of stature."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from . import metrics as M
from .pose import PoseSeq
from .video import FrameReader


def track_plate(path: Path, info: dict, seq: PoseSeq, cfg: dict, long_side: int = 640) -> np.ndarray:
    W, H = info["disp_w"], info["disp_h"]
    s = min(1.0, long_side / max(W, H))
    aw, ah = int(round(W * s / 2) * 2), int(round(H * s / 2) * 2)
    s = aw / W
    wr = M.bar_proxy(seq) * s
    body = (M.body_scale_px(seq) or H * 0.6) * s
    roi_r = max(8, int(0.07 * body))
    out = np.full((seq.T, 2), np.nan)
    prev, pts, p = None, None, None
    lk = dict(winSize=(21, 21), maxLevel=3,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))

    def seed(img, c):
        x, y = int(c[0]), int(c[1])
        m = np.zeros_like(img)
        cv2.circle(m, (x, y), roi_r, 255, -1)
        f = cv2.goodFeaturesToTrack(img, 40, 0.01, 3, mask=m)
        return f

    for i, g in enumerate(FrameReader(path, seq.fps, (aw, ah), gray=True)):
        if i >= seq.T:
            break
        anchor = wr[i]
        if p is None:
            if not np.isnan(anchor[0]):
                p = anchor.copy()
                pts = seed(g, p)
        elif prev is not None and pts is not None and len(pts) >= 3:
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, g, pts, None, **lk)
            back, st2, _ = cv2.calcOpticalFlowPyrLK(g, prev, nxt, None, **lk)
            fb = np.linalg.norm((pts - back).reshape(-1, 2), axis=1)
            good = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < 1.0)
            if good.sum() >= 3:
                d = np.median((nxt - pts).reshape(-1, 2)[good], axis=0)
                p = p + d
                pts = nxt[good].reshape(-1, 1, 2)
            else:
                pts = None
        if p is not None:
            if not np.isnan(anchor[0]) and np.linalg.norm(p - anchor) > 0.15 * body:
                p = anchor.copy()
                pts = None
            if pts is None or len(pts) < 6:
                pts = seed(g, p)
            out[i] = p / s
        prev = g
    return out
