"""--debug render: raw detections (all people, track ids), the chosen athlete, raw vs
smoothed keypoints and the primary signal. Debug output only, never in final clips."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..core.media import run_ffmpeg
from .analysis import Analysis
from .pose import SKELETON, PoseSeq, RawPose, select_main_track
from .video import FrameReader, FrameWriter


def render_debug(path: Path, info: dict, raw: RawPose, seq: PoseSeq, an: Analysis, out: Path,
                 long_side: int = 720) -> str:
    W, H = info["disp_w"], info["disp_h"]
    s = min(1.0, long_side / max(W, H))
    aw, ah = int(round(W * s / 2) * 2), int(round(H * s / 2) * 2)
    s = aw / W
    pick = select_main_track(raw)
    by_f: dict[int, list[int]] = {}
    for j, f in enumerate(raw.frames):
        by_f.setdefault(int(f), []).append(j)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.mp4")
    wr = FrameWriter(tmp, aw, ah, seq.fps, 26, "ultrafast")
    try:
        for i, img in enumerate(FrameReader(path, seq.fps, (aw, ah))):
            if i >= seq.T:
                break
            img = img.copy()
            for j in by_f.get(i, []):
                b = (raw.boxes[j] * s).astype(int)
                main = pick[i] == j
                cv2.rectangle(img, tuple(b[:2]), tuple(b[2:]), (0, 0, 255) if main else (160, 160, 160), 2)
                cv2.putText(img, f"id{raw.ids[j]} {raw.scores[j]:.2f}", (b[0], b[1] - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                for k in range(17):
                    x, y, c = raw.kps[j, k]
                    cv2.circle(img, (int(x * s), int(y * s)), 2, (0, int(255 * c), int(255 * (1 - c))), -1)
            kp = seq.kp[i] * s
            for a, b in SKELETON:
                if not (np.isnan(kp[a]).any() or np.isnan(kp[b]).any()):
                    cv2.line(img, tuple(kp[a].astype(int)), tuple(kp[b].astype(int)), (255, 255, 0), 1)
            v = an.primary[i]
            n, _ = an.rep_at(i)
            cv2.putText(img, f"f{i} {an.exercise} {an.confidence:.2f} {an.primary_name}={v:.0f} rep{n}",
                        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            wr.write(img)
    finally:
        wr.close()
    run_ffmpeg(["-i", str(tmp), "-c", "copy", "-movflags", "+faststart", str(out)])
    tmp.unlink(missing_ok=True)
    return str(out)
