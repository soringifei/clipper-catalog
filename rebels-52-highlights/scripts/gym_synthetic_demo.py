"""Render gym breakdown videos from synthetic keypoint sequences (no model needed).

    PYTHONPATH=src python3 scripts/gym_synthetic_demo.py --exercise squat --out outputs/gym_demo

Draws a mannequin video for the exercise, injects the exact keypoints through the fake
pose provider and runs the normal analysis + render. Useful to check the look/metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rebels_highlights.gym import synth
from rebels_highlights.gym.config import load_gym_config
from rebels_highlights.gym.pipeline import process_clip
from rebels_highlights.gym.pose import FakePoseProvider


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exercise", default="squat", choices=sorted(synth.GENERATORS))
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--size", default="1080x1920", help="WxH of the fake source video")
    ap.add_argument("--out", default="outputs/gym_demo")
    a = ap.parse_args()
    W, H = (int(v) for v in a.size.split("x"))
    fps = 30
    T = int(a.seconds * fps)
    kw = {"squat": dict(H=H * 0.6, x0=W * 0.5, ground=H * 0.93),
          "deadlift": dict(H=H * 0.6, x0=W * 0.5, ground=H * 0.93),
          "jump": dict(H=H * 0.55, x0=W * 0.5, ground=H * 0.93),
          "bench": dict(H=min(W, H) * 0.7, x0=W * 0.3, bench_y=H * 0.6)}.get(a.exercise, {})
    kp = synth.GENERATORS[a.exercise](T, fps, **kw)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    vid = synth.render_stick_video(out / f"synthetic_{a.exercise}.mp4", kp, W, H, fps)
    cfg = load_gym_config(overrides={"paths": {"outputs": str(out.resolve())}})
    r = process_clip(Path(vid), cfg, provider=FakePoseProvider(kp))
    print(json.dumps({"output": r["render"]["output_file"], "exercise": r["exercise"]["key"],
                      "summary": r["summary"]}, indent=1))


if __name__ == "__main__":
    main()
