#!/usr/bin/env python3
"""Write reports/environment.json (hardware, packages, ffmpeg) via core.device."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rebels_highlights.core.device import detect_environment, pick_detector, pick_device  # noqa: E402


def main() -> int:
    env = detect_environment(str(ROOT))
    env["recommended_device"] = pick_device("auto", env)
    env["recommended_detector"] = pick_detector(env["recommended_device"], "yolo11n.pt")
    out = ROOT / "reports" / "environment.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(env, indent=1))
    print(json.dumps(env, indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
