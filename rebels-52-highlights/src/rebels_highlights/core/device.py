"""Hardware detection -> device + model size choice."""
from __future__ import annotations

import os
import platform
import shutil


def detect_environment(root: str = ".") -> dict:
    env = {"os": platform.platform(), "python": platform.python_version(),
           "cpu_count": os.cpu_count(), "cuda": False, "mps": False, "gpu_name": None,
           "ram_gb": None, "free_disk_gb": round(shutil.disk_usage(root).free / 1e9, 1),
           "packages": {}}
    try:
        with open("/proc/meminfo") as fh:
            env["ram_gb"] = round(int(fh.readline().split()[1]) / 1e6, 1)
    except OSError:
        pass
    for pkg in ("numpy", "cv2", "torch", "ultralytics", "scenedetect", "paddleocr",
                "mediapipe", "yt_dlp", "imageio_ffmpeg"):
        try:
            mod = __import__(pkg)
            env["packages"][pkg] = getattr(mod, "__version__", "installed")
        except Exception:  # noqa: BLE001
            env["packages"][pkg] = None
    try:
        import torch
        env["cuda"] = torch.cuda.is_available()
        env["mps"] = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        if env["cuda"]:
            env["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001
        pass
    from .media import ffmpeg_bin, ffprobe_bin
    try:
        env["ffmpeg"] = ffmpeg_bin()
    except RuntimeError:
        env["ffmpeg"] = None
    env["ffprobe"] = ffprobe_bin()
    return env


def pick_device(requested: str, env: dict) -> str:
    if requested != "auto":
        return requested
    return "cuda" if env.get("cuda") else "mps" if env.get("mps") else "cpu"


def pick_detector(device: str, requested: str) -> str:
    """Upgrade the default nano model on GPU; keep nano on CPU."""
    if requested != "yolo11n.pt":
        return requested
    return {"cuda": "yolo11m.pt", "mps": "yolo11s.pt"}.get(device, "yolo11n.pt")
