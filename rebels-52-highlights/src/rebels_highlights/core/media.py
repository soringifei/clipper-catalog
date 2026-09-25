"""ffmpeg/ffprobe discovery and thin wrappers."""
from __future__ import annotations

import json
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Optional


@lru_cache(maxsize=1)
def ffmpeg_bin() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("ffmpeg not found: install ffmpeg or imageio-ffmpeg") from e


@lru_cache(maxsize=1)
def ffprobe_bin() -> Optional[str]:
    return shutil.which("ffprobe")


def run_ffmpeg(args: list[str], quiet: bool = True) -> None:
    cmd = [ffmpeg_bin(), "-y", "-hide_banner"]
    if quiet:
        cmd += ["-loglevel", "error"]
    subprocess.run(cmd + args, check=True)


def probe(path: str | Path) -> dict:
    """Return {duration, fps, width, height, vcodec, acodec, has_audio, format}.

    Uses ffprobe when present, else parses ``ffmpeg -i`` (bundled builds ship
    without ffprobe), else OpenCV.
    """
    path = str(path)
    fp = ffprobe_bin()
    if fp:
        out = subprocess.run([fp, "-v", "error", "-show_streams", "-show_format",
                              "-of", "json", path], capture_output=True, text=True, check=True)
        j = json.loads(out.stdout)
        v = next((s for s in j["streams"] if s["codec_type"] == "video"), {})
        a = next((s for s in j["streams"] if s["codec_type"] == "audio"), None)
        num, _, den = (v.get("avg_frame_rate") or "0/1").partition("/")
        fps = float(num) / float(den or 1) if float(den or 1) else 0.0
        return {"duration": float(j["format"].get("duration", 0)), "fps": fps,
                "width": int(v.get("width", 0)), "height": int(v.get("height", 0)),
                "vcodec": v.get("codec_name"), "acodec": a.get("codec_name") if a else None,
                "has_audio": a is not None, "format": j["format"].get("format_name", "")}
    return _probe_via_ffmpeg(path)


def _probe_via_ffmpeg(path: str) -> dict:
    import re
    r = subprocess.run([ffmpeg_bin(), "-hide_banner", "-i", path], capture_output=True, text=True)
    s = r.stderr
    info = {"duration": 0.0, "fps": 0.0, "width": 0, "height": 0, "vcodec": None,
            "acodec": None, "has_audio": False, "format": ""}
    m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", s)
    if m:
        info["duration"] = int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3])
    m = re.search(r"Input #0, ([^,]+(?:,[^,]+)*?), from", s)
    if m:
        info["format"] = m[1]
    m = re.search(r"Stream #\S+.*?Video: (\w+).*?, (\d{2,5})x(\d{2,5})", s)
    if m:
        info["vcodec"], info["width"], info["height"] = m[1], int(m[2]), int(m[3])
    m = re.search(r"([\d.]+) fps", s)
    if m:
        info["fps"] = float(m[1])
    m = re.search(r"Stream #\S+.*?Audio: (\w+)", s)
    if m:
        info["acodec"], info["has_audio"] = m[1], True
    return info
