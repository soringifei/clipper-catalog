"""Make upload copies under 9.5 MB (browser upload tooling caps a single file at 10 MB).
Two-pass libx264 at the bitrate that fits; same 1080x1920 frame, captions and -14 LUFS audio (audio copied as AAC 128k).
Usage: python upload_copy.py 2026-09-27 c01 c02 ...   -> daily/<date>/upload/cXX.mp4"""
import json, os, subprocess, sys, tempfile
from pathlib import Path

NO_WINDOW, BELOW_NORMAL = 0x08000000, 0x00004000
LIMIT = 9.4 * 1024 * 1024
day = Path(__file__).resolve().parent / sys.argv[1]
out_dir = day / "upload"
out_dir.mkdir(exist_ok=True)
flags = (NO_WINDOW | BELOW_NORMAL) if os.name == "nt" else 0
for cid in sys.argv[2:]:
    src = day / f"{cid}.mp4"
    dur = float(json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(src)],
                                          capture_output=True, text=True, creationflags=flags).stdout)["format"]["duration"])
    vk = int(LIMIT * 8 / dur / 1000 - 128 - 40)  # kbit/s for video, leaving room for audio + container
    out = out_dir / f"{cid}.mp4"
    with tempfile.TemporaryDirectory() as t:
        log = str(Path(t) / "pass")
        base = ["ffmpeg", "-v", "error", "-y", "-i", str(src), "-c:v", "libx264", "-preset", "slow", "-b:v", f"{vk}k",
                "-maxrate", f"{int(vk * 1.6)}k", "-bufsize", f"{vk * 2}k", "-pix_fmt", "yuv420p", "-profile:v", "high",
                "-threads", "4", "-passlogfile", log]
        subprocess.run(base + ["-pass", "1", "-an", "-f", "mp4", os.devnull], check=True, creationflags=flags)
        subprocess.run(base + ["-pass", "2", "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(out)],
                       check=True, creationflags=flags)
    print(cid, round(dur, 1), f"{vk} kbps", round(out.stat().st_size / 1048576, 2), "MB")
