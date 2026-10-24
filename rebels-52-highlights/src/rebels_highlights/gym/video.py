"""Video I/O via the bundled ffmpeg (no ffprobe): discovery, orientation-safe probing,
raw-frame decode pipe (auto-rotated, constant fps) and H.264 encode pipe."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np

from ..core.media import ffmpeg_bin, probe, run_ffmpeg

VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}


def discover_videos(inputs: list[str]) -> list[Path]:
    out: list[Path] = []
    for item in inputs:
        p = Path(item).expanduser()
        if p.is_dir():
            out += sorted(f for f in p.rglob("*") if f.is_file()
                          and f.suffix.lower() in VIDEO_EXTS and not f.name.startswith("."))
        elif p.is_file():
            out.append(p)
    seen, uniq = set(), []
    for f in out:
        k = str(f.resolve())
        if k not in seen:
            seen.add(k)
            uniq.append(f)
    return uniq


def rotation_of(path: str | Path) -> int:
    r = subprocess.run([ffmpeg_bin(), "-hide_banner", "-i", str(path)],
                       capture_output=True, text=True)
    m = re.search(r"displaymatrix: rotation of (-?[\d.]+) degrees", r.stderr)
    if m:
        return int(round(float(m[1]))) % 360
    m = re.search(r"rotate\s*:\s*(-?\d+)", r.stderr)
    return int(m[1]) % 360 if m else 0


def grab_frame(path: str | Path, t: float = 0.0) -> Optional[np.ndarray]:
    """One auto-rotated BGR frame at ``t`` seconds (PNG pipe, so the shape is exact)."""
    cmd = [ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-ss", f"{max(0.0, t):.3f}",
           "-i", str(path), "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"]
    r = subprocess.run(cmd, capture_output=True)
    if not r.stdout:
        return None
    return cv2.imdecode(np.frombuffer(r.stdout, np.uint8), cv2.IMREAD_COLOR)


def probe_video(path: str | Path, max_work_fps: float = 60) -> dict:
    """probe() + rotation + displayed (rotated) width/height + chosen work fps."""
    info = probe(path)
    info["rotation"] = rotation_of(path)
    frame = grab_frame(path, min(0.2, max(0.0, info["duration"] / 2)))
    if frame is None:
        frame = grab_frame(path, 0.0)
    if frame is None:
        raise RuntimeError(f"cannot decode video: {path}")
    info["disp_h"], info["disp_w"] = frame.shape[:2]
    src_fps = info.get("fps") or 30.0
    info["work_fps"] = 60.0 if (src_fps >= 49 and max_work_fps >= 60) else 30.0
    info["n_frames"] = int(round(info["duration"] * info["work_fps"]))
    return info


class FrameReader:
    """Decode frames at a constant ``fps`` (ffmpeg auto-rotates) as BGR numpy arrays.

    ``size`` = output (w, h) after scaling; ``start_s`` seeks first (frame 0 of the
    iterator is then at ``start_s``)."""

    def __init__(self, path: str | Path, fps: float, size: tuple[int, int],
                 start_s: float = 0.0, duration_s: Optional[float] = None, gray: bool = False):
        self.w, self.h = int(size[0]), int(size[1])
        self.gray = gray
        cmd = [ffmpeg_bin(), "-hide_banner", "-loglevel", "error"]
        if start_s > 0:
            cmd += ["-ss", f"{start_s:.3f}"]
        cmd += ["-i", str(path)]
        if duration_s:
            cmd += ["-t", f"{duration_s:.3f}"]
        cmd += ["-an", "-vf", f"fps={fps},scale={self.w}:{self.h}:flags=bilinear",
                "-f", "rawvideo", "-pix_fmt", "gray" if gray else "bgr24", "-"]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                     bufsize=10 ** 7)
        self.nbytes = self.w * self.h * (1 if gray else 3)

    def __iter__(self) -> Iterator[np.ndarray]:
        assert self.proc.stdout is not None
        try:
            while True:
                buf = self.proc.stdout.read(self.nbytes)
                if len(buf) < self.nbytes:
                    break
                a = np.frombuffer(buf, np.uint8)
                yield a.reshape(self.h, self.w) if self.gray else a.reshape(self.h, self.w, 3)
        finally:
            self.close()

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.wait()


class FrameWriter:
    """Raw BGR frames -> H.264 yuv420p mp4 (video only; audio is muxed afterwards)."""

    def __init__(self, out_path: str | Path, w: int, h: int, fps: float, crf: int = 20,
                 preset: str = "veryfast"):
        self.w, self.h = w, h
        cmd = [ffmpeg_bin(), "-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo",
               "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
               "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p",
               "-r", str(fps), str(out_path)]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        self.count = 0

    def write(self, frame: np.ndarray) -> None:
        assert frame.shape[:2] == (self.h, self.w), frame.shape
        assert self.proc.stdin is not None
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        self.count += 1

    def close(self) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.close()
        err = self.proc.stderr.read().decode() if self.proc.stderr else ""
        if self.proc.wait() != 0:
            raise RuntimeError(f"ffmpeg encode failed: {err[-800:]}")


def atempo_chain(sp: float) -> list[float]:
    """atempo accepts 0.5..100 per instance; chain for slower factors (product == sp)."""
    sp = max(sp, 0.0625)
    out = []
    while sp < 0.5:
        out.append(0.5)
        sp /= 0.5
    out.append(sp)
    return out


def mux_audio(video_only: Path, src: Optional[Path], segments: list[tuple[float, float, float]],
              out_path: Path, loudnorm: str, has_audio: bool) -> None:
    """Final mp4: copy video, build audio from ``segments`` = [(src_start, src_dur, speed)]
    (speed 0 = silence for src_dur seconds of output), loudnorm, AAC, +faststart."""
    total = sum(d if sp == 0 else d / sp for _, d, sp in segments)
    parts, chains = [], []
    usable = [s for s in segments if s[2] > 0 and s[1] > 0.01]
    if has_audio and src is not None and usable:
        chains.append(f"[1:a]aresample=48000,aformat=channel_layouts=stereo,"
                      f"asplit={len(usable)}" + "".join(f"[s{i}]" for i in range(len(usable))))
    k = 0
    for i, (st, d, sp) in enumerate(segments):
        if d <= 0.01:
            continue
        if sp > 0 and has_audio and src is not None:
            f = f"[s{k}]atrim=start={st:.4f}:duration={d:.4f},asetpts=PTS-STARTPTS"
            if abs(sp - 1) > 1e-3:
                f += "".join(f",atempo={t:.4f}" for t in atempo_chain(sp))
            chains.append(f + f"[p{i}]")
            k += 1
        else:
            dur = d if sp == 0 else d / sp
            chains.append(f"anullsrc=r=48000:cl=stereo,atrim=duration={dur:.4f}[p{i}]")
        parts.append(f"[p{i}]")
    if not parts:
        chains.append(f"anullsrc=r=48000:cl=stereo,atrim=duration={max(total, 0.1):.4f}[p0]")
        parts = ["[p0]"]
    real = has_audio and src is not None and bool(usable)
    norm = f"loudnorm={loudnorm}," if real and loudnorm else ""   # loudnorm chokes on pure silence
    head = ";".join(chains)
    tail = "".join(parts) + f"concat=n={len(parts)}:v=0:a=1,{{norm}}aresample=48000,apad[aout]"
    args = ["-i", str(video_only)]
    if real:
        args += ["-i", str(src)]
    for nm in ([norm, ""] if norm else [""]):
        fc = head + ";" + tail.format(norm=nm)
        try:
            run_ffmpeg(args + ["-filter_complex", fc, "-map", "0:v", "-map", "[aout]", "-c:v", "copy",
                               "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-shortest",
                               "-movflags", "+faststart", str(out_path)])
            return
        except subprocess.CalledProcessError:
            if not nm:
                raise
