"""Game ingest (yt-dlp or local file) and low-res proxy generation.

Only publicly accessible videos are downloaded through yt-dlp's normal API;
DRM-protected or access-restricted media is never circumvented - such
failures are raised to the orchestrator, which logs them.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Optional

from ..core.media import ffmpeg_bin, probe, run_ffmpeg
from ..core.models import UNKNOWN, Game, VideoInfo

MAX_HEIGHT = 1080
# Prefer mp4/h264+m4a <= 1080p, then any mp4 <= 1080p, then best <= 1080p.
FORMAT_SELECTOR = (
    f"bv*[height<={MAX_HEIGHT}][ext=mp4]+ba[ext=m4a]/"
    f"b[height<={MAX_HEIGHT}][ext=mp4]/"
    f"bv*[height<={MAX_HEIGHT}]+ba/b[height<={MAX_HEIGHT}]/b"
)


class IngestError(RuntimeError):
    """Raised when a game video cannot be obtained."""


def _meta_path(store, game_id: str) -> Path:
    return Path(store.raw) / f"{game_id}.info.json"


def _download(game: Game, dest: Path) -> dict[str, Any]:
    """Download ``game.url`` to ``dest`` with yt-dlp; return its info dict."""
    try:
        import yt_dlp
    except ImportError as e:  # pragma: no cover
        raise IngestError("yt-dlp is not installed; supply game.local_path instead") from e

    opts = {
        "format": FORMAT_SELECTOR,
        "merge_output_format": "mp4",
        "outtmpl": str(dest.with_suffix("")) + ".%(ext)s",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "retries": 3,
        "socket_timeout": 30,
        "allow_unplayable_formats": False,  # never touch DRM formats
        "ffmpeg_location": ffmpeg_bin(),
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(game.url, download=True)
    except Exception as e:  # yt_dlp.utils.DownloadError and network errors
        msg = str(e)
        hint = ""
        if any(s in msg.lower() for s in ("403", "proxy", "tunnel", "resolve", "network",
                                          "connection", "blocked")):
            hint = (" (network access to the video host appears blocked; "
                    "download it manually and set local_path in games.yaml)")
        elif "drm" in msg.lower():
            hint = " (DRM-protected: not supported)"
        raise IngestError(f"yt-dlp failed for {game.game_id} <{game.url}>{hint}: {msg}") from e
    if info is None:
        raise IngestError(f"yt-dlp returned no info for {game.url}")
    if info.get("_type") == "playlist":
        entries = [e for e in info.get("entries") or [] if e]
        if not entries:
            raise IngestError(f"empty playlist: {game.url}")
        info = entries[0]
    if not dest.exists():  # merged/remuxed file may have another extension
        cands = sorted(dest.parent.glob(dest.stem + ".*"))
        vids = [c for c in cands if c.suffix in (".mp4", ".mkv", ".webm", ".mov")]
        if not vids:
            raise IngestError(f"download produced no video file for {game.game_id}")
        src = vids[0]
        if src.suffix != ".mp4":
            run_ffmpeg(["-i", str(src), "-c", "copy", "-movflags", "+faststart", str(dest)])
            src.unlink(missing_ok=True)
        else:
            src.rename(dest)
    keep = ("id", "title", "webpage_url", "extractor", "format", "format_id", "ext",
            "duration", "fps", "width", "height", "upload_date", "uploader")
    return {k: info.get(k) for k in keep}


def ingest_game(game: Game, cfg: dict, store) -> VideoInfo:
    """Obtain the game video (local_path or yt-dlp) and return its VideoInfo."""
    meta: dict[str, Any] = {}
    if game.local_path:
        src = Path(game.local_path).expanduser()
        if not src.is_absolute():
            src = Path(cfg.get("root", ".")) / src
        if not src.exists():
            raise IngestError(f"local_path does not exist: {src}")
        path = src
    elif game.url:
        path = Path(store.raw) / f"{game.game_id}.mp4"
        mp = _meta_path(store, game.game_id)
        if path.exists() and path.stat().st_size > 0:
            meta = json.loads(mp.read_text()) if mp.exists() else {}
        else:
            meta = _download(game, path)
            mp.write_text(json.dumps(meta, indent=1))
    else:
        raise IngestError(f"game {game.game_id} has neither url nor local_path")

    p = probe(path)
    if not p.get("width") or not p.get("duration"):
        raise IngestError(f"unreadable video (no video stream/duration): {path}")
    fps = float(p.get("fps") or meta.get("fps") or 0.0)
    return VideoInfo(
        game_id=game.game_id,
        path=str(path),
        source_url=meta.get("webpage_url") or game.url,
        source_video_id=meta.get("id"),
        title=meta.get("title") or (path.stem if game.local_path else UNKNOWN),
        duration_s=float(p["duration"]),
        fps=fps,
        width=int(p["width"]),
        height=int(p["height"]),
        format=str(meta.get("format") or p.get("vcodec") or p.get("format") or UNKNOWN),
        has_audio=bool(p.get("has_audio")),
    )


def make_proxy(info: VideoInfo, cfg: dict, store) -> str:
    """Create (or reuse) a low-res/low-fps H.264 proxy; return its path."""
    pc = cfg.get("proxy", {})
    height = int(pc.get("height", 540))
    fps = float(pc.get("fps", 10))
    out = Path(store.proxies) / f"{info.game_id}_{height}p{int(fps)}.mp4"
    if out.exists() and out.stat().st_size > 0 and \
            out.stat().st_mtime >= Path(info.path).stat().st_mtime:
        return str(out)
    tmp = out.with_suffix(".tmp.mp4")
    run_ffmpeg([
        "-i", info.path, "-an", "-sn",
        "-vf", f"scale=-2:{height}:flags=area,fps={fps:g}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
        "-g", str(max(1, int(fps * 2))), "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(tmp),
    ])
    shutil.move(str(tmp), str(out))
    return str(out)


def proxy_scale(info: VideoInfo, proxy_path: Optional[str]) -> float:
    """Source px per proxy px (multiply proxy coordinates by this)."""
    if not proxy_path or not info.height:
        return 1.0
    ph = probe(proxy_path).get("height") or info.height
    return info.height / float(ph)
