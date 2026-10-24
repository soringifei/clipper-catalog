"""Download only a window around each game's &t= hint (or the full game) with yt-dlp.

Faster than pulling a 2-4 h game when you only want to test around a known
moment. Writes data/raw/<game_id>[_<start>-<end>].mp4 and prints the
`local_path` to set in config/games.yaml. Sets no metadata you did not supply.

    python scripts/fetch_windows.py                  # +-5 min around every hint
    python scripts/fetch_windows.py --minutes 10 --game game_002
    python scripts/fetch_windows.py --full           # whole games
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rebels_highlights.core.config import load_config  # noqa: E402


def fetch(url: str, out: Path, start: float | None, end: float | None) -> Path:
    import yt_dlp
    from yt_dlp.utils import download_range_func

    from rebels_highlights.core.media import ffmpeg_bin
    opts = {
        "format": "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b[height<=1080]/b",
        "merge_output_format": "mp4",
        "outtmpl": str(out.with_suffix(".%(ext)s")),
        "ffmpeg_location": ffmpeg_bin(),
        "quiet": False,
        "noprogress": True,
    }
    if start is not None:
        opts["download_ranges"] = download_range_func(None, [(start, end)])
        opts["force_keyframes_at_cuts"] = True
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/games.yaml")
    ap.add_argument("--game", action="append")
    ap.add_argument("--minutes", type=float, default=5.0, help="window half-width")
    ap.add_argument("--full", action="store_true", help="download entire games")
    args = ap.parse_args()
    cfg = load_config(args.config)
    raw = ROOT / cfg["paths"]["raw"]
    raw.mkdir(parents=True, exist_ok=True)
    failed = 0
    for g in cfg["games"]:
        if args.game and g["game_id"] not in args.game:
            continue
        hint = g.get("hint_start_s")
        if args.full or hint is None:
            start = end = None
            out = raw / f"{g['game_id']}.mp4"
        else:
            start = max(0.0, hint - args.minutes * 60)
            end = hint + args.minutes * 60
            out = raw / f"{g['game_id']}_{int(start)}-{int(end)}.mp4"
        try:
            fetch(g["url"], out, start, end)
            print(f"{g['game_id']}: local_path: {out.relative_to(ROOT)}"
                  + (f"  (source offset {int(start)} s)" if start else ""))
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"{g['game_id']}: FAILED {e}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
