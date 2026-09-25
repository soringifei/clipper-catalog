"""CLI: python -m rebels_highlights <command> [options]."""
from __future__ import annotations

import argparse
import json
import logging
import sys

from .core.config import load_config


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rebels_highlights",
                                description="#52 Bucharest Rebels highlight pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("env", "ingest", "analyze", "review", "render", "mix", "run-all", "status",
                 "export"):
        s = sub.add_parser(name)
        s.add_argument("--config", default="config/games.yaml", help="games yaml")
        s.add_argument("--player", type=int, default=None, help="jersey number (default 52)")
        s.add_argument("--game", action="append", help="limit to game id (repeatable)")
        s.add_argument("--device", default=None, help="auto|cpu|cuda|mps")
        s.add_argument("--workers", type=int, default=None)
        s.add_argument("--resume", action="store_true", default=True,
                       help="reuse cached stages (default)")
        s.add_argument("--no-resume", dest="resume", action="store_false")
        s.add_argument("--debug", action="store_true", help="also write CV debug renders")
        s.add_argument("--dry-run", action="store_true")
        s.add_argument("--approved-only", action="store_true")
        s.add_argument("--duration", type=int, action="append",
                       help="mix duration in seconds (repeatable)")
        s.add_argument("--export-dir", default=None,
                       help="copy final videos here (default ~/Desktop/Rebels52_highlights)")
    from .gym.cli import add_gym_parser
    add_gym_parser(sub)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.cmd == "gym":
        from .gym.cli import run_gym
        return run_gym(args)
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                        format="%(levelname)s %(message)s")
    overrides: dict = {}
    if args.device:
        overrides["device"] = args.device
    if args.workers:
        overrides["workers"] = args.workers
    if args.player:
        overrides["player"] = {"player_number": args.player}
    cfg = load_config(args.config, overrides)

    if args.cmd == "env":
        from .core.device import detect_environment
        print(json.dumps(detect_environment(cfg["root"]), indent=1))
        return 0

    from .pipeline import Pipeline, select_games
    pipe = Pipeline(cfg, resume=args.resume, debug=args.debug, dry_run=args.dry_run)
    games = select_games(cfg, args.game)

    if args.cmd == "export":
        dest = export_outputs(pipe.store, args.export_dir)
        print("exported to", dest)
        return 0
    if args.cmd == "status":
        for g in games:
            print(g.game_id, json.dumps(pipe.store.status(g.game_id)))
        return 0
    if args.cmd in ("ingest", "analyze", "run-all"):
        for g in games:
            if args.cmd == "ingest":
                info = pipe.ingest(g)
                print(g.game_id, "ok" if info else "FAILED (see reports/errors.jsonl)")
            else:
                cands = pipe.analyze(g)
                print(g.game_id, f"{len(cands)} candidates")
    if args.cmd in ("analyze", "review"):
        pipe.publish(games)
        print("review page:", pipe.store.review / "index.html")
    if args.cmd in ("render", "run-all"):
        res = pipe.render(games, approved_only=args.approved_only)
        print(f"rendered {len(res)} clips, QA passed {sum(r['passed'] for r in res)}")
    if args.cmd in ("mix", "run-all"):
        for m in pipe.mix(games, args.duration):
            print("mix:", m["output_file"], f"{m['duration_s']:.1f}s")
    if args.cmd == "run-all":
        from .manifests.manifest import ranked_table
        print(ranked_table(pipe.all_candidates(games)))
        if args.export_dir:
            print("exported to", export_outputs(pipe.store, args.export_dir))
    return 0


def _desktop():
    """The user's Desktop, incl. Windows OneDrive-redirected Desktops."""
    import os
    from pathlib import Path
    home = Path.home()
    for c in (Path(os.environ["OneDrive"]) / "Desktop" if os.environ.get("OneDrive") else None,
              home / "Desktop", home / "OneDrive" / "Desktop"):
        if c is not None and c.is_dir():
            return c
    return home / "Desktop"


def export_outputs(store, dest: str | None = None) -> str:
    """Copy finished videos, thumbnails and manifests to a local folder.

    Default is the Desktop so results land where the user sees them; nothing
    is uploaded anywhere.
    """
    import shutil
    from pathlib import Path
    out = Path(dest).expanduser() if dest else _desktop() / "Rebels52_highlights"
    for sub in ("approved", "single_plays", "candidates", "mixes", "manifests", "gym"):
        src = store.outputs / sub
        if not src.exists():
            continue
        for f in src.rglob("*"):
            if f.is_file() and f.suffix.lower() in (".mp4", ".jpg", ".json", ".csv"):
                if f.name.endswith((".crop.json", ".timeline.json")) or f.stem.endswith("_seed"):
                    continue
                d = out / sub / f.relative_to(src)
                d.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, d)
    review = store.review / "index.html"
    if review.exists():
        # re-point the page's relative links (review/ -> ../outputs/) at the export layout
        html = review.read_text().replace("../outputs/", "")
        (out / "review.html").write_text(html)
    return str(out)


if __name__ == "__main__":
    sys.exit(main())
