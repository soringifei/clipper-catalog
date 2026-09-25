"""CLI for the gym module.

Wire into the main CLI with::

    from .gym.cli import add_gym_parser, run_gym
    add_gym_parser(sub)            # sub = argparse subparsers
    ...
    if args.cmd == "gym": return run_gym(args)

Standalone: ``python -m rebels_highlights.gym --input clips/``.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

from .exercises import PROFILES, normalize_name


def _add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--input", "-i", action="append", default=[],
                   help="video file or folder (recursive); repeatable")
    p.add_argument("--exercise", default=None,
                   help=f"force exercise for all clips: {', '.join(PROFILES)}")
    p.add_argument("--athlete-height", type=float, default=None, help="metres (default 1.85)")
    p.add_argument("--mix", type=int, action="append", default=None, metavar="SECONDS",
                   help="also build a compilation of the best clips (repeatable, e.g. --mix 45)")
    p.add_argument("--side-by-side", nargs="+", metavar="VIDEO", default=None,
                   help="split-screen comparison: one clip (rep 1 vs last rep) or two clips")
    p.add_argument("--max-clips", type=int, default=None, help="process at most N clips")
    p.add_argument("--max-duration", type=float, default=None, help="max seconds per rendered clip")
    p.add_argument("--resume", action="store_true", help="skip clips already rendered")
    p.add_argument("--no-cache", action="store_true", help="re-run pose even if cached")
    p.add_argument("--no-slowmo", action="store_true", help="no speed ramps at the reps' peaks")
    p.add_argument("--plate-track", action="store_true", help="optical-flow bar-end tracker")
    p.add_argument("--model", default=None, help="pose weights (default yolo11n-pose.pt)")
    p.add_argument("--device", default=None, help="cpu | cuda | mps")
    p.add_argument("--gym-config", default="config/gym.yaml", help="gym yaml")
    p.add_argument("--out", default=None, help="output folder (default outputs/gym)")
    p.add_argument("--debug", action="store_true", help="also write CV debug renders")


def add_gym_parser(subparsers) -> argparse.ArgumentParser:
    p = subparsers.add_parser("gym", help="analyse gym training clips -> 9:16 breakdown videos",
                              description="Turn gym training clips into 9:16 videos with joint angles, "
                                          "bar path, velocity, ROM, reps and tempo (all estimates).")
    _add_args(p)
    return p


def build_config(args) -> dict:
    from .config import load_gym_config
    ov: dict = {}
    if args.athlete_height:
        ov.setdefault("athlete", {})["height_m"] = args.athlete_height
    if args.max_duration:
        ov.setdefault("edit", {})["max_clip_s"] = args.max_duration
    if getattr(args, "plate_track", False):
        ov.setdefault("pose", {})["plate_tracker"] = True
    if args.model:
        ov.setdefault("pose", {})["model"] = args.model
    if args.device:
        ov.setdefault("pose", {})["device"] = args.device
    if args.out:
        ov.setdefault("paths", {})["outputs"] = str(Path(args.out).resolve())
    return load_gym_config(args.gym_config, ov)


def run_gym(args) -> int:
    logging.basicConfig(level=logging.DEBUG if getattr(args, "debug", False) else logging.INFO,
                        format="%(levelname)s %(message)s")
    from .compose import build_mix, side_by_side
    from .pipeline import log_error, run_batch
    cfg = build_config(args)
    ex = None
    if args.exercise:
        ex = normalize_name(args.exercise)
        if not ex:
            print(f"unknown exercise '{args.exercise}'. valid: {', '.join(PROFILES)}", file=sys.stderr)
            return 2
    if not args.input and not args.side_by_side:
        print("nothing to do: pass --input <folder|file> and/or --side-by-side", file=sys.stderr)
        return 2
    t0 = time.time()
    results = []
    if args.input:
        results = run_batch(args.input, cfg, ex, args.max_clips, args.resume, args.debug,
                            use_cache=not args.no_cache, slowmo=not args.no_slowmo)
        for r in results:
            e = r["exercise"]
            s = r.get("summary", {})
            print(f"{Path(r['file']).name}: {e['key']} (conf {e['confidence']:.2f}, {e['source']}) "
                  f"reps={s.get('reps')} -> {r['render']['output_file']} "
                  f"[{r['render']['duration_s']:.1f}s, render {r['render'].get('render_seconds')}s]")
        if not results:
            print("no clips rendered (see reports/errors.jsonl)")
    for d in args.mix or []:
        try:
            m = build_mix(results, cfg, d, args.max_clips)
        except Exception as e:  # noqa: BLE001
            log_error(cfg, f"mix_{d}s", "mix", e)
            m = None
        if m:
            print("mix:", m["output_file"], f"{m['duration_s']:.1f}s")
        else:
            print(f"mix {d}s: not built (no rendered clips or error, see reports/errors.jsonl)")
    if args.side_by_side:
        paths = [Path(p) for p in args.side_by_side[:2]]
        try:
            r = side_by_side(paths[0], paths[1] if len(paths) > 1 else None, cfg, ex,
                             use_cache=not args.no_cache)
            print("side-by-side:", r["output_file"], f"{r['duration_s']:.1f}s")
        except Exception as e:  # noqa: BLE001
            log_error(cfg, str(paths[0]), "side_by_side", e)
            print(f"side-by-side failed: {e}", file=sys.stderr)
    print(f"done in {time.time() - t0:.1f}s")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="rebels_highlights.gym",
                                description="Gym training clips -> 9:16 movement-breakdown videos")
    _add_args(p)
    return run_gym(p.parse_args(argv))
