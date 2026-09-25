---
name: ingest-games
description: Download or register the configured full-game videos (YouTube or local files), build proxies and verify they are probe-able. Use when adding games, when ingest fails, or before a first analyze run.
---

# Ingest games

1. Check the environment: `python -m rebels_highlights env` (ffmpeg, yt-dlp, GPU). No ffprobe is fine.
2. Inspect `config/games.yaml`. Each game needs `game_id` and `url` or `local_path`.
   `hint_start_s` is a moment of interest from the shared link, not a label. Leave
   `date`/`opponent` as `unknown` unless the user gives them - never guess.
3. Run `python -m rebels_highlights ingest --resume` (or `--game game_001`).
4. Verify: `python -m rebels_highlights status`; cache `cache/<game_id>/` has the video info,
   `data/raw/` has the source, `data/proxies/` the proxy. Probe with
   `PYTHONPATH=src python3 -c "from rebels_highlights.core.media import probe; print(probe('data/raw/<file>'))"`.

## Failures
- Read `reports/errors.jsonl` (stage `ingest`).
- YouTube blocked / 403 / sign-in required (common in sandboxes): ask the user to download the
  video elsewhere and set `local_path`; do not retry in a loop, do not use unofficial mirrors.
- Corrupt/partial download: delete only that game's file in `data/raw/` and its ingest cache
  entry after asking, then re-run with `--game`.
- No real footage available: use `python scripts/make_synthetic_game.py --out data/raw/synthetic.mp4`
  and a `local_path` game entry for pipeline testing (it has ground truth in `synthetic.truth.json`).
