# rebels-52-highlights - rules for Claude

Finds and edits highlights of Bucharest Rebels #52 (MLB) from full-game videos. The user
writes Romanian; answer in Romanian unless asked otherwise, code/docs stay English.

## Non-negotiable
- Never fabricate labels, metadata (date, opponent, name), stats or confidences. Unknown -> `unknown`.
- Labels are conservative: uncertain -> lower confidence + `review_reasons` + REVIEW, or `unclear`.
  Rare labels (forced fumble, sack, TFL) need their stricter thresholds in `config/default.yaml`.
- One frame reading "52" never proves identity (`identity.min_ocr_votes`).
- Never burn CV boxes/track ids into final clips; debug overlays only via `--debug` into debug outputs.
- Music only from a licensed local file (`config/player.yaml: music_file`).
- Cache-first: stages write `cache/<game_id>/<stage>.json`; use `--resume`; never delete
  caches/outputs without asking. Failures go to `reports/errors.jsonl`, one stage never kills the run.
- Heavy deps (ultralytics, torch, paddleocr, mediapipe, sam2) are imported lazily with CPU fallback.

## Where things live
- `CONTRACTS.md` - stage function signatures; `src/rebels_highlights/core/` - models, config, store, media, device
- Stages: `ingest/ scenes/ plays/ field/ detection/ tracking/ identity/ jersey_ocr/ reid/ pose/
  events/ scoring/ rendering/ audio/ manifests/ review/ qa/`; orchestration in `pipeline.py`, CLI in `__main__.py`
- Config: `config/default.yaml` (thresholds), `config/games.yaml`, `config/player.yaml`
- Outputs: `outputs/`, `review/index.html`, `reports/` (qa.json, final_report.md, errors.jsonl)
- Synthetic test game: `python scripts/make_synthetic_game.py --out data/raw/synthetic.mp4`

## Commands
- Tests: `PYTHONPATH=src python3 -m pytest -q`
- CLI: `python -m rebels_highlights ingest|analyze|review|render|mix|run-all|status|env`
- No ffprobe here: use `core.media.probe` (parses `ffmpeg -i`).

## Skills (.claude/skills/)
ingest-games, identify-player-52, detect-football-plays, render-highlights, qa-highlights
