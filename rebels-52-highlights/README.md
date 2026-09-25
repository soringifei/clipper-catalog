# rebels-52-highlights

## Pe scurt (RO)

Pipeline Python care găsește fazele jucătorului **#52 (Middle Linebacker, Bucharest Rebels)**
în meciuri întregi de pe YouTube și produce clipuri verticale 9:16 (1080x1920) pentru
Reels/TikTok/Shorts, plus compilații de 30/45/60 s.

1. `pip install -e .` (sau `pip install -r requirements.txt`)
2. Pune linkurile în `config/games.yaml` (sau `local_path` dacă YouTube e blocat).
3. `python -m rebels_highlights run-all --resume`
4. Deschide `review/index.html`, aprobă/respinge clipuri, apasă **Export decisions** și salvează
   fișierele exportate în `review/`. Apoi `python -m rebels_highlights review`.

Etichetele sunt conservatoare: dacă nu e sigur, clipul merge la REVIEW. Nimic nu este inventat.

---

## What it does

For each configured full-game video the pipeline:

1. **ingest** - downloads with yt-dlp (or uses `local_path`), builds a low-res proxy.
2. **analyze** - scene cuts (live / replay / non-play), play segmentation from motion
   (pre-snap stillness -> synchronized snap motion -> settle), replay association, person
   detection + tracking per play, #52 identification (jersey OCR votes, team kit, re-ID,
   temporal consistency, reference photos, weak MLB position prior), event classification
   (tackle, TFL, sack, pressure, forced fumble, ...) and scoring into reviewable candidates.
3. **render** - vertical 1080x1920 H.264/AAC clips that follow #52 with a smoothed crop,
   short ident card / arrow cue, slow-motion replay of the key moment, loudness-normalised
   audio, and a thumbnail. Computer-vision boxes are never burned into final clips.
4. **mix** - 30/45/60 s compilations of the best approved clips.
5. **review** - static HTML review page + manifests + QA + final report.

## Setup

```bash
python3 -m pip install -e .            # or: pip install -r requirements.txt
# optional extras
pip install -e '.[gpu]'                # ultralytics, torch, lap  (YOLO + BoT-SORT)
pip install -e '.[ocr]'                # paddleocr, paddlepaddle  (jersey OCR)
pip install -e '.[pose]'               # mediapipe
pip install -e '.[sam]'                # sam2
pip install -e '.[dev]'                # pytest
```

ffmpeg: a system `ffmpeg` is used if on PATH, otherwise the binary bundled with
`imageio-ffmpeg`. `ffprobe` is optional (probing falls back to parsing `ffmpeg -i`).
Every heavy dependency is optional; without it a CPU/OpenCV fallback is used (slower / less
accurate). Check what is available with `python -m rebels_highlights env`
(or `python scripts/environment_report.py` -> `reports/environment.json`).

## Configuration

| File | Content |
|---|---|
| `config/games.yaml` | `game_id`, `url`, `hint_start_s` (the `&t=` of the shared link - a moment of interest, not a label), `date`, `opponent` (stay `unknown` until you supply them), `local_path` |
| `config/player.yaml` | name, team, number 52, position, reference photo dir, kit notes, optional licensed `music_file` |
| `config/default.yaml` | paths, device, all thresholds (see tuning table) |
| `reference_player_52/` | optional reference photos of #52 (see its README) |

## CLI

```bash
python -m rebels_highlights ingest    [--game game_001] [--resume]
python -m rebels_highlights analyze   [--game game_001] [--resume] [--device cpu|cuda|mps|auto] [--workers 2] [--debug]
python -m rebels_highlights review                       # build review/index.html, apply review/review_decisions.json
python -m rebels_highlights render    [--approved-only] [--dry-run] [--debug]
python -m rebels_highlights mix       [--duration 30|45|60] [--approved-only]
python -m rebels_highlights run-all   [--resume] [--game ...] [--device ...] [--workers N]
python -m rebels_highlights status                       # per-game stage status
python -m rebels_highlights env                          # hardware / package report
```

Common flags: `--config` (alternative games YAML), `--player` (alternative player YAML),
`--resume` (skip cached stages), `--game` (restrict to one or more game ids), `--device`,
`--workers`, `--debug` (debug renders with tracks/boxes into `outputs/debug/`, never into
final clips), `--dry-run` (plan only, write nothing heavy), `--approved-only`
(only AUTO_APPROVED or human-approved clips), `--duration` (mix length in seconds).
Also installed as the console script `rebels-highlights`.

Try the whole pipeline without YouTube on a synthetic game with ground truth:

```bash
python scripts/make_synthetic_game.py --out data/raw/synthetic.mp4 --seconds 60
# -> data/raw/synthetic.mp4 + data/raw/synthetic.truth.json (snaps, impacts, #52 boxes)
```

## Stages and cache layout

```
data/raw/<game_id>.*             downloaded source video
data/proxies/<game_id>*.mp4      low-res / low-fps proxy for scene + motion analysis
cache/<game_id>/<stage>.json     one JSON per stage (resumable; delete a file to recompute it)
state/<game_id>.json             stage status (done / failed + error)
reports/errors.jsonl             every stage failure (one stage never kills the run)
```

## Outputs and naming

```
outputs/candidates/        every rendered candidate clip (+ .jpg thumbnail)
outputs/approved/          clips auto-approved or approved in the review page
outputs/single_plays/      final single-play clips
outputs/mixes/             30/45/60 s compilations
outputs/thumbnails/
outputs/manifests/highlights.json, highlights.csv   all candidates, sorted by status then score
review/index.html          static review page (works from file://)
reports/qa.json            automated QA per clip
reports/final_report.md    summary, distributions, failure modes, ranked table
```

Clip file names come from `rendering.render.clip_basename()` and contain the game id, play id,
play type and #52 (e.g. `game_001_p014_solo_tackle_52`). Mixes carry their duration and a
version number. The manifest records source URL, source timestamps (start / snap / impact /
end), all confidences, review status/reasons and OCR votes for every clip.

## Review workflow

1. Open `review/index.html` in a browser (no server needed). Tabs: Auto-approved / Needs review
   / Rejected, with counts.
2. Each card shows the clip, source timestamp (links to the YouTube moment), play type and
   possible types, identity / event / highlight bars and review-reason chips.
3. **Approve** / **Reject** (+ note) are stored in the browser's localStorage.
4. **Seed #52**: opens the seed frame (or the thumbnail - then marked approximate); click on
   #52 to record a box for that play.
5. **Export decisions** downloads `review_decisions.json` (and `seeds.json` when seeds exist).
   Save both into `review/`.
6. `python -m rebels_highlights review` copies approved clips into `outputs/approved/`;
   `python -m rebels_highlights analyze --resume` re-identifies seeded plays using `review/seeds.json`.

## Tuning (config/default.yaml)

| Key | Default | Effect |
|---|---|---|
| `identity.auto_accept` | 0.80 | identity confidence needed for AUTO_APPROVED |
| `identity.review_min` | 0.55 | below this the candidate is REJECTED |
| `identity.min_ocr_votes` | 3 | frames reading "52" needed; one frame never proves identity |
| `identity.weights.*` | 0.35/0.25/0.15/0.15/0.10 | jersey OCR / re-ID / team kit / temporal / reference |
| `identity.position_prior_weight` | 0.05 | weak MLB alignment prior, never sufficient alone |
| `events.auto_label_min` | 0.70 | event confidence to commit a play type |
| `events.review_below` | 0.65 | below this: EVENT_AMBIGUOUS, needs review |
| `events.forced_fumble_auto` / `sack_auto` / `tfl_auto` | 0.85 / 0.80 / 0.80 | stricter bars for rare labels |
| `events.los_min_confidence` | 0.6 | LOS confidence needed for TFL / run-stop labels |
| `events.contact_iou`, `contact_dist_ratio` | 0.05, 0.6 | contact detection between #52 and ball carrier |
| `plays.snap_motion_jump` | 2.5 | motion jump (x pre-snap baseline) that marks a snap |
| `scenes.threshold` | 3.0 | adaptive cut detector sensitivity |
| `detection.fine_fps`, `conf`, `imgsz` | 15, 0.25, 960 | detection sampling and size |
| `render.safe_region` | 0.6 | #52 kept inside the central 60 % of the crop |
| `render.crop_smoothing`, `max_digital_zoom` | 0.85, 2.2 | virtual camera smoothness / zoom limit |
| `render.min_clip_s`, `max_clip_s` | 15, 60 | clip length bounds (also QA) |
| `mix.min_identity` | 0.80 | identity needed for a clip to enter a mix |

Move thresholds only with evidence: check `reports/final_report.md` (identity distribution,
review reasons) and the review page first.

## Known limitations

- Jersey OCR on real broadcast footage is hard (small, blurred, turned numbers). Expect many
  JERSEY_UNREADABLE candidates; reference photos and review seeds help most.
- Event labels (tackle, TFL, sack, pressure, forced fumble, ...) are heuristics from tracks,
  contact and LOS estimates. Uncertain ones stay `unclear` / in `possible_play_types` and go to review.
- Unit detection (defense vs special teams) is approximate.
- YouTube downloads can be blocked in sandboxes / CI. Download the videos elsewhere and set
  `local_path` in `config/games.yaml`.
- CPU-only runs are slow on full games; `--resume` makes restarts cheap.

## Ethics

- Never fabricate labels, stats, dates, opponents or player names: unknown stays `unknown`.
- Use only footage you are allowed to use; credit the source video (URL is kept in the manifest).
- Background music only from a local file you hold the rights to (`player.yaml: music_file`).
  Otherwise clips keep the original game audio.

## Tests

```bash
PYTHONPATH=src python3 -m pytest -q
```
