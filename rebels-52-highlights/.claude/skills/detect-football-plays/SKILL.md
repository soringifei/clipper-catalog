---
name: detect-football-plays
description: Segment full games into plays (scene cuts, snap detection, replays) and classify #52's event (tackle, TFL, sack, pressure, forced fumble, ...) conservatively. Use when plays are missed/merged, replays are duplicated, or event labels look wrong.
---

# Detect football plays and events

## Procedure
1. `python -m rebels_highlights analyze --game game_001 --resume --debug`
2. Scenes (`cache/<game>/` scenes output): hard cuts with kind `live | replay | non_play`.
   Too many/few cuts -> `scenes.threshold`, `scenes.min_scene_len_s`.
3. Plays: pre-snap stillness -> synchronized motion jump at the snap -> settle.
   Missed snaps -> lower `plays.snap_motion_jump`; merged plays -> check `plays.max_play_s`,
   `settle_motion_ratio`. Replays must get `canonical_play_id` (one candidate per real play).
4. Validate on the synthetic game: `python scripts/make_synthetic_game.py --out data/raw/synthetic.mp4`,
   compare detected snaps/impacts/replay with `data/raw/synthetic.truth.json`.
5. Events: check `play_type`, `possible_play_types`, `event_confidence`, `features`,
   `los_confidence`, `review_reasons`. Labels below `events.auto_label_min` must stay uncertain
   (`unclear` + possible types). Forced fumble / sack / TFL need `forced_fumble_auto`, `sack_auto`,
   `tfl_auto`; TFL/run-stop need `los_min_confidence`.
6. Hint: `hint_start_s` in `config/games.yaml` marks a moment of interest for review priority, not a label.

## Failures
- Stage errors: `reports/errors.jsonl`. A failed play is skipped, not fatal.
- Never upgrade a label (e.g. tackle -> forced fumble) without visual evidence; send it to REVIEW.
