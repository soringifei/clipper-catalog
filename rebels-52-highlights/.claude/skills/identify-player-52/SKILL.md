---
name: identify-player-52
description: Find and verify Bucharest Rebels #52 (MLB) in each play - jersey OCR votes, team kit, re-ID, reference photos, review seeds - and tune identity thresholds with evidence. Use when identity confidence is low, clips follow the wrong player, or many candidates are JERSEY_UNREADABLE.
---

# Identify player #52

Identity = weighted fusion (`identity.weights` in `config/default.yaml`) of jersey OCR,
re-ID, team kit, temporal consistency and reference similarity, plus a weak MLB position prior
(`position_prior_weight`, never sufficient alone). A single "52" frame never proves identity
(`identity.min_ocr_votes`).

## Procedure
1. `python -m rebels_highlights analyze --game game_001 --resume --debug`
2. Inspect `cache/game_001/` identity output: per-track `jersey_confidence`, `team_confidence`,
   `reid_confidence`, `ocr_votes`, `review_reasons`, `best_track_id`.
3. Watch the debug renders (`--debug`; boxes/track ids only there, never in final clips) to check
   the chosen track really is #52 and does not switch (TRACK_ID_SWITCH).
4. Improve evidence before touching thresholds:
   - add reference photos to `reference_player_52/` (see its README);
   - set `team_kit_notes` / `team_kit_hsv` in `config/player.yaml` after checking real frames;
   - in `review/index.html` use **Seed #52** on a few plays, export `seeds.json` into `review/`,
     re-run `analyze --resume` (seeded evidence has `source: manual_seed`).
5. Only then consider `identity.auto_accept` / `review_min`: use the identity distribution and
   tuning notes in `reports/final_report.md`, and confirm on reviewed clips. Report evidence to the user.

## Failures
- No person detections: check `detection.conf`, `imgsz`, `fine_fps`; ultralytics missing -> fallback detector is weaker.
- Wrong team: kit colours; check `team_prob` of tracks.
- Never relabel a play to #52 without evidence; mark it REVIEW instead.
