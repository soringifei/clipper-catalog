---
name: render-highlights
description: Render vertical 1080x1920 #52 highlight clips and 30/45/60 s mixes from scored candidates (smoothed crop, replay slow-mo, ident card, loudnorm audio, optional licensed music). Use when producing or re-rendering clips/mixes or fixing crop/audio problems.
---

# Render highlights

## Procedure
1. Preview what would be rendered: `python -m rebels_highlights render --dry-run`
2. Render: `python -m rebels_highlights render [--approved-only]` -> `outputs/candidates/`,
   `outputs/single_plays/`, thumbnails next to clips. File names come from `clip_basename()`.
3. Mixes: `python -m rebels_highlights mix --duration 30` (45, 60). Only clips with
   identity >= `mix.min_identity`, max `mix.max_same_game_consecutive` clips from one game in a row.
4. Debug a crop: re-render with `--debug` (debug overlays go to debug outputs only; final clips
   never contain CV boxes). Check `<clip>.crop.json` if written.
5. Tune in `config/default.yaml -> render`: `lead_in_s`, `lead_out_s`, `slowmo_speed`,
   `replay_zoom`, `crop_smoothing`, `max_digital_zoom`, `safe_region`, `min_clip_s`/`max_clip_s`,
   `loudnorm`.
6. Music only if `config/player.yaml: music_file` points to a file the user holds rights to;
   otherwise keep the game audio.
7. Always run the qa-highlights skill afterwards.

## Failures
- ffmpeg errors: `reports/errors.jsonl`; the bundled imageio-ffmpeg binary has libx264 + aac.
- #52 leaves the frame: lower `crop_smoothing` or `max_digital_zoom`; check the identity trajectory.
- Not enough good clips for a mix: `build_mix` returns None - report it, do not pad with weak clips.
