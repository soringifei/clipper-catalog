---
name: gym-analysis
description: Turn #52's gym training phone clips (squat, deadlift, bench, OHP, cleans/snatches, jumps, sprints, agility, lunges, push-ups) into cinematic 9:16 breakdown videos with joint-angle arcs, bar path, bar speed, ROM, reps and tempo (all estimates). Use when asked to analyse/edit gym or training videos, compare reps or sessions, or tune the gym renders.
---

# Gym analysis (src/rebels_highlights/gym/)

## Run
```bash
cd rebels-52-highlights
PYTHONPATH=src python3 -m rebels_highlights.gym --input <folder_or_file> [--exercise squat] \
    [--athlete-height 1.85] [--mix 45] [--side-by-side a.mp4 [b.mp4]] [--max-clips N] [--resume] [--debug]
# once wired into the main CLI: python -m rebels_highlights gym ...
```
Outputs (local only, nothing is uploaded):
- `outputs/gym/<stem>/<stem>_gym.mp4` (1080x1920 H.264 yuv420p +faststart, AAC 48k, original audio loudnorm'd,
  speed-ramped with the picture), `<stem>_gym.jpg` thumbnail, `analysis.json` (angles per frame, reps, jumps, bar path,
  summary, timings).
- `outputs/gym/gym_mix_<dur>s_vNN.mp4` - best clips (by `clip_score`), best rep window of each.
- `outputs/gym/sbs_<a>_vs_<b|reps>_vNN.mp4` - split screen: one clip = rep 1 vs last rep; two clips = best rep each;
  angle curves overlaid + ROM/tempo/speed table; normal pass then 0.5x pass.
- Pose cache: `cache/gym/<stem>-<hash>/raw_pose.npz` (all detections) - re-renders skip the model. `--no-cache` forces.
- Errors: `reports/errors.jsonl` (stage `gym/...`); one bad clip never stops the batch.

## Procedure
1. Put clips in a folder (mp4/mov/m4v/avi, iPhone HEVC + rotation OK). Run with `--max-clips 1 --debug` first;
   check `debug_<stem>.mp4` (grey boxes = other people, red = chosen athlete, raw vs smoothed keypoints).
2. Check `analysis.json -> exercise` (key, confidence, source). Wrong/unsure -> set it per file in
   `config/gym.yaml: exercise_overrides` (`"*squat*": squat`) or `--exercise`. Below `edit.label_min_confidence`
   the clip renders as neutral "TRAINING" (generic profile: no bar path/speed claims); the guess is kept in
   `extras.auto_guess_low_confidence`.
3. Render all, then `--mix 30 --mix 45` and optional `--side-by-side`.
4. Review numbers before posting: they are 2D estimates (see Limits). Never present them as lab measurements.

## What is drawn per exercise
| exercise | reps from | arcs | extra |
|---|---|---|---|
| squat | knee angle valleys | knee, hip, trunk lean | bar streak, depth (hip below knee), bar speed |
| deadlift/hinge | hip angle valleys | hip, knee, trunk | bar streak + drift, bar speed |
| bench / push-up | elbow valleys | elbow, shoulder / body line | bar streak (bench) |
| overhead press | elbow peaks (lockout) | elbow, shoulder | bar streak, speed |
| olympic (clean/snatch) | bar(wrist) height peaks | hip, knee | bar travel, peak bar speed, hip-extension deg/s |
| jump | flight phases | knee, hip | flight time -> height h = g t^2/8 (est.) |
| sprint | - | hip, knee, trunk | steps, cadence, knee drive, trunk lean |
| agility/shuffle | - | knee, hip | direction changes, stance knee angle |
| lunge | front-knee valleys | knee | |
Look: graded footage (S-curve, vignette, grain), thin glowing key-limb skeleton, glowing arcs (green->amber->red
towards end range) with condensed numbers, tapered bar-path light streak, kinetic metric pop-ins at each rep's
peak, speed ramp (to 0.3x, frame-blended, push-in + flash) into highlighted reps, type-only intro/outro.
Display font: first of `assets/fonts/{Anton,BebasNeue,Oswald}*.ttf`, else DejaVu Bold squeezed.

## Filming advice (biggest quality lever)
- Phone on a tripod at hip height, 3-5 m away, landscape or portrait, NOT moving/panning.
- Side view (90 deg) for squat, deadlift, bench, jumps, sprints; ~45 deg front-side for cleans/snatches;
  front view if you want L/R symmetry (side views report symmetry n/a).
- Whole body AND both bar ends in frame for the whole set; nobody standing in front of the athlete.
- 60 fps if possible (real frames for the slow motion, better velocity/flight time); good light.

## Tuning (config/gym.yaml)
- `athlete.height_m` (or `--athlete-height`) sets the px->m scale (segment-chain stature); all m, m/s are `est.`
- `pose.model: yolo11s-pose.pt` for better keypoints (~2.5x slower on CPU); `pose.kp_min_conf` drops weak joints.
- `pose.smoothing` one_euro (default) | savgol; `pose.one_euro.min_cutoff` lower = smoother, `beta` higher = less lag.
- `pose.plate_tracker: true` (`--plate-track`): LK optical flow on a bar-end ROI seeded at the hands (no CSRT/KCF in
  this OpenCV build), re-anchored to the wrists on drift.
- `edit.max_clip_s` (long sets trimmed to the best run of reps), `edit.peak_moments` auto|all|best|none,
  `edit.ramp_min_speed`, `edit.ramp_half_s`, `edit.ramp_zoom`, `edit.sparkline`.
- `style.accent`, `style.contrast/saturation/vignette/grain`.
- Rep counting misses: raise/lower the profile `min_rom` in `gym/exercises.py`; check the primary signal in `--debug`.

## Limits
- Single 2D camera: angles are projected (exact only in the camera plane); speeds use a body-height scale and are
  wrong if the camera pans/zooms; wrists stand in for the bar unless the plate tracker is on.
- Exercise classification is heuristic from keypoints; hard cuts inside a clip (montages) confuse tracking.
- CPU: yolo11n-pose ~15-20 fps inference at 640 px; render ~15-25 fps output.

## Tests
`PYTHONPATH=src python3 -m pytest tests/test_gym.py -q` (set `GYM_SKIP_REAL=1` to skip the YOLO smoke on
`../media/game-day.mp4`; its result lands in `reports/gym_real_smoke.json`). Synthetic demo without a model:
`PYTHONPATH=src python3 scripts/gym_synthetic_demo.py --exercise deadlift`.
