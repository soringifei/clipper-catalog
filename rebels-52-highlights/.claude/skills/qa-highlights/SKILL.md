---
name: qa-highlights
description: Quality-check rendered clips (container, 1080x1920 H.264, 15-60 s, audio, not black, #52 inside the safe region, thumbnail, manifest fields), then build manifests, review page and final report. Use after rendering or before delivering clips.
---

# QA highlights

## Procedure
1. `python -m rebels_highlights review` - builds `outputs/manifests/highlights.{json,csv}`,
   `review/index.html`, `reports/qa.json`, `reports/final_report.md` (via the pipeline).
2. Read `reports/qa.json`: `summary.failures_by_check` and each clip's `failures`.
   Checks with `null` were not evaluated (e.g. `safe_region` without crop metadata or frame size).
3. Manual QA: `PYTHONPATH=src python3 -c "from rebels_highlights.qa.qa import qa_clip; from rebels_highlights.core.config import load_config; print(qa_clip('outputs/candidates/<clip>.mp4', {}, None, load_config()))"`
4. Fix by cause:
   - `duration_ok` -> `render.lead_in_s/lead_out_s`, `min_clip_s/max_clip_s`;
   - `dims_1080x1920`, `vcodec_h264`, `audio_present` -> render/ffmpeg args;
   - `not_black` -> wrong source window or failed decode;
   - `safe_region` -> crop smoothing/zoom or wrong #52 track (see identify-player-52);
   - `manifest_fields` -> scoring output missing fields.
5. Open `review/index.html`, approve/reject, export `review_decisions.json` (+ `seeds.json`) into `review/`.
6. Final report: check that every number in `reports/final_report.md` comes from the inputs.

## Rules
- Do not deliver clips that fail QA without telling the user which checks failed.
- Tests: `PYTHONPATH=src python3 -m pytest -q`.
