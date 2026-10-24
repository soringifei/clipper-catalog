# Stage contracts (for implementers)

Package: `src/rebels_highlights` (run with `PYTHONPATH=src`). Shared types live in
`core/models.py` (dataclasses with `to_dict`/`from_dict`). Config is the dict from
`core.config.load_config()` (see `config/default.yaml`, `config/player.yaml`).
`core.store.Store(cfg)` gives paths + JSON cache; `core.media` gives `ffmpeg_bin()`,
`run_ffmpeg(args)`, `probe(path)` (no ffprobe on this box - bundled imageio-ffmpeg only).

Rules for every module:
- Heavy optional deps (ultralytics, torch, paddleocr, mediapipe, sam2) are imported lazily
  inside functions with a working CPU/OpenCV fallback. numpy, opencv-python-headless,
  scenedetect, pyyaml, pillow, imageio-ffmpeg are always available.
- Never invent metadata or labels. Uncertain -> lower confidence + review_reasons.
- Pure functions where possible; the orchestrator (`pipeline.py`) does caching.

## Public functions (exact signatures the orchestrator will call)

ingest/ingest.py
  ingest_game(game: Game, cfg, store) -> VideoInfo      # yt-dlp or local_path; raises on failure
  make_proxy(info: VideoInfo, cfg, store) -> str         # low-res/low-fps proxy mp4 path
scenes/scenes.py
  detect_scenes(video_path: str, cfg) -> list[Scene]     # hard cuts + kind live/replay/non_play
plays/plays.py
  motion_profile(video_path: str, cfg, start_s=None, end_s=None) -> dict  # {"t":[...],"motion":[...]}
  segment_plays(info: VideoInfo, scenes: list[Scene], motion: dict, cfg) -> list[Play]
  associate_replays(plays: list[Play], cfg) -> list[Play]  # sets canonical_play_id / replay_segments
field/field.py
  estimate_field(frame_bgr, tracks_at_snap: list[list[float]] | None, cfg) -> dict
      # {"yard_lines":[...], "los_x": float|None, "los_confidence": float, "homography": list|None}

tracking/track_play.py
  track_play(video_path: str, play: Play, cfg, device: str) -> dict
      # {"tracks": [Track...as dict], "fps": float, "frame_size": [w,h], "scene_cuts": [t...]}
      # detection + tracking at detection.fine_fps inside [play.start_s, play.end_s];
      # track ids never cross hard cuts; also stores per-track team_prob + embedding.
identity/identity.py
  identify_player(video_path: str, play: Play, tracking: dict, cfg, references: dict | None,
                  seed: dict | None = None) -> dict
      # {"evidence": [IdentityEvidence...as dict], "best_track_id": int|None,
      #  "identity_confidence": float, "trajectory": PlayerTrajectory as dict}
      # trajectory includes likely ball carrier/QB/returner as target_* and pose_features.
  load_references(ref_dir: str, cfg) -> dict | None     # embeddings from reference_player_52/*
  seed_from_review(store, play_id) -> dict | None        # review/seeds.json {"play_id": {"t":..,"box":[..]}}

events/events.py
  classify_event(play: Play, trajectory: PlayerTrajectory, tracking: dict, field: dict, cfg) -> EventResult
scoring/scoring.py
  build_candidate(game: Game, info: VideoInfo, play: Play, identity: dict, event: EventResult, cfg) -> Candidate
  dedupe_candidates(cands: list[Candidate]) -> list[Candidate]

rendering/render.py
  render_clip(cand: Candidate, info: VideoInfo, trajectory: PlayerTrajectory, event: EventResult,
              cfg, out_dir: str, music_file: str | None = None) -> dict
      # writes <basename>.mp4 (+ _music.mp4 when licensed music given) and <basename>.jpg
      # returns {"output_file", "thumbnail_file", "output_duration_s", "music_file"|None}
  clip_basename(cand: Candidate) -> str
  render_debug(video_path, play, tracking, identity, event, cfg, out_path) -> str
rendering/mix.py
  build_mix(clips: list[Candidate], cfg, out_dir: str, duration_s: int, version: int = 1) -> dict | None
      # {"output_file", "clip_ids", "duration_s"} or None if not enough good clips
  # render.style: epic (default, rendering/epic.py) | clean (legacy look); both write the same outputs
rendering/style.py  (shared visual engine for football + gym renderers; numpy/OpenCV/PIL only)
  get_style(cfg) -> Style; font(role, size); Grader(style)(frame, k, mono=0..1)
  spotlight(...), impact_fx(frame, k_since_impact, style, center), TextSprite / blit / Slam,
  light_streak(...), RampMap(s_a, s_b, holds, vmin, ease, fps), text_zone(occupied, H, text_h)
audio/audio.py
  audio_filter_graph(has_music: bool, cfg) -> str        # loudnorm + sidechain ducking
  sfx(name, cfg) -> np.ndarray   # synthesized boom/whoosh/subdrop/riser/rewind, cached cache/sfx/*.wav
  detect_beats(path|array) -> (beats_s, bpm); snap_to_beat(t, beats, tol=0.12)
  decode_audio / varispeed / lowpass_fft / duck_gain / place / write_wav  (numpy mixing)

manifests/manifest.py
  write_manifests(cands: list[Candidate], store) -> dict  # highlights.json + highlights.csv
review/review.py
  build_review_page(cands: list[Candidate], store) -> str # review/index.html (static, no server)
qa/qa.py
  qa_clip(path: str, cand: Candidate, trajectory: PlayerTrajectory | None, cfg) -> dict
  write_qa_report(results: list[dict], store) -> str       # reports/qa.json
