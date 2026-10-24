"""Rendering tests on a synthetic 1280x720 30 fps landscape source with audio."""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

import numpy as np
import pytest

from rebels_highlights.core.config import load_config
from rebels_highlights.core.media import ffmpeg_bin, probe
from rebels_highlights.core.models import Candidate, EventResult, PlayerTrajectory, VideoInfo
from rebels_highlights.rendering.crop import compute_crop_path
from rebels_highlights.rendering.mix import build_mix, order_clips
from rebels_highlights.rendering.render import clip_basename, render_clip

W, H, FPS, DUR = 1280, 720, 30, 12.0
SNAP, IMPACT = 3.0, 6.5
BOX_W, BOX_H = 44, 96


def p52(t: float) -> tuple[float, float]:
    """#52 centre: pre-snap stance, then flows right to meet the carrier."""
    if t < SNAP:
        return 420.0, 380.0
    if t < IMPACT:
        f = (t - SNAP) / (IMPACT - SNAP)
        return 420.0 + 300 * f, 380.0 - 40 * f
    return 720.0 + 20 * (t - IMPACT), 340.0 + 10 * (t - IMPACT)


def carrier(t: float) -> tuple[float, float]:
    if t < SNAP:
        return 900.0, 330.0
    if t < IMPACT:
        f = (t - SNAP) / (IMPACT - SNAP)
        return 900.0 - 150 * f, 330.0 + 10 * f
    return 750.0 + 15 * (t - IMPACT), 340.0 + 10 * (t - IMPACT)


def box(c):
    return [c[0] - BOX_W / 2, c[1] - BOX_H / 2, c[0] + BOX_W / 2, c[1] + BOX_H / 2]


@pytest.fixture(scope="module")
def workdir(tmp_path_factory):
    return tmp_path_factory.mktemp("render")


@pytest.fixture(scope="module")
def source(workdir) -> str:
    import cv2
    path = str(workdir / "src.mp4")
    cmd = [ffmpeg_bin(), "-y", "-hide_banner", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "pipe:0",
           "-f", "lavfi", "-i", f"sine=frequency=330:sample_rate=48000:duration={DUR}",
           "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-shortest", path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    base = np.zeros((H, W, 3), np.uint8)
    base[:] = (40, 120, 40)
    for x in range(0, W, 80):
        cv2.line(base, (x, 0), (x, H), (230, 230, 230), 2)
    for i in range(int(DUR * FPS)):
        t = i / FPS
        f = base.copy()
        for c, col in ((p52(t), (20, 20, 200)), (carrier(t), (200, 90, 20))):
            b = [int(v) for v in box(c)]
            cv2.rectangle(f, (b[0], b[1]), (b[2], b[3]), col, -1)
        cv2.putText(f, "52", (int(p52(t)[0]) - 15, int(p52(t)[1])), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2)
        p.stdin.write(f.tobytes())
    p.stdin.close()
    assert p.wait() == 0
    return path


@pytest.fixture(scope="module")
def cfg():
    return load_config(overrides={"render": {"preset": "veryfast", "crf": 26}})


def make_case(source: str, idx: int, play_type: str, unit: str, tier: str, score: float,
              game: str = "game_001"):
    ts = [i / 15 for i in range(int(1.0 * 15), int(11.0 * 15))]
    traj = PlayerTrajectory(play_id=f"{game}_p{idx:03d}", track_id=7, times=ts,
                            boxes=[box(p52(t)) for t in ts],
                            target_times=[t for t in ts if t >= 2.0],
                            target_boxes=[box(carrier(t)) for t in ts if t >= 2.0],
                            target_role="ball_carrier")
    ev = EventResult(play_id=traj.play_id, play_type=play_type, event_confidence=0.85,
                     impact_s=IMPACT, contact_point=[735.0, 340.0], primary_actor_score=0.8)
    cand = Candidate(clip_id=f"{game}_c{idx:03d}", game_id=game, play_id=traj.play_id,
                     unit=unit, play_type=play_type, source_start_s=2.0, snap_time_s=SNAP,
                     impact_time_s=IMPACT, source_end_s=9.0, track_id=7,
                     identity_confidence=0.91, event_confidence=0.85,
                     overall_highlight_score=score, tier=tier, review_status="AUTO_APPROVED",
                     caption=f"#52 {play_type.replace('_', ' ')}")
    info = VideoInfo(game_id=game, path=source, duration_s=DUR, fps=FPS, width=W, height=H,
                     has_audio=True)
    return cand, info, traj, ev


def test_basename():
    c = Candidate(clip_id="x", game_id="game_003", play_id="game_003_p014",
                  unit="special_teams", play_type="special_teams_tackle", snap_time_s=3725.4,
                  identity_confidence=0.873)
    assert clip_basename(c) == "unknown-date_game-003_p014_special-teams_tackle_52_01h02m05s_c087"
    c.date, c.opponent, c.unit, c.play_type = "2025-05-10", "Cluj Crusaders", "defense", "tackle_for_loss"
    assert clip_basename(c) == "2025-05-10_rebels-vs-cluj-crusaders_p014_defense_tfl_52_01h02m05s_c087"


def test_crop_safe_region_and_split(cfg):
    cand, info, traj, ev = make_case("unused", 1, "solo_tackle", "defense", "B", 0.6)
    path = compute_crop_path(traj, ev, W, H, 1.0, 10.0, cfg)
    assert path.mode == "vertical"
    half = path.safe_region * path.crop_w / 2
    inside = np.abs(path.px - path.cx) <= half + 1e-6
    assert inside.mean() > 0.95
    # boxes also stay fully inside the crop
    assert np.all(path.cx - path.crop_w / 2 >= -1e-6) and np.all(path.cx + path.crop_w / 2 <= W + 1e-6)
    # far-apart target at impact -> split mode
    far = PlayerTrajectory(play_id="p", track_id=1, times=traj.times, boxes=traj.boxes,
                           target_times=traj.times,
                           target_boxes=[[1150, 300, 1194, 396]] * len(traj.times))
    sp = compute_crop_path(far, ev, W, H, 1.0, 10.0, cfg)
    assert sp.mode == "split"
    assert sp.crop_w > path.crop_w


@pytest.fixture(scope="module")
def rendered(source, cfg, workdir):
    out = []
    specs = [(1, "sack", "defense", "S", 0.92), (2, "solo_tackle", "defense", "B", 0.65),
             (3, "special_teams_tackle", "special_teams", "A", 0.80)]
    times = []
    for idx, pt, unit, tier, sc in specs:
        cand, info, traj, ev = make_case(source, idx, pt, unit, tier, sc)
        t0 = time.time()
        res = render_clip(cand, info, traj, ev, cfg, str(workdir / "clips"))
        times.append(time.time() - t0)
        cand.output_file, cand.thumbnail_file = res["output_file"], res["thumbnail_file"]
        cand.output_duration_s = res["output_duration_s"]
        out.append((cand, res, traj))
    print(f"render times: {[round(t, 1) for t in times]} s")
    return out


def test_render_clip(rendered):
    cand, res, _ = rendered[0]
    assert Path(res["output_file"]).is_file()
    assert Path(res["thumbnail_file"]).is_file()
    assert res["music_file"] is None
    pi = probe(res["output_file"])
    assert (pi["width"], pi["height"]) == (1080, 1920)
    assert pi["vcodec"] == "h264" and pi["acodec"] == "aac" and pi["has_audio"]
    assert 15.0 <= pi["duration"] <= 60.0
    assert abs(pi["duration"] - res["output_duration_s"]) < 0.3
    assert res["crop_mode"] == "vertical" and "POOR_VERTICAL_CROP" not in res["review_reasons"]
    import cv2
    th = cv2.imread(res["thumbnail_file"])
    assert th.shape[:2] == (1920, 1080)
    # seed frame (landscape, pre-snap) for manual seeding
    seed = cv2.imread(res["seed_frame"])
    assert seed.shape[:2] == (H, W) and res["seed_frame_size"] == [W, H]
    assert res["seed_frame_t"] < SNAP
    # QA crop sidecar: #52 inside the central safe region for >95% of real-time frames
    import json
    meta = json.loads(Path(res["crop_file"]).read_text())
    assert Path(res["crop_file"]).name == Path(res["output_file"]).stem + ".crop.json"
    ok = []
    for f in meta["frames"]:
        x1, _, x2, _ = f["crop"]
        half = meta["safe_region"] * (x2 - x1) / 2
        ok.append(abs(f["player_x"] - (x1 + x2) / 2) <= half + 1e-3)
    assert len(ok) > 100 and np.mean(ok) > 0.95


def test_music_version(source, cfg, workdir):
    music = str(workdir / "music.m4a")
    subprocess.run([ffmpeg_bin(), "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
                    "-i", "sine=frequency=660:sample_rate=48000:duration=5", "-c:a", "aac", music],
                   check=True)
    cand, info, traj, ev = make_case(source, 9, "open_field_tackle", "defense", "A", 0.7)
    res = render_clip(cand, info, traj, ev, cfg, str(workdir / "music"), music_file=music)
    assert res["music_file"] and Path(res["music_file"]).is_file()
    assert res["music_file"].endswith("_music.mp4")
    pm = probe(res["music_file"])
    assert pm["has_audio"] and abs(pm["duration"] - res["output_duration_s"]) < 0.3


def test_build_mix(rendered, cfg, workdir):
    clips = [c for c, _, _ in rendered]
    seq = order_clips(clips, cfg)
    assert seq[0].play_type == "sack"            # strongest S/A opens
    assert seq[-1].play_type == "special_teams_tackle"  # next strongest closes
    res = build_mix(clips, cfg, str(workdir / "mixes"), 30)
    assert res is not None
    assert Path(res["output_file"]).is_file()
    assert Path(res["output_file"]).name.startswith("52_MLB_Bucharest-Rebels_mix_games-001-001_")
    pi = probe(res["output_file"])
    assert pi["duration"] <= 30.0 + 0.05
    assert pi["duration"] >= 15.0
    assert (pi["width"], pi["height"]) == (1080, 1920) and pi["has_audio"]
    # clips below identity threshold / not approved are excluded
    clips[1].review_status = "REVIEW"
    res2 = build_mix(clips, cfg, str(workdir / "mixes"), 30, version=2)
    assert res2 is None or clips[1].clip_id not in res2["clip_ids"]


def test_render_debug(source, cfg, workdir):
    from rebels_highlights.core.models import Play
    from rebels_highlights.rendering.render import render_debug
    cand, info, traj, ev = make_case(source, 5, "sack", "defense", "S", 0.9)
    play = Play(play_id=traj.play_id, game_id="game_001", start_s=5.5, end_s=7.0,
                estimated_snap_s=SNAP, unit="defense")
    tracking = {"tracks": [{"track_id": 7, "scene_id": 0, "times": traj.times, "boxes": traj.boxes,
                            "team_prob": 0.9},
                           {"track_id": 9, "scene_id": 0, "times": traj.target_times,
                            "boxes": traj.target_boxes, "team_prob": 0.1}]}
    identity = {"best_track_id": 7, "identity_confidence": 0.91, "trajectory": traj.to_dict(),
                "evidence": [{"track_id": 7, "scene_id": 0, "identity_confidence": 0.91,
                              "ocr_votes": {"52": 5}}]}
    out = render_debug(source, play, tracking, identity, ev, cfg, str(workdir / "dbg" / "d.mp4"))
    pi = probe(out)
    assert (pi["width"], pi["height"]) == (W, H) and pi["duration"] > 1.0


# ------------------------------------------------------------------ epic style engine
def test_style_switch(cfg):
    from rebels_highlights.rendering.style import get_style
    assert get_style(cfg).name == "epic" and get_style(cfg).epic
    assert get_style({"render": {"style": "clean"}}).name == "clean"
    assert get_style({"render": {"style": "nonsense"}}).name == "epic"
    st = get_style({"render": {"epic": {"grade": 0.5, "ramp_speed": 0.33}}})
    assert st.grade == 0.5 and st.ramp_speed == 0.33


def test_clean_style_render(source, workdir):
    import json
    cfg_c = load_config(overrides={"render": {"preset": "ultrafast", "crf": 30, "style": "clean"}})
    cand, info, traj, ev = make_case(source, 11, "solo_tackle", "defense", "B", 0.6)
    res = render_clip(cand, info, traj, ev, cfg_c, str(workdir / "clean"))
    assert res["style"] == "clean"
    tl = json.loads(Path(res["timeline_file"]).read_text())
    assert [s["kind"] for s in tl["segments"]][:2] == ["ident", "live"]
    assert 15.0 <= probe(res["output_file"])["duration"] <= 60.0


def test_epic_timeline(rendered):
    import json
    cand, res, _ = rendered[0]
    assert res["style"] == "epic"
    tl = json.loads(Path(res["timeline_file"]).read_text())
    kinds = [s["kind"] for s in tl["segments"]]
    assert kinds[:6] == ["coldopen", "flash", "title", "rewind", "intro", "live"]
    assert "replay" in kinds and kinds[-1] == "endcard"
    assert "REPLAY" not in json.dumps(tl)
    co = tl["segments"][0]
    assert 0.6 <= co["end"] - co["start"] <= 1.0
    # hook: first impact inside the first second, then live + replay impacts
    assert tl["impacts_out"][0] < 1.0 and len(tl["impacts_out"]) >= 3
    ec = tl["segments"][-1]
    assert 1.5 <= ec["end"] - ec["start"] <= 2.7
    assert 15.0 <= tl["duration"] <= 60.0
    # text never sits on the impact moment of the live play... except the
    # post-contact word, which is placed in a band away from the contact
    assert any(t["text"] == "title" for t in tl["text"])
    names = {s["name"] for s in tl["sfx"]}
    assert {"boom", "whoosh", "subdrop", "riser"} <= names


def test_caption_label_only_from_caption(source, cfg):
    from rebels_highlights.rendering.epic import EpicClip, caption_label
    assert caption_label("#52 | MLB | BUCHAREST REBELS") is None
    assert caption_label("#52 | MLB | BUCHAREST REBELS — SACK") == "SACK"
    from rebels_highlights.rendering.style import get_style
    cand, info, traj, ev = make_case(source, 12, "sack", "defense", "S", 0.9)
    cand.caption = "#52 | MLB | BUCHAREST REBELS"          # no label -> no label slam
    clip = EpicClip(cand, info, traj, ev, cfg, get_style(cfg), None)
    tags = [s.tag for s in clip.build_text()]
    clip.reader.close()
    assert "label" not in tags and "title" in tags and "end" in tags
    # unknown player name is never drawn
    assert clip.name is None


def test_speed_ramp_monotonic():
    from rebels_highlights.rendering.style import RampMap
    r = RampMap(10.0, 16.0, [(13.0, 13.3)], vmin=0.3, ease=0.35, fps=30)
    fr = r.frames()
    assert np.all(np.diff(fr) > 0)                      # strictly monotonic source time
    assert fr[0] == pytest.approx(10.0) and fr[-1] <= 16.0
    sp = np.diff(fr) * 30
    assert sp.min() == pytest.approx(0.3, abs=0.02)     # floor reached around contact
    assert sp[:30].min() > 0.99 and sp[-10:].min() > 0.99  # 1x before/after
    hold_out = r.out(13.3) - r.out(13.0)
    assert 0.8 <= hold_out <= 1.2                       # ~1 s of slow motion
    assert r.duration > 6.0 and r.n == round(r.duration * 30)
    # out/src are inverse maps
    assert r.src(r.out(14.2)) == pytest.approx(14.2, abs=1e-3)


def test_sfx_generation(tmp_path):
    from rebels_highlights.audio import audio as au
    cfg_t = {"root": str(tmp_path), "paths": {"cache": "cache"}}
    for name in au.SFX_NAMES:
        x = au.sfx(name, cfg_t)
        assert x.ndim == 2 and x.shape[1] == 2 and len(x) > 1000
        assert 0.5 < np.abs(x).max() <= 0.91 and np.all(np.isfinite(x))
    files = sorted(p.name for p in (tmp_path / "cache" / "sfx").glob("*.wav"))
    assert len(files) == len(au.SFX_NAMES)
    boom = au.synth_sfx("boom")[:, 0]
    spec = np.abs(np.fft.rfft(boom))
    f = np.fft.rfftfreq(len(boom), 1 / au.SAMPLE_RATE)
    assert 25 <= f[np.argmax(spec)] <= 90                # deep boom
    x2, sr = au.read_wav(tmp_path / "cache" / "sfx" / files[0])
    assert sr == au.SAMPLE_RATE


def test_beat_detection_and_snapping():
    from rebels_highlights.audio.audio import detect_beats, snap_to_beat, tile_beats
    sr, bpm, off = 22050, 124.0, 0.21
    x = np.zeros(sr * 16, np.float32)
    rng = np.random.default_rng(0)
    P = 60 / bpm
    for k in range(int(16 / P)):
        i = int((off + k * P) * sr)
        x[i:i + 400] += np.hanning(400) * rng.normal(0, 1, 400)
    beats, est = detect_beats(x, sr)
    assert abs(est - bpm) / bpm < 0.03
    err = [min(abs(b - (off + k * P)) for k in range(int(16 / P))) for b in beats[2:-2]]
    assert np.median(err) < 0.03
    assert snap_to_beat(beats[5] + 0.08, beats) == beats[5]
    assert snap_to_beat(beats[5] + 0.2, beats, 0.12) == beats[5] + 0.2
    assert tile_beats([0.1, 0.6], 1.0, 3.0)[:4] == [0.1, 0.6, 1.1, 1.6]


def test_plan_snaps_cuts_to_beats(source, cfg):
    from rebels_highlights.rendering.epic import plan_epic
    cand, info, traj, ev = make_case(source, 13, "sack", "defense", "S", 0.9)
    beats = list(np.arange(0.03, 60, 0.5))                # 120 BPM grid, off the nominal cuts
    plain = plan_epic(cand, ev, cfg, DUR)
    snapped = plan_epic(cand, ev, cfg, DUR, beats=beats)
    assert len(snapped["beat_snapped"]) >= 3
    seg = {s["kind"]: s for s in snapped["segments"]}
    pseg = {s["kind"]: s for s in plain["segments"]}
    for k in snapped["beat_snapped"]:
        e = seg[k]["out_end"]
        assert min(abs(b - e) for b in beats) <= 1 / 30 + 1e-6, k
        assert abs(e - pseg[k]["out_end"]) <= 0.12 + 1 / 30 + 0.5  # small nudges only (cumulative)
    assert 15.0 <= snapped["duration"] <= 60.0 and 15.0 <= plain["duration"] <= 60.0


def test_mix_align_to_beats():
    from rebels_highlights.rendering.mix import align_to_beats
    pieces = [(0.0, 5.0, 0), (2.0, 7.93, 1), (1.0, 6.0, 2)]
    beats = list(np.arange(0.0, 30, 0.5))
    out = align_to_beats(pieces, beats, 30)
    t = 0.0
    for i, (a, b, ci) in enumerate(out[:-1]):
        t += b - a
        assert min(abs(t - x) for x in beats) < 1 / 30 + 1e-6
    assert out[-1] == pieces[-1]
