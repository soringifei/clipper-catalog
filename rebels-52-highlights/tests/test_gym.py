"""Tests for the gym module: angle math, rep segmentation, classifier, jump height,
crop smoothing, video I/O, an end-to-end render with a fake pose provider and (when the
YOLO pose weights load) a real smoke run on media/game-day.mp4."""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pytest

from rebels_highlights.core.media import probe
from rebels_highlights.gym import metrics as M
from rebels_highlights.gym import synth
from rebels_highlights.gym.analysis import analyze
from rebels_highlights.gym.config import exercise_override, load_gym_config
from rebels_highlights.gym.exercises import classify, normalize_name
from rebels_highlights.gym.pose import FakePoseProvider, build_sequence, fill_gaps, one_euro, savgol
from rebels_highlights.gym.render import clamp_window, plan_edit, ramp_speed, smooth_centers
from rebels_highlights.gym.video import atempo_chain, probe_video

ROOT = Path(__file__).resolve().parents[1]
GAME_DAY = ROOT.parent / "media" / "game-day.mp4"


def _cfg(tmp_path: Path, **extra) -> dict:
    ov = {"paths": {"outputs": str(tmp_path / "out"), "cache": str(tmp_path / "cache")}}
    for k, v in extra.items():
        ov.setdefault(k, {}).update(v)
    return load_gym_config(overrides=ov)


def _seq(kp: np.ndarray, cfg: dict, fps: float = 30.0, w: int = 1280, h: int = 1100):
    info = {"n_frames": len(kp), "work_fps": fps, "disp_w": w, "disp_h": h}
    return build_sequence(FakePoseProvider(kp).run(None, info, cfg), cfg)


# ---------------------------------------------------------------------------
# angle math
# ---------------------------------------------------------------------------
def test_angle_3pt_basic():
    assert M.angle_3pt([1, 0], [0, 0], [0, 1]) == pytest.approx(90)
    assert M.angle_3pt([-1, 0], [0, 0], [1, 0]) == pytest.approx(180)
    assert M.angle_3pt([1, 1], [0, 0], [1, 0]) == pytest.approx(45)
    a = M.angle_3pt(np.array([[1, 0], [np.nan, 0]]), np.zeros((2, 2)), np.array([[0, 1], [0, 1]]))
    assert a[0] == pytest.approx(90) and np.isnan(a[1])


def test_angle_vs_vertical_image_coords():
    # y grows downwards: a segment going straight up the image is 0 deg
    assert M.angle_vs_vertical([0, 100], [0, 0]) == pytest.approx(0)
    assert M.angle_vs_vertical([0, 100], [100, 0]) == pytest.approx(45)
    assert M.angle_vs_vertical([0, 0], [100, 0]) == pytest.approx(90)


def test_joint_angles_on_synthetic_squat(tmp_path):
    cfg = _cfg(tmp_path)
    seq = _seq(synth.squat(150, 30, reps=3), cfg)
    ang = M.joint_angles(seq)
    knee = ang["knee_l"][~np.isnan(ang["knee_l"])]
    assert knee.max() > 170 and knee.min() < 75          # standing -> deep squat
    assert np.nanmax(ang["trunk"]) < 60


# ---------------------------------------------------------------------------
# rep segmentation
# ---------------------------------------------------------------------------
def test_rep_segmentation_sinusoid():
    fps, reps, T = 30.0, 5, 450
    t = np.arange(T)
    rng = np.random.default_rng(0)
    knee = 130 + 45 * np.cos(2 * np.pi * reps * t / T) + rng.normal(0, 1.5, T)
    out = M.segment_reps(knee, fps, "valley", min_rom=30)
    assert len(out) == reps
    for r in out:
        assert r.rom_deg == pytest.approx(90, abs=8)
        assert r.min_deg == pytest.approx(85, abs=6)
        assert r.start < r.turn < r.end
        assert r.ecc_s == pytest.approx(r.con_s, abs=0.35)       # symmetric motion
        assert r.peak_ang_vel_dps > 100
    # valleys land on the true bottoms (period 90 frames, first at 45)
    assert [r.turn for r in out] == pytest.approx([45 + 90 * i for i in range(reps)], abs=4)


def test_rep_segmentation_ignores_wobble_and_peak_polarity():
    fps = 30.0
    t = np.arange(300)
    wobble = 170 + 6 * np.sin(2 * np.pi * t / 20)
    assert M.segment_reps(wobble, fps, "valley", min_rom=25) == []
    press = 90 + 40 * np.sin(2 * np.pi * 3 * t / 300 - np.pi / 2)   # starts at rack (bottom)
    out = M.segment_reps(press, fps, "peak", min_rom=30)
    assert len(out) == 3 and all(r.rom_deg > 70 for r in out)


def test_zigzag_alternates():
    x = np.array([0, 5, 1, 6, 0, 7, 0], float)
    piv = M.zigzag(x, 3)
    kinds = [k for _, k in piv]
    assert all(a != b for a, b in zip(kinds, kinds[1:]))
    assert len(piv) >= 5


# ---------------------------------------------------------------------------
# classifier
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,expected", [
    ("squat", "squat"), ("deadlift", "deadlift"), ("bench", "bench"), ("jump", "jump"),
    ("sprint", "sprint"), ("overhead_press", "overhead_press"), ("olympic", "olympic"),
    ("lunge", "lunge"), ("agility", "agility"), ("pushup", "pushup"),
])
def test_classifier_synthetic(tmp_path, name, expected):
    cfg = _cfg(tmp_path)
    seq = _seq(synth.GENERATORS[name](300, 30), cfg)
    key, conf, scores = classify(seq)
    assert key == expected, scores
    assert conf >= cfg["edit"]["label_min_confidence"]


def test_classifier_low_confidence_is_neutral(tmp_path):
    cfg = _cfg(tmp_path)
    kp = synth.squat(200, 30, reps=4)
    kp[:, :, :2] = kp[0, :, :2]          # frozen athlete: no movement at all
    an = analyze(_seq(kp, cfg), cfg)
    assert not an.label_visible and an.summary["exercise"] == "TRAINING"


def test_exercise_overrides(tmp_path):
    cfg = _cfg(tmp_path)
    cfg["exercise_overrides"] = {"*squat*": "squat", "IMG_04*.MOV": "clean"}
    assert exercise_override(cfg, "/x/Heavy_SQUAT_day.mp4") == "squat"
    assert normalize_name(exercise_override(cfg, "IMG_0412.MOV")) == "olympic"
    assert exercise_override(cfg, "other.mp4") is None
    assert normalize_name("Bench Press") == "bench" and normalize_name("nonsense") is None
    an = analyze(_seq(synth.deadlift(200, 30), cfg), cfg, exercise="squat", source="cli")
    assert an.exercise == "squat" and an.exercise_source == "cli" and an.confidence == 1.0


# ---------------------------------------------------------------------------
# jump + scale
# ---------------------------------------------------------------------------
def test_jump_height_formula():
    assert M.jump_height_from_flight(0.5) == pytest.approx(9.81 * 0.25 / 8)
    assert M.jump_height_from_flight(0.0) == 0.0


def test_detect_jumps_synthetic(tmp_path):
    cfg = _cfg(tmp_path)
    seq = _seq(synth.jump(300, 30, jumps=3, flight_s=0.5), cfg)
    jumps = M.detect_jumps(seq, M.body_scale_px(seq))
    assert len(jumps) == 3
    for j in jumps:
        assert j.flight_s == pytest.approx(0.5, abs=0.07)
        assert j.height_m_est == pytest.approx(0.31, abs=0.08)


def test_metres_per_px_from_athlete_height(tmp_path):
    cfg = _cfg(tmp_path)
    seq = _seq(synth.squat(120, 30, H=700), cfg)          # stature 700 px
    mpp = M.metres_per_px(seq, 1.85)
    assert mpp == pytest.approx(1.85 / 700, rel=0.12)


def test_symmetry():
    assert M.symmetry_pct(100, 100) == 100
    assert M.symmetry_pct(90, 110) == pytest.approx(80)
    assert M.symmetry_pct(float("nan"), 3) is None


# ---------------------------------------------------------------------------
# smoothing / crop
# ---------------------------------------------------------------------------
def test_crop_smoothing_reduces_jitter_and_stays_in_bounds():
    rng = np.random.default_rng(1)
    t = np.arange(300)
    truth = 600 + 200 * np.sin(t / 60)
    noisy = truth + rng.normal(0, 25, len(t))
    noisy[50:60] = np.nan
    sm = smooth_centers(noisy, 0.9, 30)
    assert not np.isnan(sm).any()
    assert np.std(np.diff(sm)) < 0.2 * np.std(np.diff(np.nan_to_num(noisy, nan=600)))
    assert np.abs(sm - truth)[20:-20].mean() < 25         # zero-phase: no big lag
    cl = clamp_window(np.array([0.0, 500, 5000]), 400, 1920)
    assert cl.tolist() == [200, 500, 1720]
    assert clamp_window(np.array([10.0]), 3000, 1920).tolist() == [960]


def test_one_euro_savgol_and_gaps():
    rng = np.random.default_rng(2)
    x = np.linspace(0, 100, 200) + rng.normal(0, 3, 200)
    oe = one_euro(x[:, None], 30, 1.0, 0.0)[:, 0]
    assert np.std(np.diff(oe)) < np.std(np.diff(x))
    sg = savgol(x, 9, 2)
    assert np.std(np.diff(sg)) < np.std(np.diff(x))
    g = np.array([0, 1, np.nan, np.nan, 4, np.nan, np.nan, np.nan, np.nan, 9], float)
    f = fill_gaps(g, 2)
    assert f[2] == pytest.approx(2) and f[3] == pytest.approx(3)
    assert np.isnan(f[5:9]).all()                       # long gap stays empty


def test_ramp_speed_profile():
    d = np.linspace(-1, 1, 201)
    v = ramp_speed(d, 0.6, 0.3)
    assert v.min() == pytest.approx(0.3) and v[0] == pytest.approx(1) and v[-1] == pytest.approx(1)
    assert np.all(np.diff(v[:100]) <= 1e-9)             # slows monotonically into the peak


def test_atempo_chain():
    for sp in (0.3, 0.5, 0.8, 1.0, 0.2):
        ch = atempo_chain(sp)
        assert math.prod(ch) == pytest.approx(sp) and all(0.5 <= c <= 100 for c in ch)


def test_plan_edit_trims_long_clip(tmp_path):
    cfg = _cfg(tmp_path, edit={"max_clip_s": 20})
    seq = _seq(synth.squat(2400, 30, reps=20), cfg)          # 80 s raw clip
    an = analyze(seq, cfg)
    plan = plan_edit(an, cfg)
    assert plan.duration_s <= 20.5 and "trimmed" in plan.note
    assert len(plan.peak_reps) <= 3


# ---------------------------------------------------------------------------
# video I/O
# ---------------------------------------------------------------------------
def test_rotation_metadata_handled(tmp_path):
    kp = synth.squat(30, 30, H=400, x0=640, ground=680)
    p = synth.render_stick_video(tmp_path / "rot.mp4", kp, 1280, 720, 30, audio=False, rotate_meta=90)
    info = probe_video(p)
    assert (info["width"], info["height"]) == (720, 1280)        # stored portrait
    assert (info["disp_w"], info["disp_h"]) == (1280, 720)       # displayed landscape
    assert info["rotation"] in (90, 270)


# ---------------------------------------------------------------------------
# end to end (fake pose provider)
# ---------------------------------------------------------------------------
def test_end_to_end_render_with_fake_pose(tmp_path):
    from rebels_highlights.gym.pipeline import process_clip
    cfg = _cfg(tmp_path)
    fps, T = 30, 150
    kp = synth.squat(T, fps, reps=3, H=440, x0=640, ground=690)
    vid = synth.render_stick_video(tmp_path / "squat_set.mp4", kp, 1280, 720, fps)
    t0 = time.time()
    r = process_clip(Path(vid), cfg, provider=FakePoseProvider(kp))
    took = time.time() - t0
    out = Path(r["render"]["output_file"])
    assert out.exists() and out.parent == tmp_path / "out" / "squat_set"
    info = probe(out)
    assert (info["width"], info["height"]) == (1080, 1920)
    assert info["vcodec"] == "h264" and info["has_audio"] and info["acodec"] == "aac"
    assert info["duration"] == pytest.approx(r["render"]["duration_s"], abs=0.25)
    # 5 s raw + intro/outro + 3 speed ramps
    assert 9 < info["duration"] < 20
    assert Path(r["render"]["thumbnail_file"]).exists()
    aj = json.loads((out.parent / "analysis.json").read_text())
    assert aj["exercise"]["key"] == "squat" and len(aj["reps"]) == 3
    assert aj["summary"]["below_parallel_reps"] == "3/3"
    assert "m_per_px_est" in aj["scale"] and aj["summary"].get("peak_bar_speed_mps_est")
    print(f"\n[e2e] rendered {info['duration']:.1f}s in {took:.1f}s")


# ---------------------------------------------------------------------------
# real smoke (YOLO pose on the stock clip) - skipped if weights can't load
# ---------------------------------------------------------------------------
@pytest.mark.skipif(os.environ.get("GYM_SKIP_REAL") == "1" or not GAME_DAY.exists(),
                    reason="real smoke disabled or media missing")
def test_real_smoke_game_day(tmp_path):
    from rebels_highlights.gym.pipeline import run_batch
    from rebels_highlights.gym.pose import YoloPoseProvider
    cfg = _cfg(tmp_path)
    cfg["paths"]["models"] = str(ROOT / "cache/models")
    try:
        YoloPoseProvider(cfg).model()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"yolo pose model unavailable: {e}")
    t0 = time.time()
    res = run_batch([str(GAME_DAY)], cfg, max_clips=1)
    took = time.time() - t0
    report = {"seconds": round(took, 1)}
    if res:
        r = res[0]
        report.update(track_fraction=r["track_fraction"], exercise=r["exercise"]["key"],
                      confidence=r["exercise"]["confidence"], duration=r["render"]["duration_s"],
                      render_s=r["render"]["render_seconds"], pose_s=r["timing"]["pose_s"])
        info = probe(r["render"]["output_file"])
        assert (info["width"], info["height"]) == (1080, 1920) and info["has_audio"]
    else:   # a clip without a trackable person is a valid outcome - it must fail cleanly
        errs = (ROOT / "reports/errors.jsonl")
        report["error"] = errs.read_text().strip().splitlines()[-1][:300] if errs.exists() else "?"
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports/gym_real_smoke.json").write_text(json.dumps(report, indent=1))
    print("\n[real smoke]", json.dumps(report))
