"""Ingest / scenes / plays / field on a synthetic football-like video."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from rebels_highlights.core.config import load_config
from rebels_highlights.core.media import probe
from rebels_highlights.core.models import Game, Play, Scene, VideoInfo
from rebels_highlights.core.store import Store
from rebels_highlights.field.field import estimate_field, project_along_field
from rebels_highlights.ingest.ingest import IngestError, ingest_game, make_proxy
from rebels_highlights.plays.plays import (associate_replays, motion_profile,
                                           segment_plays)
from rebels_highlights.scenes.scenes import detect_scenes

W, H, FPS = 640, 360, 30
STATIC_S, MOVE_S, STOP_S, CUT_S = 3.0, 4.0, 2.0, 3.0
SNAP_T = STATIC_S
CUT_T = STATIC_S + MOVE_S + STOP_S


def _field_bg() -> np.ndarray:
    img = np.zeros((H, W, 3), np.uint8)
    img[:] = (40, 140, 50)  # BGR grass green
    for x in range(40, W, 80):  # yard lines
        cv2.line(img, (x, 20), (x + 10, H - 20), (245, 245, 245), 3)
    return img


def _write_video(path: str) -> None:
    rng = np.random.default_rng(0)
    n_players = 14
    # two lines of players facing each other around x=320
    pos = np.array([[300 - 10 * (i % 2), 60 + 20 * i] if i < 7 else
                    [340 + 10 * (i % 2), 60 + 20 * (i - 7)] for i in range(n_players)],
                   dtype=float)
    vel = rng.uniform(-220, 220, size=(n_players, 2))
    vel[np.abs(vel) < 80] = 120.0
    bg = _field_bg()
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    total = int((STATIC_S + MOVE_S + STOP_S + CUT_S) * FPS)
    for f in range(total):
        t = f / FPS
        if t < CUT_T:
            if STATIC_S <= t < STATIC_S + MOVE_S:
                pos += vel / FPS
                for k in range(2):
                    bounce = (pos[:, k] < 15) | (pos[:, k] > (W, H)[k] - 30)
                    vel[bounce, k] *= -1
                    pos[:, k] = np.clip(pos[:, k], 15, (W, H)[k] - 30)
            img = bg.copy()
            for i, (x, y) in enumerate(pos):
                col = (30, 30, 200) if i < 7 else (220, 220, 220)
                cv2.rectangle(img, (int(x), int(y)), (int(x) + 12, int(y) + 26), col, -1)
        else:  # hard cut: scoreboard-like graphic, no grass
            img = np.zeros((H, W, 3), np.uint8)
            img[:] = (90, 40, 20)
            cv2.putText(img, "REBELS 14 - 7", (120, 190), cv2.FONT_HERSHEY_SIMPLEX, 1.6,
                        (255, 255, 255), 3)
        vw.write(img)
    vw.release()


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("rh")
    video = str(tmp / "game.mp4")
    _write_video(video)
    cfg = load_config()
    cfg["root"] = str(tmp / "proj")
    store = Store(cfg)
    return {"video": video, "cfg": cfg, "store": store, "tmp": tmp}


@pytest.fixture(scope="module")
def ingested(env):
    game = Game(game_id="game_900", local_path=env["video"])
    info = ingest_game(game, env["cfg"], env["store"])
    info.proxy_path = make_proxy(info, env["cfg"], env["store"])
    return info


def test_ingest_local_path(ingested, env):
    assert ingested.path == env["video"]
    assert ingested.width == W and ingested.height == H
    assert abs(ingested.fps - FPS) < 0.5
    assert abs(ingested.duration_s - (CUT_T + CUT_S)) < 0.3
    assert ingested.has_audio is False
    assert ingested.source_url is None


def test_ingest_errors(env):
    with pytest.raises(IngestError):
        ingest_game(Game(game_id="game_901"), env["cfg"], env["store"])
    with pytest.raises(IngestError):
        ingest_game(Game(game_id="game_902", local_path=str(env["tmp"] / "missing.mp4")),
                    env["cfg"], env["store"])


def test_proxy_probeable(ingested, env):
    p = probe(ingested.proxy_path)
    assert p["height"] == env["cfg"]["proxy"]["height"]
    assert abs(p["fps"] - env["cfg"]["proxy"]["fps"]) < 0.5
    assert abs(p["duration"] - ingested.duration_s) < 0.5
    # second call reuses the cached proxy
    assert make_proxy(ingested, env["cfg"], env["store"]) == ingested.proxy_path


def test_scenes(ingested, env):
    scenes = detect_scenes(ingested.proxy_path, env["cfg"])
    assert len(scenes) >= 2
    cut = [s for s in scenes if abs(s.start_s - CUT_T) < 0.5]
    assert cut, [s.to_dict() for s in scenes]
    assert cut[0].kind == "non_play"
    live = [s for s in scenes if s.end_s <= CUT_T + 0.5]
    assert live and all(s.kind == "live" for s in live)


def test_plays(ingested, env):
    cfg = env["cfg"]
    scenes = detect_scenes(ingested.proxy_path, cfg)
    motion = motion_profile(ingested.proxy_path, cfg)
    assert len(motion["t"]) == len(motion["motion"]) == len(motion["camera_motion"]) > 50
    plays = associate_replays(segment_plays(ingested, scenes, motion, cfg), cfg)
    assert plays, motion
    p = plays[0]
    assert p.play_id == "game_900_p001"
    assert p.estimated_snap_s is not None and abs(p.estimated_snap_s - SNAP_T) < 0.6
    assert p.start_s >= 0 and p.end_s <= CUT_T + 0.1
    assert cfg["plays"]["min_play_s"] <= p.end_s - p.start_s <= cfg["plays"]["max_play_s"]
    assert p.end_s >= SNAP_T + MOVE_S - 1.0
    assert 0.5 <= p.confidence <= 1.0
    # nothing detected inside the non_play scene
    assert all(q.start_s < CUT_T for q in plays)


def test_motion_window(ingested, env):
    m = motion_profile(ingested.proxy_path, env["cfg"], start_s=2.0, end_s=5.0)
    assert m["t"][0] >= 2.0 and m["t"][-1] <= 5.01


def test_associate_replays():
    info = VideoInfo(game_id="g", path="x", duration_s=200)
    scenes = [Scene(0, 0.0, 12.0, "live"), Scene(1, 12.0, 20.0, "replay"),
              Scene(2, 100.0, 108.0, "replay")]
    t = np.arange(0.0, 110.0, 0.1)
    m = np.where(((t > 3) & (t < 7)) | ((t > 14) & (t < 18)) | ((t > 102) & (t < 106)),
                 5.0, 0.2)
    plays = associate_replays(segment_plays(
        info, scenes, {"t": t.tolist(), "motion": m.tolist()}, {}), {})
    assert len(plays) == 3
    live, rep, lone = plays
    assert live.canonical_play_id is None and len(live.replay_segments) == 1
    assert live.replay_segments[0]["start"] == rep.start_s
    assert rep.canonical_play_id == live.play_id
    assert lone.canonical_play_id is None  # > 60 s after the live play


def test_field_and_los():
    frame = _field_bg()
    boxes = []
    for i in range(7):  # offense at x~300, defense at x~345, spread across field
        boxes.append([290.0, 60 + 35 * i, 302.0, 86 + 35 * i])
        boxes.append([340.0, 60 + 35 * i, 352.0, 86 + 35 * i])
    boxes += [[200, 150, 212, 176], [450, 200, 462, 226]]  # backs / safeties
    f = estimate_field(frame, boxes, {})
    assert len(f["yard_lines"]) >= 4
    assert f["homography"] is not None
    assert f["los_x"] is not None and 305 <= f["los_x"] <= 340
    assert f["los_confidence"] >= 0.5
    assert project_along_field(100, 180, f) < project_along_field(500, 180, f)
    # scattered players -> unsure
    rng = np.random.default_rng(1)
    rand = [[x, y, x + 12, y + 26] for x, y in rng.uniform([20, 20], [600, 320], (14, 2))]
    g = estimate_field(frame, rand, {})
    assert g["los_confidence"] < f["los_confidence"]
    assert estimate_field(None, boxes, {})["los_x"] is None


def test_classify_replay_heuristic():
    from rebels_highlights.scenes.scenes import DEFAULTS, classify_scenes
    rng = np.random.default_rng(3)
    h_live = [np.abs(rng.normal(1, 0.1, 480)).astype(np.float32) for _ in range(4)]
    h_other = [np.eye(1, 480, 7, dtype=np.float32)[0] for _ in range(4)]

    def st(green, hist, change=3.0, dup=0, pairs=10):
        return {"green": [green] * 4, "hist": hist, "dup": dup, "pairs": pairs,
                "change": [change] * 4}

    spans = [(0, 30), (30, 36), (36, 37), (37, 60), (60, 66)]
    stats = [st(0.7, h_live),                   # live
             st(0.7, h_live),                   # short, similar, then a wipe -> replay
             st(0.1, h_other),                  # wipe / graphic
             st(0.7, h_live),                   # live (long)
             st(0.6, h_other, dup=6, pairs=10)]  # slow-motion signature -> replay
    assert classify_scenes(spans, stats, DEFAULTS) == [
        "live", "replay", "non_play", "live", "replay"]
