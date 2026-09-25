"""Synthetic-footage tests for detection/tracking/OCR/identity (numpy + OpenCV only)."""
from __future__ import annotations

import os

import cv2
import numpy as np
import pytest

os.environ.setdefault("REBELS_OCR_BACKEND", "digits")
os.environ.setdefault("REBELS_POSE_BACKEND", "box")

from rebels_highlights.core.config import load_config  # noqa: E402
from rebels_highlights.core.models import IdentityEvidence, Play, PlayerTrajectory, Track  # noqa: E402
from rebels_highlights.detection.detector import PersonDetector, referee_score  # noqa: E402
from rebels_highlights.identity import identity as ident  # noqa: E402
from rebels_highlights.jersey_ocr.ocr import read_jersey, vote  # noqa: E402
from rebels_highlights.pose.pose import pose_features  # noqa: E402
from rebels_highlights.tracking.track_play import track_play  # noqa: E402

W, H, FPS = 640, 360, 15
REBELS = (40, 30, 170)     # BGR dark red
OPP = (190, 110, 20)       # BGR blue
PANTS = (210, 210, 210)
PW, PH = 44, 100

# number, colour, start (x, y) of box top-left, velocity px/frame
PLAYERS = [
    ("52", REBELS, (400, 120), (-1.2, 0.3)),
    ("32", REBELS, (470, 230), (-0.8, -0.2)),
    ("90", REBELS, (520, 20), (-0.9, 0.1)),
    ("17", OPP, (150, 60), (1.0, 0.2)),
    ("88", OPP, (220, 200), (0.6, -0.3)),
    ("45", OPP, (60, 230), (1.1, 0.0)),
]


def draw_player(img, x, y, number, colour, w=PW, h=PH):
    x, y = int(round(x)), int(round(y))
    cv2.circle(img, (x + w // 2, y + 9), 9, (90, 120, 170), -1)            # head
    cv2.rectangle(img, (x, y + 18), (x + w, y + int(h * 0.62)), colour, -1)  # jersey
    cv2.rectangle(img, (x + 3, y + int(h * 0.62)), (x + w - 3, y + h), PANTS, -1)
    if number:
        font, sc, th = cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2
        (tw, tht), _ = cv2.getTextSize(number, font, sc, th)
        cv2.putText(img, number, (x + (w - tw) // 2, y + 24 + tht), font, sc, (255, 255, 255), th)


def field(shift=0.0, shade=(40, 150, 45)):
    img = np.full((H, W, 3), shade, np.uint8)
    for gx in range(-80, W + 80, 80):  # yard lines (move with the camera pan)
        xx = int(gx - shift) % (W + 80) - 40
        cv2.line(img, (xx, 0), (xx, H), (235, 235, 235), 2)
    return img


def player_box(p, f, pan=0.0):
    (x0, y0), (vx, vy) = p[2], p[3]
    x, y = x0 + vx * f - pan * f, y0 + vy * f
    return [x, y, x + PW, y + PH]


def render_video(path, n=45, pan=0.7, cut_at=None):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (W, H))
    assert vw.isOpened()
    for f in range(n):
        if cut_at is not None and f >= cut_at:  # different camera: bluish crowd shot
            img = np.full((H, W, 3), (120, 60, 90), np.uint8)
            cv2.rectangle(img, (0, 120), (W, H), (60, 170, 90), -1)
            for i, p in enumerate(PLAYERS[:3]):
                draw_player(img, 100 + 150 * i, 170, p[0], p[1])
        else:
            img = field(shift=pan * f)
            for p in PLAYERS:
                b = player_box(p, f, pan)
                draw_player(img, b[0], b[1], p[0], p[1])
        vw.write(img)
    vw.release()


def make_cfg(tmp):
    cfg = load_config()
    cfg["root"] = str(tmp)
    cfg["detection"]["backend"] = "opencv"
    cfg["detection"]["fine_fps"] = FPS
    cfg["tracking"]["tracker"] = "iou"
    return cfg


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("synth")
    vid = tmp / "game.avi"
    render_video(vid)
    cfg = make_cfg(tmp)
    play = Play(play_id="g_p001", game_id="g", start_s=0.0, end_s=45 / FPS, estimated_snap_s=0.2,
                scene_ids=[0], unit="defense")
    trk = track_play(str(vid), play, cfg, "cpu")
    return {"vid": str(vid), "cfg": cfg, "play": play, "trk": trk, "tmp": tmp}


def gt_track(trk, pidx, frame=20, pan=0.7):
    """Track id whose box at ``frame`` best overlaps ground-truth player ``pidx``."""
    t = frame / FPS
    gt = player_box(PLAYERS[pidx], frame, pan)
    best, bi = None, 0.0
    for tr in trk["tracks"]:
        for tt, b in zip(tr["times"], tr["boxes"]):
            if abs(tt - t) < 0.5 / FPS:
                i = ident._iou(b, gt)
                if i > bi:
                    best, bi = tr["track_id"], i
    return best


def crop_of(number, colour=REBELS):
    img = field()
    draw_player(img, 100, 100, number, colour)
    return img[100:100 + PH, 100:100 + PW].copy()


# ---------------------------------------------------------------------------
def test_ocr_reads_52_vs_32():
    r52, r32 = read_jersey(crop_of("52")), read_jersey(crop_of("32", OPP))
    assert r52 and r52[0][0] == "52" and r52[0][1] > 0.5
    assert r32 and r32[0][0] == "32" and r32[0][1] > 0.5
    for n in ("17", "88", "45"):
        assert read_jersey(crop_of(n, OPP))[0][0] == n


def test_ocr_blank_jersey_has_no_confident_reading():
    r = read_jersey(crop_of(""))
    assert not r or r[0][1] < 0.4


def test_vote_thresholds():
    one = vote([[("52", 0.9)], [], [], []], min_votes=3)
    assert one["votes"]["52"] == 1 and one["jersey_confidence"] < 0.2
    many = vote([[("52", 0.9)]] * 5 + [[]] * 2, min_votes=3)
    assert many["votes"] == {"52": 5, "unknown": 2} and many["jersey_confidence"] > 0.7
    mixed = vote([[("52", 0.9)]] * 3 + [[("32", 0.9)]] * 3, min_votes=3)
    assert mixed["jersey_confidence"] < many["jersey_confidence"] * 0.7
    other = vote([[("32", 0.9)]] * 6, min_votes=3)
    assert other["best"] == "32" and other["jersey_confidence"] == 0.0


def test_detector_fallback_and_referee_flag():
    img = field()
    for p in PLAYERS:
        draw_player(img, *p[2], p[0], p[1])
    det = PersonDetector(cfg={"detection": {}}, backend="opencv")
    assert det.backend == "opencv"
    dets = det.detect(img)
    assert len([d for d in dets if not d["referee"] and not d["sideline"]]) == len(PLAYERS)
    ref = np.full((PH, PW, 3), 255, np.uint8)
    for x in range(0, PW, 8):
        ref[:, x:x + 4] = 15
    assert referee_score(ref) >= 0.5
    assert referee_score(crop_of("52")) < 0.5


def test_tracker_keeps_ids(synth):
    trk = synth["trk"]
    assert trk["tracker"] == "iou" and trk["frame_size"] == [W, H]
    assert not trk["scene_cuts"]
    for tr in trk["tracks"]:
        Track.from_dict(tr)
    for pidx in range(len(PLAYERS)):
        ids = {gt_track(trk, pidx, f) for f in (2, 15, 30, 42)}
        assert len(ids) == 1 and None not in ids, (PLAYERS[pidx][0], ids)
    # the two teams end up in different colour clusters
    c52 = next(t for t in trk["tracks"] if t["track_id"] == gt_track(trk, 0))["team_cluster"]
    c17 = next(t for t in trk["tracks"] if t["track_id"] == gt_track(trk, 3))["team_cluster"]
    assert c52 != c17


def test_track_ids_never_cross_hard_cut(tmp_path):
    vid = tmp_path / "cut.avi"
    render_video(vid, n=30, cut_at=15)
    cfg = make_cfg(tmp_path)
    play = Play(play_id="g_p002", game_id="g", start_s=0.0, end_s=2.0, scene_ids=[3, 4])
    trk = track_play(str(vid), play, cfg, "cpu")
    assert len(trk["scene_cuts"]) == 1 and abs(trk["scene_cuts"][0] - 1.0) < 0.1
    for tr in trk["tracks"]:
        assert all(t < 1.0 for t in tr["times"]) or all(t >= 1.0 - 1e-6 for t in tr["times"])
    assert {tr["scene_id"] for tr in trk["tracks"]} == {3, 4}


def test_identity_picks_52(synth):
    res = ident.identify_player(synth["vid"], synth["play"], synth["trk"], synth["cfg"], None)
    tid52 = gt_track(synth["trk"], 0)
    assert res["best_track_id"] == tid52
    ev = {e["track_id"]: e for e in res["evidence"]}
    IdentityEvidence.from_dict(ev[tid52])
    assert ev[tid52]["ocr_votes"].get("52", 0) >= synth["cfg"]["identity"]["min_ocr_votes"]
    assert res["identity_confidence"] > synth["cfg"]["identity"]["review_min"]
    assert "IDENTITY_LOW" not in ev[tid52]["review_reasons"]
    tid32 = gt_track(synth["trk"], 1)
    assert ev[tid32]["identity_confidence"] < synth["cfg"]["identity"]["review_min"]
    assert ev[tid32]["ocr_votes"].get("32", 0) >= 3
    # teams resolved from the #52 reads: Rebels high, opponents low
    assert res["team_probs"][tid52] > 0.7 and res["team_probs"][gt_track(synth["trk"], 3)] < 0.3
    traj = PlayerTrajectory.from_dict(res["trajectory"])
    assert traj.track_id == tid52 and len(traj.times) >= 40
    assert traj.target_role in ("ball_carrier", "qb") and traj.target_boxes
    assert {"stance", "acceleration", "deceleration", "direction_change", "fall"} <= set(traj.pose_features)


def test_identity_single_frame_never_suffices(synth):
    cfg = synth["cfg"]
    tid52 = gt_track(synth["trk"], 0)
    tr = next(t for t in synth["trk"]["tracks"] if t["track_id"] == tid52)
    one = {"tracks": [dict(tr, times=tr["times"][:1], boxes=tr["boxes"][:1])], "fps": FPS}
    res = ident.identify_player(synth["vid"], synth["play"], one, cfg, None)
    assert res["identity_confidence"] < cfg["identity"]["review_min"]


def test_manual_seed(synth):
    tid32 = gt_track(synth["trk"], 1)
    seed = {"t": 20 / FPS, "box": player_box(PLAYERS[1], 20, 0.7)}
    res = ident.identify_player(synth["vid"], synth["play"], synth["trk"], synth["cfg"], None, seed=seed)
    assert res["best_track_id"] == tid32
    ev = next(e for e in res["evidence"] if e["track_id"] == tid32)
    assert ev["source"] == "manual_seed" and res["identity_confidence"] >= 0.9


def test_seed_from_review_and_references(tmp_path):
    from rebels_highlights.core.store import Store
    cfg = make_cfg(tmp_path)
    store = Store(cfg)
    (store.review / "seeds.json").write_text('{"g_p001": {"t": 1.5, "box": [1, 2, 30, 90]}}')
    assert ident.seed_from_review(store, "g_p001") == {"t": 1.5, "box": [1.0, 2.0, 30.0, 90.0]}
    assert ident.seed_from_review(store, "nope") is None
    assert ident.load_references(str(tmp_path / "empty_refs"), cfg) is None
    refs = tmp_path / "refs"
    refs.mkdir()
    cv2.imwrite(str(refs / "a.png"), crop_of("52"))
    r = ident.load_references(str(refs), cfg)
    assert r is not None and len(r["prototypes"]) == 1


def test_pose_fall_from_box_aspect():
    times = [i / 10 for i in range(20)]
    boxes = [[100 + 5 * i, 100, 140 + 5 * i, 200] if i < 10 else [150, 170, 250, 210] for i in range(20)]
    f = pose_features(None, boxes, times)
    assert f["fall"] > 0.6 and 0.9 <= f["fall_t"] <= 1.1
    assert pose_features(None, boxes[:9], times[:9])["fall"] == 0.0
