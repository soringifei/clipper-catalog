"""Synthetic tests for events (features + rules) and scoring."""
from __future__ import annotations

import copy

import numpy as np
import pytest

from rebels_highlights.core.models import Game, Play, PlayerTrajectory, VideoInfo, PLAY_TYPES
from rebels_highlights.events.events import classify_event, rule_scores
from rebels_highlights.scoring.scoring import build_candidate, dedupe_candidates

FPS = 15
H, W = 100.0, 40.0
Y = 600.0

CFG = {
    "root": "/nonexistent_root_for_tests",
    "paths": {"cache": "cache"},
    "player": {"player_name": "unknown", "team_name": "Bucharest Rebels", "player_number": 52,
               "position_short": "MLB"},
    "identity": {"auto_accept": 0.80, "review_min": 0.55, "min_ocr_votes": 3},
    "events": {"auto_label_min": 0.70, "review_below": 0.65, "forced_fumble_auto": 0.85,
               "sack_auto": 0.80, "tfl_auto": 0.80, "los_min_confidence": 0.6,
               "contact_iou": 0.05, "contact_dist_ratio": 0.6},
    "scoring": {
        "tiers": {"S": ["forced_fumble", "fumble_recovery", "sack"],
                  "A": ["tackle_for_loss", "open_field_tackle", "special_teams_tackle"],
                  "B": ["run_stop_near_los", "qb_pressure", "block_shed", "solo_tackle"],
                  "C": ["assisted_tackle", "pursuit", "coverage_play", "kickoff_coverage",
                        "punt_coverage", "tackle"]},
        "weights": {"primary_actor": 0.30, "football_impact": 0.25, "visibility": 0.15,
                    "visual_quality": 0.10, "editability": 0.10, "identity": 0.10}},
    "render": {"lead_in_s": 2.5, "lead_out_s": 1.5},
}


def box(x, y=Y, h=H, w=W):
    return [x - w / 2, y - h / 2, x + w / 2, y + h / 2]


def times(t0=0.0, t1=6.0):
    return list(np.round(np.arange(t0, t1 + 1e-9, 1 / FPS), 4))


def lerp_path(ts, keys):
    """keys: list of (t, x, y) -> positions at ts."""
    kt = [k[0] for k in keys]
    return [(float(np.interp(t, kt, [k[1] for k in keys])),
             float(np.interp(t, kt, [k[2] for k in keys]))) for t in ts]


def make_scene(contact_x=1100.0, contact_t=2.5, tackle=True, target_role="ball_carrier",
               extra_tracks=(), los_x=1000.0, los_conf=0.9):
    """#52 starts deep on the left (defense), offense on the right of the LOS."""
    ts = times()
    p52 = lerp_path(ts, [(0, 700, Y), (1.0, 700, Y), (contact_t, contact_x - 25, Y),
                         (6.0, contact_x - 20, Y)])
    if tackle:
        pt = lerp_path(ts, [(0, 1200, Y), (1.0, 1200, Y), (contact_t, contact_x + 5, Y),
                            (6.0, contact_x + 10, Y)])
    else:
        pt = lerp_path(ts, [(0, 1200, Y), (1.0, 1200, Y), (6.0, 1200 + 0, 300)])
    b52 = [box(x, y) for x, y in p52]
    bt = []
    for t, (x, y) in zip(ts, pt):
        if tackle and t > contact_t + 0.3:
            bt.append(box(x, y + 30, h=40, w=100))   # carrier on the ground
        else:
            bt.append(box(x, y))
    traj = PlayerTrajectory(play_id="g1_p1", track_id=1, times=ts, boxes=b52,
                            target_times=ts, target_boxes=bt, target_role=target_role)
    tracks = [
        {"track_id": 1, "times": ts, "boxes": b52, "team_prob": 0.9},
        {"track_id": 2, "times": ts, "boxes": bt, "team_prob": 0.1},
        # offensive linemen at snap, far from the action vertically
        {"track_id": 3, "times": ts, "boxes": [box(1040, 150)] * len(ts), "team_prob": 0.1},
        {"track_id": 4, "times": ts, "boxes": [box(1060, 1000)] * len(ts), "team_prob": 0.15},
    ] + list(extra_tracks)
    tracking = {"tracks": tracks, "fps": FPS, "frame_size": [1920, 1080]}
    field = {"los_x": los_x, "los_confidence": los_conf, "yard_lines": [], "homography": None}
    play = Play(play_id="g1_p1", game_id="g1", start_s=0.0, end_s=6.0, estimated_snap_s=1.0,
                snap_confidence=0.8, unit="defense", unit_confidence=0.8)
    return play, traj, tracking, field


def test_all_types_scored():
    play, traj, tracking, field = make_scene()
    ev = classify_event(play, traj, tracking, field, CFG)
    for k in PLAY_TYPES:
        assert f"score_{k}" in ev.features
    assert all(isinstance(v, float) for v in ev.features.values())


def test_a_clean_solo_tfl_committed():
    play, traj, tracking, field = make_scene()
    ev = classify_event(play, traj, tracking, field, CFG)
    assert ev.features["has_contact"] == 1.0
    assert ev.features["contact_gain_h"] < -0.3
    assert ev.play_type == "tackle_for_loss", (ev.play_type, ev.features)
    assert ev.event_confidence >= 0.80
    assert "UNCERTAIN_LOS" not in ev.review_reasons
    assert ev.primary_actor_score >= 0.5
    assert ev.impact_s == pytest.approx(2.5, abs=0.3)
    assert ev.contact_point is not None


def test_b_low_los_confidence_tackle_possible_tfl():
    play, traj, tracking, field = make_scene(los_conf=0.3)
    ev = classify_event(play, traj, tracking, field, CFG)
    assert ev.play_type == "tackle"
    assert "tackle_for_loss" in ev.possible_play_types
    assert "UNCERTAIN_LOS" in ev.review_reasons


def test_c_pile_without_ball_evidence_never_forced_fumble():
    pile = [{"track_id": 10 + i, "times": times(), "team_prob": tp,
             "boxes": [box(1100 + dx, Y + dy)] * len(times())}
            for i, (dx, dy, tp) in enumerate([(30, 40, 0.9), (-30, -40, 0.1), (50, -30, 0.9),
                                              (-50, 50, 0.1), (10, 70, 0.9)])]
    play, traj, tracking, field = make_scene(extra_tracks=pile)
    traj.pose_features = {"fall": 1.0, "deceleration": 1.0}
    ev = classify_event(play, traj, tracking, field, CFG)
    assert ev.play_type != "forced_fumble"
    assert "forced_fumble" not in ev.possible_play_types
    assert ev.features["pile_size"] >= 4
    # a ball track that stays with the carrier is not a fumble either
    ts = times()
    ball = {"track_id": 99, "role": "ball", "times": ts, "team_prob": 0.5,
            "boxes": [[b[0] + 10, b[1] + 30, b[0] + 25, b[1] + 40] for b in traj.target_boxes]}
    tracking["tracks"].append(ball)
    ev2 = classify_event(play, traj, tracking, field, CFG)
    assert ev2.play_type != "forced_fumble"


def test_c2_forced_fumble_needs_ball_separation():
    play, traj, tracking, field = make_scene()
    ts = times()
    bxs = []
    for t, b in zip(ts, traj.target_boxes):
        if t <= 2.5:
            bxs.append([b[0] + 10, b[1] + 30, b[0] + 25, b[1] + 40])
        else:  # ball squirts away after contact
            dx = (t - 2.5) * 400
            bxs.append([1150 + dx, 620, 1165 + dx, 630])
    tracking["tracks"].append({"track_id": 99, "role": "ball", "times": ts, "boxes": bxs,
                               "team_prob": 0.5})
    ev = classify_event(play, traj, tracking, field, CFG)
    assert ev.features["target_separation_event"] == 1.0
    assert ev.play_type == "forced_fumble"


def test_d_assisted_tackle():
    ts = times()
    mate_path = lerp_path(ts, [(0, 800, 300), (1.0, 800, 300), (2.5, 1110, Y - 40),
                               (6.0, 1112, Y - 40)])
    mate = {"track_id": 20, "times": ts, "team_prob": 0.85,
            "boxes": [box(x, y) for x, y in mate_path]}
    play, traj, tracking, field = make_scene(contact_x=1300, extra_tracks=[mate])
    # move the mate to the new contact point
    mate_path = lerp_path(ts, [(0, 800, 300), (1.0, 800, 300), (2.6, 1310, Y - 40),
                               (6.0, 1312, Y - 40)])
    mate["boxes"] = [box(x, y) for x, y in mate_path]
    ev = classify_event(play, traj, tracking, field, CFG)
    assert ev.features["defenders_at_contact"] >= 1
    assert ev.features["score_assisted_tackle"] > ev.features["score_solo_tackle"]
    assert ev.play_type in ("assisted_tackle", "tackle_for_loss", "tackle")
    assert ev.play_type != "solo_tackle"


def test_e_pursuit_without_contact():
    play, traj, tracking, field = make_scene(tackle=False)
    # #52 chases hard but the target outruns him (never within contact range)
    ts = traj.times
    tgt = lerp_path(ts, [(0, 1200, Y), (1.0, 1200, Y), (6.0, 1800, 300)])
    traj.target_boxes = [box(x, y) for x, y in tgt]
    p52 = lerp_path(ts, [(0, 700, Y), (1.0, 700, Y), (6.0, 1650, 380)])
    traj.boxes = [box(x, y) for x, y in p52]
    tracking["tracks"][0]["boxes"] = traj.boxes
    tracking["tracks"][1]["boxes"] = traj.target_boxes
    ev = classify_event(play, traj, tracking, field, CFG)
    assert ev.features["has_contact"] == 0.0
    assert ev.play_type == "pursuit", (ev.play_type, ev.features.get("score_pursuit"))
    assert not any("tackle" in p for p in [ev.play_type])


# ------------------------------------------------------------------ scoring
def _identity(conf, votes=5, traj=None):
    return {"best_track_id": 1, "identity_confidence": conf,
            "evidence": [{"track_id": 1, "scene_id": 0, "jersey_confidence": 0.9,
                          "team_confidence": 0.95, "reid_confidence": 0.8,
                          "identity_confidence": conf, "ocr_votes": {"52": votes}}],
            "trajectory": traj.to_dict() if traj else {}}


def _cand(id_conf, los_conf=0.9, play_id="g1_p1", start=0.0):
    play, traj, tracking, field = make_scene(los_conf=los_conf)
    play = copy.deepcopy(play)
    play.play_id = play_id
    ev = classify_event(play, traj, tracking, field, CFG)
    game = Game(game_id="g1", url="https://example.invalid/v", date="unknown", opponent="unknown")
    info = VideoInfo(game_id="g1", path="x.mp4", source_url="https://example.invalid/v",
                     source_video_id="vid", width=1920, height=1080)
    return build_candidate(game, info, play, _identity(id_conf, traj=traj), ev, CFG), ev


def test_f_scoring_statuses_and_captions():
    c, ev = _cand(0.92)
    assert c.review_status == "AUTO_APPROVED" and c.manual_review is False
    assert c.caption == "#52 | MLB | BUCHAREST REBELS — TFL"
    assert c.tier == "A"
    assert c.date == "unknown" and c.opponent == "unknown"
    assert c.clip_id == "g1_p1"
    assert c.source_start_s == pytest.approx(0.0)  # snap 1.0 - 2.5 clipped to play start
    assert c.source_end_s == pytest.approx(ev.impact_s + 1.5, abs=1e-3)
    for k in ("identity_confidence", "event_confidence", "primary_actor_score",
              "visibility_score", "visual_quality_score", "football_impact_score",
              "editability_score", "overall_highlight_score"):
        assert 0.0 <= getattr(c, k) <= 1.0
    c2, _ = _cand(0.70)
    assert c2.review_status == "REVIEW" and "IDENTITY_LOW" in c2.review_reasons
    c3, _ = _cand(0.40)
    assert c3.review_status == "REJECTED"
    # uncertain LOS -> generic label, caption must not claim TFL, needs review
    c4, _ = _cand(0.92, los_conf=0.3)
    assert "TFL" not in c4.caption
    assert c4.review_status == "REVIEW"


def test_f_caption_neutral_when_low_confidence():
    from rebels_highlights.core.models import EventResult
    from rebels_highlights.scoring.scoring import caption_for
    ev = EventResult(play_id="x", play_type="sack", event_confidence=0.5)
    assert caption_for(ev, CFG) == "#52 | MLB | BUCHAREST REBELS"
    ev = EventResult(play_id="x", play_type="unclear", event_confidence=0.9)
    assert caption_for(ev, CFG) == "#52 | MLB | BUCHAREST REBELS"


def test_f_dedupe():
    a, _ = _cand(0.92, play_id="g1_p1")
    b, _ = _cand(0.85, play_id="g1_p1r")          # replay window overlaps fully
    c, _ = _cand(0.95, play_id="g2_p1")
    c.game_id = "g2"                              # other game, never merged
    out = dedupe_candidates([a, b, c])
    assert len(out) == 3
    assert a.review_status == "AUTO_APPROVED"
    assert b.review_status == "REJECTED" and "DUPLICATE_PLAY" in b.review_reasons
    assert "DUPLICATE_PLAY" not in c.review_reasons


def test_rule_scores_cover_all_types():
    s = rule_scores({}, Play(play_id="p", game_id="g", start_s=0, end_s=1))
    assert set(s) == set(PLAY_TYPES)
