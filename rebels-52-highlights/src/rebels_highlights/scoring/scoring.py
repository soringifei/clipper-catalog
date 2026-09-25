"""Candidate building + scoring + dedupe.

Each metric is kept separate on the Candidate (identity, event, primary actor,
visibility, visual quality, football impact, editability); only
``overall_highlight_score`` combines them, with the weights in
``cfg['scoring']['weights']``.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ..core.models import UNKNOWN, Candidate, EventResult, Game, Play, VideoInfo

TIER_SCORES = {"S": 1.0, "A": 0.8, "B": 0.6, "C": 0.4, "D": 0.2}
GENERIC_LABELS = {"tackle", "pursuit", "unclear", "other", "other_defense", "other_special_teams"}
BLOCKING_REASONS = {
    "IDENTITY_LOW", "TRACK_ID_SWITCH", "EVENT_AMBIGUOUS", "POSSIBLE_FORCED_FUMBLE",
    "UNCERTAIN_LOS", "DUPLICATE_PLAY", "PLAYER_NOT_PRIMARY_ACTOR", "BALL_NOT_CLEARLY_VISIBLE",
}
CAPTIONS = {
    "tackle_for_loss": "TFL", "sack": "SACK", "special_teams_tackle": "SPECIAL TEAMS TACKLE",
    "open_field_tackle": "OPEN-FIELD TACKLE", "run_stop_near_los": "RUN STOP",
    "forced_fumble": "FORCED FUMBLE", "fumble_recovery": "FUMBLE RECOVERY",
    "qb_pressure": "QB PRESSURE", "block_shed": "BLOCK SHED", "solo_tackle": "TACKLE",
    "assisted_tackle": "TACKLE", "tackle": "TACKLE", "pursuit": "PURSUIT",
    "coverage_play": "COVERAGE", "kickoff_coverage": "KICKOFF COVERAGE",
    "punt_coverage": "PUNT COVERAGE",
}


def _c(v: float) -> float:
    return float(min(1.0, max(0.0, v))) if v == v else 0.0


def tier_for(play_type: str, cfg: dict) -> str:
    for tier, labels in (cfg.get("scoring", {}).get("tiers") or {}).items():
        if play_type in (labels or []):
            return str(tier)
    return "D"


def caption_for(event: EventResult, cfg: dict) -> str:
    p = cfg.get("player", {})
    num = p.get("player_number", 52)
    pos = p.get("position_short", "MLB")
    team = str(p.get("team_name", "Bucharest Rebels")).upper()
    base = f"#{num} | {pos} | {team}"
    min_c = float(cfg.get("events", {}).get("auto_label_min", 0.70))
    label = CAPTIONS.get(event.play_type)
    if not label or event.event_confidence < min_c:
        return base
    f = event.features or {}
    if event.play_type == "run_stop_near_los" and f.get("read_downhill", 0.0) >= 0.5:
        label = "READ → DOWNHILL → FINISH"
    elif event.play_type in ("solo_tackle", "tackle", "assisted_tackle") \
            and f.get("path_length_h", 0.0) >= 5.0 and f.get("has_contact"):
        label = "PURSUIT → CLOSE → TACKLE"
    return f"{base} — {label}"


def _best_evidence(identity: dict) -> dict:
    ev = identity.get("evidence") or []
    best = identity.get("best_track_id")
    match = [e for e in ev if e.get("track_id") == best]
    pool = match or ev
    return max(pool, key=lambda e: e.get("identity_confidence", 0.0)) if pool else {}


def _visibility(play: Play, times: list, boxes: list) -> tuple[float, float]:
    """(fraction of play time #52 is tracked, median box height px)."""
    if not times:
        return 0.0, 0.0
    t = np.sort(np.asarray(times, dtype=float))
    dur = max(1e-3, play.end_s - play.start_s)
    t = t[(t >= play.start_s - 0.1) & (t <= play.end_s + 0.1)]
    if len(t) == 0:
        return 0.0, 0.0
    gaps = np.diff(t)
    step = float(np.median(gaps)) if len(gaps) else 0.1
    covered = float(np.sum(np.minimum(gaps, 0.5)[gaps <= 0.5])) + step
    b = np.asarray(boxes, dtype=float).reshape(-1, 4)
    h = float(np.median(b[:, 3] - b[:, 1])) if len(b) else 0.0
    return _c(covered / dur), h


def _box_near(times: list, boxes: list, t: Optional[float], tol: float = 0.3):
    if not times or t is None:
        return None
    ts = np.asarray(times, dtype=float)
    i = int(np.argmin(np.abs(ts - t)))
    return boxes[i] if abs(ts[i] - t) <= tol else None


def build_candidate(game: Game, info: VideoInfo, play: Play, identity: dict,
                    event: EventResult, cfg: dict) -> Candidate:
    identity = identity or {}
    pcfg = cfg.get("player", {})
    icfg = cfg.get("identity", {})
    ecfg = cfg.get("events", {})
    rcfg = cfg.get("render", {})
    weights = cfg.get("scoring", {}).get("weights", {})
    tier_scores = {**TIER_SCORES, **(cfg.get("scoring", {}).get("tier_scores") or {})}
    reasons: list[str] = []

    best = _best_evidence(identity)
    id_conf = float(identity.get("identity_confidence", best.get("identity_confidence", 0.0)) or 0.0)
    traj = identity.get("trajectory") or {}
    times, boxes = traj.get("times") or [], traj.get("boxes") or []

    # --- source window
    lead_in = float(rcfg.get("lead_in_s", 2.5))
    lead_out = float(rcfg.get("lead_out_s", 1.5))
    snap = play.estimated_snap_s
    impact = event.impact_s
    anchor = snap if snap is not None else impact
    start = max(play.start_s, anchor - lead_in) if anchor is not None else play.start_s
    end = min(play.end_s, impact + lead_out) if impact is not None else play.end_s
    if end <= start:
        start, end = play.start_s, play.end_s

    # --- metrics
    vis_frac, h_px = _visibility(play, times, boxes)
    visibility = _c(vis_frac * (0.7 + 0.3 * _c(h_px / 80.0)))
    visual_quality = _c(h_px / 120.0)
    tier = tier_for(event.play_type, cfg)
    impact_score = float(tier_scores.get(tier, 0.2))
    if event.play_type not in GENERIC_LABELS:
        impact_score *= _c(event.event_confidence)
    lead_ok = _c((anchor - play.start_s) / lead_in) if (anchor is not None and lead_in > 0) else 0.5
    frame_h = float(info.height or 0)
    co_vis = 0.7
    b52 = _box_near(times, boxes, impact)
    btg = _box_near(traj.get("target_times") or [], traj.get("target_boxes") or [], impact)
    if frame_h > 0 and b52 is not None:
        crop_w = frame_h * 9.0 / 16.0
        if btg is not None:
            spread = max(b52[2], btg[2]) - min(b52[0], btg[0])
        else:
            spread = b52[2] - b52[0]
        co_vis = _c(crop_w / spread) if spread > crop_w else 1.0
    editability = _c(0.5 * lead_ok + 0.5 * co_vis)
    if co_vis < 0.6:
        reasons.append("POOR_VERTICAL_CROP")
    if visual_quality < 0.4:
        reasons.append("LOW_RESOLUTION")
    if visibility < 0.4:
        reasons.append("OCCLUSION")

    ocr_votes = dict(best.get("ocr_votes") or {})
    if ocr_votes.get(str(pcfg.get("player_number", 52)), 0) < int(icfg.get("min_ocr_votes", 3)):
        reasons.append("JERSEY_UNREADABLE")
    if id_conf < float(icfg.get("auto_accept", 0.80)):
        reasons.append("IDENTITY_LOW")
    reasons += list(best.get("review_reasons") or [])
    reasons += list(event.review_reasons or [])
    reasons = [r for r in dict.fromkeys(reasons)]

    metrics = {
        "primary_actor": _c(event.primary_actor_score), "football_impact": impact_score,
        "visibility": visibility, "visual_quality": visual_quality,
        "editability": editability, "identity": _c(id_conf),
    }
    wsum = sum(float(w) for w in weights.values()) or 1.0
    overall = sum(float(weights.get(k, 0.0)) * v for k, v in metrics.items()) / wsum

    # --- review status
    blocking = [r for r in reasons if r in BLOCKING_REASONS]
    ev_ok = event.event_confidence >= float(ecfg.get("review_below", 0.65)) and event.play_type != "unclear"
    if id_conf < float(icfg.get("review_min", 0.55)):
        status = "REJECTED"
    elif id_conf >= float(icfg.get("auto_accept", 0.80)) and ev_ok and not blocking:
        status = "AUTO_APPROVED"
    else:
        status = "REVIEW"

    cand = Candidate(
        clip_id=play.play_id, game_id=game.game_id, play_id=play.play_id,
        source_url=info.source_url or game.url, source_video_id=info.source_video_id,
        date=game.date or UNKNOWN, opponent=game.opponent or UNKNOWN,
        player_name=str(pcfg.get("player_name", UNKNOWN) or UNKNOWN),
        player_number=int(pcfg.get("player_number", 52)),
        position=str(pcfg.get("position_short", "MLB")), unit=play.unit,
        play_type=event.play_type, possible_play_types=list(event.possible_play_types),
        source_start_s=round(float(start), 3), snap_time_s=snap, impact_time_s=impact,
        source_end_s=round(float(end), 3),
        output_duration_s=round(float(end - start), 3),
        track_id=identity.get("best_track_id", traj.get("track_id")),
        jersey_confidence=float(best.get("jersey_confidence", 0.0)),
        team_confidence=float(best.get("team_confidence", 0.0)),
        reid_confidence=float(best.get("reid_confidence", 0.0)),
        identity_confidence=round(id_conf, 4), event_confidence=round(float(event.event_confidence), 4),
        primary_actor_score=round(metrics["primary_actor"], 4), visibility_score=round(visibility, 4),
        visual_quality_score=round(visual_quality, 4), football_impact_score=round(impact_score, 4),
        editability_score=round(editability, 4), overall_highlight_score=round(overall, 4),
        tier=tier, manual_review=status != "AUTO_APPROVED", review_status=status,
        review_reasons=reasons, ocr_votes=ocr_votes, caption=caption_for(event, cfg),
        replay_segments=[dict(s) for s in (play.replay_segments or [])],
    )
    # non-serialised hint for dedupe (asdict ignores dynamic attributes)
    cand.canonical_play_id = play.canonical_play_id or play.play_id  # type: ignore[attr-defined]
    return cand


def _overlap_frac(a: Candidate, b: Candidate) -> float:
    inter = min(a.source_end_s, b.source_end_s) - max(a.source_start_s, b.source_start_s)
    shortest = min(a.source_end_s - a.source_start_s, b.source_end_s - b.source_start_s)
    return inter / shortest if shortest > 0 and inter > 0 else 0.0


def _canon(c: Candidate) -> str:
    return getattr(c, "canonical_play_id", None) or c.play_id


def dedupe_candidates(cands: list[Candidate]) -> list[Candidate]:
    """Merge duplicates of the same play within a game; returns ALL candidates
    (merged-away ones REJECTED with DUPLICATE_PLAY) in the original order."""
    n = len(cands)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            a, b = cands[i], cands[j]
            if a.game_id != b.game_id:
                continue
            if _canon(a) == _canon(b) or a.play_id == _canon(b) or b.play_id == _canon(a) \
                    or _overlap_frac(a, b) > 0.5:
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    for idx in groups.values():
        if len(idx) < 2:
            continue
        keep = max(idx, key=lambda i: (cands[i].identity_confidence,
                                       cands[i].overall_highlight_score))
        k = cands[keep]
        for i in idx:
            if i == keep:
                continue
            c = cands[i]
            if "DUPLICATE_PLAY" not in c.review_reasons:
                c.review_reasons.append("DUPLICATE_PLAY")
            c.review_status = "REJECTED"
            c.manual_review = False
            for seg in c.replay_segments:
                if seg not in k.replay_segments:
                    k.replay_segments.append(dict(seg))
    return cands
