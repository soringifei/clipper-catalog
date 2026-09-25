"""Event classification for #52 (stage 1 rules + optional stage-2 blend).

Every type in ``PLAY_TYPES`` gets a scored hypothesis. A specific label is only
committed when its confidence clears its threshold (``events.auto_label_min``,
stricter ``sack_auto``/``tfl_auto``/``forced_fumble_auto``) and its evidence
gates pass (LOS confidence for LOS-dependent labels; contact + ball possession
disruption for forced fumbles). Otherwise a safe generic label is committed
('tackle', 'pursuit' or 'unclear') and the specific candidates are listed in
``possible_play_types`` with review reasons.
"""
from __future__ import annotations

from typing import Optional

from ..core.models import PLAY_TYPES, EventResult, Play, PlayerTrajectory
from . import temporal_model
from .features import compute_features, isfinite_dict

GENERIC = {"tackle", "pursuit", "unclear", "other", "other_defense", "other_special_teams"}
LOS_DEPENDENT = {"tackle_for_loss", "sack", "run_stop_near_los"}
FUMBLE_TYPES = {"forced_fumble", "fumble_recovery"}
POSSIBLE_MIN = 0.30
PRIORITY = [
    "forced_fumble", "fumble_recovery", "sack", "tackle_for_loss", "special_teams_tackle",
    "open_field_tackle", "run_stop_near_los", "block_shed", "qb_pressure", "solo_tackle",
    "assisted_tackle", "kickoff_coverage", "punt_coverage", "return_block", "coverage_play",
    "pursuit", "other_defense", "other_special_teams", "tackle", "other", "unclear",
]


def _c(v: float) -> float:
    return float(min(1.0, max(0.0, v)))


def rule_scores(f: dict[str, float], play: Play) -> dict[str, float]:
    """Stage-1 hypothesis confidences in [0, 1] for every PLAY_TYPE."""
    g = lambda k, d=0.0: float(f.get(k, d) or 0.0)  # noqa: E731
    contact = g("has_contact")
    down = max(g("target_down"), 0.8 * g("target_decel"), 0.8 * g("pose_fall"))
    tackle = contact * (0.5 + 0.5 * down)
    n_def = g("defenders_at_contact")
    solo = 1.0 if n_def == 0 else (0.35 if n_def == 1 else 0.1)
    los_c = g("los_confidence") if g("los_known") else 0.0
    gain = g("contact_gain_h")
    behind = _c((-gain - 0.1) / 0.6) if g("los_known") else 0.0
    near = _c(1.0 - (abs(gain) - 0.5) / 1.5) if g("los_known") else 0.0
    beyond = _c((gain - 1.5) / 2.0) if g("los_known") else 0.5
    qb = g("target_is_qb")
    st = g("special_teams_signature")
    dmin = g("min_distance_h", 99.0)
    path = g("path_length_h")
    close = g("closing_speed_h")
    los_fac = 0.8 + 0.2 * los_c
    pursuit = _c(path / 5.0) * _c(0.4 + close / 2.0) * _c(0.5 + 0.5 * max(0.0, g("pursuit_angle_cos")))
    s = {k: 0.0 for k in PLAY_TYPES}
    s["tackle"] = tackle
    s["solo_tackle"] = tackle * solo * (0.7 + 0.3 * g("first_to_contact")) * (1 - 0.5 * st)
    s["assisted_tackle"] = tackle * (min(1.0, 0.55 + 0.2 * n_def) if n_def >= 1 else 0.1) * (1 - 0.5 * st)
    s["open_field_tackle"] = tackle * g("open_field") * (0.3 + 0.7 * beyond) * (1 - 0.5 * st)
    s["tackle_for_loss"] = tackle * behind * (1 - qb) * los_fac * (1 - 0.7 * st)
    s["sack"] = tackle * qb * max(behind, 0.6 if g("los_known") else 0.0) * los_fac
    s["run_stop_near_los"] = tackle * near * (1 - behind) * (1 - qb) * los_fac * (1 - 0.7 * st)
    s["qb_pressure"] = qb * g("qb_pressure") * (1 - 0.5 * contact * down) * _c(1.6 - dmin / 1.5)
    s["forced_fumble"] = contact * g("target_separation_event")
    s["fumble_recovery"] = g("ball_recovered_by_52")
    s["block_shed"] = g("block_shed") * (0.6 + 0.4 * contact)
    s["pursuit"] = pursuit * (1 - 0.6 * contact)
    s["coverage_play"] = g("coverage_drop") * (1 - contact) * 0.7
    s["other_defense"] = 0.2 * (1 - st)
    s["special_teams_tackle"] = st * tackle
    cov_st = st * (1 - contact) * _c(path / 5.0) * 0.6
    s["kickoff_coverage"] = cov_st
    s["punt_coverage"] = cov_st
    s["return_block"] = st * g("opponents_at_contact") * (1 - contact) * 0.3
    s["other_special_teams"] = 0.3 * st
    s["other"] = 0.1
    s["unclear"] = 0.15
    return {k: round(_c(v), 4) for k, v in s.items()}


def _threshold(label: str, ev: dict) -> float:
    base = float(ev.get("auto_label_min", 0.70))
    return {"sack": float(ev.get("sack_auto", 0.80)),
            "tackle_for_loss": float(ev.get("tfl_auto", 0.80)),
            "forced_fumble": float(ev.get("forced_fumble_auto", 0.85)),
            "fumble_recovery": float(ev.get("forced_fumble_auto", 0.85))}.get(label, base)


def _gates_ok(label: str, f: dict[str, float], los_ok: bool) -> bool:
    if label in LOS_DEPENDENT and not los_ok:
        return False
    if label in FUMBLE_TYPES:
        # never without contact AND possession-disruption evidence from a ball track
        return bool(f.get("has_contact")) and bool(f.get("ball_track")) \
            and f.get("target_separation_event", 0.0) >= 0.5
    return True


def primary_actor(f: dict[str, float]) -> float:
    n_def = float(f.get("defenders_at_contact", 0.0))
    if f.get("has_contact"):
        return _c(0.4 * f.get("first_to_contact", 0.0) + 0.3 / (1.0 + n_def)
                  + 0.3 * f.get("closing_share", 0.0))
    # no contact: pursuit/coverage involvement only
    prox = _c(1.5 - f.get("min_distance_h", 9.0) / 3.0)
    return _c(0.5 * prox + 0.5 * f.get("closing_share", 0.0)) * 0.8


def classify_event(play: Play, trajectory: PlayerTrajectory, tracking: Optional[dict],
                   field: Optional[dict], cfg: dict) -> EventResult:
    ev = cfg.get("events", {})
    comp = compute_features(play, trajectory, tracking, field, cfg)
    f = isfinite_dict(comp["features"])
    res = EventResult(play_id=play.play_id, impact_s=comp["impact_s"],
                      contact_point=comp["contact_point"], los_x=comp["los_x"],
                      los_confidence=comp["los_confidence"])
    los_ok = bool(f.get("los_known")) and res.los_confidence >= float(ev.get("los_min_confidence", 0.6))

    scores = rule_scores(f, play)
    scores, blended = temporal_model.maybe_blend(scores, f, cfg)
    f["stage2_blended"] = float(blended)
    for k, v in scores.items():
        f[f"score_{k}"] = float(v)

    reasons: list[str] = []
    specific = sorted(((v, k) for k, v in scores.items() if k not in GENERIC), reverse=True)
    committed: Optional[str] = None
    conf = 0.0
    # among hypotheses above their threshold, prefer the most specific /
    # highest-impact label; if any such hypothesis fails its evidence gate
    # (uncertain LOS, no ball evidence) fall back to a safe generic label.
    passing = [(k, v) for v, k in specific if v >= _threshold(k, ev)]
    blocked = [k for k, v in passing if not _gates_ok(k, f, los_ok)]
    if passing and not blocked:
        committed, conf = min(passing, key=lambda kv: (PRIORITY.index(kv[0]), -kv[1]))
    if committed is None:
        down = max(f.get("target_down", 0.0), f.get("target_decel", 0.0))
        if f.get("has_contact") and down >= 0.5:
            committed, conf = "tackle", scores["tackle"]
        elif scores["pursuit"] >= POSSIBLE_MIN:
            committed, conf = "pursuit", scores["pursuit"]
        else:
            committed, conf = "unclear", max(scores["unclear"], scores["tackle"] * 0.5)
        if any(v >= POSSIBLE_MIN for v, k in specific) or committed == "unclear":
            reasons.append("EVENT_AMBIGUOUS")
    elif committed == "tackle_for_loss" and scores["sack"] >= _threshold("sack", ev) * 0.8 \
            or committed == "sack" and scores["tackle_for_loss"] >= _threshold("tackle_for_loss", ev) * 0.8:
        reasons.append("EVENT_AMBIGUOUS")  # QB role unsure: sack vs TFL

    possible: list[str] = []
    for v, k in specific:
        if k == committed or v < POSSIBLE_MIN:
            continue
        possible.append("possible_" + k if k in FUMBLE_TYPES else k)

    # possible forced fumble without ball-track confirmation (pose/ball hints)
    ff_hint = f.get("has_contact") and (
        (f.get("ball_track") and 0.0 < f.get("target_separation_event", 0.0) < 0.5)
        or f.get("pose_ball_loose", 0.0) > 0.5 or f.get("pose_strip", 0.0) > 0.5)
    if ff_hint and "possible_forced_fumble" not in possible and committed != "forced_fumble":
        possible.append("possible_forced_fumble")
    if "possible_forced_fumble" in possible or "possible_fumble_recovery" in possible:
        reasons.append("POSSIBLE_FORCED_FUMBLE")
        if f.get("ball_visible_frac", 0.0) < 0.5:
            reasons.append("BALL_NOT_CLEARLY_VISIBLE")

    considered_los = any(scores[k] >= POSSIBLE_MIN or _raw_los_score(k, f, play) >= POSSIBLE_MIN
                         for k in LOS_DEPENDENT)
    if not los_ok and considered_los:
        reasons.append("UNCERTAIN_LOS")
        for k in LOS_DEPENDENT:
            if _raw_los_score(k, f, play) >= POSSIBLE_MIN and k not in possible and k != committed:
                possible.append(k)

    res.primary_actor_score = round(primary_actor(f), 4)
    if res.primary_actor_score < 0.5:
        reasons.append("PLAYER_NOT_PRIMARY_ACTOR")
    res.play_type = committed
    res.event_confidence = round(float(conf), 4)
    res.possible_play_types = possible
    res.features = f
    res.review_reasons = list(dict.fromkeys(reasons))
    return res


def _raw_los_score(label: str, f: dict[str, float], play: Play) -> float:
    """LOS-dependent score as if the LOS were fully trusted (for UNCERTAIN_LOS)."""
    if not f.get("los_known"):
        return 0.0
    g = dict(f)
    g["los_confidence"] = 1.0
    return rule_scores(g, play).get(label, 0.0)
