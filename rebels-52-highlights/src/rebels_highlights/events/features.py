"""Per-play kinematic features for #52 (numpy only).

All distances are normalised by the mean #52 box height ("H") so features are
scale-invariant across broadcast zoom levels; speeds are in H per second.
Every value in the returned dict is a float (booleans are 0.0/1.0) so the dict
can be stored in ``EventResult.features`` and fed to the stage-2 model.

Sign convention for LOS-relative positions: the offense is the team opposite
to #52 (team_prob on the other side of 0.5). ``offense_side`` is +1 when the
offense centroid at the snap is at larger along-field coordinate than the LOS
(-1 otherwise). ``contact_gain_h`` is the ball carrier's gain at contact in H
units (negative = behind the LOS, i.e. in the offensive backfield).
"""
from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np

from ..core.models import Play, PlayerTrajectory

NAN = float("nan")
CONTACT_RADIUS_H = 1.2      # defenders "at the contact" (assist vs solo)
OPEN_FIELD_RADIUS_H = 3.0   # nobody else near -> open-field tackle
QB_PRESSURE_H = 1.5


# ---------------------------------------------------------------- helpers
def _arr(boxes) -> np.ndarray:
    a = np.asarray(boxes, dtype=float)
    return a.reshape(-1, 4) if a.size else np.zeros((0, 4))


def _centers(b: np.ndarray) -> np.ndarray:
    return np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2], axis=1) if len(b) else np.zeros((0, 2))


def _iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / ua) if ua > 0 else 0.0


def _resample(times, boxes, at: np.ndarray, tol: float = 0.35) -> np.ndarray:
    """Linear-interpolate boxes onto ``at``; NaN outside the sampled span (+tol)."""
    t = np.asarray(times, dtype=float)
    b = _arr(boxes)
    out = np.full((len(at), 4), np.nan)
    if len(t) == 0 or len(b) != len(t):
        return out
    order = np.argsort(t)
    t, b = t[order], b[order]
    for k in range(4):
        out[:, k] = np.interp(at, t, b[:, k])
    valid = (at >= t[0] - tol) & (at <= t[-1] + tol)
    # also invalidate long gaps
    idx = np.clip(np.searchsorted(t, at), 1, max(1, len(t) - 1))
    if len(t) > 1:
        gap = t[idx] - t[idx - 1]
        valid &= ~((gap > 1.0) & (at > t[0]) & (at < t[-1]))
    out[~valid] = np.nan
    return out


def _box_at(track: dict, t: float, tol: float = 0.25) -> Optional[list[float]]:
    ts = track.get("times") or []
    if not ts:
        return None
    ts_a = np.asarray(ts, dtype=float)
    i = int(np.argmin(np.abs(ts_a - t)))
    if abs(ts_a[i] - t) > tol:
        return None
    return list(map(float, track["boxes"][i]))


def _velocity(c: np.ndarray, t: np.ndarray) -> np.ndarray:
    if len(c) < 2:
        return np.zeros_like(c)
    v = np.gradient(c, t, axis=0)
    return np.nan_to_num(v)


def project_along_field(x: float, y: float, field: dict) -> float:
    """Local fallback: along-field coordinate (pixel x when no homography)."""
    H = field.get("homography") if field else None
    if H is not None:
        try:
            m = np.asarray(H, dtype=float).reshape(3, 3)
            p = m @ np.array([x, y, 1.0])
            if abs(p[2]) > 1e-9:
                return float(p[0] / p[2])
        except Exception:
            pass
    return float(x)


def _clip01(v: float) -> float:
    return float(min(1.0, max(0.0, v))) if v == v else 0.0


# ---------------------------------------------------------------- main
def compute_features(play: Play, traj: PlayerTrajectory, tracking: Optional[dict],
                     field: Optional[dict], cfg: dict) -> dict[str, Any]:
    """Return {"features": {str: float}, "impact_s", "contact_point", "los_x",
    "los_confidence"} (non-float extras for EventResult)."""
    ev = cfg.get("events", {})
    c_ratio = float(ev.get("contact_dist_ratio", 0.6))
    c_iou = float(ev.get("contact_iou", 0.05))
    tracking = tracking or {}
    field = field or {}
    tracks = tracking.get("tracks") or []
    f: dict[str, float] = {}
    out: dict[str, Any] = {"features": f, "impact_s": None, "contact_point": None,
                           "los_x": field.get("los_x"),
                           "los_confidence": float(field.get("los_confidence") or 0.0)}

    t = np.asarray(traj.times, dtype=float)
    b = _arr(traj.boxes)
    if len(t) < 2 or len(b) != len(t):
        f["n_samples"] = float(len(t))
        f["has_contact"] = 0.0
        return out
    order = np.argsort(t)
    t, b = t[order], b[order]
    H = float(np.mean(b[:, 3] - b[:, 1])) or 1.0
    c52 = _centers(b)
    f["n_samples"] = float(len(t))
    f["box_h_px"] = H
    snap = play.estimated_snap_s if play.estimated_snap_s is not None else float(t[0])
    f["snap_s"] = float(snap)

    # #52 own team side
    my_track = next((tr for tr in tracks if tr.get("track_id") == traj.track_id), None)
    my_side = (float(my_track.get("team_prob", 0.5)) > 0.5) if my_track else True

    def same_team(tr) -> bool:
        return (float(tr.get("team_prob", 0.5)) > 0.5) == my_side

    # ---- target distance / contact
    tb = _resample(traj.target_times, traj.target_boxes, t) if traj.target_times else np.full((len(t), 4), np.nan)
    has_target = bool(np.isfinite(tb[:, 0]).any())
    f["has_target"] = float(has_target)
    f["target_is_qb"] = float(traj.target_role == "qb")
    f["target_is_returner"] = float(traj.target_role == "returner")
    impact_i: Optional[int] = None
    contact = False
    if has_target:
        ct = _centers(np.nan_to_num(tb))
        d = np.linalg.norm(c52 - ct, axis=1) / H
        d[~np.isfinite(tb[:, 0])] = np.nan
        ious = np.array([_iou(b[i], tb[i]) if np.isfinite(tb[i, 0]) else 0.0 for i in range(len(t))])
        is_c = (np.nan_to_num(d, nan=1e9) < c_ratio) | (ious > c_iou)
        f["start_distance_h"] = float(d[np.isfinite(d)][0])
        mi = int(np.nanargmin(d))
        f["min_distance_h"] = float(d[mi])
        if is_c.any():
            contact = True
            impact_i = int(np.argmax(is_c))
            f["contact_frames"] = float(is_c.sum())
        else:
            impact_i = mi
            f["contact_frames"] = 0.0
        f["has_contact"] = float(contact)
        f["impact_rel_snap_s"] = float(t[impact_i] - snap)
        # closing speed over the second before impact
        dd = np.gradient(np.nan_to_num(d, nan=np.nanmax(d)), t)
        w = (t >= t[impact_i] - 1.0) & (t <= t[impact_i])
        close = -dd[w] if w.any() else np.array([0.0])
        f["closing_speed_h"] = float(np.mean(close))
        f["max_closing_speed_h"] = float(np.max(close))
        # target velocity before/after contact
        vt = np.linalg.norm(_velocity(ct, t), axis=1) / H
        vt[~np.isfinite(tb[:, 0])] = np.nan
        wb = (t >= t[impact_i] - 0.5) & (t < t[impact_i])
        wa = (t > t[impact_i] + 0.2) & (t <= t[impact_i] + 1.0)
        vb = float(np.nanmean(vt[wb])) if np.isfinite(vt[wb]).any() else NAN
        va = float(np.nanmean(vt[wa])) if np.isfinite(vt[wa]).any() else NAN
        f["target_speed_before_h"] = vb if vb == vb else 0.0
        f["target_speed_after_h"] = va if va == va else 0.0
        f["target_decel"] = _clip01((vb - va) / max(vb, 1e-3)) if (vb == vb and va == va and vb > 0.3) else 0.0
        # target "down": box aspect (h/w) collapses or height drops after contact
        th = tb[:, 3] - tb[:, 1]
        tw = np.maximum(tb[:, 2] - tb[:, 0], 1e-3)
        asp = th / tw
        pre = (t <= t[impact_i]) & (t >= t[impact_i] - 1.0) & np.isfinite(asp)
        post = (t > t[impact_i]) & (t <= t[impact_i] + 2.0) & np.isfinite(asp)
        if pre.any() and post.any():
            a0, a1 = float(np.median(asp[pre])), float(np.min(asp[post]))
            h0, h1 = float(np.median(th[pre])), float(np.min(th[post]))
            f["target_aspect_drop"] = _clip01(1 - a1 / max(a0, 1e-3))
            f["target_height_drop"] = _clip01(1 - h1 / max(h0, 1e-3))
            f["target_down"] = _clip01(max((f["target_aspect_drop"] - 0.2) / 0.3,
                                           (f["target_height_drop"] - 0.15) / 0.25))
        else:
            f["target_aspect_drop"] = f["target_height_drop"] = f["target_down"] = 0.0
        out["impact_s"] = float(t[impact_i])
        cp = (c52[impact_i] + ct[impact_i]) / 2 if contact else ct[impact_i]
        out["contact_point"] = [float(cp[0]), float(cp[1])]
        f["qb_pressure"] = float(traj.target_role == "qb" and f["min_distance_h"] <= QB_PRESSURE_H)
    else:
        f["has_contact"] = 0.0
        f["target_down"] = 0.0
        f["target_decel"] = 0.0
        f["qb_pressure"] = 0.0
        f["closing_speed_h"] = 0.0

    impact_t = float(t[impact_i]) if impact_i is not None else float(t[-1])
    cp_xy = np.array(out["contact_point"]) if out["contact_point"] else c52[-1]

    # ---- identify target + other players at impact
    tgt_box_imp = tb[impact_i] if (impact_i is not None and has_target) else None
    others = []  # (track, box_at_impact, is_teammate)
    target_track = None
    for tr in tracks:
        if tr.get("role") == "ball" or tr.get("track_id") == traj.track_id:
            continue
        bx = _box_at(tr, impact_t)
        if bx is None:
            continue
        if _iou(bx, b[impact_i if impact_i is not None else -1]) > 0.7:
            continue  # duplicate of #52
        if tgt_box_imp is not None and np.isfinite(tgt_box_imp[0]) and target_track is None \
                and _iou(bx, tgt_box_imp) > 0.5:
            target_track = tr
            continue
        others.append((tr, bx, same_team(tr)))

    def near(bx, r):
        c = np.array([(bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2])
        return np.linalg.norm(c - cp_xy) / H <= r

    n_def = sum(1 for tr, bx, mate in others if mate and near(bx, CONTACT_RADIUS_H))
    n_opp = sum(1 for tr, bx, mate in others if not mate and near(bx, CONTACT_RADIUS_H))
    n_open = sum(1 for tr, bx, mate in others if near(bx, OPEN_FIELD_RADIUS_H))
    f["defenders_at_contact"] = float(n_def)
    f["opponents_at_contact"] = float(n_opp)
    f["players_near_contact"] = float(n_open)
    f["pile_size"] = float(n_def + n_opp + (2 if contact else 0))
    f["open_field"] = float(n_open <= 1)

    # first to contact + closing-speed share among defenders
    first = 1.0
    share = 1.0
    if has_target and impact_i is not None:
        ct_all = _centers(np.nan_to_num(tb))
        my_close = max(0.0, f.get("closing_speed_h", 0.0))
        tot = my_close
        for tr, _bx, mate in others:
            if not mate:
                continue
            ob = _resample(tr["times"], tr["boxes"], t, tol=0.2)
            ok = np.isfinite(ob[:, 0]) & np.isfinite(tb[:, 0])
            if not ok.any():
                continue
            od = np.full(len(t), np.nan)
            od[ok] = np.linalg.norm(_centers(ob[ok]) - ct_all[ok], axis=1) / H
            oc = ok & (od < c_ratio)
            if contact and oc.any() and t[int(np.argmax(oc))] < t[impact_i] - 0.1:
                first = 0.0
            w = ok & (t >= t[impact_i] - 1.0) & (t <= t[impact_i])
            if w.sum() >= 2:
                odd = np.gradient(od[w], t[w]) if w.sum() > 1 else np.array([0.0])
                tot += max(0.0, float(-np.mean(odd)))
        share = my_close / tot if tot > 1e-6 else (1.0 if not others else 0.5)
    f["first_to_contact"] = first if contact else 0.0
    f["closing_share"] = float(share)

    # ---- LOS relative
    los_x = field.get("los_x")
    f["los_known"] = float(los_x is not None)
    f["los_confidence"] = float(field.get("los_confidence") or 0.0)
    snap_t = snap
    offense_side = 0.0
    if los_x is not None:
        los_a = project_along_field(float(los_x), float(np.mean(c52[:, 1])), field)
        offs = []
        for tr in tracks:
            if tr.get("role") == "ball" or tr.get("track_id") == traj.track_id or same_team(tr):
                continue
            bx = _box_at(tr, snap_t, tol=0.5)
            if bx is not None:
                offs.append(project_along_field((bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2, field))
        if offs:
            offense_side = 1.0 if np.mean(offs) > los_a else -1.0
        else:
            i0 = int(np.argmin(np.abs(t - snap_t)))
            x52 = project_along_field(c52[i0, 0], c52[i0, 1], field)
            offense_side = -1.0 if x52 > los_a else 1.0
        f["offense_side"] = offense_side
        # target / contact gain
        gx = project_along_field(float(cp_xy[0]), float(cp_xy[1]), field)
        # scale: H in field coords when homography exists (approx via 1px offset)
        hx = abs(project_along_field(float(cp_xy[0]) + H, float(cp_xy[1]), field) - gx) or H
        gain = (los_a - gx) * offense_side / hx
        f["contact_gain_h"] = float(gain)
        f["contact_behind_los"] = float(gain < -0.3)
        f["contact_near_los"] = float(abs(gain) <= 1.5)
        f["contact_beyond_los"] = float(gain > 1.5)
        # #52 depth on the defensive side (positive = off the ball)
        depth = np.array([(project_along_field(x, y, field) - los_a) * (-offense_side) / hx for x, y in c52])
    else:
        # no LOS: depth relative to own start position, direction unknown -> use
        # movement toward the target's start as "downhill"
        f["offense_side"] = 0.0
        f["contact_gain_h"] = 0.0
        f["contact_behind_los"] = f["contact_near_los"] = f["contact_beyond_los"] = 0.0
        if has_target:
            ct0 = _centers(np.nan_to_num(tb))[np.isfinite(tb[:, 0])][0]
            dirv = np.sign(c52[0, 0] - ct0[0]) or 1.0
        else:
            dirv = 1.0
        depth = (c52[:, 0] - c52[0, 0]) * dirv / H
    w_early = (t >= snap_t) & (t <= snap_t + 1.5)
    if w_early.sum() >= 2:
        de = depth[w_early]
        drop = float(de[-1] - de[0])
    else:
        drop = 0.0
    f["early_drop_h"] = drop
    f["coverage_drop"] = float(drop > 1.0)
    f["read_downhill"] = float(drop < -0.5)

    # ---- pursuit
    seg = c52[(t >= snap_t) & (t <= impact_t + 1e-6)]
    if len(seg) < 2:
        seg = c52
    steps = np.diff(seg, axis=0)
    f["path_length_h"] = float(np.linalg.norm(steps, axis=1).sum() / H)
    mv = steps[np.linalg.norm(steps, axis=1) > 0.05 * H]
    if len(mv) >= 2:
        ang = np.arctan2(mv[:, 1], mv[:, 0])
        da = np.angle(np.exp(1j * np.diff(ang)))
        f["angle_change_rad"] = float(np.abs(da).sum())
    else:
        f["angle_change_rad"] = 0.0
    lead = 0.0
    if has_target:
        ct = _centers(np.nan_to_num(tb))
        v52 = _velocity(c52, t)
        cos = []
        for i in range(len(t)):
            if not (snap_t <= t[i] <= impact_t) or not np.isfinite(tb[i, 0]):
                continue
            j = int(np.argmin(np.abs(t - min(t[i] + 0.5, impact_t))))
            to = ct[j] - c52[i]
            n1, n2 = np.linalg.norm(v52[i]), np.linalg.norm(to)
            if n1 > 0.3 * H and n2 > 1e-6:
                cos.append(float(v52[i] @ to / (n1 * n2)))
        lead = float(np.mean(cos)) if cos else 0.0
        ii = impact_i if impact_i is not None else len(t) - 1
        f["projected_intercept_err_h"] = float(np.linalg.norm(c52[ii] - ct[ii]) / H)
    f["pursuit_angle_cos"] = lead
    sp52 = np.linalg.norm(_velocity(c52, t), axis=1) / H
    f["max_speed_h"] = float(np.max(sp52))

    # ---- ball evidence (only when tracking has a ball-flagged track)
    ball = next((tr for tr in tracks if tr.get("role") == "ball"), None)
    f["ball_track"] = float(ball is not None)
    f["ball_visible_frac"] = 0.0
    f["target_separation_event"] = 0.0
    f["ball_recovered_by_52"] = 0.0
    f["throw_detected"] = 0.0
    if ball is not None and has_target and ball.get("times"):
        bb = _resample(ball["times"], ball["boxes"], t, tol=0.15)
        ok = np.isfinite(bb[:, 0])
        f["ball_visible_frac"] = float(ok.mean())
        if ok.any():
            bc = _centers(np.nan_to_num(bb))
            ct = _centers(np.nan_to_num(tb))
            dtb = np.where(ok & np.isfinite(tb[:, 0]), np.linalg.norm(bc - ct, axis=1) / H, np.nan)
            d52b = np.where(ok, np.linalg.norm(bc - c52, axis=1) / H, np.nan)
            pre = (t < impact_t) & (t >= impact_t - 1.0)
            post = (t > impact_t) & (t <= impact_t + 1.5)
            held = np.nanmedian(dtb[pre]) < 0.6 if np.isfinite(dtb[pre]).any() else False
            loose = np.nanmax(dtb[post]) > 1.5 if np.isfinite(dtb[post]).any() else False
            if held and loose and contact:
                f["target_separation_event"] = 1.0
            elif held and not contact and loose:
                f["throw_detected"] = 1.0
            late = t > impact_t + 0.5
            if f["target_separation_event"] and np.isfinite(d52b[late]).any() \
                    and np.nanmin(d52b[late]) < 0.5:
                f["ball_recovered_by_52"] = 1.0

    # ---- block shed: contact with an opponent (not target) before, then separate + close
    shed = 0.0
    for tr in tracks:
        if tr is target_track or tr.get("role") == "ball" or tr.get("track_id") == traj.track_id \
                or same_team(tr):
            continue
        ob = _resample(tr["times"], tr["boxes"], t, tol=0.2)
        ok = np.isfinite(ob[:, 0]) & (t < impact_t)
        if ok.sum() < 3:
            continue
        od = np.full(len(t), np.inf)
        od[ok] = np.linalg.norm(_centers(ob[ok]) - c52[ok], axis=1) / H
        eng = ok & (od < c_ratio * 1.2)
        if eng.sum() < 2:
            continue
        last = int(np.where(eng)[0][-1])
        after = ok & (np.arange(len(t)) > last)
        if after.any() and np.max(od[after]) > 1.0 and \
                (contact or f.get("min_distance_h", 9) < 1.0):
            # engaged for >= 0.3 s and freed himself before the tackle
            if t[last] - t[int(np.where(eng)[0][0])] >= 0.3:
                shed = 1.0
                break
    f["block_shed"] = shed

    # ---- special-teams signature
    st = 1.0 if play.unit == "special_teams" else 0.0
    W = float((tracking.get("frame_size") or [0, 0])[0] or 0)
    if not st and W > 0 and tracks:
        dur = _coverage_run_s(tracks, W)
        f["coverage_run_s"] = dur
        if dur >= 3.0:
            st = 0.8
    f["special_teams_signature"] = st

    # pose features (fall/lean/...) passed through as floats
    for k, v in (traj.pose_features or {}).items():
        try:
            f[f"pose_{k}"] = float(v)
        except (TypeError, ValueError):
            pass
    return out


def _coverage_run_s(tracks: list[dict], W: float, bin_s: float = 0.5) -> float:
    """Longest run (s) where players span > 60% of frame width and mostly run
    the same horizontal direction (kick/punt coverage signature)."""
    ts = [x for tr in tracks for x in (tr.get("times") or [])]
    if not ts:
        return 0.0
    t0, t1 = min(ts), max(ts)
    bins = np.arange(t0, t1 + bin_s, bin_s)
    best = run = 0.0
    for a in bins:
        xs, dirs = [], []
        for tr in tracks:
            if tr.get("role") == "ball":
                continue
            tt = np.asarray(tr.get("times") or [], dtype=float)
            m = (tt >= a) & (tt < a + bin_s)
            if m.sum() < 1:
                continue
            bx = np.asarray(tr["boxes"], dtype=float)[m]
            cx = (bx[:, 0] + bx[:, 2]) / 2
            xs.append(float(cx.mean()))
            # direction over a slightly wider window
            m2 = (tt >= a - bin_s / 2) & (tt < a + bin_s * 1.5)
            if m2.sum() >= 2:
                b2 = np.asarray(tr["boxes"], dtype=float)[m2]
                c2 = (b2[:, 0] + b2[:, 2]) / 2
                dx = c2[-1] - c2[0]
                if abs(dx) > 2:
                    dirs.append(np.sign(dx))
        ok = len(xs) >= 4 and (max(xs) - min(xs)) / W > 0.6 and len(dirs) >= 3 \
            and abs(float(np.mean(dirs))) > 0.7
        run = run + bin_s if ok else 0.0
        best = max(best, run)
    return float(best)


FEATURE_KEYS = [
    "has_contact", "min_distance_h", "start_distance_h", "closing_speed_h", "max_closing_speed_h",
    "target_decel", "target_down", "target_is_qb", "target_is_returner", "defenders_at_contact",
    "opponents_at_contact", "players_near_contact", "open_field", "first_to_contact",
    "closing_share", "los_known", "los_confidence", "contact_gain_h", "contact_behind_los",
    "contact_near_los", "contact_beyond_los", "early_drop_h", "coverage_drop", "read_downhill",
    "path_length_h", "angle_change_rad", "pursuit_angle_cos", "max_speed_h", "qb_pressure",
    "ball_track", "target_separation_event", "ball_recovered_by_52", "block_shed",
    "special_teams_signature",
]


def feature_vector(features: dict[str, float]) -> np.ndarray:
    v = np.array([float(features.get(k, 0.0) or 0.0) for k in FEATURE_KEYS])
    v[~np.isfinite(v)] = 0.0
    return v


def isfinite_dict(d: dict[str, float]) -> dict[str, float]:
    return {k: (float(v) if (isinstance(v, (int, float)) and math.isfinite(v)) else 0.0)
            for k, v in d.items()}
