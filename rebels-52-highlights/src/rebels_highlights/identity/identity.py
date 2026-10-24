"""Who is #52? Per-track evidence fusion for one play.

For every track from ``track_play`` we collect independent evidence:

* jersey_confidence    temporal voting of per-frame jersey OCR (>= min_ocr_votes
                       consistent reads needed; one frame never suffices)
* team_confidence      P(Rebels) from torso-colour clusters; the Rebels cluster
                       is ``player.team_kit_hsv`` if set, else the cluster whose
                       tracks read the target number most often
* reid_confidence      similarity to the game prototype (high-confidence #52
                       tracks from this game, ``references['game_prototype']``)
                       or to strong-jersey tracks of this play, discounted by how
                       similar the rest of the team looks (kit colours are shared)
* temporal_confidence  coverage/density of the track, penalised by suspected
                       ID switches
* reference_confidence similarity to reference_player_52/ images
* position_prior       weak MLB prior: ~4-9 yards off the ball, near the middle

``identity_confidence`` = weighted mean over the components that are
available (cfg ``identity.weights`` + ``position_prior_weight``), capped
below ``review_min`` while the target number has < ``min_ocr_votes`` reads.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from ..core.models import IdentityEvidence, Play, PlayerTrajectory
from ..jersey_ocr import ocr as _ocr
from ..pose.pose import pose_features
from ..reid import reid as _reid

LOW_RES_PX = 60
OCR_MIN_PX = 36


# --------------------------------------------------------------------------
def load_references(ref_dir: str, cfg: dict) -> Optional[dict]:
    """Embedding prototypes from reference_player_52/* (None if no usable images)."""
    return _reid.load_reference_prototypes(ref_dir, cfg)


def seed_from_review(store, play_id: str) -> Optional[dict]:
    """review/seeds.json -> {"t": float, "box": [x1,y1,x2,y2]} for this play, or None."""
    p = Path(store.review) / "seeds.json"
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text())
    except (OSError, ValueError):
        return None
    s = d.get(play_id) if isinstance(d, dict) else None
    if not isinstance(s, dict) or "t" not in s or "box" not in s or len(s["box"]) != 4:
        return None
    return {"t": float(s["t"]), "box": [float(v) for v in s["box"]]}


def build_game_prototype(pairs: Sequence[tuple[dict, dict]], min_conf: float = 0.8) -> Optional[list[float]]:
    """Mean embedding of confidently identified #52 tracks across a game.

    ``pairs``: [(identity_result, tracking_result), ...]. Only jersey-backed or
    manually seeded identities above ``min_conf`` contribute.
    """
    embs = []
    for ident, trk in pairs:
        if not ident or ident.get("best_track_id") is None or ident.get("identity_confidence", 0) < min_conf:
            continue
        ev = next((e for e in ident.get("evidence", []) if e["track_id"] == ident["best_track_id"]), None)
        if not ev or (ev.get("jersey_confidence", 0) < 0.5 and ev.get("source") != "manual_seed"):
            continue
        for t in trk.get("tracks", []):
            if t["track_id"] in ident.get("trajectory_track_ids", [ident["best_track_id"]]) and t.get("embedding"):
                embs.append(t["embedding"])
    m = _reid.mean_embedding(embs)
    return m.tolist() if m is not None else None


# --------------------------------------------------------------------------
def _center(b):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def _h(b):
    return max(1.0, b[3] - b[1])


def _iou(a, b) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def _tkey(t: float) -> float:
    return round(float(t), 3)


def _box_at(track: dict, t: float, tol: float = 0.08) -> Optional[list[float]]:
    ts = track["times"]
    if not ts:
        return None
    i = int(np.argmin(np.abs(np.asarray(ts) - t)))
    return track["boxes"][i] if abs(ts[i] - t) <= tol else None


def _sample_idx(n: int, k: int) -> list[int]:
    if n <= k:
        return list(range(n))
    return sorted(set(np.linspace(0, n - 1, k).astype(int).tolist()))


# --------------------------------------------------------------------------
def _ocr_tracks(video_path: str, tracks: list[dict], cfg: dict, frames: Optional[dict] = None) -> dict:
    """Per-track per-frame jersey readings. ``frames`` may pre-supply {t: frame}."""
    k = int(cfg.get("identity", {}).get("ocr_samples", 12))
    plan: dict[int, list[int]] = {}
    need = set()
    for tr in tracks:
        idx = [i for i in _sample_idx(len(tr["times"]), max(k * 2, k))
               if _h(tr["boxes"][i]) >= OCR_MIN_PX]
        idx = [idx[j] for j in _sample_idx(len(idx), k)] if idx else []
        plan[tr["track_id"]] = idx
        need.update(tr["times"][i] for i in idx)
    if frames is None:
        from ..tracking.track_play import read_frames_at
        frames = read_frames_at(video_path, sorted(need)) if need else {}
    fr = {_tkey(t): f for t, f in frames.items()}
    out: dict[int, list] = {}
    for tr in tracks:
        reads = []
        for i in plan[tr["track_id"]]:
            f = fr.get(_tkey(tr["times"][i]))
            if f is None:
                continue
            crop = _reid.crop_box(f, tr["boxes"][i])
            reads.append(_ocr.read_jersey(crop) if crop.size else [])
        out[tr["track_id"]] = reads
    return out


def _resolve_teams(tracks: list[dict], votes: dict[int, dict], target: str) -> Optional[int]:
    """Pick the Rebels colour cluster; update team_prob in place. Returns cluster id."""
    rebels = next((t.get("rebels_cluster") for t in tracks if t.get("rebels_cluster") is not None), None)
    if rebels is None:
        score: dict[int, float] = defaultdict(float)
        for t in tracks:
            c = t.get("team_cluster")
            v = votes.get(t["track_id"], {})
            if c is None:
                continue
            score[c] += v.get("n_target", 0) * t.get("team_cluster_conf", 0.5)
        if score and max(score.values()) > 0:
            rebels = max(score, key=score.get)
    if rebels is None:
        return None
    for t in tracks:
        c = t.get("team_cluster")
        if c is None:
            t["team_prob"] = 0.5
            continue
        p = float(t.get("team_cluster_conf", 0.5))
        t["team_prob"] = p if c == rebels else 1.0 - p
    return rebels


def _formation(tracks: list[dict], t0: float, team_known: bool):
    if not team_known:
        return None
    reb, opp = [], []
    for t in tracks:
        b = _box_at(t, t0, tol=0.35)
        if b is None:
            continue
        (reb if t.get("team_prob", 0.5) >= 0.6 else opp if t.get("team_prob", 0.5) <= 0.4 else []).append((t, b))
    if len(reb) < 3 or len(opp) < 3:
        return None
    R = np.array([_center(b) for _, b in reb])
    O = np.array([_center(b) for _, b in opp])
    u = R.mean(0) - O.mean(0)
    n = np.linalg.norm(u)
    if n < 1e-6:
        return None
    u /= n
    perp = np.array([-u[1], u[0]])
    los = (np.max(O @ u) + np.min(R @ u)) / 2.0
    lat0 = float(np.median(R @ perp))
    yard = float(np.median([_h(b) for _, b in reb + opp])) * 0.49
    return {"u": u, "perp": perp, "los": float(los), "lat0": lat0, "yard": max(yard, 1.0),
            "opp_center": O.mean(0), "reb": reb, "opp": opp}


def _position_prior(track: dict, form: Optional[dict], t0: float) -> Optional[float]:
    if form is None:
        return None
    b = _box_at(track, t0, tol=0.35)
    if b is None or track.get("team_prob", 0.5) < 0.5:
        return 0.0 if b is not None else None
    c = np.array(_center(b))
    depth = (c @ form["u"] - form["los"]) / form["yard"]
    lat = (c @ form["perp"] - form["lat0"]) / form["yard"]
    return float(math.exp(-0.5 * ((depth - 6.0) / 2.5) ** 2) * math.exp(-0.5 * (lat / 5.0) ** 2))


def _occlusion(track: dict, by_t: dict[float, list]) -> float:
    occl = 0
    for t, b in zip(track["times"], track["boxes"]):
        for tid, ob in by_t.get(_tkey(t), []):
            if tid != track["track_id"] and _iou(b, ob) > 0.3:
                occl += 1
                break
    return occl / max(1, len(track["times"]))


def _temporal(track: dict, play_dur: float, fps: float) -> float:
    ts = track["times"]
    if len(ts) < 2:
        return 0.0
    span = ts[-1] - ts[0]
    cov = min(1.0, span / max(0.6 * play_dur, 1e-3))
    dens = min(1.0, len(ts) / max(1.0, span * fps + 1))
    stab = 1.0 - 0.7 * float(track.get("id_switch_score") or 0.0)
    return float(np.clip((0.6 * cov + 0.4 * dens) * stab, 0, 1))


def _reid_discriminative(emb, proto, team_embs: list) -> Optional[float]:
    """Similarity to ``proto`` relative to the teammates' similarity.

    Returns None (evidence unavailable) when the embedding cannot tell this
    track apart from its teammates - e.g. colour histograms of identical kits.
    """
    s = _reid.cosine(emb, proto)
    base = _reid.sim_to_conf(s)
    others = [_reid.cosine(e, proto) for e in team_embs]
    if not others:
        return base * 0.5
    mu, sd = float(np.mean(others)), float(np.std(others))
    if sd < 0.03 and abs(s - mu) < 0.03:
        return None
    z = (s - mu) / (sd + 0.02)
    return float(base * np.clip(0.5 + z / 4.0, 0, 1))


# --------------------------------------------------------------------------
def _find_target(tracks: list[dict], chain: set[int], play: Play, form: Optional[dict],
                 team_known: bool) -> tuple[Optional[dict], str]:
    if not tracks:
        return None, "unknown"
    t_all = sorted({_tkey(t) for tr in tracks for t in tr["times"]})
    if len(t_all) < 3:
        return None, "unknown"
    late0 = t_all[0] + 0.6 * (t_all[-1] - t_all[0])
    by_t: dict[float, list] = defaultdict(list)
    for tr in tracks:
        for t, b in zip(tr["times"], tr["boxes"]):
            by_t[_tkey(t)].append((tr["track_id"], b))
    cands = [tr for tr in tracks if tr["track_id"] not in chain
             and (not team_known or tr.get("team_prob", 0.5) < 0.5)]
    best, best_s = None, float("inf")
    for c in cands:
        ds = []
        for t, b in zip(c["times"], c["boxes"]):
            if _tkey(t) < late0:
                continue
            others = [ob for tid, ob in by_t[_tkey(t)] if tid != c["track_id"]]
            if len(others) < 2:
                continue
            cc = _center(b)
            ds.append(np.mean([math.hypot(_center(o)[0] - cc[0], _center(o)[1] - cc[1]) / _h(o)
                               for o in others]))
        if len(ds) < 3:
            continue
        s = float(np.mean(ds))
        if s < best_s:
            best, best_s = c, s
    if best is None:
        return None, "unknown"
    if play.unit == "special_teams":
        return best, "returner"
    role = "ball_carrier"
    if form is not None:
        snap = play.estimated_snap_s if play.estimated_snap_s is not None else best["times"][0]
        b0 = _box_at(best, snap, tol=0.4)
        b1 = _box_at(best, snap + 1.2, tol=0.4)
        if b0 is not None and b1 is not None:
            c0, c1 = np.array(_center(b0)), np.array(_center(b1))
            u = form["u"]
            depth0 = (form["los"] - c0 @ u) / form["yard"]          # yards behind LOS, offense side
            lat = abs(c0 @ form["perp"] - form["lat0"]) / form["yard"]
            retreat = ((c0 - c1) @ u) / form["yard"]                # moving away from defence
            if 0.5 <= depth0 <= 9 and lat <= 4 and retreat >= 1.0:
                role = "qb"
    return best, role


def _empty(play: Play) -> dict:
    return {"evidence": [], "best_track_id": None, "identity_confidence": 0.0,
            "trajectory": PlayerTrajectory(play_id=play.play_id, track_id=None).to_dict(),
            "team_probs": {}, "rebels_cluster": None, "trajectory_track_ids": [],
            "ocr_backend": _ocr.backend_name()}


def identify_player(video_path: str, play, tracking: dict, cfg: dict,
                    references: Optional[dict] = None, seed: Optional[dict] = None,
                    frames: Optional[dict] = None) -> dict:
    """Fuse identity evidence for every track; pick #52's track and trajectory.

    ``frames`` (optional {t: frame}) lets callers/tests skip video decoding.
    """
    if isinstance(play, dict):
        play = Play.from_dict(play)
    tracks = [t for t in (tracking or {}).get("tracks", []) if t.get("times")]
    if not tracks:
        return _empty(play)
    ic = cfg.get("identity", {})
    W = dict(ic.get("weights", {}))
    w_pos = float(ic.get("position_prior_weight", 0.05))
    min_votes = int(ic.get("min_ocr_votes", 3))
    review_min = float(ic.get("review_min", 0.55))
    auto_accept = float(ic.get("auto_accept", 0.8))
    target = str((cfg.get("player") or {}).get("player_number", 52))
    fps = float(tracking.get("fps") or cfg.get("detection", {}).get("fine_fps", 15))
    play_dur = max(1e-3, play.end_s - play.start_s)
    references = references or {}

    # ---- jersey OCR + voting
    reads = _ocr_tracks(video_path, tracks, cfg, frames)
    votes = {tid: _ocr.vote(r, target=target, min_votes=min_votes) for tid, r in reads.items()}

    # ---- team resolution
    rebels = _resolve_teams(tracks, votes, target)
    team_known = rebels is not None or any(t.get("rebels_cluster") is not None for t in tracks)

    # ---- appearance
    protos = references.get("prototypes")
    game_proto = references.get("game_prototype")
    strong = [t for t in tracks if votes[t["track_id"]]["jersey_confidence"] >= 0.5 and t.get("embedding")]
    team_embs = [t["embedding"] for t in tracks if t.get("embedding") and
                 (not team_known or t.get("team_prob", 0.5) >= 0.5)]

    snap = play.estimated_snap_s if play.estimated_snap_s is not None else tracks[0]["times"][0]
    form = _formation(tracks, snap, team_known)
    by_t: dict[float, list] = defaultdict(list)
    for tr in tracks:
        for t, b in zip(tr["times"], tr["boxes"]):
            by_t[_tkey(t)].append((tr["track_id"], b))

    evidence: dict[int, IdentityEvidence] = {}
    for tr in tracks:
        tid = tr["track_id"]
        v = votes[tid]
        emb = tr.get("embedding")
        comps: dict[str, Optional[float]] = {
            "jersey_ocr": v["jersey_confidence"],
            "team_kit": float(tr.get("team_prob", 0.5)),
            "temporal": _temporal(tr, play_dur, fps),
            "reid": None, "reference": None,
        }
        others = [e for e in team_embs if e is not emb]
        if emb is not None and game_proto is not None:
            comps["reid"] = _reid_discriminative(emb, game_proto, others)
        elif emb is not None and any(s is not tr for s in strong):
            proto = _reid.mean_embedding([s["embedding"] for s in strong if s is not tr])
            comps["reid"] = _reid_discriminative(emb, proto, others)
        if emb is not None and protos:
            best_p = max(protos, key=lambda p: _reid.cosine(emb, p))
            comps["reference"] = _reid_discriminative(emb, best_p, others)
        pos = _position_prior(tr, form, snap)
        num = den = 0.0
        for k, c in comps.items():
            if c is None:
                continue
            num += float(W.get(k, 0.0)) * c
            den += float(W.get(k, 0.0))
        if pos is not None:
            num += w_pos * pos
            den += w_pos
        conf = num / den if den > 0 else 0.0
        # jersey gate: never identity from one frame / without the number
        if v["n_target"] < min_votes:
            strong_app = max(comps["reid"] or 0.0, comps["reference"] or 0.0) >= 0.8
            conf = min(conf, (auto_accept if strong_app else review_min) - 0.01)
            if v["n_target"] == 0 and not strong_app:
                conf *= 0.6
        if v["best"] not in (None, target) and v["votes"].get(v["best"], 0) >= min_votes \
                and v["votes"].get(v["best"], 0) > v["n_target"]:
            conf *= 0.3  # consistently reads as a different number
        if team_known and tr.get("team_prob", 0.5) < 0.3:
            conf *= 0.5
        reasons = []
        if not _ocr.jersey_readable(v):
            reasons.append("JERSEY_UNREADABLE")
        if float(tr.get("id_switch_score") or 0) >= 0.5:
            reasons.append("TRACK_ID_SWITCH")
        if _occlusion(tr, by_t) >= 0.3:
            reasons.append("OCCLUSION")
        if float(np.median([_h(b) for b in tr["boxes"]])) < LOW_RES_PX:
            reasons.append("LOW_RESOLUTION")
        conf = float(np.clip(conf, 0, 1))
        if conf < review_min:
            reasons.insert(0, "IDENTITY_LOW")
        evidence[tid] = IdentityEvidence(
            track_id=tid, scene_id=int(tr.get("scene_id", 0)),
            jersey_confidence=round(v["jersey_confidence"], 4),
            team_confidence=round(comps["team_kit"], 4),
            reid_confidence=round(comps["reid"] or 0.0, 4),
            temporal_confidence=round(comps["temporal"], 4),
            reference_confidence=round(comps["reference"] or 0.0, 4),
            position_prior=round(pos or 0.0, 4),
            identity_confidence=round(conf, 4),
            ocr_votes={k: int(x) for k, x in v["votes"].items()},
            review_reasons=reasons)

    # ---- choose #52
    best_tid: Optional[int] = None
    if seed and seed.get("box") is not None and seed.get("t") is not None:
        best_iou = 0.0
        for tr in tracks:
            b = _box_at(tr, float(seed["t"]), tol=max(0.5, 1.5 / fps))
            if b is None:
                continue
            i = _iou(b, seed["box"])
            if i > best_iou:
                best_iou, best_tid = i, tr["track_id"]
        if best_tid is not None and best_iou >= 0.1:
            ev = evidence[best_tid]
            ev.source = "manual_seed"
            ev.identity_confidence = round(max(ev.identity_confidence, 0.9), 4)
            ev.review_reasons = [r for r in ev.review_reasons if r != "IDENTITY_LOW"]
        else:
            best_tid = None
    if best_tid is None:
        cand = max(evidence.values(), key=lambda e: e.identity_confidence)
        if cand.identity_confidence >= 0.25 and (cand.ocr_votes.get(target, 0) > 0 or
                                                 cand.reid_confidence >= 0.8 or
                                                 cand.reference_confidence >= 0.8):
            best_tid = cand.track_id

    tmap = {t["track_id"]: t for t in tracks}
    chain = [best_tid] if best_tid is not None else []
    if best_tid is not None:
        chain = _propagate(best_tid, tracks, votes, target, min_votes, team_known)
    traj = PlayerTrajectory(play_id=play.play_id, track_id=best_tid)
    if chain:
        pts = sorted((t, b) for tid in chain for t, b in zip(tmap[tid]["times"], tmap[tid]["boxes"]))
        seen = set()
        for t, b in pts:
            if _tkey(t) in seen:
                continue
            seen.add(_tkey(t))
            traj.times.append(float(t))
            traj.boxes.append([float(x) for x in b])
        tgt, role = _find_target(tracks, set(chain), play, form, team_known)
        if tgt is not None:
            traj.target_times = [float(t) for t in tgt["times"]]
            traj.target_boxes = [[float(x) for x in b] for b in tgt["boxes"]]
            traj.target_role = role
            traj_target_id = tgt["track_id"]
        else:
            traj_target_id = None
        traj.pose_features = _pose(video_path, traj, cfg, frames)
        if len(chain) > 1:
            for tid in chain[1:]:
                e = evidence[tid]
                if e.source != "manual_seed":
                    e.identity_confidence = round(max(e.identity_confidence,
                                                      0.9 * evidence[best_tid].identity_confidence), 4)
                    if e.identity_confidence >= review_min:
                        e.review_reasons = [r for r in e.review_reasons if r != "IDENTITY_LOW"]
                    e.review_reasons.append("OCCLUSION") if "OCCLUSION" not in e.review_reasons else None
    else:
        traj_target_id = None

    ev_list = sorted(evidence.values(), key=lambda e: -e.identity_confidence)
    best_conf = evidence[best_tid].identity_confidence if best_tid is not None else 0.0
    return {"evidence": [e.to_dict() for e in ev_list], "best_track_id": best_tid,
            "identity_confidence": float(best_conf), "trajectory": traj.to_dict(),
            "trajectory_track_ids": chain, "target_track_id": traj_target_id,
            "team_probs": {int(t["track_id"]): round(float(t.get("team_prob", 0.5)), 4) for t in tracks},
            "rebels_cluster": rebels, "ocr_backend": _ocr.backend_name(),
            "review_reasons": list(evidence[best_tid].review_reasons) if best_tid is not None
            else ["IDENTITY_LOW"]}


def _propagate(best: int, tracks: list[dict], votes: dict, target: str, min_votes: int,
               team_known: bool, max_gap: float = 1.5) -> list[int]:
    """Link #52 across occlusion splits inside the same cut-free segment."""
    tmap = {t["track_id"]: t for t in tracks}
    chain = [best]
    used = {best}

    def ok(a: dict, b: dict, forward: bool) -> float:
        if b["track_id"] in used or b.get("segment", b.get("scene_id")) != a.get("segment", a.get("scene_id")):
            return -1
        vb = votes[b["track_id"]]
        if vb["best"] not in (None, target) and vb["votes"].get(vb["best"], 0) >= min_votes:
            return -1
        if team_known and (a.get("team_prob", 0.5) >= 0.5) != (b.get("team_prob", 0.5) >= 0.5):
            return -1
        if forward:
            gap = b["times"][0] - a["times"][-1]
            pa, pb = a["boxes"][-1], b["boxes"][0]
        else:
            gap = a["times"][0] - b["times"][-1]
            pa, pb = a["boxes"][0], b["boxes"][-1]
        if gap < -0.2 or gap > max_gap:
            return -1
        ca, cb = _center(pa), _center(pb)
        d = math.hypot(ca[0] - cb[0], ca[1] - cb[1]) / _h(pa)
        if d > 1.0 + 6.0 * max(gap, 0):  # ~6 body heights/s max displacement
            return -1
        sim = _reid.cosine(a.get("embedding"), b.get("embedding"))
        if sim < 0.85:
            return -1
        return sim - 0.05 * d

    for forward in (True, False):
        cur = tmap[best]
        while True:
            opts = [(ok(cur, b, forward), b) for b in tracks]
            opts = [(s, b) for s, b in opts if s >= 0]
            if not opts:
                break
            s, nxt = max(opts, key=lambda x: x[0])
            used.add(nxt["track_id"])
            chain.append(nxt["track_id"])
            cur = nxt
    return chain


def _pose(video_path: str, traj: PlayerTrajectory, cfg: dict, frames: Optional[dict]) -> dict:
    idx = _sample_idx(len(traj.times), 16)
    try:
        if frames is None:
            from ..tracking.track_play import read_frames_at
            fr = read_frames_at(video_path, [traj.times[i] for i in idx])
        else:
            fr = frames
        fr = {_tkey(t): f for t, f in fr.items()}
        crops_by_i = {}
        for i in idx:
            f = fr.get(_tkey(traj.times[i]))
            if f is not None:
                crops_by_i[i] = _reid.crop_box(f, traj.boxes[i], pad=0.1)
        sel = [i for i in idx if crops_by_i.get(i) is not None]
        feats = pose_features([crops_by_i[i] for i in sel] if sel else None,
                              traj.boxes, traj.times, cfg,
                              crop_times=[traj.times[i] for i in sel])
    except Exception:  # noqa: BLE001 - pose is optional
        feats = pose_features(None, traj.boxes, traj.times, cfg)
    return feats
