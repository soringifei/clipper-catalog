"""Stage orchestration. Every stage is cached per game under cache/<game_id>/.

Pass A (discovery, proxy):  ingest -> proxy -> scenes -> motion -> plays
Pass B (fine, candidates):  tracking -> identity -> field -> events -> scoring
Output:                      render -> qa -> manifests -> review -> mixes
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from .core.config import games_from_config, game_hint
from .core.device import detect_environment, pick_detector, pick_device
from .core.models import (Candidate, EventResult, Game, Play, PlayerTrajectory,
                          Scene, VideoInfo)
from .core.store import Store

log = logging.getLogger("rebels_highlights")


class Pipeline:
    def __init__(self, cfg: dict, resume: bool = True, debug: bool = False,
                 dry_run: bool = False):
        self.cfg = cfg
        self.store = Store(cfg)
        self.resume = resume
        self.debug = debug
        self.dry_run = dry_run
        self.env = detect_environment(cfg["root"])
        self.device = pick_device(cfg.get("device", "auto"), self.env)
        cfg["detection"]["model"] = pick_detector(self.device, cfg["detection"]["model"])
        (self.store.reports / "environment.json").write_text(
            json.dumps({**self.env, "device": self.device,
                        "detector": cfg["detection"]["model"]}, indent=1))

    # ------------------------------------------------------------------ A
    def ingest(self, game: Game) -> Optional[VideoInfo]:
        from .ingest.ingest import ingest_game, make_proxy
        st = self.store

        def _run():
            info = ingest_game(game, self.cfg, st)
            info.proxy_path = make_proxy(info, self.cfg, st)
            return info.to_dict()

        d = st.run_stage(game.game_id, "ingest", _run, self.resume)
        return VideoInfo.from_dict(d) if d else None

    def discover(self, game: Game, info: VideoInfo) -> list[Play]:
        from .plays.plays import associate_replays, motion_profile, segment_plays
        from .scenes.scenes import detect_scenes
        st, gid, video = self.store, game.game_id, info.proxy_path or info.path

        scenes = st.run_stage(gid, "scenes", lambda: [s.to_dict() for s in
                              detect_scenes(video, self.cfg)], self.resume) or []
        motion = st.run_stage(gid, "motion", lambda: motion_profile(video, self.cfg),
                              self.resume)
        if motion is None:
            return []
        plays = st.run_stage(gid, "plays", lambda: [p.to_dict() for p in associate_replays(
            segment_plays(info, [Scene.from_dict(s) for s in scenes], motion, self.cfg),
            self.cfg)], self.resume) or []
        return [Play.from_dict(p) for p in plays]

    # ------------------------------------------------------------------ B
    def analyze_play(self, game: Game, info: VideoInfo, play: Play,
                     references: Optional[dict]) -> Optional[dict]:
        from .events.events import classify_event
        from .field.field import estimate_field
        from .identity.identity import identify_player, seed_from_review
        from .tracking.track_play import track_play
        st, gid, pid = self.store, game.game_id, play.play_id

        tracking = st.run_stage(gid, f"tracking/{pid}", lambda: track_play(
            info.path, play, self.cfg, self.device), self.resume)
        if not tracking:
            return None
        seed = seed_from_review(st, pid)
        # A new manual seed invalidates the cached identity for this play.
        ident_key = f"identity/{pid}" + ("_seeded" if seed else "")
        identity = st.run_stage(gid, ident_key, lambda: identify_player(
            info.path, play, tracking, self.cfg, references, seed), self.resume)
        if not identity or identity.get("best_track_id") is None:
            return {"play": play, "identity": identity, "event": None}
        traj = PlayerTrajectory.from_dict(identity["trajectory"])

        def _field():
            frame = _frame_at(info.path, play.estimated_snap_s or play.start_s)
            boxes = _boxes_at(tracking, play.estimated_snap_s or play.start_s)
            return estimate_field(frame, boxes, self.cfg) if frame is not None else {
                "los_x": None, "los_confidence": 0.0, "yard_lines": [], "homography": None}

        field = st.run_stage(gid, f"field/{pid}", _field, self.resume) or {
            "los_x": None, "los_confidence": 0.0, "yard_lines": [], "homography": None}
        event = st.run_stage(gid, f"events/{pid}" + ("_seeded" if seed else ""),
                             lambda: classify_event(play, traj, tracking, field,
                                                    self.cfg).to_dict(), self.resume)
        if self.debug and event:
            from .rendering.render import render_debug
            out = st.outputs / "debug" / f"{pid}.mp4"
            out.parent.mkdir(exist_ok=True)
            try:
                render_debug(info.path, play, tracking, identity,
                             EventResult.from_dict(event), self.cfg, str(out))
            except Exception as e:  # noqa: BLE001
                st.log_error(gid, "debug_render", e, play_id=pid)
        return {"play": play, "identity": identity,
                "event": EventResult.from_dict(event) if event else None}

    def analyze(self, game: Game) -> list[Candidate]:
        from .identity.identity import load_references
        from .scoring.scoring import build_candidate
        st, gid = self.store, game.game_id
        info = self.ingest(game)
        if not info:
            return []
        plays = self.discover(game, info)
        canonical = [p for p in plays if not p.canonical_play_id]
        hint = game_hint(self.cfg, gid)
        if hint is not None:  # the shared &t= moment is analysed first
            canonical.sort(key=lambda p: 0 if p.start_s - 30 <= hint <= p.end_s + 30 else 1)
        refs = load_references(str(Path(self.cfg["root"]) /
                                   self.cfg["player"]["reference_images_dir"]), self.cfg)
        cands = []
        for play in canonical:
            try:
                res = self.analyze_play(game, info, play, refs)
            except Exception as e:  # noqa: BLE001
                st.log_error(gid, "analyze_play", e, play_id=play.play_id)
                continue
            if not res or not res["event"]:
                continue
            cands.append(build_candidate(game, info, play, res["identity"],
                                         res["event"], self.cfg))
        st.save(gid, "candidates", [c.to_dict() for c in cands])
        return cands

    # ------------------------------------------------------------ output
    def all_candidates(self, games: list[Game]) -> list[Candidate]:
        from .scoring.scoring import dedupe_candidates
        cands = []
        for g in games:
            cands += [Candidate.from_dict(c) for c in self.store.load(g.game_id, "candidates", [])]
        return dedupe_candidates(cands)

    def render(self, games: list[Game], approved_only: bool = False) -> list[dict]:
        from .qa.qa import qa_clip, write_qa_report
        from .rendering.render import render_clip
        from .review.review import apply_review_decisions
        st = self.store
        decisions = apply_review_decisions(st) or {}
        music = self.cfg["player"].get("music_file") or None
        if music and not Path(music).is_absolute():
            music = str(Path(self.cfg["root"]) / music)
        if music and not Path(music).exists():
            st.log_error(None, "render", f"music_file not found: {music}; clean audio only")
            music = None
        cands = self.all_candidates(games)
        qa_results = []
        for c in cands:
            dec = (decisions.get(c.play_id) or {}).get("decision")
            if dec == "approve":
                c.review_status, c.manual_review = "AUTO_APPROVED", False
            elif dec == "reject":
                c.review_status = "REJECTED"
            if c.review_status == "REJECTED" or (approved_only and c.review_status != "AUTO_APPROVED"):
                continue
            info = VideoInfo.from_dict(st.load(c.game_id, "ingest"))
            ident = self._identity(c.game_id, c.play_id)
            ev = self._event(c.game_id, c.play_id)
            if not ident or not ev:
                continue
            traj = PlayerTrajectory.from_dict(ident["trajectory"])
            sub = "single_plays" if c.review_status == "AUTO_APPROVED" else "candidates"
            if self.dry_run:
                log.info("dry-run: would render %s -> %s", c.clip_id, sub)
                continue
            try:
                out = render_clip(c, info, traj, ev, self.cfg, str(st.outputs / sub), music)
            except Exception as e:  # noqa: BLE001
                st.log_error(c.game_id, "render", e, play_id=c.play_id)
                continue
            c.output_file = out["output_file"]
            c.thumbnail_file = out["thumbnail_file"]
            c.output_duration_s = out["output_duration_s"]
            for k in ("seed_frame", "seed_frame_size", "seed_frame_t"):
                setattr(c, k, out.get(k))
            q = qa_clip(c.output_file, c, traj, self.cfg)
            qa_results.append(q)
            if not q["passed"]:
                c.review_reasons = sorted(set(c.review_reasons) | {"QA_FAILED"})
            elif c.review_status == "AUTO_APPROVED":
                _link(Path(c.output_file), st.outputs / "approved")
        # persist rendered paths back into per-game candidate caches
        by_game: dict[str, list] = {}
        for c in cands:
            by_game.setdefault(c.game_id, []).append(c.to_dict())
        for gid, lst in by_game.items():
            st.save(gid, "candidates", lst)
        write_qa_report(qa_results, st)
        self.publish(games, qa_results)
        return qa_results

    def mix(self, games: list[Game], durations: Optional[list[int]] = None) -> list[dict]:
        from .rendering.mix import build_mix
        qa = _load_json(self.store.reports / "qa.json", {})
        passed = {r["clip_id"] for r in qa.get("results", []) if r.get("passed")}
        clips = [c for c in self.all_candidates(games)
                 if c.review_status == "AUTO_APPROVED" and c.output_file
                 and c.clip_id in passed]
        mixes = []
        for i, d in enumerate(durations or self.cfg["mix"]["durations"]):
            try:
                m = build_mix(clips, self.cfg, str(self.store.outputs / "mixes"), d)
            except Exception as e:  # noqa: BLE001
                self.store.log_error(None, "mix", e, duration=d)
                continue
            if m:
                mixes.append(m)
        (self.store.outputs / "manifests" / "mixes.json").write_text(json.dumps(mixes, indent=1))
        return mixes

    def publish(self, games: list[Game], qa_results: Optional[list[dict]] = None) -> None:
        from .manifests.manifest import final_report, write_manifests
        from .review.review import build_review_page
        cands = self.all_candidates(games)
        write_manifests(cands, self.store)
        build_review_page(cands, self.store)
        status = {g.game_id: self.store.status(g.game_id) for g in games}
        if qa_results is None:
            qa_results = _load_json(self.store.reports / "qa.json", {}).get("results", [])
        final_report(cands, status, qa_results, self.store)

    # ------------------------------------------------------------ helpers
    def _identity(self, gid: str, pid: str) -> Optional[dict]:
        return self.store.load(gid, f"identity/{pid}_seeded") or self.store.load(gid, f"identity/{pid}")

    def _event(self, gid: str, pid: str) -> Optional[EventResult]:
        d = self.store.load(gid, f"events/{pid}_seeded") or self.store.load(gid, f"events/{pid}")
        return EventResult.from_dict(d) if d else None


def select_games(cfg: dict, only: Optional[list[str]]) -> list[Game]:
    return games_from_config(cfg, only)


def _frame_at(path: str, t: float):
    import cv2
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def _boxes_at(tracking: dict, t: float) -> list[list[float]]:
    out = []
    for tr in tracking.get("tracks", []):
        times = tr.get("times") or []
        if not times:
            continue
        i = min(range(len(times)), key=lambda k: abs(times[k] - t))
        if abs(times[i] - t) < 0.5:
            out.append(tr["boxes"][i])
    return out


def _link(src: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for f in [src, src.with_suffix(".jpg"), src.with_name(src.stem + "_music.mp4")]:
        if f.exists():
            d = dst_dir / f.name
            if not d.exists():
                try:
                    d.hardlink_to(f)
                except OSError:
                    import shutil
                    shutil.copy2(f, d)


def _load_json(p: Path, default):
    return json.loads(p.read_text()) if p.exists() else default
