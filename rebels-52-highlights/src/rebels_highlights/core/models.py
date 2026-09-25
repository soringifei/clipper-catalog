"""Shared data contracts between pipeline stages.

Every stage reads/writes these as JSON (via ``to_dict``/``from_dict``) under
``cache/<game_id>/<stage>.json`` so stages can be re-run independently.
Times are seconds in the *source* video. Boxes are ``[x1, y1, x2, y2]`` in
source pixels. Every confidence/score is a float in [0, 1].
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Optional

UNKNOWN = "unknown"

# Play-type vocabulary (see events/). "possible_*" types only ever appear in
# ``possible_play_types``, never as the committed ``play_type``.
DEFENSE_TYPES = [
    "solo_tackle", "assisted_tackle", "open_field_tackle", "tackle_for_loss",
    "run_stop_near_los", "sack", "qb_pressure", "forced_fumble",
    "fumble_recovery", "block_shed", "pursuit", "coverage_play", "other_defense",
]
SPECIAL_TEAMS_TYPES = [
    "kickoff_coverage", "punt_coverage", "special_teams_tackle", "return_block",
    "other_special_teams",
]
OTHER_TYPES = ["tackle", "other", "unclear"]
PLAY_TYPES = DEFENSE_TYPES + SPECIAL_TEAMS_TYPES + OTHER_TYPES

REVIEW_REASONS = [
    "IDENTITY_LOW", "JERSEY_UNREADABLE", "TRACK_ID_SWITCH", "OCCLUSION",
    "LOW_RESOLUTION", "EVENT_AMBIGUOUS", "POSSIBLE_FORCED_FUMBLE",
    "UNCERTAIN_LOS", "DUPLICATE_PLAY", "POOR_VERTICAL_CROP",
    "PLAYER_NOT_PRIMARY_ACTOR", "BALL_NOT_CLEARLY_VISIBLE",
]

REVIEW_STATUSES = ["AUTO_APPROVED", "REVIEW", "REJECTED"]


class _Serializable:
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)  # type: ignore[arg-type]

    @classmethod
    def from_dict(cls, d: dict[str, Any]):
        known = {f.name for f in fields(cls)}  # type: ignore[arg-type]
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class Game(_Serializable):
    game_id: str                      # game_001, game_002, ...
    url: Optional[str] = None
    local_path: Optional[str] = None  # used instead of url when supplied
    date: str = UNKNOWN
    opponent: str = UNKNOWN


@dataclass
class VideoInfo(_Serializable):
    game_id: str
    path: str
    source_url: Optional[str] = None
    source_video_id: Optional[str] = None
    title: str = UNKNOWN
    duration_s: float = 0.0
    fps: float = 0.0
    width: int = 0
    height: int = 0
    format: str = UNKNOWN
    has_audio: bool = False
    proxy_path: Optional[str] = None


@dataclass
class Scene(_Serializable):
    scene_id: int
    start_s: float
    end_s: float
    kind: str = "live"  # live | replay | non_play (scoreboard, sideline, break)


@dataclass
class Play(_Serializable):
    play_id: str                   # game_001_p014
    game_id: str
    start_s: float
    end_s: float
    estimated_snap_s: Optional[float] = None
    snap_confidence: float = 0.0
    scene_ids: list[int] = field(default_factory=list)
    unit: str = UNKNOWN            # defense | special_teams | offense | unknown
    unit_confidence: float = 0.0
    canonical_play_id: Optional[str] = None  # set when this is a duplicate/replay
    replay_segments: list[dict[str, float]] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class Detection(_Serializable):
    t: float
    frame_idx: int
    box: list[float]
    score: float
    track_id: Optional[int] = None


@dataclass
class Track(_Serializable):
    """One tracklet within a single scene (IDs never cross hard cuts)."""
    track_id: int
    scene_id: int
    times: list[float] = field(default_factory=list)
    boxes: list[list[float]] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    team_prob: float = 0.5         # P(Bucharest Rebels)
    embedding: Optional[list[float]] = None  # mean appearance embedding


@dataclass
class IdentityEvidence(_Serializable):
    track_id: int
    scene_id: int
    jersey_confidence: float = 0.0
    team_confidence: float = 0.0
    reid_confidence: float = 0.0
    temporal_confidence: float = 0.0
    reference_confidence: float = 0.0
    position_prior: float = 0.0
    identity_confidence: float = 0.0
    ocr_votes: dict[str, int] = field(default_factory=dict)
    source: str = "auto"  # auto | manual_seed
    review_reasons: list[str] = field(default_factory=list)


@dataclass
class PlayerTrajectory(_Serializable):
    """#52's path through one play, plus the likely ball carrier/QB."""
    play_id: str
    track_id: Optional[int]
    times: list[float] = field(default_factory=list)
    boxes: list[list[float]] = field(default_factory=list)
    target_times: list[float] = field(default_factory=list)   # ball carrier / QB
    target_boxes: list[list[float]] = field(default_factory=list)
    target_role: str = UNKNOWN  # ball_carrier | qb | returner | unknown
    pose_features: dict[str, float] = field(default_factory=dict)


@dataclass
class EventResult(_Serializable):
    play_id: str
    play_type: str = "unclear"
    possible_play_types: list[str] = field(default_factory=list)
    event_confidence: float = 0.0
    impact_s: Optional[float] = None
    contact_point: Optional[list[float]] = None  # [x, y] source px
    los_x: Optional[float] = None
    los_confidence: float = 0.0
    primary_actor_score: float = 0.0
    features: dict[str, float] = field(default_factory=dict)
    review_reasons: list[str] = field(default_factory=list)


@dataclass
class Candidate(_Serializable):
    """Scored, reviewable highlight candidate (one per canonical play)."""
    clip_id: str
    game_id: str
    play_id: str
    source_url: Optional[str] = None
    source_video_id: Optional[str] = None
    date: str = UNKNOWN
    opponent: str = UNKNOWN
    player_name: str = UNKNOWN
    player_number: int = 52
    position: str = "MLB"
    unit: str = UNKNOWN
    play_type: str = "unclear"
    possible_play_types: list[str] = field(default_factory=list)
    source_start_s: float = 0.0
    snap_time_s: Optional[float] = None
    impact_time_s: Optional[float] = None
    source_end_s: float = 0.0
    output_duration_s: float = 0.0
    track_id: Optional[int] = None
    jersey_confidence: float = 0.0
    team_confidence: float = 0.0
    reid_confidence: float = 0.0
    identity_confidence: float = 0.0
    event_confidence: float = 0.0
    primary_actor_score: float = 0.0
    visibility_score: float = 0.0
    visual_quality_score: float = 0.0
    football_impact_score: float = 0.0
    editability_score: float = 0.0
    overall_highlight_score: float = 0.0
    tier: str = "D"
    manual_review: bool = True
    review_status: str = "REVIEW"
    review_reasons: list[str] = field(default_factory=list)
    ocr_votes: dict[str, int] = field(default_factory=dict)
    caption: str = ""
    output_file: Optional[str] = None
    thumbnail_file: Optional[str] = None
    replay_segments: list[dict[str, float]] = field(default_factory=list)
    canonical_play_id: Optional[str] = None
    seed_frame: Optional[str] = None          # landscape pre-snap JPG for manual seeding
    seed_frame_size: Optional[list[int]] = None
    seed_frame_t: Optional[float] = None
