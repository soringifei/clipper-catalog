import csv
import json
from pathlib import Path

import pytest

from rebels_highlights.core.config import load_config
from rebels_highlights.core.media import run_ffmpeg
from rebels_highlights.core.models import Candidate, PlayerTrajectory
from rebels_highlights.core.store import Store
from rebels_highlights.manifests.manifest import (CSV_COLUMNS, final_report, ranked_table,
                                                  write_manifests)
from rebels_highlights.qa.qa import qa_clip, write_qa_report
from rebels_highlights.review.review import apply_review_decisions, build_review_page


@pytest.fixture
def env(tmp_path):
    cfg = load_config(overrides={"root": str(tmp_path)})
    return cfg, Store(cfg)


def _cands():
    base = dict(game_id="game_001", source_url="https://www.youtube.com/watch?v=abc",
                source_video_id="abc", play_type="solo_tackle",
                possible_play_types=["tackle_for_loss"], source_start_s=3725.4, source_end_s=3745.0,
                ocr_votes={"52": 4, "32": 1}, review_reasons=["EVENT_AMBIGUOUS"])
    return [
        Candidate(clip_id="c_rej", play_id="game_001_p003", review_status="REJECTED",
                  overall_highlight_score=0.9, identity_confidence=0.3, **base),
        Candidate(clip_id="c_rev", play_id="game_001_p002", review_status="REVIEW",
                  overall_highlight_score=0.5, identity_confidence=0.6123456, **base),
        Candidate(clip_id="c_auto_lo", play_id="game_001_p004", review_status="AUTO_APPROVED",
                  overall_highlight_score=0.4, identity_confidence=0.85, **base),
        Candidate(clip_id="c_auto_hi", play_id="game_001_p001", review_status="AUTO_APPROVED",
                  overall_highlight_score=0.8, identity_confidence=0.92, **base),
    ]


def test_manifest_roundtrip_and_csv(env):
    cfg, store = env
    out = write_manifests(_cands(), store)
    rows = json.loads(Path(out["json"]).read_text())
    assert [r["clip_id"] for r in rows] == ["c_auto_hi", "c_auto_lo", "c_rev", "c_rej"]
    back = [Candidate.from_dict(r) for r in rows]
    assert back[0].ocr_votes == {"52": 4, "32": 1}
    with open(out["csv"], newline="") as fh:
        rd = csv.DictReader(fh)
        for col in ["clip_id", "source_url", "possible_play_types", "ocr_votes",
                    "overall_highlight_score", "review_reasons", "caption", "thumbnail_file"]:
            assert col in rd.fieldnames
        assert set(CSV_COLUMNS) <= set(rd.fieldnames)
        recs = list(rd)
    assert recs[2]["identity_confidence"] == "0.612"
    assert json.loads(recs[0]["possible_play_types"]) == ["tackle_for_loss"]
    table = ranked_table(_cands())
    assert "c_auto_hi" in table.splitlines()[2]
    rep = final_report(_cands(), {"game_001": {"status": "done", "plays": 12},
                                  "game_002": {"status": "failed", "error": "HTTP 403"}}, [], store)
    txt = Path(rep).read_text()
    assert "Processed: 1" in txt and "Failed: 1" in txt and "Plays analysed: 12" in txt
    assert "not enough evidence" in txt


def test_review_page(env):
    cfg, store = env
    page = Path(build_review_page(_cands(), store)).read_text()
    for g in ("AUTO_APPROVED", "REVIEW", "REJECTED"):
        assert f'id="{g}"' in page
    assert page.count('<article class="card"') == 4
    assert "t=3725s" in page and "01:02:05" in page
    assert "Export decisions" in page and "Seed #52" in page
    assert apply_review_decisions(store) == {}
    (store.review / "review_decisions.json").write_text(json.dumps(
        {"game_001_p001": {"decision": "approve", "note": "x"}, "bad": {"decision": "?"}}))
    assert apply_review_decisions(store) == {"game_001_p001": {"decision": "approve", "note": "x"}}


def _make_clip(path: Path, seconds: float):
    run_ffmpeg(["-f", "lavfi", "-i", f"testsrc2=size=1080x1920:rate=30:duration={seconds}",
                "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
                "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-shortest", str(path)])
    run_ffmpeg(["-ss", "1", "-i", str(path), "-frames:v", "1", str(path.with_suffix(".jpg"))])


def test_qa(env, tmp_path):
    cfg, store = env
    good, short = tmp_path / "good.mp4", tmp_path / "short.mp4"
    _make_clip(good, 16)
    _make_clip(short, 5)
    cand = _cands()[3]
    traj = PlayerTrajectory(play_id=cand.play_id, track_id=1)
    r1 = qa_clip(str(good), cand, traj, cfg)
    assert r1["passed"], r1
    assert r1["checks"]["dims_1080x1920"] and r1["checks"]["audio_present"] and r1["checks"]["not_black"]
    r2 = qa_clip(str(short), cand, None, cfg)
    assert not r2["passed"] and "duration_ok" in r2["failures"]
    # crop metadata: #52 outside safe band -> fail
    good.with_suffix(".crop.json").write_text(json.dumps(
        {"frames": [{"t": i, "crop": [0, 0, 405, 720], "player_x": 20} for i in range(10)]}))
    r3 = qa_clip(str(good), cand, None, cfg)
    assert "safe_region" in r3["failures"]
    rep = json.loads(Path(write_qa_report([r1, r2, r3], store)).read_text())
    assert rep["summary"]["total"] == 3 and rep["summary"]["passed"] == 1
