"""Highlight manifests (JSON + CSV), ranked text table and the final Markdown report.

Everything written here is derived strictly from the candidates / status dicts
passed in. Nothing is inferred or invented: missing values stay ``unknown``/empty.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Optional

from ..core.models import Candidate, REVIEW_STATUSES

CSV_COLUMNS = [
    "clip_id", "game_id", "play_id", "source_url", "source_video_id", "date", "opponent",
    "player_name", "player_number", "position", "unit", "play_type", "possible_play_types",
    "source_start_s", "snap_time_s", "impact_time_s", "source_end_s", "output_duration_s",
    "track_id", "jersey_confidence", "team_confidence", "reid_confidence",
    "identity_confidence", "event_confidence", "primary_actor_score", "visibility_score",
    "visual_quality_score", "football_impact_score", "editability_score",
    "overall_highlight_score", "tier", "manual_review", "review_status", "review_reasons",
    "ocr_votes", "output_file", "thumbnail_file", "caption",
]

_STATUS_RANK = {s: i for i, s in enumerate(REVIEW_STATUSES)}


def _as_dict(c: Any) -> dict:
    if isinstance(c, dict):
        return dict(c)
    if hasattr(c, "to_dict"):
        return c.to_dict()
    raise TypeError(f"unsupported candidate type {type(c)}")


def sort_candidates(cands: Iterable[Any]) -> list[dict]:
    ds = [_as_dict(c) for c in cands]
    return sorted(ds, key=lambda d: (_STATUS_RANK.get(d.get("review_status"), 99),
                                     -float(d.get("overall_highlight_score") or 0.0),
                                     str(d.get("clip_id", ""))))


def _cell(v: Any) -> Any:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return round(v, 3)
    if isinstance(v, (list, dict)):
        return json.dumps(v, ensure_ascii=False, sort_keys=isinstance(v, dict))
    return v


def write_manifests(cands: list[Candidate], store) -> dict:
    """Write outputs/manifests/highlights.json and highlights.csv."""
    out_dir = Path(store.outputs) / "manifests"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = sort_candidates(cands)
    jp = out_dir / "highlights.json"
    tmp = jp.with_suffix(".tmp")
    tmp.write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    tmp.replace(jp)
    extra = sorted({k for r in rows for k in r} - set(CSV_COLUMNS))
    cols = CSV_COLUMNS + extra
    cp = out_dir / "highlights.csv"
    with cp.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: _cell(r.get(k)) for k in cols})
    return {"json": str(jp), "csv": str(cp)}


def load_manifest(store) -> list[Candidate]:
    jp = Path(store.outputs) / "manifests" / "highlights.json"
    if not jp.exists():
        return []
    return [Candidate.from_dict(d) for d in json.loads(jp.read_text())]


# ---------------------------------------------------------------------------
def _f(v: Any, nd: int = 2) -> str:
    return "-" if v is None else f"{float(v):.{nd}f}"


def ranked_table(cands: list[Candidate]) -> str:
    rows = sort_candidates(cands)
    head = ["rank", "clip", "game", "play_type", "identity", "event", "primary_actor",
            "highlight", "review_status"]
    body = [[str(i + 1), str(r.get("clip_id", "")), str(r.get("game_id", "")),
             str(r.get("play_type", "")), _f(r.get("identity_confidence")),
             _f(r.get("event_confidence")), _f(r.get("primary_actor_score")),
             _f(r.get("overall_highlight_score")), str(r.get("review_status", ""))]
            for i, r in enumerate(rows)]
    widths = [max(len(h), *(len(b[i]) for b in body)) if body else len(h)
              for i, h in enumerate(head)]
    line = lambda cells: "  ".join(c.ljust(w) for c, w in zip(cells, widths)).rstrip()  # noqa: E731
    out = [line(head), "  ".join("-" * w for w in widths)]
    out += [line(b) for b in body]
    return "\n".join(out)


def _read_errors(store) -> list[dict]:
    p = Path(store.reports) / "errors.jsonl"
    errs = []
    if p.exists():
        for ln in p.read_text().splitlines():
            try:
                errs.append(json.loads(ln))
            except json.JSONDecodeError:
                continue
    return errs


def _histogram(vals: list[float], edges=(0.0, 0.4, 0.55, 0.7, 0.8, 0.9, 1.0001)) -> list[tuple[str, int]]:
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        n = sum(1 for v in vals if lo <= v < hi)
        out.append((f"{lo:.2f}-{min(hi, 1.0):.2f}", n))
    return out


def _tuning(rows: list[dict], cfg_ident: dict, qa_results: list[dict]) -> list[str]:
    """Evidence-based suggestions only. Returns an empty-ish note when evidence is thin."""
    recs: list[str] = []
    n = len(rows)
    if n < 5:
        return [f"Only {n} candidate(s): not enough evidence to recommend threshold changes."]
    auto = float(cfg_ident.get("auto_accept", 0.80))
    rmin = float(cfg_ident.get("review_min", 0.55))
    ids = [float(r.get("identity_confidence") or 0) for r in rows]
    near_auto = [v for v in ids if auto - 0.05 <= v < auto]
    near_min = [v for v in ids if rmin - 0.05 <= v < rmin]
    if len(near_auto) >= max(3, 0.2 * n):
        recs.append(f"{len(near_auto)}/{n} candidates sit just below identity.auto_accept={auto:.2f} "
                    f"(within 0.05). Review them manually first; lower auto_accept only if most "
                    f"turn out to be #52.")
    if len(near_min) >= max(3, 0.2 * n):
        recs.append(f"{len(near_min)}/{n} candidates sit just below identity.review_min={rmin:.2f}. "
                    f"Spot-check REJECTED ones before lowering review_min.")
    unread = sum(1 for r in rows if "JERSEY_UNREADABLE" in (r.get("review_reasons") or []))
    if unread >= 0.5 * n:
        recs.append(f"JERSEY_UNREADABLE on {unread}/{n} candidates: add reference photos to "
                    f"reference_player_52/ and/or seed #52 in the review page (review/seeds.json).")
    amb = sum(1 for r in rows if "EVENT_AMBIGUOUS" in (r.get("review_reasons") or []))
    if amb >= 0.5 * n:
        recs.append(f"EVENT_AMBIGUOUS on {amb}/{n} candidates: event labels are heuristic; keep "
                    f"events.auto_label_min as is and confirm labels during review.")
    crop = sum(1 for q in qa_results if not (q.get("checks") or {}).get("safe_region", True))
    if qa_results and crop >= max(2, 0.25 * len(qa_results)):
        recs.append(f"{crop}/{len(qa_results)} clips failed the safe-region QA check: consider "
                    f"lowering render.crop_smoothing or render.max_digital_zoom.")
    if not recs:
        recs.append("No threshold change is supported by the current evidence.")
    return recs


def final_report(cands: list[Candidate], games_status: dict, qa_results: list[dict], store,
                 cfg: Optional[dict] = None, mixes: Optional[list[dict]] = None,
                 plays_analysed: Optional[int] = None) -> str:
    """Write reports/final_report.md. ``games_status``: {game_id: status dict or str}.

    Optional keys read from ``games_status`` values: ``status`` (done/failed/...),
    ``plays`` (int), ``error``. ``mixes`` may be given directly; otherwise
    ``outputs/mixes/*.mp4`` is listed.
    """
    rows = sort_candidates(cands)
    cfg = cfg or {}
    ident_cfg = (cfg.get("identity") or {})
    lines: list[str] = ["# Rebels #52 highlights - final report", ""]

    # games
    done, failed = [], []
    total_plays = 0
    have_plays = False
    for gid, st in sorted((games_status or {}).items()):
        s = st if isinstance(st, dict) else {"status": st}
        status = str(s.get("status", "unknown"))
        (failed if status.lower() in ("failed", "error") else done).append((gid, s))
        if isinstance(s.get("plays"), int):
            total_plays += s["plays"]
            have_plays = True
    if plays_analysed is not None:
        total_plays, have_plays = plays_analysed, True
    lines += ["## Games", f"- Processed: {len(done)}", f"- Failed: {len(failed)}"]
    for gid, s in done:
        lines.append(f"  - {gid}: {s.get('status', 'unknown')}"
                     + (f", {s['plays']} plays" if isinstance(s.get('plays'), int) else ""))
    for gid, s in failed:
        lines.append(f"  - {gid}: FAILED - {str(s.get('error', 'see reports/errors.jsonl'))[:200]}")
    lines.append(f"- Plays analysed: {total_plays if have_plays else 'not reported'}")
    lines.append("")

    # candidates
    status_c = Counter(r.get("review_status") for r in rows)
    auto_th = float(ident_cfg.get("auto_accept", 0.80))
    high = [r for r in rows if float(r.get("identity_confidence") or 0) >= auto_th
            and float(r.get("event_confidence") or 0) >= float((cfg.get("events") or {}).get("auto_label_min", 0.70))]
    selected = [r for r in rows if r.get("output_file")]
    lines += ["## #52 candidates",
              f"- #52 appearances (candidate plays): {len(rows)}",
              f"- High-confidence (identity >= {auto_th:.2f} and event >= auto_label_min): {len(high)}",
              f"- AUTO_APPROVED: {status_c.get('AUTO_APPROVED', 0)}",
              f"- Review required: {status_c.get('REVIEW', 0)}",
              f"- REJECTED: {status_c.get('REJECTED', 0)}",
              f"- Rendered highlight clips: {len(selected)}", ""]

    # play types
    pt = Counter(r.get("play_type", "unclear") for r in rows if r.get("review_status") != "REJECTED")
    lines += ["## Play-type distribution (non-rejected)"]
    lines += [f"- {k}: {v}" for k, v in pt.most_common()] or ["- none"]
    lines.append("")

    # identity distribution
    ids = [float(r.get("identity_confidence") or 0) for r in rows]
    if ids:
        lines += ["## Identity confidence distribution"]
        lines += [f"- {b}: {n}" for b, n in _histogram(ids)]
        lines.append("")

    # best clips
    best = [r for r in rows if r.get("review_status") != "REJECTED"][:5]
    lines += ["## Best clips"]
    if best:
        for r in best:
            lines.append(f"- {r.get('clip_id')} ({r.get('game_id')}, {r.get('play_type')}, "
                         f"score {_f(r.get('overall_highlight_score'))}, {r.get('review_status')})"
                         + (f" -> `{r['output_file']}`" if r.get("output_file") else ""))
    else:
        lines.append("- none")
    lines.append("")

    # mixes
    lines += ["## Mixes"]
    mix_list = mixes
    if mix_list is None:
        mix_dir = Path(store.outputs) / "mixes"
        mix_list = [{"output_file": str(p)} for p in sorted(mix_dir.glob("*.mp4"))] if mix_dir.exists() else []
    if mix_list:
        for m in mix_list:
            if not m:
                continue
            extra = f" ({m['duration_s']} s, {len(m.get('clip_ids', []))} clips)" if m.get("duration_s") else ""
            lines.append(f"- `{m.get('output_file')}`{extra}")
    else:
        lines.append("- none built")
    lines.append("")

    # QA
    if qa_results:
        passed = sum(1 for q in qa_results if q.get("passed"))
        lines += ["## QA", f"- Clips checked: {len(qa_results)}, passed: {passed}, "
                  f"failed: {len(qa_results) - passed}"]
        fails = Counter(f for q in qa_results for f in q.get("failures", []))
        lines += [f"  - {k}: {v}" for k, v in fails.most_common()]
        lines.append("")

    # failure modes
    errs = _read_errors(store)
    reasons = Counter(x for r in rows for x in (r.get("review_reasons") or []))
    lines += ["## Known failure modes"]
    if errs:
        by = Counter((e.get("stage"), (str(e.get("error", ""))[:120])) for e in errs)
        lines.append(f"- Pipeline errors logged in reports/errors.jsonl: {len(errs)}")
        for (stage, msg), n in by.most_common(10):
            lines.append(f"  - [{stage}] x{n}: {msg}")
    else:
        lines.append("- No pipeline errors logged.")
    if reasons:
        lines.append("- Review reasons across candidates:")
        lines += [f"  - {k}: {v}" for k, v in reasons.most_common()]
    lines.append("")

    lines += ["## Recommended tuning"]
    lines += [f"- {t}" for t in _tuning(rows, ident_cfg, qa_results or [])]
    lines.append("")

    lines += ["## Ranked candidates", "", "```", ranked_table(cands) if rows else "(no candidates)", "```", ""]
    p = Path(store.reports) / "final_report.md"
    p.write_text("\n".join(lines))
    return str(p)
