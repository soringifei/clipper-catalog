"""Text-only quality triage for explicitly reviewed public NASA transcript payloads.

Unavailable, partial or malformed evaluations remain UNVERIFIED; drafts may be
retained locally but never acquire a passing score. No visual/audio acceptance.
No evaluator is loaded or called without a receipt bound to outgoing text.
"""
import importlib.util
import re
import hashlib
import json
import math
from datetime import datetime, timezone
from urllib.parse import urlsplit
from pathlib import Path

JEV_CALL = Path(r"C:\Users\sgife\dev\sorinOS\scripts\jev_call.py")
HOOK_LABELS = ["no hook, slow start", "weak hook", "ok hook", "strong hook", "irresistible hook"]
DESC_LABELS = ["useless or misleading", "weak", "ok", "good", "excellent, specific and honest"]
THRESH = {"hook": 1.5, "whole": 0.4, "slop": 0.6, "desc": 1.0}
HARD_HOOK = 1.0  # auto candidates are dropped only below this hook score (1.0-1.5 = flagged, kept)
_jev = None


def payload(item):
    """The exact text fields allowed to leave this process after public review."""
    return {key: re.sub(r"\s+", " ", str(item.get(key) or "")).strip()[:limit]
            for key, limit in (("opening", 300), ("text", 2500), ("description", 300))}


def payload_sha256(item):
    raw = json.dumps(payload(item), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def source_evidence_valid(item):
    """Require an explicit local review receipt bound to the actual outgoing text.

    A source name/URL alone is not proof. This validates an already supplied
    receipt; it does not independently establish rights or public availability.
    """
    evidence = item.get("sourceEvidence")
    if not isinstance(evidence, dict) or evidence.get("status") != "VERIFIED_PUBLIC":
        return False
    try:
        url = urlsplit(evidence.get("sourceUrl", ""))
        host = (url.hostname or "").lower()
        if url.scheme != "https" or url.username or url.password or url.port not in (None, 443):
            return False
        if not (host == "nasa.gov" or host.endswith(".nasa.gov")):
            return False
        reviewed = datetime.fromisoformat(evidence.get("reviewedAt", "").replace("Z", "+00:00"))
        if reviewed.tzinfo is None or reviewed > datetime.now(timezone.utc):
            return False
    except (ValueError, TypeError, AttributeError):
        return False
    reviewer = evidence.get("reviewedBy")
    return (isinstance(reviewer, str) and bool(reviewer.strip())
            and evidence.get("contentSha256") == payload_sha256(item))


def _default_evaluate():
    global _jev
    if _jev is None:
        try:
            spec = importlib.util.spec_from_file_location("jev_call", JEV_CALL)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            # Called only after every outgoing item has a valid bound receipt.
            _jev = lambda state, questions, purpose: mod.evaluate(
                state, questions, purpose, public_data=True)
        except Exception:
            _jev = False
    return _jev or None


def first_seconds(words, start, secs=2.0):
    """Words starting in the first `secs` seconds of the clip (timed words: {'w','s','e'})."""
    return " ".join(w["w"] for w in words if w["s"] < start + secs).strip()


def verdict(j):
    if not j:
        return None
    why = []
    if j.get("hook") is not None and j["hook"] < THRESH["hook"]:
        why.append(f"hook {j['hook']:.1f}")
    if j.get("whole") is not None and j["whole"] < THRESH["whole"]:
        why.append(f"incomplete {j['whole']:.2f}")
    if j.get("slop") is not None and j["slop"] > THRESH["slop"]:
        why.append(f"formulaic text {j['slop']:.2f}")
    if j.get("desc") is not None and j["desc"] < THRESH["desc"]:
        why.append(f"description {j['desc']:.1f}")
    drop = bool(why) and not (len(why) == 1 and why[0].startswith("hook") and j["hook"] >= HARD_HOOK)
    return {"pass": not why, "drop": drop, "why": why}


def score(items, evaluate=None):
    """Only explicitly reviewed public items may be sent. Missing results remain UNVERIFIED. items: [{'key', 'opening', 'text', 'description'}] -> {key: {hook, whole, slop, desc, pass, why}} (missing = unavailable)."""
    items = [dict(it, **payload(it)) for it in items if source_evidence_valid(it)]
    if not items:
        return {}
    evaluate = evaluate or _default_evaluate()
    if not evaluate:
        return {}
    out = {}
    for i in range(0, len(items), 2):
        pair = items[i:i + 2]
        parts, questions = [], {}
        for n, it in enumerate(pair):
            tag = f"c{n}"
            text = re.sub(r"\s+", " ", str(it.get("text") or ""))[:2500]
            parts.append(f"CLIP {tag}\nFirst 2 seconds: \"{it.get('opening') or text[:60]}\"\nFull transcript: \"{text}\""
                         + (f"\nPost description: \"{it['description'][:300]}\"" if it.get("description") else ""))
            questions[f"{tag}_hook"] = {"type": "score", "criteria": HOOK_LABELS,
                                        "instructions": f"Clip {tag}: how strongly do the first 2 seconds make a TikTok viewer keep watching?"}
            questions[f"{tag}_whole"] = {"type": "boolean",
                                         "instructions": f"Clip {tag}: is it one complete, self-contained idea a viewer understands without context?"}
            questions[f"{tag}_slop"] = {"type": "boolean",
                                        "instructions": f"Clip {tag}: judging only the supplied transcript and description, does the wording feel formulaic or AI-generated? Do not infer visual or audio quality."}
            if it.get("description"):
                questions[f"{tag}_desc"] = {"type": "score", "criteria": DESC_LABELS,
                                            "instructions": f"Clip {tag}: quality of the post description (specific, honest, makes people want to watch)."}
        state = ("Short vertical clips cut from public NASA podcasts for a TikTok page (HookHaus). Judge each clip strictly.\n\n"
                 + "\n\n".join(parts))[:12000]
        try:
            ans = evaluate(state, questions, "clip-gate")
        except Exception:
            continue
        if not isinstance(ans, dict):
            continue
        for n, it in enumerate(pair):
            def g(k, field):
                answer = ans.get(f"c{n}_{k}")
                return answer.get(field) if isinstance(answer, dict) else None
            j = {"hook": g("hook", "score"), "whole": g("whole", "probability"), "slop": g("slop", "probability"), "desc": g("desc", "score")}
            limits = {"hook": 4, "whole": 1, "slop": 1, "desc": 4}
            j = {k: (round(float(v), 2) if type(v) in (int, float) and math.isfinite(v)
                     and 0 <= v <= limits[k] else None) for k, v in j.items()}
            required = ("hook", "whole", "slop", "desc") if it.get("description") else ("hook", "whole", "slop")
            if any(j[k] is None for k in required):
                continue
            j.update(verdict(j))
            out[it["key"]] = j
    return out


def candidate_item(c, key=0):
    copy = c.get("copy") or c.get("llm") or {}
    return {"key": key, "opening": first_seconds(c.get("words") or [], c.get("start", 0)) or c.get("hook", ""),
            "text": c.get("text", ""), "description": str(copy.get("description") or "").strip(),
            "sourceEvidence": c.get("sourceEvidence")}


def gate(cands, evaluate=None, drop=True):
    """Scores candidates (make_daily dicts with words/start/text/hook/llm/copy); sets c['jev'].
    drop=True removes failing auto candidates; editor picks are never dropped (flag only). Returns kept list."""
    items = [candidate_item(c, k) for k, c in enumerate(cands)]
    res = score(items, evaluate)
    kept = []
    for k, c in enumerate(cands):
        c["jev"] = res.get(k)
        c["jev_status"] = "SCORED_TEXT_ONLY" if c["jev"] else "UNVERIFIED"
        if drop and c["jev"] and c["jev"]["drop"] and not c.get("copy"):
            continue
        kept.append(c)
    return kept


def opening_from_item(it):
    """For already-rendered clips (captions.json): first ~2 s of the transcript from its words/s rate."""
    wps = next((float(m.group(1)) for n in it.get("quality_notes", []) for m in [re.match(r"([\d.]+) words/s", str(n))] if m), 3.0)
    words = str(it.get("transcript") or it.get("hook_line") or "").split()
    return " ".join(words[:max(3, round(2 * wps))])
