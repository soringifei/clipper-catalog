"""HookHaus text gate with Jev (typed evaluator; public transcript snippets only).

Jev receives no video frames or audio here. Its scores cannot establish visual
quality, framing, sharpness, subtitle appearance, or final clip acceptance.

Per candidate (2 candidates = 8 questions per Jev call):
  hook   score 0-4  hook strength of the first ~2 s
  whole  boolean    a complete, self-contained idea
  slop   boolean    transcript/description reads like formulaic AI copy
  desc   score 0-4  description quality (only when a description exists)
Verdict: pass unless hook < 1.5, whole < 0.4, slop > 0.6 or desc < 1.0 (flagged in review.html);
auto candidates are DROPPED only for hook < 1.0 or any other failed check. Jev unavailable -> everything passes
unchanged (fail open, jev = None). Uses C:\\Users\\sgife\\dev\\sorinOS\\scripts\\jev_call.py (24 h cache + usage log).
"""
import importlib.util
import re
from pathlib import Path

JEV_CALL = Path(r"C:\Users\sgife\dev\sorinOS\scripts\jev_call.py")
HOOK_LABELS = ["no hook, slow start", "weak hook", "ok hook", "strong hook", "irresistible hook"]
DESC_LABELS = ["useless or misleading", "weak", "ok", "good", "excellent, specific and honest"]
THRESH = {"hook": 1.5, "whole": 0.4, "slop": 0.6, "desc": 1.0}
HARD_HOOK = 1.0  # auto candidates are dropped only below this hook score (1.0-1.5 = flagged, kept)
_jev = None


def _default_evaluate():
    global _jev
    if _jev is None:
        try:
            spec = importlib.util.spec_from_file_location("jev_call", JEV_CALL)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            # The clips are cut from public NASA podcasts: the transcript text is public by source.
            if "public_data" in mod.evaluate.__code__.co_varnames:
                _jev = lambda s, q, p: mod.evaluate(s, q, p, public_data=True)
            else:
            # This pipeline uses published NASA transcripts and public post copy.
            _jev = lambda state, questions, purpose: mod.evaluate(state, questions, purpose, public_data=True)
        except Exception:
            _jev = False
    return _jev or (lambda *a, **k: None)


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
    """items: [{'key', 'opening', 'text', 'description'}] -> {key: {hook, whole, slop, desc, pass, why}} (missing = unavailable)."""
    evaluate = evaluate or _default_evaluate()
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
        ans = evaluate(state, questions, "clip-gate")
        if not ans:
            continue
        for n, it in enumerate(pair):
            g = lambda k, f: (ans.get(f"c{n}_{k}") or {}).get(f)
            j = {"hook": g("hook", "score"), "whole": g("whole", "probability"), "slop": g("slop", "probability"), "desc": g("desc", "score")}
            j = {k: (round(float(v), 2) if isinstance(v, (int, float)) else None) for k, v in j.items()}
            if j["hook"] is None and j["whole"] is None:
                continue
            j.update(verdict(j))
            out[it["key"]] = j
    return out


def gate(cands, evaluate=None, drop=True):
    """Scores candidates (make_daily dicts with words/start/text/hook/llm/copy); sets c['jev'].
    drop=True removes failing auto candidates; editor picks are never dropped (flag only). Returns kept list."""
    items = []
    for k, c in enumerate(cands):
        L = c.get("copy") or c.get("llm") or {}
        items.append({"key": k, "opening": first_seconds(c.get("words") or [], c.get("start", 0)) or c.get("hook", ""),
                      "text": c.get("text", ""), "description": str(L.get("description") or "").strip()})
    res = score(items, evaluate)
    kept = []
    for k, c in enumerate(cands):
        c["jev"] = res.get(k)
        if drop and c["jev"] and c["jev"]["drop"] and not c.get("copy"):
            continue
        kept.append(c)
    return kept


def opening_from_item(it):
    """For already-rendered clips (captions.json): first ~2 s of the transcript from its words/s rate."""
    wps = next((float(m.group(1)) for n in it.get("quality_notes", []) for m in [re.match(r"([\d.]+) words/s", str(n))] if m), 3.0)
    words = str(it.get("transcript") or it.get("hook_line") or "").split()
    return " ".join(words[:max(3, round(2 * wps))])
