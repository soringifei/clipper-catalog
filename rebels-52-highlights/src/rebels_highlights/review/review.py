"""Static, self-contained review page (review/index.html) + reading exported decisions.

The page works from file:// with no server. Decisions and #52 seeds are kept in
the browser's localStorage and exported as JSON downloads that the user saves
into ``review/``:

* ``review/review_decisions.json``  {play_id: {"decision": "approve"|"reject", "note": str}}
* ``review/seeds.json``             {play_id: {"t": s, "box": [x1,y1,x2,y2], ...}}

Seed boxes are in source pixels when the candidate carries ``seed_frame`` +
``seed_frame_size`` (+ ``seed_frame_t``); otherwise the thumbnail is used and the
seed is flagged ``"approximate": true`` (the pipeline must treat it as a hint).
"""
from __future__ import annotations

import html
import json
import os
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from ..core.models import REVIEW_STATUSES

_GROUP_TITLES = {"AUTO_APPROVED": "Auto-approved", "REVIEW": "Needs review", "REJECTED": "Rejected"}


def _as_dict(c: Any) -> dict:
    return dict(c) if isinstance(c, dict) else c.to_dict()


def hhmmss(s: Optional[float]) -> str:
    if s is None:
        return "--:--:--"
    s = max(0, int(round(float(s))))
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def youtube_link(url: Optional[str], t: Optional[float]) -> Optional[str]:
    if not url or t is None:
        return None
    u = urlparse(url)
    host = (u.netloc or "").lower()
    if not any(h in host for h in ("youtube.com", "youtu.be")):
        return None
    q = [(k, v) for k, v in parse_qsl(u.query) if k != "t"]
    q.append(("t", f"{max(0, int(float(t)))}s"))
    return urlunparse(u._replace(query=urlencode(q)))


def _rel(path: Optional[str], root: Path, review_dir: Path) -> Optional[str]:
    if not path:
        return None
    p = Path(path)
    if not p.is_absolute():
        p = root / p
    return Path(os.path.relpath(p, review_dir)).as_posix()


def _bar(label: str, v: Any) -> str:
    try:
        f = max(0.0, min(1.0, float(v or 0)))
    except (TypeError, ValueError):
        f = 0.0
    cls = "hi" if f >= 0.8 else "mid" if f >= 0.55 else "lo"
    return (f'<div class="bar"><span class="bl">{html.escape(label)}</span>'
            f'<span class="bt"><span class="bf {cls}" style="width:{f * 100:.0f}%"></span></span>'
            f'<span class="bv">{f:.2f}</span></div>')


def _card(d: dict, root: Path, review_dir: Path) -> str:
    e = html.escape
    pid = str(d.get("play_id") or d.get("clip_id"))
    t0 = d.get("source_start_s")
    link = youtube_link(d.get("source_url"), t0)
    ts = e(hhmmss(t0)) + (f"–{e(hhmmss(d.get('source_end_s')))}" if d.get("source_end_s") else "")
    ts_html = f'<a href="{e(link)}" target="_blank" rel="noopener">{ts}</a>' if link else ts
    thumb = _rel(d.get("thumbnail_file"), root, review_dir)
    video = _rel(d.get("output_file"), root, review_dir)
    seed_img = _rel(d.get("seed_frame"), root, review_dir)
    fsize = d.get("seed_frame_size") or d.get("frame_size") or [0, 0]
    seed_t = d.get("seed_frame_t", d.get("impact_time_s", d.get("snap_time_s", t0)))
    exact = bool(seed_img and fsize and fsize[0])
    reasons = "".join(f'<span class="chip">{e(r)}</span>' for r in d.get("review_reasons") or [])
    poss = ", ".join(d.get("possible_play_types") or [])
    media = (f'<video controls preload="none" poster="{e(thumb or "")}" src="{e(video)}"></video>'
             if video else (f'<img src="{e(thumb)}" alt="thumbnail">' if thumb
                            else '<div class="noimg">no render yet</div>'))
    return f'''<article class="card" data-play="{e(pid)}" data-status="{e(str(d.get("review_status")))}"
 data-seed-img="{e(seed_img or thumb or "")}" data-seed-exact="{int(exact)}"
 data-seed-t="{e(str(seed_t if seed_t is not None else ""))}" data-fw="{int(fsize[0] or 0)}" data-fh="{int(fsize[1] or 0)}">
 <div class="media">{media}</div>
 <div class="body">
  <div class="top"><span class="clip">{e(str(d.get("clip_id")))}</span><span class="tier">{e(str(d.get("tier", "")))}</span></div>
  <div class="meta">{e(str(d.get("game_id")))} · {e(str(d.get("opponent", "unknown")))} · {e(str(d.get("date", "unknown")))}</div>
  <div class="meta">Source {ts_html}</div>
  <div class="ptype">{e(str(d.get("play_type", "unclear")))}</div>
  {f'<div class="poss">possible: {e(poss)}</div>' if poss else ''}
  {_bar("identity", d.get("identity_confidence"))}
  {_bar("event", d.get("event_confidence"))}
  {_bar("highlight", d.get("overall_highlight_score"))}
  <div class="chips">{reasons}</div>
  <textarea class="note" placeholder="note"></textarea>
  <div class="btns"><button class="ok" data-act="approve">Approve</button>
  <button class="no" data-act="reject">Reject</button><button class="seed" data-act="seed">Seed #52</button></div>
  <div class="dec"></div>
 </div></article>'''


_CSS = """
:root{--bg:#0d1117;--panel:#161b22;--line:#30363d;--fg:#e6edf3;--mut:#8b949e;--acc:#f0b429;
--ok:#2ea043;--no:#da3633;--mid:#d29922}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.4 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
header{position:sticky;top:0;z-index:5;background:#010409e6;border-bottom:1px solid var(--line);
padding:12px 16px;display:flex;flex-wrap:wrap;gap:10px;align-items:center}
h1{font-size:18px;margin:0 12px 0 0;letter-spacing:.5px}h1 b{color:var(--acc)}
.tab{background:var(--panel);border:1px solid var(--line);color:var(--fg);padding:6px 12px;
border-radius:20px;cursor:pointer}.tab.on{border-color:var(--acc);color:var(--acc)}
.tab .n{opacity:.7;margin-left:4px}.sp{flex:1}
.exp{background:var(--acc);color:#111;border:0;border-radius:6px;padding:7px 12px;font-weight:600;cursor:pointer}
section{padding:16px}section h2{font-size:15px;color:var(--mut);margin:0 0 12px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:14px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;overflow:hidden;display:flex;flex-direction:column}
.card.approve{border-color:var(--ok)}.card.reject{border-color:var(--no);opacity:.75}
.media{background:#000;aspect-ratio:9/16;max-height:420px;display:flex;align-items:center;justify-content:center}
.media video,.media img{width:100%;height:100%;object-fit:contain}.noimg{color:var(--mut)}
.body{padding:10px 12px;display:flex;flex-direction:column;gap:5px}
.top{display:flex;justify-content:space-between}.clip{font-weight:600;word-break:break-all}
.tier{background:var(--acc);color:#111;border-radius:4px;padding:0 6px;font-weight:700}
.meta{color:var(--mut);font-size:12px}.meta a{color:var(--acc)}
.ptype{font-weight:700;text-transform:uppercase;letter-spacing:.4px}.poss{color:var(--mut);font-size:12px}
.bar{display:flex;align-items:center;gap:6px;font-size:12px}.bl{width:62px;color:var(--mut)}
.bt{flex:1;height:7px;background:#21262d;border-radius:4px;overflow:hidden}
.bf{display:block;height:100%}.bf.hi{background:var(--ok)}.bf.mid{background:var(--mid)}.bf.lo{background:var(--no)}
.bv{width:32px;text-align:right}.chips{display:flex;flex-wrap:wrap;gap:4px}
.chip{font-size:10px;background:#30363d;border-radius:10px;padding:1px 7px}
.note{background:#0d1117;color:var(--fg);border:1px solid var(--line);border-radius:6px;min-height:30px;resize:vertical}
.btns{display:flex;gap:6px}.btns button{flex:1;border:1px solid var(--line);background:#21262d;color:var(--fg);
border-radius:6px;padding:6px;cursor:pointer}.btns .ok:hover{background:var(--ok)}.btns .no:hover{background:var(--no)}
.btns .seed:hover{background:var(--acc);color:#111}.dec{font-size:12px;color:var(--acc)}
.hide{display:none}
#modal{position:fixed;inset:0;background:#000d;display:none;z-index:9;align-items:center;justify-content:center;flex-direction:column;gap:8px}
#modal.on{display:flex}#modal img{max-width:95vw;max-height:80vh;cursor:crosshair;border:2px solid var(--acc)}
#modal p{color:var(--fg);margin:0;max-width:90vw;text-align:center}
#modal button{background:#21262d;color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:6px 14px}
"""

_JS = r"""
const KEY='rebels52_review_v1', SKEY='rebels52_seeds_v1';
const load=k=>{try{return JSON.parse(localStorage.getItem(k)||'{}')}catch(e){return {}}};
const save=(k,v)=>{try{localStorage.setItem(k,JSON.stringify(v))}catch(e){}};
let dec=load(KEY), seeds=load(SKEY);
function paint(){document.querySelectorAll('.card').forEach(c=>{const p=c.dataset.play,d=dec[p];
 c.classList.remove('approve','reject');let s=[];
 if(d){c.classList.add(d.decision);s.push(d.decision.toUpperCase())}
 if(seeds[p])s.push('seed @ '+(+seeds[p].t).toFixed(1)+'s'+(seeds[p].approximate?' (approx)':''));
 c.querySelector('.dec').textContent=s.join(' · ');
 const n=c.querySelector('.note');if(d&&d.note&&!n.value)n.value=d.note;});}
function dl(name,obj){const a=document.createElement('a');
 a.href=URL.createObjectURL(new Blob([JSON.stringify(obj,null,1)],{type:'application/json'}));
 a.download=name;document.body.appendChild(a);a.click();a.remove();}
document.querySelectorAll('.tab[data-g]').forEach(t=>t.onclick=()=>{
 document.querySelectorAll('.tab[data-g]').forEach(x=>x.classList.toggle('on',x===t));
 document.querySelectorAll('section').forEach(s=>s.classList.toggle('hide',t.dataset.g!=='ALL'&&s.id!==t.dataset.g));});
let cur=null;const M=document.getElementById('modal'),MI=M.querySelector('img');
document.querySelectorAll('.card button').forEach(b=>b.onclick=()=>{
 const c=b.closest('.card'),p=c.dataset.play,a=b.dataset.act,note=c.querySelector('.note').value;
 if(a==='seed'){if(!c.dataset.seedImg){alert('No seed frame or thumbnail for this play.');return}
  cur=c;MI.src=c.dataset.seedImg;M.querySelector('p').textContent=(c.dataset.seedExact==='1'?
  'Click on #52 (source frame).':'Click on #52. Only the rendered thumbnail is available: seed will be marked approximate.');
  M.classList.add('on');return}
 dec[p]={decision:a,note:note};save(KEY,dec);paint();});
MI.onclick=ev=>{if(!cur)return;const r=MI.getBoundingClientRect();
 const nx=(ev.clientX-r.left)/r.width,ny=(ev.clientY-r.top)/r.height;
 const fw=+cur.dataset.fw||MI.naturalWidth,fh=+cur.dataset.fh||MI.naturalHeight;
 const x=nx*fw,y=ny*fh,exact=cur.dataset.seedExact==='1';
 const s={t:cur.dataset.seedT===''?null:+cur.dataset.seedT,box:[x-40,y-80,x+40,y+80].map(v=>Math.round(v)),
  frame_size:[fw,fh],norm_xy:[+nx.toFixed(4),+ny.toFixed(4)]};
 if(!exact){s.approximate=true;s.image='thumbnail'}
 seeds[cur.dataset.play]=s;save(SKEY,seeds);M.classList.remove('on');cur=null;paint();};
M.querySelector('button').onclick=()=>{M.classList.remove('on');cur=null};
document.getElementById('exp').onclick=()=>{
 document.querySelectorAll('.card').forEach(c=>{const p=c.dataset.play,n=c.querySelector('.note').value;
  if(dec[p])dec[p].note=n;});save(KEY,dec);
 dl('review_decisions.json',dec);if(Object.keys(seeds).length)setTimeout(()=>dl('seeds.json',seeds),300);};
document.getElementById('clr').onclick=()=>{if(confirm('Clear all local decisions and seeds?')){dec={};seeds={};save(KEY,dec);save(SKEY,seeds);
 document.querySelectorAll('.note').forEach(n=>n.value='');paint();}};
paint();
"""


def build_review_page(cands: list, store) -> str:
    review_dir = Path(store.review)
    review_dir.mkdir(parents=True, exist_ok=True)
    root = Path(store.root)
    from ..manifests.manifest import sort_candidates
    rows = sort_candidates([_as_dict(c) for c in cands])
    groups = {s: [r for r in rows if r.get("review_status") == s] for s in REVIEW_STATUSES}
    other = [r for r in rows if r.get("review_status") not in REVIEW_STATUSES]
    groups["REVIEW"] += other  # unknown status -> needs a human
    tabs = [f'<button class="tab on" data-g="ALL">All<span class="n">{len(rows)}</span></button>']
    secs = []
    for s in REVIEW_STATUSES:
        tabs.append(f'<button class="tab" data-g="{s}">{_GROUP_TITLES[s]}'
                    f'<span class="n">{len(groups[s])}</span></button>')
        cards = "\n".join(_card(d, root, review_dir) for d in groups[s]) or \
            '<p class="meta">No clips in this group.</p>'
        secs.append(f'<section id="{s}"><h2>{_GROUP_TITLES[s]} ({len(groups[s])})</h2>'
                    f'<div class="grid">{cards}</div></section>')
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Rebels #52 review</title><style>{_CSS}</style></head><body>
<header><h1>Rebels <b>#52</b> review</h1>{''.join(tabs)}<span class="sp"></span>
<button class="exp" id="exp">Export decisions</button><button class="tab" id="clr">Clear</button></header>
{''.join(secs)}
<div id="modal"><p></p><img alt="seed frame"><button>Cancel</button></div>
<footer class="meta" style="padding:16px">Save exported <code>review_decisions.json</code> and
<code>seeds.json</code> into the <code>review/</code> folder, then run
<code>python -m rebels_highlights review</code> / <code>analyze --resume</code>.</footer>
<script>{_JS}</script></body></html>"""
    out = review_dir / "index.html"
    out.write_text(page, encoding="utf-8")
    return str(out)


def apply_review_decisions(store) -> dict:
    """Read review/review_decisions.json -> {play_id: {"decision", "note"}}.

    Only "approve"/"reject" decisions are kept; malformed entries are ignored.
    The CLI uses the result to copy approved clips into outputs/approved/.
    """
    p = Path(store.review) / "review_decisions.json"
    if not p.exists():
        return {}
    try:
        raw = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    out = {}
    for pid, d in (raw or {}).items():
        if isinstance(d, str):
            d = {"decision": d}
        if not isinstance(d, dict):
            continue
        decision = str(d.get("decision", "")).lower()
        if decision in ("approve", "approved"):
            decision = "approve"
        elif decision in ("reject", "rejected"):
            decision = "reject"
        else:
            continue
        out[str(pid)] = {"decision": decision, "note": str(d.get("note") or "")}
    return out
