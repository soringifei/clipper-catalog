"""HookHaus daily batch: up to 7 vertical talking-head clips from rights-cleared long sources.

Prepares only. It never publishes or schedules anything on TikTok; publishing is Sorin's decision.

  python make_daily.py                     # today's batch (default 7 clips)
  python make_daily.py --date 2026-09-27 --force   # rebuild that date (frees its used.json entries)
  python make_daily.py --transcribe-only   # warm the transcript cache
  python make_daily.py --review-only --date 2026-09-27   # rebuild review.html from captions.json

Pipeline: local faster-whisper transcript (cached per source) -> sentence windows of 20-45 s ->
heuristic hook/standalone/energy score -> optional local Ollama rerank + copy -> pause tightening,
face-aware 9:16 framing, word-timed ASS captions, -14 LUFS audio -> cover per clip -> captions.json,
review.html, QA frames, used.json. Runs hidden (pythonw); light CPU settings while a game is running.
"""
import argparse, datetime as dt, html, importlib.util, json, math, os, re, shutil, subprocess, sys, tempfile, time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CLIPPER = ROOT.parent
TRANSCRIPTS = ROOT / "transcripts"
USED = ROOT / "used.json"
LOG = ROOT / "daily.log"
W, H = 1080, 1920
CAPTION_BOTTOM = 1390  # caption baseline; TikTok's caption/UI overlay covers roughly the bottom 25%
NO_WINDOW = 0x08000000
BELOW_NORMAL = 0x00004000
GAME_PREFIXES = ("dota2", "league of legends", "leagueclient", "nba2k", "cs2")
OLLAMA = "http://127.0.0.1:11434"
FILLER_START = {"so", "and", "but", "um", "uh", "yeah", "well", "okay", "ok", "like", "because", "or", "right", "anyway", "also", "then"}
BACKREF_START = {"that", "it", "this", "he", "she", "they", "those", "these", "which", "there"}
STRONG = {"never", "first", "only", "most", "best", "worst", "scared", "afraid", "died", "dead", "crazy", "secret", "wrong",
          "actually", "nobody", "everyone", "every", "mistake", "fail", "failed", "failure", "risk", "dangerous", "fire", "explode",
          "exploded", "impossible", "surprised", "amazing", "incredible", "love", "hate", "fear", "honestly", "truth", "real",
          "biggest", "hardest", "favorite", "weird", "funny", "problem", "why", "how", "what", "mom", "kids", "home"}
CAPTION_DROP = {"um", "uh", "uhm", "erm", "hmm", "mm"}
BANNED = ["you won't believe", "won't believe", "mind-blowing", "mind blowing", "game-changer", "game changer", "insane",
          "must watch", "must-watch", "wait for it", "delve", "unleash", "journey", "buckle up", "🔥"]


def log(msg):
    line = f"{dt.datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass
    if sys.stdout:
        try:
            print(line, flush=True)
        except Exception:
            pass


def game_running():
    import psutil
    for p in psutil.process_iter(["name"]):
        n = (p.info.get("name") or "").lower()
        if n.startswith(GAME_PREFIXES):
            return n
    return None


def lower_priority():
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass


def ffbin(name):
    p = os.environ.get(name.upper()) or shutil.which(name)
    if not p:
        raise SystemExit(f"{name} not found on PATH")
    return p


def run(cmd, cwd=None, check=True, capture=True):
    flags = (NO_WINDOW | BELOW_NORMAL) if os.name == "nt" else 0
    r = subprocess.run(cmd, cwd=cwd, capture_output=capture, text=True, encoding="utf-8", errors="replace", creationflags=flags)
    if check and r.returncode != 0:
        raise RuntimeError(f"command failed ({r.returncode}): {' '.join(map(str, cmd))[:400]}\n{(r.stderr or '')[-1500:]}")
    return r


def load_sources():
    data = json.loads((ROOT / "sources.json").read_text(encoding="utf-8"))
    out = []
    for s in data["sources"]:
        s["abspath"] = (ROOT / s["path"]).resolve()
        out.append(s)
    return out


def probe(path):
    r = run([ffbin("ffprobe"), "-v", "error", "-show_entries", "format=duration:stream=codec_type,width,height,r_frame_rate",
             "-of", "json", str(path)])
    j = json.loads(r.stdout)
    v = next(s for s in j["streams"] if s["codec_type"] == "video")
    return {"w": int(v["width"]), "h": int(v["height"]), "dur": float(j["format"]["duration"])}


# ---------------------------------------------------------------- transcription (cached per source)

def whisper_model_dir(size):
    snap = Path.home() / ".cache" / "huggingface" / "hub" / f"models--Systran--faster-whisper-{size}" / "snapshots"
    if snap.exists():
        for p in snap.iterdir():
            if (p / "model.bin").exists():
                return str(p)
    return None


WHISPER_CLI = Path(r"C:\Users\sgife\dev\sorinOS\runtime\whisper-cuda\whisper-cli.exe")
WHISPER_MODEL = Path(r"C:\Users\sgife\dev\sorinOS\runtime\whisper\ggml-large-v3-turbo-q5_0.bin")


def whisper_cpp_words(wav):
    """Word-level timings from whisper.cpp: one segment per word (-ml 1 -sow), token probabilities from -ojf."""
    base = wav.with_suffix("")
    run([str(WHISPER_CLI), "-m", str(WHISPER_MODEL), "-f", str(wav), "-l", "en", "-ml", "1", "-sow", "-ojf", "-mc", "0",
         "-of", str(base), "-np", "-t", "4"])
    j = json.loads(Path(str(base) + ".json").read_text(encoding="utf-8"))
    words = []
    for seg in j["transcription"]:
        t = seg["text"].strip()
        if not t or t.startswith("[") or t.startswith("("):
            continue
        toks = [k for k in seg.get("tokens", []) if not k.get("text", "").startswith("[_")
                and re.search(r"[A-Za-z0-9]", k.get("text", ""))]
        p = min((k.get("p", 1) for k in toks), default=1)
        words.append({"s": seg["offsets"]["from"] / 1000, "e": seg["offsets"]["to"] / 1000, "w": t, "p": round(p, 3)})
    Path(str(base) + ".json").unlink(missing_ok=True)
    return words


def tighten_word_starts(words):
    """ASR often stretches a word over the silence before it; cap each word's length so cuts land on speech."""
    for w in words:
        cap = 0.18 + 0.085 * len(w["w"])
        if w["e"] - w["s"] > cap:
            w["s"] = round(w["e"] - cap, 3)
    return words


def transcribe(src, light):
    TRANSCRIPTS.mkdir(exist_ok=True)
    out = TRANSCRIPTS / f"{src['id']}.json"
    if out.exists():
        return json.loads(out.read_text(encoding="utf-8"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    log(f"transcribing {src['id']} (light={light})")
    wav = Path(tempfile.gettempdir()) / f"hookhaus-{src['id']}.wav"
    run([ffbin("ffmpeg"), "-v", "error", "-y", "-i", str(src["abspath"]), "-vn", "-ac", "1", "-ar", "16000", str(wav)])
    t0 = time.time()
    words, name = None, None
    if not light and WHISPER_CLI.exists() and WHISPER_MODEL.exists():
        try:
            words = whisper_cpp_words(wav)
            name = "whisper.cpp large-v3-turbo q5_0 (CUDA)"
        except Exception as e:
            log(f"whisper.cpp failed ({e}); falling back to faster-whisper CPU")
    if words is None:
        from faster_whisper import WhisperModel
        size = "small" if light or not whisper_model_dir("medium") else "medium"
        model = WhisperModel(whisper_model_dir(size), device="cpu", compute_type="int8", cpu_threads=4 if light else 8,
                             num_workers=1, local_files_only=True)
        name = f"faster-whisper-{size} (CPU int8)"
        log(f"transcribing {src['id']} with {name}")
        segs, _ = model.transcribe(str(wav), language=src.get("lang", "en"), beam_size=1 if light else 5,
                                   word_timestamps=True, vad_filter=True, condition_on_previous_text=False)
        words = []
        for s in segs:
            for w in s.words or []:
                t = w.word.strip()
                if t:
                    words.append({"s": round(w.start, 3), "e": round(w.end, 3), "w": t, "p": round(w.probability, 3)})
    words = tighten_word_starts(words)
    res = {"source": src["id"], "model": name, "language": src.get("lang", "en"), "seconds": round(time.time() - t0, 1),
           "words": words}
    out.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    try:
        wav.unlink()
    except OSError:
        pass
    log(f"transcribed {src['id']}: {len(words)} words in {res['seconds']} s")
    return res


# ---------------------------------------------------------------- candidate windows + heuristic score

def sentences(words):
    out, cur = [], []
    for i, w in enumerate(words):
        cur.append(w)
        gap = words[i + 1]["s"] - w["e"] if i + 1 < len(words) else 99
        if re.search(r"[.?!][\"')]*$", w["w"]) or gap > 1.6:
            out.append({"s": cur[0]["s"], "e": cur[-1]["e"], "words": cur, "text": " ".join(x["w"] for x in cur),
                        "closed": bool(re.search(r"[.?!][\"')]*$", w["w"]))})
            cur = []
    if cur:
        out.append({"s": cur[0]["s"], "e": cur[-1]["e"], "words": cur, "text": " ".join(x["w"] for x in cur), "closed": False})
    return out


def norm(w):
    return re.sub(r"[^a-z0-9']", "", w.lower())


def score_window(sents):
    text = " ".join(s["text"] for s in sents)
    words = [w for s in sents for w in s["words"]]
    dur = sents[-1]["e"] - sents[0]["s"]
    first = sents[0]
    fw = [norm(w["w"]) for w in first["words"]]
    notes, sc = [], 0.0
    if fw and fw[0] in FILLER_START:
        sc -= 3; notes.append("filler start")
    if fw and fw[0] in BACKREF_START:
        sc -= 2; notes.append("back-reference start")
    if len(fw) < 4:
        sc -= 1
    if len(fw) > 36:
        sc -= 0.7; notes.append("long first sentence")
    hook_words = [norm(w["w"]) for w in words if w["s"] < first["s"] + 3.0]
    if first["text"].rstrip().endswith("?"):
        sc -= 1.0; notes.append("starts on the interviewer's question")
    if {"you", "your"} & set(hook_words):
        sc += 0.8
    if any(re.search(r"\d", w) for w in hook_words) or {"one", "two", "three", "hundred", "thousand", "million"} & set(hook_words):
        sc += 0.7
    sc += min(3, len(STRONG & set(fw))) * 0.8
    if not sents[-1]["closed"]:
        sc -= 2; notes.append("open ending")
    if sents[-1]["text"].rstrip().endswith("?"):
        sc -= 2; notes.append("ends on a question")
    low = text.lower()
    if any(s["text"].rstrip().endswith("?") for s in sents[1:-1]):
        sc -= 2.5; notes.append("new question mid-clip")
    for ph in ("next question", "great question", "question is for", "our next", "thank you so much", "welcome to the",
               "thanks for joining", "we're down to", "last question", "thank you both"):
        if ph in low:
            sc -= 3; notes.append(f"moderator line: {ph}")
    dis = sum(1 for w in fw if w in CAPTION_DROP)
    if dis:
        sc -= min(2, dis); notes.append("disfluent hook")
    for ref in ("as i said", "like i said", "as i mentioned", "we talked about", "earlier", "that question"):
        if ref in low:
            sc -= 1.5; notes.append(f"refers back: {ref}")
    wps = len(words) / max(dur, 1)
    if wps < 1.8:
        sc -= 2; notes.append(f"slow {wps:.1f} w/s")
    elif 2.3 <= wps <= 3.8:
        sc += 0.8
    gaps = sum(max(0, words[i + 1]["s"] - words[i]["e"] - 1.0) for i in range(len(words) - 1))
    if gaps / max(dur, 1) > 0.12:
        sc -= 1.5; notes.append("long pauses")
    lowconf = sum(1 for w in words if w.get("p", 1) < 0.4) / max(1, len(words))
    if lowconf > 0.12:
        sc -= 2; notes.append("low ASR confidence")
    if 25 <= dur <= 40:
        sc += 0.5
    return sc, notes, wps


def candidates(src, tr, used):
    words = tr["words"]
    if not words:
        return []
    total = words[-1]["e"]
    lo, hi = src.get("skip_head", 30), total - src.get("skip_tail", 30)
    ss = [s for s in sentences(words) if s["s"] >= lo and s["e"] <= hi]
    cuts = source_cuts(src)
    out = []
    for i in range(len(ss)):
        for j in range(i, len(ss)):
            dur = ss[j]["e"] - ss[i]["s"]
            if dur > 44:
                break
            if dur < 21 or not ss[j]["closed"]:
                continue
            if any(ss[k + 1]["s"] - ss[k]["e"] > 3.0 for k in range(i, j)):
                break  # a long silence usually means a topic or speaker change
            win = [trim_lead(ss[i])] + ss[i + 1:j + 1]
            a, b = win[0]["s"], ss[j]["e"]
            if b - a < 20:
                continue
            if overlaps_used(src["id"], a, b, used):
                continue
            sc, notes, wps = score_window(win)
            if late_switch(win, cuts):
                sc -= 2.5; notes = notes + ["camera/speaker switch late in clip"]
            out.append({"source": src["id"], "start": a, "end": b, "dur": round(b - a, 2), "h_score": round(sc, 2),
                        "notes": notes, "wps": round(wps, 2), "text": " ".join(s["text"] for s in win),
                        "hook": win[0]["text"], "words": [w for s in win for w in s["words"]],
                        "sents": [s["words"] for s in win]})
    out.sort(key=lambda c: -c["h_score"])
    picked = []
    for c in out:  # overlapping alternatives are allowed (the LLM picks between them); near-duplicates are not
        if any(abs(c["start"] - p["start"]) <= 4 and abs(c["end"] - p["end"]) <= 4 for p in picked):
            continue
        if sum(1 for p in picked if c["start"] < p["end"] and c["end"] > p["start"]) >= 2:
            continue
        picked.append(c)
    return picked


LEAD_DROP = {"well", "so", "uh", "um", "and", "but", "yeah", "okay", "ok", "oh", "now", "anyway", "right", "absolutely", "yes",
             "sure", "definitely", "wow"}


def source_cuts(src):
    """Scene cuts for the whole source (cached). In multi-camera interviews a cut + new sentence usually = new speaker."""
    TRANSCRIPTS.mkdir(exist_ok=True)
    f = TRANSCRIPTS / f"{src['id']}.cuts.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    r = run([ffbin("ffmpeg"), "-hide_banner", "-nostats", "-i", str(src["abspath"]), "-an", "-vf",
             "scale=160:-2,select='gt(scene,0.28)',showinfo", "-f", "null", "-"], check=False)
    cuts = [round(float(x), 2) for x in re.findall(r"pts_time:([\d.]+)", r.stderr or "")]
    f.write_text(json.dumps(cuts), encoding="utf-8")
    return cuts


def late_switch(win, cuts):
    a, b = win[0]["s"], win[-1]["e"]
    for c in cuts:
        if a + 0.45 * (b - a) < c < b - 0.3:
            if any(c - 0.4 <= s["s"] <= c + 1.5 for s in win[1:]):
                return True
    return False


def trim_lead(sent):
    """Start on the first real word: drop 'Well,', 'So,', 'Uh,', 'You know,', 'I mean,' at the head of the hook."""
    ws = list(sent["words"])
    while len(ws) > 4:
        n0, n1 = norm(ws[0]["w"]), norm(ws[1]["w"])
        if n0 in LEAD_DROP:
            ws = ws[1:]
        elif (n0, n1) in (("you", "know"), ("i", "mean")) and re.search(r",$", ws[1]["w"]):
            ws = ws[2:]
        else:
            break
    if len(ws) == len(sent["words"]):
        return sent
    ws = [dict(w) for w in ws]
    ws[0]["w"] = ws[0]["w"][:1].upper() + ws[0]["w"][1:]
    return {**sent, "s": ws[0]["s"], "words": ws, "text": " ".join(w["w"] for w in ws)}


def overlaps_used(sid, a, b, used):
    for u in used:
        if u["source"] != sid:
            continue
        inter = min(b, u["end"]) - max(a, u["start"])
        if inter > 0.25 * min(b - a, u["end"] - u["start"]):
            return True
    return False


# ---------------------------------------------------------------- local LLM rerank + copy (free, optional)

LLM_PROMPT = """You pick short-form clips from a long interview. Rate this excerpt as a standalone 20-45 second vertical video.
Speaker context: {speakers}.

Excerpt, one numbered sentence per line:
{text}

Return JSON only:
{{"drop_first": 0-2 (how many leading sentences to cut because they are the interviewer/moderator talking or a weak lead-in),
 "drop_last": 0-2 (how many trailing sentences to cut because they are the interviewer/moderator, start a new topic, or trail off),
 then rate the excerpt AFTER those cuts:
 "hook": 1-10 (do the first words grab attention on their own, without context?),
 "standalone": 1-10 (one complete idea, understandable with zero context, ends on a finished thought?),
 "interest": 1-10 (would a curious stranger watch to the end?),
 "reject_reason": "" or a short reason if this should not be posted,
 "cover_text": "a 3-5 word phrase a person would actually say about this excerpt, not a list of keywords, no commas, no clickbait. Example: 'Why the runway won'",
 "description": "ONE plain sentence, max 110 characters, shaped like '<who> on <the specific point>.' Example: 'STS-1 pilot Bob Crippen on why a runway landing beat splashing down.' Follow the speaker rule above. Never use: highlights, emphasizes, reflects, discusses, shares, delves, journey, incredible, inspiring. No emoji, no hype, no question to the audience.",
 "hashtags": ["#tag1", "#tag2", "#tag3"] (max 3, specific and relevant)}}"""


def ollama_available(start=True):
    if os.environ.get("SORINOS_NO_LOCAL_MODELS") == "1":
        return []
    for attempt in range(8):
        try:
            with urllib.request.urlopen(OLLAMA + "/api/tags", timeout=5) as r:
                return [m["name"] for m in json.loads(r.read())["models"]]
        except Exception:
            exe = shutil.which("ollama") or str(Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe")
            if attempt == 0 and start and Path(exe).exists():
                log("ollama not reachable; starting a hidden 'ollama serve'")
                subprocess.Popen([exe, "serve"], creationflags=(NO_WINDOW | 0x00000008) if os.name == "nt" else 0,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(3)
    return []


def llm_rate(c, speakers, model):
    numbered = "\n".join(f"[{i + 1}] " + " ".join(w["w"] for w in s) for i, s in enumerate(c["sents"]))
    body = {"model": model, "prompt": LLM_PROMPT.format(speakers=speakers, text=numbered), "format": "json",
            "stream": False, "keep_alive": "3m", "options": {"temperature": 0.2, "num_ctx": 2048}}
    req = urllib.request.Request(OLLAMA + "/api/generate", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=240) as r:
        out = json.loads(json.loads(r.read())["response"])
    for k in ("hook", "standalone", "interest", "drop_first", "drop_last"):
        try:
            out[k] = float(out.get(k, 0))
        except (TypeError, ValueError):
            out[k] = 0.0
    return out


def apply_trims(c):
    """Apply the LLM's drop_first/drop_last (interviewer lines, trailing new topic) when >= 20 s remain."""
    L = c.get("llm") or {}
    df, dl = int(max(0, min(2, L.get("drop_first", 0)))), int(max(0, min(2, L.get("drop_last", 0))))
    if not (df or dl) or df + dl >= len(c["sents"]):
        return
    ss = c["sents"][df:len(c["sents"]) - dl]
    first = trim_lead({"s": ss[0][0]["s"], "e": ss[0][-1]["e"], "words": ss[0], "text": " ".join(w["w"] for w in ss[0])})
    ss = [first["words"]] + ss[1:]
    a, b = ss[0][0]["s"], ss[-1][-1]["e"]
    if b - a < 20:
        c["notes"].append(f"llm suggested trim {df}/{dl} not applied (would be {b - a:.0f} s)")
        return
    c.update({"start": a, "end": b, "dur": round(b - a, 2), "sents": ss, "words": [w for s in ss for w in s],
              "hook": first["text"], "text": " ".join(" ".join(w["w"] for w in s) for s in ss)})
    c["notes"].append(f"trimmed {df} leading / {dl} trailing sentence(s) per llm")


DESC_BANNED = ("highlight", "emphasiz", "reflects", "discuss", "shares", "delve", "journey", "incredible", "inspiring")


def clean_copy(text, limit=160, strict=False):
    text = re.sub(r"[\U0001F000-\U0001FAFF☀-➿️]", "", text or "").strip()
    for b in BANNED + (list(DESC_BANNED) if strict else []):
        if b in text.lower():
            return ""
    text = re.sub(r"\s+", " ", text)
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0].rstrip(",;:") + "."


def clean_tags(tags):
    out = []
    for t in tags or []:
        t = "#" + re.sub(r"[^a-z0-9]", "", str(t).lower())
        if len(t) > 2 and t not in out and t not in ("#fyp", "#foryou", "#viral", "#foryoupage"):
            out.append(t)
    return out[:3] or ["#space", "#astronaut"]


# ---------------------------------------------------------------- edit decision: pause tightening + timeline map

def keep_intervals(words, start, end, max_gap=0.55, pad=0.14):
    """Cut dead air: gaps between words longer than max_gap shrink to about 2*pad."""
    ws = [w for w in words if w["s"] >= start - 0.01 and w["e"] <= end + 0.01]
    iv = [[max(0, ws[0]["s"] - 0.06), ws[0]["e"]]]
    for w in ws[1:]:
        if w["s"] - iv[-1][1] > max_gap:
            iv[-1][1] += pad
            iv.append([w["s"] - pad, w["e"]])
        else:
            iv[-1][1] = w["e"]
    iv[-1][1] += 0.35
    return [(round(a, 3), round(b, 3)) for a, b in iv]


def mapper(iv):
    offs, acc = [], 0.0
    for a, b in iv:
        offs.append((a, b, acc))
        acc += b - a

    def f(t):
        for a, b, o in offs:
            if t <= b:
                return o + max(0.0, t - a)
        return acc
    return f, acc


# ---------------------------------------------------------------- framing (shot-aware, face-aware 9:16)

def scene_cuts(path, start, end):
    r = run([ffbin("ffmpeg"), "-hide_banner", "-nostats", "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}", "-i", str(path),
             "-an", "-vf", "scale=320:-2,select='gt(scene,0.28)',showinfo", "-f", "null", "-"], check=False)
    return [start + float(x) for x in re.findall(r"pts_time:([\d.]+)", r.stderr or "")]


def sample_faces(path, start, end, info, fps=2):
    """Face boxes (source pixels) every 1/fps s. YuNet DNN detector (opencv_zoo, local file); Haar only as fallback."""
    import cv2
    yunet = ROOT / "models" / "face_detection_yunet_2023mar.onnx"
    det = cv2.FaceDetectorYN.create(str(yunet), "", (960, 540), 0.82, 0.3, 50) if yunet.exists() else None
    casc = None if det else cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    out = []
    with tempfile.TemporaryDirectory() as t:
        run([ffbin("ffmpeg"), "-v", "error", "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}", "-i", str(path),
             "-vf", f"fps={fps},scale=960:-2", str(Path(t) / "f%04d.jpg")])
        for i, fp in enumerate(sorted(Path(t).glob("f*.jpg"))):
            img = cv2.imread(str(fp))
            s = info["w"] / img.shape[1]
            if det:
                det.setInputSize((img.shape[1], img.shape[0]))
                _, faces = det.detect(img)
                found = [] if faces is None else [tuple(f[:4]) for f in faces if f[2] >= 28]
            else:
                g = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
                found = list(casc.detectMultiScale(g, 1.1, 6, minSize=(48, 48)))
            out.append((start + (i + 0.5) / fps, [(float(x) * s, float(y) * s, float(w) * s, float(h) * s) for (x, y, w, h) in found]))
    return out


def people_in(frames, W0):
    boxes = [b for _, bs in frames for b in bs]
    if not boxes:
        return []
    maxw = max(b[2] for b in boxes)
    boxes = [b for b in boxes if b[2] >= 0.55 * maxw]  # drop small false positives (posters, models, patches)
    clusters = []
    for b in sorted(boxes, key=lambda b: b[0] + b[2] / 2):
        cx = b[0] + b[2] / 2
        for c in clusters:
            if abs(c["cx"] - cx) < 0.08 * W0:
                c["b"].append(b); c["cx"] = sum(x[0] + x[2] / 2 for x in c["b"]) / len(c["b"]); break
        else:
            clusters.append({"cx": cx, "b": [b]})
    need = max(1, math.ceil(0.34 * len(frames)))
    people = []
    for c in clusters:
        if len(c["b"]) >= need:
            bs = c["b"]
            med = lambda i: sorted(x[i] for x in bs)[len(bs) // 2]
            people.append({"x": med(0), "y": med(1), "w": med(2), "h": med(3)})
    return people


def plan_frame(path, start, end, info):
    sw, sh = info["w"], info["h"]
    cw = int(round(sh * 9 / 16 / 2) * 2)
    bounds = [start] + [c for c in scene_cuts(path, start, end) if start + 0.3 < c < end - 0.3] + [end]
    shots = [{"a": bounds[i], "b": bounds[i + 1]} for i in range(len(bounds) - 1)]
    frames = sample_faces(path, start, end, info)
    regions = []
    for s in shots:
        fr = [f for f in frames if s["a"] <= f[0] < s["b"]]
        s["n"] = len(fr)
        ppl = people_in(fr, sw) if fr else None
        if ppl is None:
            s["mode"] = None; continue
        if not ppl:
            s["mode"] = "fit"; continue
        left, right = min(p["x"] for p in ppl), max(p["x"] + p["w"] for p in ppl)
        fw = max(p["w"] for p in ppl)
        s["face_y"] = sum(p["y"] + p["h"] / 2 for p in ppl) / len(ppl) / sh
        s["people"] = len(ppl)
        if len(ppl) == 1 or right - left + 0.3 * fw <= cw:
            cx = (left + right) / 2
            s["mode"], s["x"] = "crop", int(min(max(0, cx - cw / 2), sw - cw)) // 2 * 2
        else:
            s["mode"] = "two"
            m = 0.9 * fw
            regions.append((max(0, left - m), min(sw, right + m), s["face_y"] * sh))
    for i, s in enumerate(shots):  # very short shots with no sample inherit a neighbour
        if s["mode"] is None:
            nb = shots[i - 1] if i else (shots[i + 1] if i + 1 < len(shots) else None)
            s.update({k: v for k, v in (nb or {"mode": "fit"}).items() if k not in ("a", "b", "n")})
    band = None
    if any(s["mode"] == "fit" for s in shots):
        band = {"x": 0, "y": 0, "w2": sw // 2 * 2, "h2": sh // 2 * 2, "fit": True}
    elif regions:
        x0, x1 = min(r[0] for r in regions), max(r[1] for r in regions)
        w2 = int((x1 - x0) // 2 * 2)
        h2 = int(min(sh, w2 * 1.05) // 2 * 2)
        fy = sum(r[2] for r in regions) / len(regions)
        band = {"x": int(x0) // 2 * 2, "y": int(min(max(0, fy - 0.36 * h2), sh - h2) // 2 * 2), "w2": w2, "h2": h2, "fit": False}
    if band:
        band["bh"] = int(round(W * band["h2"] / band["w2"] / 2) * 2)
        band["by"] = max(180, (H - band["bh"]) // 2 - 140)
    modes = [s["mode"] for s in shots]
    crop_shots = [s for s in shots if s["mode"] == "crop"]
    pick = max(crop_shots or shots, key=lambda s: s["b"] - s["a"])
    note = (f"{len(shots)} shot(s): " + ", ".join(f"{modes.count(m)} {m}" for m in ("crop", "two", "fit") if m in modes)
            + " (crop = face-centred 9:16, two = both speakers kept in a band, fit = whole frame over blurred fill)")
    return {"shots": shots, "band": band, "cw": cw, "note": note, "cover_t": pick["a"] + min(1.2, (pick["b"] - pick["a"]) / 2),
            "cover_shot": pick}


# ---------------------------------------------------------------- captions (ASS, word-timed, max 2 lines)

def ass_time(t):
    t = max(0.0, t)
    return f"{int(t // 3600)}:{int(t % 3600 // 60):02d}:{t % 60:05.2f}"


def caption_pages(words, fmap):
    ws = []
    for w in words:
        if norm(w["w"]) in CAPTION_DROP:
            continue
        txt = w["w"].replace("{", "(").replace("}", ")")
        ws.append({"s": fmap(w["s"]), "e": fmap(w["e"]), "w": txt})
    pages, cur = [], []
    for i, w in enumerate(ws):
        cur.append(w)
        chars = len(" ".join(x["w"] for x in cur))
        nxt = ws[i + 1] if i + 1 < len(ws) else None
        brk = (nxt is None or re.search(r"[.?!,;:]$", w["w"]) and len(cur) >= 2 or re.search(r"[.?!]$", w["w"])
               or (nxt and nxt["s"] - w["e"] > 0.45) or len(cur) >= 5
               or (nxt and chars + 1 + len(nxt["w"]) > 24))
        if brk:
            pages.append(cur); cur = []
    return pages


def write_ass(path, pages, total, bottom_y, credit):
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,Arial,76,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,6,2,2,90,170,{H - bottom_y},1
Style: Credit,Arial,30,&H40FFFFFF,&H40FFFFFF,&H80000000,&H00000000,-1,0,0,0,100,100,0,0,1,2,0,7,60,60,250,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    ev = [f"Dialogue: 0,{ass_time(0)},{ass_time(total)},Credit,,0,0,0,,{credit}"]
    for pi, pg in enumerate(pages):
        nxt_start = pages[pi + 1][0]["s"] if pi + 1 < len(pages) else total
        page_end = min(nxt_start, pg[-1]["e"] + 0.6)
        for wi, w in enumerate(pg):
            a = w["s"] if wi else pg[0]["s"]
            b = pg[wi + 1]["s"] if wi + 1 < len(pg) else page_end
            if b - a < 0.02:
                continue
            txt = " ".join((r"{\c&H0000D7FF&}" + x["w"] + r"{\c&H00FFFFFF&}") if k == wi else x["w"] for k, x in enumerate(pg))
            ev.append(f"Dialogue: 1,{ass_time(a)},{ass_time(b)},Cap,,0,0,0,,{txt}")
    Path(path).write_text(head + "\n".join(ev) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- render

def crop_chain(inp, out, plan, info, x_expr):
    return (f"[{inp}]crop={plan['cw']}:{info['h']}:x='{x_expr}':y=0,scale={W}:{H}:flags=lanczos,"
            f"unsharp=5:5:0.45:5:5:0,setsar=1[{out}]")


def band_chain(inp, out, plan):
    b = plan["band"]
    return (f"[{inp}]split=2[{out}s1][{out}s2];[{out}s1]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},"
            f"boxblur=26:2,eq=brightness=-0.22:saturation=0.75[{out}bg];[{out}s2]crop={b['w2']}:{b['h2']}:{b['x']}:{b['y']},"
            f"scale={W}:{b['bh']}:flags=lanczos,unsharp=5:5:0.4:5:5:0[{out}fg];[{out}bg][{out}fg]overlay=0:{b['by']},setsar=1[{out}]")


def video_graph(plan, info, sel, fmap):
    """Whole-clip graph: select (pause cuts) -> per-shot crop x, band shots overlaid only while they are on screen."""
    pre = f"[0:v]select='{sel}',setpts=N/FRAME_RATE/TB,fps=30[src]"
    shots = plan["shots"]
    crops = [s for s in shots if s["mode"] == "crop"]
    bands = [s for s in shots if s["mode"] != "crop"]
    if not bands:
        x_expr = str(crops[-1]["x"])
        for s in reversed(crops[:-1]):
            x_expr = f"if(lt(t,{fmap(s['b']):.3f}),{s['x']},{x_expr})"
        return pre + ";" + crop_chain("src", "base", plan, info, x_expr)
    if not crops:
        return pre + ";" + band_chain("src", "base", plan)
    x_expr = str(crops[-1]["x"])
    for s in reversed(crops[:-1]):
        x_expr = f"if(lt(t,{fmap(s['b']):.3f}),{s['x']},{x_expr})"
    en = "+".join(f"between(t,{fmap(s['a']):.3f},{fmap(s['b']) - 0.001:.3f})" for s in bands)
    return (pre + ";[src]split=2[c0][b0];" + crop_chain("c0", "cv", plan, info, x_expr) + ";" + band_chain("b0", "bv", plan)
            + f";[cv][bv]overlay=0:0:enable='{en}'[base]")


def still_graph(plan, info, shot):
    if shot["mode"] == "crop":
        return crop_chain("0:v", "base", plan, info, str(shot["x"]))
    return band_chain("0:v", "base", plan)


def caption_bottom(plan):
    if not plan["band"] or all(s["mode"] == "crop" for s in plan["shots"]):
        return CAPTION_BOTTOM
    b = plan["band"]
    bottom = b["by"] + b["bh"]
    return min(CAPTION_BOTTOM, bottom + 210) if b["bh"] < 900 else min(CAPTION_BOTTOM, bottom - 30)


def measure_loudness(src, start, span, asel):
    r = run([ffbin("ffmpeg"), "-hide_banner", "-ss", f"{start:.3f}", "-t", f"{span:.3f}", "-i", str(src), "-filter_complex",
             f"[0:a]aselect='{asel}',asetpts=N/SR/TB,highpass=f=70,loudnorm=I=-14:TP=-1.5:LRA=11:print_format=json",
             "-f", "null", "-"], check=True)
    return json.loads(r.stderr[r.stderr.rindex("{"):r.stderr.rindex("}") + 1])


def render_clip(c, src, info, plan, out_mp4, work, light, credit):
    iv = keep_intervals(c["words"], c["start"], c["end"])
    t0 = iv[0][0]
    fmap0, total = mapper(iv)
    # plan shots are in source time; the graph runs on the input-seeked timeline, so map relative to t0
    span = iv[-1][1] - t0 + 0.2
    sel = "+".join(f"between(t,{a - t0:.3f},{b - t0:.3f})" for a, b in iv)
    graph_v = video_graph(plan, info, sel, fmap0)
    ass = work / (out_mp4.stem + ".ass")
    pages = caption_pages(c["words"], fmap0)
    write_ass(ass, pages, total, caption_bottom(plan), credit)
    m = measure_loudness(src["abspath"], t0, span, sel)
    ln = (f"loudnorm=I=-14:TP=-1.5:LRA=11:measured_I={m['input_i']}:measured_TP={m['input_tp']}:measured_LRA={m['input_lra']}:"
          f"measured_thresh={m['input_thresh']}:offset={m['target_offset']}:linear=true")
    graph = (graph_v + f";[base]subtitles={ass.name}[v];"
             f"[0:a]aselect='{sel}',asetpts=N/SR/TB,highpass=f=70,{ln},aresample=48000[a]")
    (work / (out_mp4.stem + ".graph.txt")).write_text(graph, encoding="utf-8")
    enc_gpu = ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", "21", "-b:v", "0", "-maxrate", "12M", "-bufsize", "24M"]
    enc_cpu = ["-c:v", "libx264", "-preset", "veryfast" if light else "medium", "-crf", "20", "-threads", "4" if light else "0"]
    base = [ffbin("ffmpeg"), "-v", "error", "-y", "-ss", f"{t0:.3f}", "-t", f"{span:.3f}", "-i", str(src["abspath"]),
            "-/filter_complex", out_mp4.stem + ".graph.txt", "-map", "[v]", "-map", "[a]"]
    if light:
        base[1:1] = ["-filter_threads", "2", "-filter_complex_threads", "2"]
    tail = ["-pix_fmt", "yuv420p", "-profile:v", "high", "-r", "30", "-c:a", "aac", "-b:a", "160k", "-ar", "48000",
            "-t", f"{total:.3f}", "-movflags", "+faststart", out_mp4.name]
    try:
        run(base + (enc_cpu if light else enc_gpu) + tail, cwd=work)
    except RuntimeError as e:
        if light:
            raise
        log(f"nvenc failed, retrying on CPU: {str(e)[:300]}")
        run(base + enc_cpu + tail, cwd=work)
    shutil.move(str(work / out_mp4.name), str(out_mp4))
    return {"intervals": iv, "cut_seconds": round((iv[-1][1] - iv[0][0]) - total, 2), "duration": round(total, 2),
            "pages": len(pages), "input_lufs": m["input_i"]}


def load_cover_module():
    spec = importlib.util.spec_from_file_location("make_cover", CLIPPER / "assets" / "make-cover.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_cover(src, info, plan, text, out_jpg, work):
    from PIL import Image
    shot = plan["cover_shot"]
    tmp = work / (out_jpg.stem + "-raw.png")
    run([ffbin("ffmpeg"), "-v", "error", "-y", "-ss", f"{plan['cover_t']:.3f}", "-i", str(src["abspath"]), "-filter_complex",
         still_graph(plan, info, shot), "-map", "[base]", "-frames:v", "1", str(tmp)])
    img = Image.open(tmp).convert("RGB")
    mc = load_cover_module()
    if shot["mode"] == "crop":
        face = shot.get("face_y", 0.3) * H
    else:
        b = plan["band"]
        face = b["by"] + (shot.get("face_y", 0.4) * info["h"] - b["y"]) / b["h2"] * b["bh"]
    y = int(min(max(face + 560, 980), 1250))
    mc.draw_text(img, text, y, 132)
    img.save(out_jpg, quality=90)
    tmp.unlink(missing_ok=True)


def qa_clip(mp4, qa_dir):
    info = probe(mp4)
    r = run([ffbin("ffmpeg"), "-hide_banner", "-nostats", "-i", str(mp4), "-af", "ebur128=peak=true", "-f", "null", "-"])
    mi = re.findall(r"I:\s+(-?[\d.]+) LUFS", r.stderr)
    lufs = float(mi[-1]) if mi else None
    frames = []
    for tag, t in (("a", 0.4), ("b", info["dur"] / 2), ("c", max(0, info["dur"] - 0.6))):
        f = qa_dir / f"{mp4.stem}-{tag}.jpg"
        run([ffbin("ffmpeg"), "-v", "error", "-y", "-ss", f"{t:.2f}", "-i", str(mp4), "-frames:v", "1", "-vf", "scale=540:-2", str(f)])
        frames.append(str(f.relative_to(mp4.parent)).replace("\\", "/"))
    ok = 20 <= info["dur"] <= 45 and info["w"] == W and info["h"] == H and lufs is not None and abs(lufs + 14) <= 1.0
    return {"duration": round(info["dur"], 2), "size": f"{info['w']}x{info['h']}", "lufs": lufs, "frames": frames, "pass": ok}


# ---------------------------------------------------------------- state, review page, board

def load_used():
    if USED.exists():
        return json.loads(USED.read_text(encoding="utf-8"))
    return {"_note": "Segments already used in a batch or published. status: batch | published | rejected.", "entries": []}


def save_used(u):
    USED.write_text(json.dumps(u, ensure_ascii=False, indent=1), encoding="utf-8")


def fmt_ts(t):
    return f"{int(t // 60)}:{t % 60:04.1f}"


def write_review(day_dir, items, day):
    cards = []
    for it in items:
        e = html.escape
        notes = "".join(f"<li>{e(n)}</li>" for n in it["quality_notes"])
        cards.append(f"""<article class="card">
  <video src="{e(it['file'])}" poster="{e(it['cover'])}" controls preload="none" playsinline></video>
  <div class="meta">
    <label class="pick"><input type="checkbox" data-id="{e(it['id'])}"> {e(it['id'])} &middot; {it['duration']:.1f} s</label>
    <p class="hook">&ldquo;{e(it['hook_line'])}&rdquo;</p>
    <p class="desc">{e(it['description'])}</p>
    <p class="tags">{e(' '.join(it['hashtags']))}</p>
    <p class="small">Source: <a href="{e(it['source_url'])}">{e(it['source_title'])}</a>, {e(it['source_range'])}</p>
    <p class="small">Rights: {e(it['rights_basis'])}</p>
    <details><summary>Quality notes</summary><ul>{notes}</ul></details>
    <a class="small" href="{e(it['cover'])}">cover</a>
  </div>
</article>""")
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>HookHaus {day}</title>
<style>
:root{{--bg:#0e0f11;--card:#17191c;--text:#eceff3;--muted:#9aa3ad;--line:#262a2f;--accent:#ffd400}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.45 system-ui,Segoe UI,Arial,sans-serif}}
header{{padding:20px 16px 8px;max-width:1500px;margin:auto}} h1{{margin:0 0 4px;font-size:22px}} header p{{margin:0;color:var(--muted)}}
main{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:16px;padding:16px;max-width:1500px;margin:auto}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden}}
video{{width:100%;aspect-ratio:9/16;background:#000;display:block}} .meta{{padding:12px 14px}}
.hook{{font-weight:600;margin:.4em 0}} .desc{{margin:.4em 0}} .tags{{color:var(--accent);margin:.3em 0}}
.small{{color:var(--muted);font-size:12.5px}} a{{color:#8ab4ff}} .pick{{font-weight:600}} input{{accent-color:var(--accent)}}
#sel{{position:sticky;bottom:0;background:#000c;padding:10px 16px;text-align:center}} button{{font:inherit;padding:6px 12px}}
</style></head><body>
<header><h1>HookHaus &middot; {day} &middot; {len(items)} clips</h1>
<p>Prepared only, nothing is published or scheduled. Tick clips for option B, then copy the list into the board task.</p></header>
<main>
{chr(10).join(cards)}
</main>
<div id="sel"><button id="copy">Copy ticked clip ids</button> <span id="out" class="small"></span></div>
<script>
document.getElementById('copy').onclick=()=>{{const ids=[...document.querySelectorAll('input[data-id]:checked')].map(i=>i.dataset.id).join(', ');
document.getElementById('out').textContent=ids||'none ticked';try{{navigator.clipboard.writeText(ids)}}catch(e){{}}}};
</script></body></html>"""
    (day_dir / "review.html").write_text(page, encoding="utf-8")


def board_task(day, n, day_dir):
    cli = Path(r"C:\Users\sgife\dev\sorinOS\scripts\shared_tasks.py")
    if not cli.exists():
        log("board CLI missing; skipped board task")
        return None
    tid = f"hookhaus-publish-{day.replace('-', '')}"
    rel = f"clipper/daily/{day}/review.html"
    blocker = (f"Cere aprobarea lui Sorin: {n} clipuri HookHaus gata ({rel}). VARIANTE: A) public tot, la ~2h "
               f"intre 10:00 si 22:00 B) doar cele bifate C) nu azi. RECOMANDARE: A.")
    r = run([sys.executable.replace("pythonw", "python"), str(cli), "create", "--id", tid, "--title", f"HookHaus: aprobare publicare {day}",
             "--owner", "sorin", "--status", "blocked", "--blocker", blocker,
             "--next-action", f"Deschide {day_dir / 'review.html'} si alege A/B/C; postarea o face pasul de publicare doar dupa aprobare.",
             "--provenance", f"make_daily.py batch {day}: {n} clips, captions.json in {day_dir}"], check=False)
    if r.returncode != 0:
        log(f"board create {tid}: {(r.stderr or r.stdout).strip()[:300]}")
    else:
        log(f"board task {tid} created")
    return tid


# ---------------------------------------------------------------- editor picks (manual edit decisions)

def load_picks(path, src_by, trs):
    """picks.json: [{source, segments: [[a, b], ...], cover_text, description, hashtags, fixes?, why?}].
    Each segment keeps the words that START inside [a, b); gaps between segments become hard cuts."""
    out = []
    for p in json.loads(Path(path).read_text(encoding="utf-8-sig")):
        s = src_by[p["source"]]
        words = []
        for k, (a, b) in enumerate(p["segments"]):
            seg = [dict(w) for w in trs[p["source"]]["words"] if a - 0.02 <= w["s"] < b]
            if k == 0:
                seg = trim_lead({"s": seg[0]["s"], "e": seg[-1]["e"], "words": seg, "text": ""})["words"]
            words += seg
        for w in words:
            for bad, good in {**s.get("asr_fixes", {}), **p.get("fixes", {})}.items():
                w["w"] = re.sub(rf"(?<![\w]){re.escape(bad)}(?![\w])", good, w["w"])
        words[0]["w"] = words[0]["w"][:1].upper() + words[0]["w"][1:]
        spoken = sum(b - a for a, b in p["segments"])
        text = " ".join(w["w"] for w in words)
        out.append({"source": s["id"], "start": words[0]["s"], "end": words[-1]["e"], "dur": round(spoken, 2), "h_score": 0.0,
                    "final": 10.0, "rejected": False, "llm": None, "copy": p, "notes": ["editor pick: " + p.get("why", "")],
                    "wps": round(len(words) / max(spoken, 1), 2), "text": text, "hook": sentences(words)[0]["text"],
                    "words": words, "sents": [words]})
    return out


# ---------------------------------------------------------------- main

def select(pool, n, per_source):
    chosen, count = [], {}

    def adjusted(c):  # neighbouring answers tend to repeat a topic: spread picks across the source
        near = sum(1 for p in chosen if p["source"] == c["source"] and abs(p["start"] - c["start"]) < 120)
        return c["final"] - 0.8 * near

    left = list(pool)
    while left and len(chosen) < n:
        ok = [c for c in left if count.get(c["source"], 0) < per_source
              and not any(p["source"] == c["source"] and c["start"] < p["end"] and c["end"] > p["start"] for p in chosen)]
        if not ok:
            break
        c = max(ok, key=adjusted)
        left.remove(c)
        chosen.append(c); count[c["source"]] = count.get(c["source"], 0) + 1
    # alternate sources in posting order
    by = {}
    for c in chosen:
        by.setdefault(c["source"], []).append(c)
    order = []
    while any(by.values()):
        for k in list(by):
            if by[k]:
                order.append(by[k].pop(0))
    return order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=dt.date.today().isoformat())
    ap.add_argument("--count", type=int, default=7)
    ap.add_argument("--force", action="store_true", help="rebuild this date: frees its used.json entries")
    ap.add_argument("--transcribe-only", action="store_true")
    ap.add_argument("--review-only", action="store_true")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--no-board", action="store_true")
    ap.add_argument("--picks", help="editor picks JSON (manual segments + copy); skips auto selection")
    ap.add_argument("--wait-for-game", type=int, default=0, help="minutes to wait for a running game to close")
    a = ap.parse_args()
    lower_priority()
    day, day_dir = a.date, ROOT / a.date
    if a.review_only:
        items = json.loads((day_dir / "captions.json").read_text(encoding="utf-8"))["clips"]
        write_review(day_dir, items, day)
        return
    waited = 0
    while a.wait_for_game and game_running() and waited < a.wait_for_game:
        log(f"game running ({game_running()}); waiting 5 min")
        time.sleep(300); waited += 5
    light = bool(game_running())
    log(f"=== batch {day} (light={light})")
    sources = [s for s in load_sources() if s["enabled"]]
    used = load_used()
    if a.force:
        used["entries"] = [u for u in used["entries"] if not (u.get("date") == day and u.get("status") == "batch")]
    if day_dir.exists() and (day_dir / "captions.json").exists() and not a.force and not a.transcribe_only:
        log(f"{day_dir} already has a batch; use --force to rebuild"); return
    pool, infos, trs = [], {}, {}
    for s in sources:
        try:
            infos[s["id"]] = probe(s["abspath"])
        except Exception as e:
            log(f"skip {s['id']}: cannot probe ({str(e)[:160]})"); continue
        tr = transcribe(s, light)
        trs[s["id"]] = tr
        if a.transcribe_only:
            continue
        for w in tr["words"]:  # known ASR misspellings of names/terms (sources.json asr_fixes)
            for bad, good in s.get("asr_fixes", {}).items():
                w["w"] = re.sub(rf"\b{re.escape(bad)}\b", good, w["w"])
        cs = candidates(s, tr, used["entries"])
        log(f"{s['id']}: {len(cs)} non-overlapping candidate windows")
        pool += cs[:18]
    if a.transcribe_only:
        return
    src_by = {s["id"]: s for s in sources}
    model = None
    if a.picks:
        chosen = load_picks(a.picks, src_by, trs)
        log(f"editor picks: {len(chosen)} from {a.picks}")
        day_dir.mkdir(exist_ok=True)
    else:
        pool = [c for c in pool if c["h_score"] > -1.5]
        models = [] if a.no_llm else ollama_available()
        model = next((m for m in (["gemma3:4b"] if light else ["gemma3:12b", "gemma3:4b"]) if m in models), None)
        for c in pool:
            c["llm"] = None
            if model:
                try:
                    c["llm"] = llm_rate(c, src_by[c["source"]].get("speakers", ""), model)
                except Exception as e:
                    log(f"llm failed for {c['source']}@{c['start']:.0f}: {str(e)[:120]}")
            L = c["llm"]
            if L:
                apply_trims(c)
                c["final"] = 0.25 * c["h_score"] + 0.45 * L["hook"] + 0.35 * L["standalone"] + 0.2 * L["interest"]
                c["rejected"] = L["hook"] < 7 or L["standalone"] < 7 or bool(str(L.get("reject_reason") or "").strip())
            else:
                c["final"] = c["h_score"]
                c["rejected"] = c["h_score"] < 1.0
        if model:  # free the GPU before rendering
            try:
                urllib.request.urlopen(urllib.request.Request(OLLAMA + "/api/generate", data=json.dumps(
                    {"model": model, "keep_alive": 0}).encode(), headers={"Content-Type": "application/json"}), timeout=20).read()
            except Exception:
                pass
        good = [c for c in pool if not c["rejected"]]
        log(f"pool {len(pool)}, passed {len(good)} (llm={model})")
        day_dir.mkdir(exist_ok=True)
        (day_dir / "pool.json").write_text(json.dumps(
            [{k: c[k] for k in ("source", "start", "end", "dur", "h_score", "final", "rejected", "llm", "notes", "text")}
             for c in sorted(pool, key=lambda c: -c["final"])], ensure_ascii=False, indent=1), encoding="utf-8")
        per_source = a.count if len(sources) == 1 else max(4, math.ceil(a.count * 0.6))
        chosen = select(good, a.count, per_source)
    if not chosen:
        log("no candidate passed the bar; no batch today"); return
    day_dir.mkdir(exist_ok=True)
    work, qa = day_dir / "_work", day_dir / "qa"
    work.mkdir(exist_ok=True); qa.mkdir(exist_ok=True)
    items = []
    for i, c in enumerate(chosen, 1):
        s, info = src_by[c["source"]], infos[c["source"]]
        cid = f"c{i:02d}"
        out = day_dir / f"{cid}.mp4"
        try:
            fr = plan_frame(s["abspath"], c["start"], c["end"], info)
            cx = (c.get("copy") or {}).get("crop_cx")  # editor override: centre the 9:16 crop on the speaker (fraction of width)
            if cx:
                for sh in fr["shots"]:
                    if sh["mode"] == "crop":
                        sh["x"] = int(min(max(0, cx * info["w"] - fr["cw"] / 2), info["w"] - fr["cw"])) // 2 * 2
            r = render_clip(c, s, info, fr, out, work, light, s.get("credit", ""))
            L = c.get("copy") or c["llm"] or {}
            cover_text = re.sub(r"[.!]+$", "", clean_copy(L.get("cover_text", ""), 60))
            if not cover_text or "," in cover_text or not (2 <= len(cover_text.split()) <= 5):
                cover_text = re.sub(r"[,.;:?!]+$", "", " ".join(c["hook"].split()[:4]))
            make_cover(s, info, fr, cover_text, day_dir / f"{cid}-cover.jpg", work)
            q = qa_clip(out, qa)
        except Exception as e:
            log(f"{cid} failed: {str(e)[:600]}"); continue
        if c.get("copy"):
            desc = c["copy"]["description"].strip()
        else:
            desc = clean_copy(L.get("description", ""), 130, strict=True) or f"“{clean_copy(c['hook'], 110)}”"
        credit_line = f"{desc} Source: NASA. Not endorsed by NASA."
        notes = [fr["note"], f"cut {r['cut_seconds']} s of dead air", f"{c['wps']} words/s",
                 f"loudness {q['lufs']} LUFS", "auto QA pass" if q["pass"] else "AUTO QA FAIL"] + c["notes"]
        if c["llm"]:
            notes.append(f"llm hook {L['hook']:.0f} / standalone {L['standalone']:.0f} / interest {L['interest']:.0f}")
        items.append({"id": cid, "file": out.name, "cover": f"{cid}-cover.jpg", "duration": q["duration"],
                      "hook_line": c["hook"], "cover_text": cover_text, "description": credit_line,
                      "hashtags": clean_tags(L.get("hashtags")), "source": s["id"], "source_title": s["title"],
                      "source_url": s["url"], "source_start": round(c["start"], 2), "source_end": round(c["end"], 2),
                      "source_range": f"{fmt_ts(c['start'])}-{fmt_ts(c['end'])}", "rights_basis": s["rights"],
                      "transcript": c["text"], "framing": [(round(x["a"], 2), round(x["b"], 2), x["mode"]) for x in fr["shots"]], "qa": q, "quality_notes": notes,
                      "score": round(c["final"], 2), "status": "prepared, not published", "selection": "editor" if c.get("copy") else "auto"})
        used["entries"].append({"date": day, "id": cid, "source": s["id"], "start": round(c["start"], 2),
                                "end": round(c["end"], 2), "status": "batch"})
        log(f"{cid} ok {q['duration']} s {'/'.join(x['mode'] for x in fr['shots'])} lufs={q['lufs']} :: {c['hook'][:70]}")
    meta = {"date": day, "generated": dt.datetime.now().isoformat(timespec="seconds"), "light_mode": light, "llm": model,
            "requested": a.count, "delivered": len(items), "published": False,
            "note": "Prepared only. Publishing needs Sorin's approval on the board task.", "clips": items}
    (day_dir / "captions.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    write_review(day_dir, items, day)
    save_used(used)
    shutil.rmtree(work, ignore_errors=True)
    log(f"batch {day}: {len(items)}/{a.count} clips -> {day_dir / 'review.html'}")
    if items and not a.no_board:
        board_task(day, len(items), day_dir)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import traceback
        log("FATAL " + traceback.format_exc()[-2000:])
        raise
