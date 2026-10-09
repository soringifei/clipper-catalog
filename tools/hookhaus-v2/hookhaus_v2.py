"""HookHaus v2: long 16:9 interview -> 3 vertical clips + before/after variants.

Local only. Steps:
  1. auto-editor removes silences (render + v3 timeline for time mapping)
  2. faster-whisper on GPU transcribes the cut with word timestamps
  3. deterministic scoring picks N non-overlapping, sentence-aligned moments
  4. OpenCV face tracking drives a smoothed 9:16 crop with punch-in zooms
  5. word-by-word ASS subtitles are burned in with ffmpeg/libass
  6. before/after: raw 16:9 source at the same moment -> transition -> final

Moment scores are editing heuristics, not a prediction of views.
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_WHISPER_PY = Path(r"C:\Users\sgife\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe")
OUT_W, OUT_H = 1080, 1920

HOOK_WORDS = {
    "you", "your", "never", "always", "why", "how", "what", "secret", "mistake", "first",
    "biggest", "only", "surprising", "incredible", "amazing", "most", "best", "worst",
    "imagine", "actually", "really", "universe", "discover", "discovered",
}
WEAK_OPENERS = {"and", "so", "but", "because", "which", "or", "um", "uh", "that", "then", "also"}


# ---------------------------------------------------------------- timeline

def load_timeline(path):
    """auto-editor v3 timeline -> (fps, [(cut_start, dur, src_offset)] in frames)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    fps = Fraction(data["timebase"])
    clips = sorted((c["start"], c["dur"], c["offset"]) for c in data["v"][0])
    return fps, clips


def cut_to_source(t, fps, clips):
    """Map a time (s) in the silence-cut video back to the original source (s)."""
    frame = t * float(fps)
    for start, dur, offset in clips:
        if start <= frame < start + dur:
            return (offset + frame - start) / float(fps)
    if clips and frame >= clips[-1][0] + clips[-1][1]:
        start, dur, offset = clips[-1]
        return (offset + dur) / float(fps)
    return clips[0][2] / float(fps) if clips else t


# ---------------------------------------------------------------- moments

def split_sentences(words, max_gap=1.2):
    """Group word dicts into sentences on terminal punctuation or long pauses."""
    sentences, cur = [], []
    for i, w in enumerate(words):
        cur.append(w)
        nxt = words[i + 1] if i + 1 < len(words) else None
        gap = (nxt["start"] - w["end"]) if nxt else 0
        if re.search(r"[.?!]$", w["text"]) or gap > max_gap or nxt is None:
            sentences.append({
                "start": cur[0]["start"], "end": cur[-1]["end"],
                "text": " ".join(x["text"] for x in cur).strip(), "words": cur,
            })
            cur = []
    return sentences


def _tokens(text):
    return re.findall(r"[a-z0-9']+", text.lower())


def score_window(sents):
    """Heuristic score for a candidate clip made of consecutive sentences."""
    first = sents[0]
    dur = sents[-1]["end"] - first["start"]
    n_words = sum(len(s["words"]) for s in sents)
    toks = _tokens(first["text"])
    reasons, score = [], 0.0
    density = n_words / max(dur, 0.1)
    score += min(density, 4.0)
    reasons.append(f"density {density:.2f} w/s")
    if toks and toks[0] in WEAK_OPENERS:
        score -= 1.5
        reasons.append(f"weak opener '{toks[0]}'")
    if "?" in first["text"]:
        score += 1.0
        reasons.append("question hook")
    if any(t.isdigit() for t in toks):
        score += 0.7
        reasons.append("number in hook")
    hits = sorted(set(toks) & HOOK_WORDS)
    if hits:
        score += min(len(hits), 3) * 0.4
        reasons.append("hook words " + ",".join(hits))
    if 3 <= len(toks) <= 14:
        score += 0.6
        reasons.append("short first line")
    return round(score, 3), reasons


def pick_moments(sentences, n=3, min_dur=15.0, max_dur=30.0):
    """Greedy best-first choice of n non-overlapping sentence-aligned windows."""
    cands = []
    for i in range(len(sentences)):
        for j in range(i, len(sentences)):
            dur = sentences[j]["end"] - sentences[i]["start"]
            if dur > max_dur:
                break
            if dur >= min_dur:
                score, reasons = score_window(sentences[i:j + 1])
                cands.append({"i": i, "j": j, "start": sentences[i]["start"], "end": sentences[j]["end"],
                              "score": score, "reasons": reasons})
    cands.sort(key=lambda c: (-c["score"], c["start"]))
    chosen = []
    for c in cands:
        if all(c["end"] <= o["start"] or c["start"] >= o["end"] for o in chosen):
            chosen.append(c)
        if len(chosen) == n:
            break
    chosen.sort(key=lambda c: c["start"])
    for c in chosen:
        c["text"] = " ".join(s["text"] for s in sentences[c["i"]:c["j"] + 1])
        c["sentence_starts"] = [s["start"] for s in sentences[c["i"]:c["j"] + 1]]
    return chosen


def pad_bounds(moments, total, pre=0.08, post=0.25):
    """Breathing room around each moment without running into its neighbours."""
    out = []
    for k, m in enumerate(moments):
        lo = (moments[k - 1]["end"] + m["start"]) / 2 if k else 0.0
        hi = (m["end"] + moments[k + 1]["start"]) / 2 if k + 1 < len(moments) else total
        out.append((max(lo, m["start"] - pre), min(hi, m["end"] + post)))
    return out


# ---------------------------------------------------------------- reframing

def fill_track(samples, n_frames, default):
    """samples: {frame_idx: value}. Linear interpolation, edges held."""
    if not samples:
        return [default] * n_frames
    keys = sorted(samples)
    out = []
    k = 0
    for f in range(n_frames):
        while k + 1 < len(keys) and keys[k + 1] <= f:
            k += 1
        a = keys[k]
        if f <= keys[0]:
            out.append(samples[keys[0]])
        elif k + 1 >= len(keys):
            out.append(samples[a])
        else:
            b = keys[k + 1]
            r = (f - a) / (b - a)
            out.append(samples[a] + (samples[b] - samples[a]) * r)
    return out


def median_filter(xs, win=9):
    half = win // 2
    out = []
    for i in range(len(xs)):
        w = sorted(xs[max(0, i - half):i + half + 1])
        out.append(w[len(w) // 2])
    return out


def virtual_camera(targets, deadzone, gain=0.12, max_step=12.0):
    """Camera that only moves when the subject leaves a deadzone, eased and speed-capped."""
    if not targets:
        return []
    cam = targets[0]
    out = []
    for t in targets:
        err = t - cam
        if abs(err) > deadzone:
            step = (err - deadzone * (1 if err > 0 else -1)) * gain
            cam += max(-max_step, min(max_step, step))
        out.append(cam)
    return out


def crop_box(cx, cy, zoom, w, h):
    """9:16 crop inside a w x h frame, centered on (cx, cy) and clamped."""
    ch = h / zoom
    cw = ch * 9 / 16
    if cw > w:
        cw, ch = w, w * 16 / 9
    x = min(max(cx - cw / 2, 0), w - cw)
    y = min(max(cy - ch / 2, 0), h - ch)
    return int(round(x)), int(round(y)), int(round(cw)), int(round(ch))


def zoom_schedule(times, sentence_starts, punch=1.15):
    """Alternate 1.0 / punch zoom at each sentence start (jump-cut punch-ins)."""
    out = []
    for t in times:
        idx = sum(1 for s in sentence_starts if s <= t + 1e-6) - 1
        out.append(punch if idx % 2 == 1 else 1.0)
    return out


def detect_faces(video, every=3):
    """Return (fps, n_frames, w, h, {frame: (cx, cy)}) of the largest face."""
    import cv2

    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    samples, f = {}, 0
    scale = 480 / w
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if f % every == 0:
            small = cv2.resize(frame, None, fx=scale, fy=scale)
            gray = cv2.equalizeHist(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
            faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
            if len(faces):
                x, y, fw, fh = max(faces, key=lambda r: r[2] * r[3])
                samples[f] = ((x + fw / 2) / scale, (y + fh / 2) / scale)
        f += 1
    cap.release()
    return fps, f, w, h, samples


def reframe(segment, out_path, ass_name, sentence_starts, ffmpeg="ffmpeg"):
    """Face-tracked 9:16 crop of `segment`, subtitles burned, audio loudness-normalised."""
    import cv2

    fps, n, w, h, samples = detect_faces(segment)
    xs = fill_track({k: v[0] for k, v in samples.items()}, n, w / 2)
    ys = fill_track({k: v[1] for k, v in samples.items()}, n, h * 0.4)
    cam_x = virtual_camera(median_filter(xs), deadzone=w * 0.05)
    cam_y = virtual_camera(median_filter(ys), deadzone=h * 0.06, max_step=6.0)
    zooms = zoom_schedule([i / fps for i in range(n)], sentence_starts)

    out_path = Path(out_path)
    cmd = [ffmpeg, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{OUT_W}x{OUT_H}",
           "-r", f"{fps:.6f}", "-i", "-", "-i", str(Path(segment).resolve()),
           "-map", "0:v", "-map", "1:a", "-vf", f"subtitles={ass_name}",
           "-af", "loudnorm=I=-14:TP=-1.5:LRA=11", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-shortest",
           "-movflags", "+faststart", out_path.name]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, cwd=out_path.parent)
    cap = cv2.VideoCapture(str(segment))
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        # face sits in the upper third of a vertical frame: aim slightly below it
        x, y, cw, ch = crop_box(cam_x[i], cam_y[i] + h * 0.12 / zooms[i], zooms[i], w, h)
        crop = frame[y:y + ch, x:x + cw]
        proc.stdin.write(cv2.resize(crop, (OUT_W, OUT_H), interpolation=cv2.INTER_CUBIC).tobytes())
    cap.release()
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg reframe failed for {segment}")
    return {"frames": n, "face_samples": len(samples), "face_hit_rate": round(len(samples) / max(1, -(-n // 3)), 3)}


# ---------------------------------------------------------------- subtitles

ASS_HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Word,Arial Black,84,&H00FFFFFF,&H00FFFFFF,&H00000000,&H64000000,1,0,0,0,100,100,0,0,1,7,3,2,90,90,560,1
Style: Label,Arial Black,64,&H00FFFFFF,&H00FFFFFF,&H00000000,&H96000000,1,0,0,0,100,100,2,0,3,18,0,8,60,60,150,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
HIGHLIGHT = r"{\c&H00E5FF&\fscx112\fscy112}"  # BGR: yellow
RESET = r"{\c&HFFFFFF&\fscx100\fscy100}"


def ass_time(t):
    t = max(0.0, t)
    cs = int(round(t * 100))
    return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


def clean_word(text):
    return re.sub(r"[{}\\]", "", text).strip().upper()


def chunk_words(words, max_words=3, max_gap=0.5):
    chunks, cur = [], []
    for i, w in enumerate(words):
        cur.append(w)
        nxt = words[i + 1] if i + 1 < len(words) else None
        if (len(cur) >= max_words or nxt is None or re.search(r"[.?!,]$", w["text"])
                or nxt["start"] - w["end"] > max_gap):
            chunks.append(cur)
            cur = []
    return chunks


def build_ass(words, offset=0.0, labels=()):
    """Word-by-word karaoke: each word gets its own event, highlighted within its chunk.

    `words` are in absolute times; `offset` is subtracted (clip start).
    `labels` is a sequence of (start, end, text) placed at the top.
    """
    lines = [ASS_HEADER]
    for start, end, text in labels:
        lines.append(f"Dialogue: 1,{ass_time(start)},{ass_time(end)},Label,,0,0,0,,{text}\n")
    chunks = chunk_words(words)
    for ci, chunk in enumerate(chunks):
        for k, w in enumerate(chunk):
            start = w["start"] - offset
            if k + 1 < len(chunk):
                end = chunk[k + 1]["start"] - offset
            else:  # linger briefly, but never under the next chunk
                end = w["end"] - offset + 0.25
                if ci + 1 < len(chunks):
                    end = min(end, chunks[ci + 1][0]["start"] - offset)
            parts = [(HIGHLIGHT + clean_word(x["text"]) + RESET) if x is w else clean_word(x["text"])
                     for x in chunk]
            lines.append(f"Dialogue: 0,{ass_time(start)},{ass_time(max(end, start + 0.08))},Word,,0,0,0,,"
                         + " ".join(parts) + "\n")
    return "".join(lines)


# ---------------------------------------------------------------- orchestration

def sh(cmd, **kw):
    print("+", " ".join(str(c) for c in cmd), file=sys.stderr)
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def ffprobe_duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                          str(path)], capture_output=True, text=True, check=True).stdout
    return float(out.strip())


def before_after(source, src_start, final, out_path, before_s=2.5, fade=0.35):
    """Raw 16:9 letterboxed with BEFORE label -> slide transition -> final with AFTER label."""
    out_path = Path(out_path)
    ass = out_path.with_suffix(".ass")
    ass.write_text(build_ass([], labels=[
        (0, before_s - fade, "BEFORE · RAW 16:9"),
        (before_s, before_s + 1.4, "AFTER · 9:16 EDIT"),
    ]), encoding="utf-8")
    fc = (
        f"[0:v]trim=0:{before_s},setpts=PTS-STARTPTS,fps=30000/1001,scale={OUT_W}:-2,"
        f"pad={OUT_W}:{OUT_H}:0:(oh-ih)/2:color=0x0b1220,setsar=1,format=yuv420p[b];"
        f"[1:v]fps=30000/1001,setsar=1,format=yuv420p[a];"
        f"[b][a]xfade=transition=slideleft:duration={fade}:offset={before_s - fade},subtitles={ass.name}[v];"
        f"[0:a]atrim=0:{before_s},asetpts=PTS-STARTPTS,aresample=48000[ba];"
        f"[1:a]aresample=48000[aa];[ba][aa]acrossfade=d={fade}[aout]"
    )
    sh(["ffmpeg", "-v", "error", "-y", "-ss", f"{src_start:.3f}", "-t", f"{before_s + 0.5}",
        "-i", Path(source).resolve(), "-i", Path(final).resolve(), "-filter_complex", fc,
        "-map", "[v]", "-map", "[aout]", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", out_path.name],
       cwd=out_path.parent)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="16:9 source video (public domain / licensed)")
    ap.add_argument("--out", default=str(HERE / "out"))
    ap.add_argument("--moments", type=int, default=3)
    ap.add_argument("--min-dur", type=float, default=15.0)
    ap.add_argument("--max-dur", type=float, default=30.0)
    ap.add_argument("--whisper-python", default=str(DEFAULT_WHISPER_PY))
    ap.add_argument("--whisper-model", default="small")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--auto-editor", default=shutil.which("auto-editor") or str(HERE / ".venv/Scripts/auto-editor.exe"))
    ap.add_argument("--margin", default="0.2s")
    args = ap.parse_args(argv)

    out = Path(args.out)
    work = out / "work"
    work.mkdir(parents=True, exist_ok=True)
    src = Path(args.input).resolve()

    # 1. silence cut: render and export the same edit as a v3 timeline
    cut, tl = work / "cut.mp4", work / "cut.v3"
    for extra, dest in ((["--export", "v3"], tl), ([], cut)):
        sh([args.auto_editor, src, "--margin", args.margin, *extra, "--progress", "none", "-o", dest])
    fps, clips = load_timeline(tl)
    src_dur, cut_dur = ffprobe_duration(src), ffprobe_duration(cut)

    # 2. transcription on GPU (separate interpreter with faster-whisper)
    wav, tjson = work / "cut.wav", work / "transcript.json"
    sh(["ffmpeg", "-v", "error", "-y", "-i", cut, "-ac", "1", "-ar", "16000", wav])
    sh([args.whisper_python, HERE / "transcribe_gpu.py", wav, tjson, "--model", args.whisper_model,
        "--device", args.device])
    transcript = json.loads(tjson.read_text(encoding="utf-8"))
    words = transcript["words"]

    # 3. moments
    sentences = split_sentences(words)
    moments = pick_moments(sentences, args.moments, args.min_dur, args.max_dur)
    if len(moments) < args.moments:
        print(f"only {len(moments)} moments fit {args.min_dur}-{args.max_dur}s", file=sys.stderr)

    manifest = {
        "source": str(src), "source_seconds": round(src_dur, 2), "cut_seconds": round(cut_dur, 2),
        "silence_removed_seconds": round(src_dur - cut_dur, 2), "auto_editor_margin": args.margin,
        "transcription": {k: transcript[k] for k in ("model", "device", "compute_type", "seconds", "language")},
        "words": len(words), "sentences": len(sentences), "moments": [],
    }
    for n, (m, (s, e)) in enumerate(zip(moments, pad_bounds(moments, cut_dur)), 1):
        seg = work / f"moment-{n}-seg.mp4"
        sh(["ffmpeg", "-v", "error", "-y", "-i", cut, "-ss", f"{s:.3f}", "-to", f"{e:.3f}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "12", "-c:a", "pcm_s16le", seg.with_suffix(".mkv")])
        seg = seg.with_suffix(".mkv")
        clip_words = [w for w in words if s <= w["start"] < e]
        ass = out / f"moment-{n}.ass"
        ass.write_text(build_ass(clip_words, offset=s), encoding="utf-8")
        final = out / f"moment-{n}-final.mp4"
        stats = reframe(seg, final, ass.name, [x - s for x in m["sentence_starts"]])
        src_start = cut_to_source(s, fps, clips)
        ba = out / f"moment-{n}-before-after.mp4"
        before_after(src, src_start, final, ba)
        manifest["moments"].append({
            "n": n, "cut_start": round(s, 2), "cut_end": round(e, 2), "source_start": round(src_start, 2),
            "source_end": round(cut_to_source(e, fps, clips), 2), "seconds": round(e - s, 2),
            "score": m["score"], "reasons": m["reasons"], "text": m["text"], "reframe": stats,
            "final": final.name, "before_after": ba.name,
            "final_seconds": round(ffprobe_duration(final), 2), "before_after_seconds": round(ffprobe_duration(ba), 2),
        })
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"moments": len(manifest["moments"]), "device": manifest["transcription"]["device"],
                      "out": str(out)}))


if __name__ == "__main__":
    main()
