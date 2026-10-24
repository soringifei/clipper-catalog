"""Rebuild one clip from a known YouTube source with the HookHaus brand layer.
usage: python rebuild_clip.py JOBS.json NAME   (never uploads; writes out/NAME/)"""
import json, re, subprocess, sys, pathlib
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import brand_layer as bl
def run(cmd, **kw):
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, encoding="utf-8", errors="replace", **kw)
    if r.returncode: raise SystemExit(f"FAIL {cmd[0]}: {r.stderr[-400:]}")
    return r.stdout
def sent_end(s):
    s = s.strip()
    return bool(re.search(r"[.?!][\"')]*$", s)) and not s.rstrip("\"')").endswith("...")
def tok(s): return re.sub(r"[^a-z0-9]", "", s.lower())
def main():
    jobs = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")); name = sys.argv[2]; j = jobs[name]
    out = HERE / "out" / name; src = out / "src"; src.mkdir(parents=True, exist_ok=True)
    url = f"https://www.youtube.com/watch?v={j['id']}"
    a, b = max(0, j["vtt"] - 8), j["vtt"] + 48
    if not (src / "section.mkv").exists():
        run(["yt-dlp", "-f", "bv*[height<=1080][vcodec^=avc1]+ba/b[height<=1080]", "--download-sections", f"*{a}-{b}", "--no-playlist",
             "-o", src / "section.%(ext)s", url])
    wide = src / "wide_h264.mp4"
    run(["ffmpeg", "-v", "error", "-y", "-i", src / "section.mkv", "-c:v", "libx264", "-crf", "18", "-preset", "veryfast", "-c:a", "aac", wide])
    wav = src / "wide.wav"; run(["ffmpeg", "-v", "error", "-y", "-i", wide, "-ac", "1", "-ar", "16000", wav])
    run([sys.executable, HERE / "transcribe_gpu.py", wav, src / "wide.json", "--model", "medium", "--device", "cpu"])
    words = json.loads((src / "wide.json").read_text(encoding="utf-8"))["words"]
    want = [tok(w) for w in j["start_phrase"].split()]
    idx = next((i for i in range(len(words)) if [tok(x["text"]) for x in words[i:i + len(want)]] == want), None)
    if idx is None: raise SystemExit("start phrase not found in whisper words: " + " ".join(w["text"] for w in words[:60]))
    t0 = words[idx]["start"]; end = None
    if j.get("end_phrase"):
        ew = [tok(w) for w in j["end_phrase"].split()]
        e = next((i for i in range(idx, len(words)) if [tok(x["text"]) for x in words[i:i + len(ew)]] == ew), None)
        if e is None: raise SystemExit("end phrase not found")
        end = words[e + len(ew) - 1]["end"]
    for w in ([] if end else words[idx:]):
        d = w["end"] - t0
        if d >= j.get("min", 14) and sent_end(w["text"]): end = w["end"]; break
        if d >= j.get("max", 32): end = w["end"]; break
    if end is None: end = words[-1]["end"]
    nxt_start = next((w["start"] for w in words if w["start"] > end), None)
    clip = out / "clip_in.mp4"
    run(["ffmpeg", "-v", "error", "-y", "-ss", f"{max(0, t0 - 0.12):.2f}", "-to", f"{min(end + 0.3, nxt_start - 0.03) if nxt_start else end + 0.3:.2f}", "-i", wide, "-c:v", "libx264", "-crf", "14",
         "-preset", "veryfast", "-c:a", "aac", "-b:a", "192k", clip])
    title = run(["yt-dlp", "--get-title", "--no-playlist", url]).strip().splitlines()[0]
    cmd = [sys.executable, HERE / "make_branded.py", "--input", clip, "--out", out, "--hook", j["hook"], "--question", j["question"],
           "--tags", j["tags"], "--source-id", j["id"], "--source-title", title, "--source-channel", j["channel"], "--device", "cpu",
           "--whisper-model", "medium", "--keep-pauses"]
    if j.get("layout"): cmd += ["--layout", j["layout"]]
    for attempt in range(3):  # whisper word times can drift: confirm the clip really opens on the start phrase
        last = run(cmd, cwd=HERE).strip().splitlines()[-1]
        tw = json.loads((out / "work" / "transcript.json").read_text(encoding="utf-8"))["words"]
        tw = bl.drop_tail_hallucination(tw)
        k = next((i for i in range(len(tw)) if [tok(x["text"]) for x in tw[i:i + len(want)]] == want), None)
        head = k is not None and k > 0 and tw[k]["start"] >= 0.35
        tail = not sent_end(tw[-1]["text"])
        to = None
        if tail:  # never end mid-sentence: cut after the last word that closes a sentence
            ends = [w for w in tw if sent_end(w["text"])]
            if ends and ends[-1]["end"] >= 8:
                nxt = next((w["start"] for w in tw if w["start"] >= ends[-1]["end"] and w is not ends[-1]), None)
                to = ends[-1]["end"] + 0.3
                if nxt is not None:
                    to = min(to, max(ends[-1]["end"] + 0.05, nxt - 0.03))
        if not head and to is None:
            break
        ss = tw[k]["start"] - 0.12 if head else 0.0
        tmp = out / "clip_tmp.mp4"
        extra = ["-to", f"{to:.2f}"] if to else []
        run(["ffmpeg", "-v", "error", "-y", "-ss", f"{ss:.2f}", *extra, "-i", clip, "-c:v", "libx264", "-crf", "14",
             "-preset", "veryfast", "-c:a", "aac", "-b:a", "192k", tmp])
        tmp.replace(clip)
    print(last)
main()