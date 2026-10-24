"""Branded HookHaus clip from ONE already-cut section of a licensed source.

    python make_branded.py --input section.mp4 --out out/dimon-1 \
        --hook "*Head in the sand*, you lose" --question "Do you look at the numbers you avoid?" \
        --tags business,entrepreneurship,founder --source-id TEhy1JtzxIc \
        --source-title "Conversation with Jamie Dimon..." --source-channel "World Economic Forum"

Writes final.mp4, cover.jpg, caption.txt, manifest.json and QA frames at 0/1/2/5 s. It never uploads
or publishes anything. Runs at below-normal priority and on CPU unless --device cuda is given.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import brand_layer as bl  # noqa: E402
import hookhaus_v2 as hv  # noqa: E402
from PIL import Image  # noqa: E402


def letterbox_pct(path):
    """Share of the frame height that is black bars (ffmpeg cropdetect on 3 s). None if not measurable."""
    import re
    r = subprocess.run(["ffmpeg", "-v", "info", "-ss", "1", "-t", "3", "-i", str(path), "-vf",
                        "cropdetect=limit=24:round=2", "-f", "null", "-"], capture_output=True, text=True)
    crops = re.findall(r"crop=(\d+):(\d+):(\d+):(\d+)", r.stderr)
    dims = re.search(r"Stream #0:\d.*?Video:.*?, (\d{2,5})x(\d{2,5})", r.stderr)
    if not crops or not dims:
        return None
    full_h = int(dims.group(2))
    return round(100 * (1 - int(crops[-1][1]) / full_h), 1)


def low_priority():
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hook", required=True, help="max 9 words, key words between *stars*")
    ap.add_argument("--question", default="")
    ap.add_argument("--tags", default="")
    ap.add_argument("--source-id", required=True)
    ap.add_argument("--source-title", required=True)
    ap.add_argument("--source-channel", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--whisper-model", default="small")
    ap.add_argument("--cover-t", type=float, default=1.2)
    ap.add_argument("--cover-title", default="", help="defaults to the hook; keep it to 3 lines")
    ap.add_argument("--allow-weak-hook", action="store_true")
    ap.add_argument("--layout", choices=["auto", "single", "stack"], default="auto",
                help="stack = two speakers one above the other (auto when faces are far apart)")
    ap.add_argument("--keep-pauses", action="store_true", help="skip silence removal")
    args = ap.parse_args(argv)

    problems = bl.check_hook(args.hook)
    if problems and not args.allow_weak_hook:
        sys.exit("hook rejected: " + "; ".join(problems))
    low_priority()

    out = Path(args.out).resolve()
    work = out / "work"
    work.mkdir(parents=True, exist_ok=True)
    fonts = out / "fonts"
    fonts.mkdir(exist_ok=True)
    for f in bl.FONT_DIR.glob("*.otf"):
        shutil.copy2(f, fonts / f.name)
    mark = out / "mark.png"
    bl.draw_mark(112).save(mark)
    src = Path(args.input).resolve()
    bars = letterbox_pct(src)
    if bars is not None and bars >= 3:
        print(f"WARNING: source has {bars}% black bars; the 9:16 crop will show them. Pick another source.",
              file=sys.stderr)

    cut, tl = work / "cut.mp4", work / "cut.v3"
    auto = shutil.which("auto-editor") or str(HERE / ".venv/Scripts/auto-editor.exe")
    if args.keep_pauses:  # no silence removal: use when auto-editor clips quiet words
        hv.sh(["ffmpeg", "-v", "error", "-y", "-i", src, "-c:v", "libx264", "-preset", "veryfast", "-crf", "12",
               "-c:a", "aac", "-b:a", "192k", cut])
    else:
        for extra, dest in ((["--export", "v3"], tl), ([], cut)):
            hv.sh([auto, src, "--margin", "0.2s", *extra, "--progress", "none", "-o", dest])
    wav, tjson = work / "cut.wav", work / "transcript.json"
    hv.sh(["ffmpeg", "-v", "error", "-y", "-i", cut, "-ac", "1", "-ar", "16000", wav])
    hv.sh([sys.executable, HERE / "transcribe_gpu.py", wav, tjson, "--model", args.whisper_model,
           "--device", args.device])
    transcript = json.loads(tjson.read_text(encoding="utf-8"))
    words = bl.tidy_words(transcript["words"])
    sentences = hv.split_sentences(words)
    cut_dur = hv.ffprobe_duration(cut)

    seg = work / "seg.mkv"
    hv.sh(["ffmpeg", "-v", "error", "-y", "-i", cut, "-c:v", "libx264", "-preset", "veryfast", "-crf", "12",
           "-c:a", "pcm_s16le", seg])
    layout = hv.pick_layout(seg, args.layout)
    ass = out / "clip.ass"
    ass.write_text(bl.build_branded_ass(words, args.hook, stack=(layout == "stack")), encoding="utf-8")
    cover_raw = work / "cover_raw.png"
    stats = hv.reframe(seg, out / "final.mp4", ass.name, [s["start"] for s in sentences],
                       fontsdir="fonts", mark_png=mark, cover_path=cover_raw, cover_t=args.cover_t, layout=layout)
    with Image.open(cover_raw) as frame:
        cover = bl.make_cover(frame.copy(), args.cover_title or args.hook, out / "cover.jpg")

    source = {"title": args.source_title, "channel": args.source_channel,
              "url": f"https://youtu.be/{args.source_id}"}
    tags = [t for t in args.tags.split(",") if t]
    caption = bl.build_caption(args.hook, args.question, tags, source)
    (out / "caption.txt").write_text(caption, encoding="utf-8")

    qa = out / "qa"
    qa.mkdir(exist_ok=True)
    for t in (0.0, 1.0, 2.0, 5.0):
        hv.sh(["ffmpeg", "-v", "error", "-y", "-ss", str(t), "-i", out / "final.mp4", "-frames:v", "1",
               qa / f"t{int(t)}s.png"])
    manifest = {
        "source": source, "license": "CC BY (as declared on YouTube by the uploader; rights not independently verified)",
        "hook": args.hook, "hook_problems": problems, "source_black_bars_pct": bars, "cut_seconds": round(cut_dur, 2),
        "final_seconds": round(hv.ffprobe_duration(out / "final.mp4"), 2), "reframe": stats, "cover": cover,
        "transcript_words": len(words), "text": " ".join(s["text"] for s in sentences),
        "caption_chars": len(caption), "published": False,
    }
    framing = []
    if stats.get("layout") != "stack" and stats.get("face_x_spread_pct", 0) > 40:
        framing.append(f"faces spread over {stats['face_x_spread_pct']}% of the frame (wide two-shot): a 9:16 crop cannot follow the speaker")
    if stats.get("layout") != "stack" and 0 < stats.get("face_med_width_pct", 100) < 8:
        framing.append(f"faces only {stats['face_med_width_pct']}% of frame width: too small for a vertical crop")
    if stats.get("layout") != "stack" and stats.get("face_hit_rate", 1) < 0.5:
        framing.append(f"face found in only {stats['face_hit_rate']:.0%} of samples")
    manifest["framing_problems"] = framing
    manifest["ready"] = not framing and not problems
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"out": str(out), "final_seconds": manifest["final_seconds"],
                      "face_hit_rate": stats["face_hit_rate"], "hook_problems": problems,
                      "framing_problems": framing, "ready": manifest["ready"]}))


if __name__ == "__main__":
    main()
