"""HookHaus brand layer for the vertical clips made by hookhaus_v2.

Hook card (first seconds), lime word-by-word subtitles, the H mark, the cover and the caption.
Everything is deterministic. The hook wording is written or reviewed by a person; `check_hook`
only flags obvious problems and does not predict views.
"""
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FONT_DIR = ROOT / "fonts"
BLACK = FONT_DIR / "InterDisplay-Black.otf"
BOLD = FONT_DIR / "Inter-Bold.otf"

LIME = (198, 255, 61)  # brand/mark.svg #C6FF3D
INK = (17, 17, 19)  # #111113
ORANGE = (255, 90, 31)  # #FF5A1F
WHITE = (255, 255, 255)
ASS_LIME = "&H3DFFC6&"  # ASS colours are BGR
ASS_WHITE = "&HFFFFFF&"
W, H = 1080, 1920

HOOK_MAX_WORDS = 9
HOOK_MAX_CHARS = 60
HOOK_END = 2.8  # seconds the hook card stays on screen
TENSION = r"\b(never|stop|why|how|nobody|mistake|lose|lost|wrong|secret|fail|failed|broke|cost|worth)\b"

ASS_HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Word,Inter Display Black,88,&H00FFFFFF,&H00FFFFFF,&H00111113,&H64000000,0,0,0,0,100,100,0,0,1,8,3,2,90,150,560,1
Style: Hook,Inter Display Black,96,&H00FFFFFF,&H00FFFFFF,&H00111113,&H64000000,0,0,0,0,100,100,0,0,1,9,4,8,70,150,215,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


# ------------------------------------------------------------------ hook text

def split_markup(text):
    """'*$100M* in a year' -> [('$100M', True), (' in a year', False)]."""
    return [(part, i % 2 == 1) for i, part in enumerate(text.split("*")) if part]


def plain(text):
    return text.replace("*", "")


def check_hook(text):
    """Return a list of problems; an empty list means no obvious problem was found."""
    t = plain(text).strip()
    words = t.split()
    problems = []
    if len(words) > HOOK_MAX_WORDS:
        problems.append(f"{len(words)} words, max {HOOK_MAX_WORDS}")
    if len(t) > HOOK_MAX_CHARS:
        problems.append(f"{len(t)} characters, max {HOOK_MAX_CHARS}")
    if text.count("*") < 2:
        problems.append("no highlighted key word (*word*)")
    if not (re.search(r"[0-9$%€£]", t) or "?" in t or re.search(TENSION, t, re.I)):
        problems.append("no number, question or tension word")
    return problems


def hook_words(text):
    """Words with their highlight flag; punctuation typed right after a word stays attached to it."""
    out = []
    for seg, hl in split_markup(text):
        for k, w in enumerate(seg.split()):
            glued = k == 0 and not seg[0].isspace() and out and not re.search(r"[A-Za-z0-9$€£%]", w)
            if glued:
                out[-1] = (out[-1][0] + w, out[-1][1])
            else:
                out.append((w, hl))
    return out


def wrap_words(words, max_chars=15):
    lines, cur, n = [], [], 0
    for w, hl in words:
        add = len(w) + (1 if cur else 0)
        if cur and n + add > max_chars:
            lines.append(cur)
            cur, n, add = [], 0, len(w)
        cur.append((w, hl))
        n += add
    if cur:
        lines.append(cur)
    return lines


def _ass_safe(word):
    return re.sub(r"[{}\\]", "", word).upper()


PROFANITY = re.compile(r"^(f+u+c+k\w*|sh+i+t\w*|b+i+t+c+h\w*|a+s+h+o+l+e\w*|d+a+m+n\w*)$", re.I)


def drop_tail_hallucination(words):
    """Whisper invents 'Thank you.' / 'Thanks for watching.' over trailing silence or music."""
    ws = list(words)
    while len(ws) >= 3 and re.sub(r"[^a-z ]", "", " ".join(w["text"] for w in ws[-2:]).lower()) in ("thank you", "thanks for"):
        ws = ws[:-2]
    if len(ws) >= 4 and re.sub(r"[^a-z ]", "", " ".join(w["text"] for w in ws[-3:]).lower()) == "thanks for watching":
        ws = ws[:-3]
    return ws


def tidy_words(words):
    """Glue a lone '%' to the number before it and star out profanity in the burned-in subtitles."""
    out = []
    for w in drop_tail_hallucination(words):
        w = dict(w)
        core = re.sub(r"[^\w]", "", w["text"])
        if w["text"].strip() == "%" and out:
            out[-1]["text"] += "%"
            out[-1]["end"] = w["end"]
            continue
        if core and PROFANITY.match(core):
            w["text"] = w["text"].replace(core, core[0] + "*" * (len(core) - 1))
        out.append(w)
    return out


def hook_event(text, start=0.0, end=HOOK_END, stack=False):
    from hookhaus_v2 import ass_time

    lines = []
    for line in wrap_words(hook_words(text)):
        lines.append(" ".join(
            (f"{{\\c{ASS_LIME}}}{_ass_safe(w)}{{\\c{ASS_WHITE}}}" if hl else _ass_safe(w)) for w, hl in line))
    body = "\\N".join(lines)
    where = "{\\pos(540,545)}" if stack else ""
    pop = where + "{\\fad(90,260)\\fscx90\\fscy90\\t(0,170,\\fscx100\\fscy100)}"
    return f"Dialogue: 2,{ass_time(start)},{ass_time(end)},Hook,,0,0,0,,{pop}{body}\n"


def build_branded_ass(words, hook, offset=0.0, hook_end=HOOK_END, stack=False):
    """Hook card + word-by-word subtitles with the lime highlight."""
    from hookhaus_v2 import ass_time, chunk_words, clean_word

    lines = [ASS_HEADER, hook_event(hook, 0.0, hook_end, stack)]
    hi = f"{{\\c{ASS_LIME}\\fscx112\\fscy112}}"
    reset = f"{{\\c{ASS_WHITE}\\fscx100\\fscy100}}"
    chunks = chunk_words(words)
    seam = "{\\an5\\pos(540,985)}" if stack else ""
    for ci, chunk in enumerate(chunks):
        for k, w in enumerate(chunk):
            start = w["start"] - offset
            if k + 1 < len(chunk):
                end = chunk[k + 1]["start"] - offset
            else:
                end = w["end"] - offset + 0.25
                if ci + 1 < len(chunks):
                    end = min(end, chunks[ci + 1][0]["start"] - offset)
            parts = [(hi + clean_word(x["text"]) + reset) if x is w else clean_word(x["text"]) for x in chunk]
            lines.append(f"Dialogue: 0,{ass_time(start)},{ass_time(max(end, start + 0.08))},Word,,0,0,0,,"
                         + seam + " ".join(parts) + "\n")
    return "".join(lines)


# ------------------------------------------------------------------ mark

def _bezier(p0, p1, p2, n=24):
    pts = []
    for i in range(1, n + 1):
        t = i / n
        pts.append(((1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t * t * p2[0],
                    (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t * t * p2[1]))
    return pts


def draw_mark(size=512):
    """The H mark from brand/mark.svg, drawn with PIL (no SVG renderer needed)."""
    k = 2
    s = 1000 * k
    layer = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    for x in (245, 650):
        d.rectangle([x * k, 265 * k, (x + 105) * k, (265 + 470) * k], fill=INK + (255,))
    path = [(350, 470), (548, 470)]
    path += _bezier((548, 470), (612, 470), (612, 534))
    path += [(612, 560)]
    path += _bezier((612, 560), (612, 628), (548, 628))
    path += _bezier((548, 628), (484, 628), (484, 568))
    d.line([(x * k, y * k) for x, y in path], fill=ORANGE + (255,), width=88 * k, joint="curve")
    layer = layer.rotate(6, center=(s // 2, s // 2), resample=Image.BICUBIC)  # SVG rotate(-6) = 6 deg counter-clockwise
    base = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    ImageDraw.Draw(base).ellipse([0, 0, s - 1, s - 1], fill=LIME + (255,))
    base.alpha_composite(layer)
    mask = Image.new("L", (s, s), 0)
    ImageDraw.Draw(mask).ellipse([0, 0, s - 1, s - 1], fill=255)
    base.putalpha(mask)
    return base.resize((size, size), Image.LANCZOS)


# ------------------------------------------------------------------ cover

def _fit_title(words, font_path, max_w=860, max_lines=4, start=136, floor=84):
    size = start
    while size >= floor:
        font = ImageFont.truetype(str(font_path), size)
        lines, cur = [], []
        for w, hl in words:
            trial = cur + [(w, hl)]
            if cur and font.getlength(" ".join(x[0] for x in trial).upper()) > max_w:
                lines.append(cur)
                cur = [(w, hl)]
            else:
                cur = trial
        if cur:
            lines.append(cur)
        if len(lines) <= max_lines and all(font.getlength(" ".join(x[0] for x in ln).upper()) <= max_w for ln in lines):
            return font, lines, size
        size -= 4
    raise ValueError("title does not fit the cover; shorten it")


def make_cover(frame, title, out_path, kicker="CLIPS PEOPLE FINISH"):
    """1080x1920 cover from a clip frame. Title and mark stay inside the centre 3:4 band that the
    profile grid shows, and away from TikTok's bottom caption area."""
    img = ImageOps.fit(frame.convert("RGB"), (W, H), Image.LANCZOS).convert("RGBA")
    column = Image.new("RGBA", (1, H))
    for y in range(H):
        a = 0.16 + (min(0.72, (y - 820) / 560 * 0.72) if y > 820 else 0.0)
        column.putpixel((0, y), (8, 8, 10, int(255 * min(a, 0.88))))
    img.alpha_composite(column.resize((W, H), Image.NEAREST))

    font, lines, size = _fit_title(hook_words(title), BLACK)
    lh = int(size * 1.04)
    block_h = lh * len(lines)
    bottom = 1440
    top = bottom - block_h
    text = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(text)
    for li, line in enumerate(lines):
        x = 70
        for w, hl in line:
            word = w.upper()
            d.text((x, top + li * lh), word, font=font, fill=(LIME if hl else WHITE) + (255,))
            x += font.getlength(word + " ")
    shadow = text.filter(ImageFilter.GaussianBlur(14))
    sh_px = shadow.split()[3].point(lambda v: min(255, int(v * 1.1)))
    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    shadow.putalpha(sh_px)
    img.alpha_composite(shadow)
    img.alpha_composite(text)
    d = ImageDraw.Draw(img)
    d.rectangle([70, top - 44, 70 + 190, top - 44 + 14], fill=LIME + (255,))
    kfont = ImageFont.truetype(str(BOLD), 40)
    d.text((70, bottom + 30), kicker, font=kfont, fill=WHITE + (214,))
    mark = draw_mark(124)
    img.alpha_composite(mark, (70, 290))
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(out, quality=94)
    return {"font_size": size, "lines": len(lines)}


# ------------------------------------------------------------------ caption

def build_caption(hook, question, tags, source=None):
    """Hook, question, 3-4 tags, then the short attribution (kept: CC BY needs it)."""
    tags = [t if t.startswith("#") else "#" + t for t in tags][:4]
    parts = [plain(hook).strip(), (question or "").strip(), " ".join(tags)]
    if source:
        parts.append(f"Source: {source['title']} ({source['channel']}) {source['url']} "
                     f"CC BY 3.0. Edited excerpt; reframed and captioned.")
    return "\n".join(p for p in parts if p)
