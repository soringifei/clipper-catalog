#!/usr/bin/env python3
"""Generate a synthetic broadcast-like "game" video + ground truth for pipeline tests.

    python scripts/make_synthetic_game.py --out data/raw/synthetic.mp4 --seconds 60

Content (1280x720, 30 fps, AAC crowd audio):
* green field, white yard lines every 5 yd, side-view static camera;
* offense (white kit, dark numbers) vs Bucharest Rebels defense (dark kit, white numbers);
  Rebels #52 lines up as MLB ~5 yd behind the defensive line;
* up to 3 plays: ~3 s static pre-snap, sudden synchronized motion at the snap, #52
  pursues and contacts the ball carrier, who "falls" (box becomes wide/short), settle;
* hard cuts to a dark scoreboard card (2 s) between plays;
* play 2 is followed by a 0.5x slow-motion zoomed replay of itself.

Ground truth goes to ``<out>.truth.json`` (snap/impact times, segments, #52 boxes).
Frames are drawn with OpenCV and piped to ffmpeg (bundled imageio-ffmpeg is fine).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rebels_highlights.core.media import ffmpeg_bin  # noqa: E402

W, H, FPS = 1280, 720, 30
YD = 16                      # px per yard
LOS_X = 600                  # line of scrimmage (offense left, defense right)
PW, PH = 38, 88              # standing player box
PLAY_LEN, CARD_LEN, PRESNAP = 10.0, 2.0, 3.0
REPLAY_SPEED, REPLAY_ZOOM = 0.5, 1.3

REBELS = dict(body=(60, 35, 30), num=(255, 255, 255), trim=(0, 140, 230))   # dark navy, white numbers
OPP = dict(body=(240, 240, 240), num=(40, 30, 30), trim=(40, 40, 170))      # white, dark numbers

# (role, team, number, base_x, base_y)
ROSTER = [
    *[("OL", "opp", n, LOS_X - 22, y) for n, y in zip((72, 74, 70, 75, 76), (290, 335, 380, 425, 470))],
    ("QB", "opp", 12, LOS_X - 90, 380), ("RB", "opp", 28, LOS_X - 150, 395),
    ("WR", "opp", 81, LOS_X - 25, 170), ("WR", "opp", 88, LOS_X - 25, 610),
    ("TE", "opp", 85, LOS_X - 22, 515),
    *[("DL", "reb", n, LOS_X + 24, y) for n, y in zip((91, 94, 97, 99), (305, 360, 410, 465))],
    ("MLB", "reb", 52, LOS_X + 24 + 5 * YD, 385),
    ("OLB", "reb", 44, LOS_X + 24 + 5 * YD, 250), ("OLB", "reb", 33, LOS_X + 24 + 5 * YD, 530),
    ("CB", "reb", 21, LOS_X + 60, 170), ("CB", "reb", 24, LOS_X + 60, 610),
    ("S", "reb", 7, LOS_X + 230, 330), ("S", "reb", 3, LOS_X + 230, 470),
]

# Per-play scenario: impact point (relative to LOS) and time from snap to impact.
SCENARIOS = [
    dict(name="run_stop_near_los", imp=(LOS_X + 35, 350), t_imp=2.3),
    dict(name="tackle_for_loss", imp=(LOS_X - 25, 440), t_imp=1.9),
    dict(name="open_field_tackle", imp=(LOS_X + 150, 300), t_imp=2.8),
]


def ease(s: float) -> float:
    s = min(max(s, 0.0), 1.0)
    return 1 - (1 - s) ** 2          # sudden start at the snap, decelerating


def field_background() -> np.ndarray:
    img = np.zeros((H, W, 3), np.uint8)
    img[:] = (40, 120, 45)
    for i, x in enumerate(range(0, W, 5 * YD)):
        if i % 2:
            img[:, x:x + 5 * YD] = (45, 132, 50)
    cv2.rectangle(img, (0, 0), (W, 95), (60, 60, 60), -1)          # stands
    cv2.rectangle(img, (0, 690), (W, H), (60, 60, 60), -1)
    cv2.line(img, (0, 100), (W, 100), (245, 245, 245), 4)
    cv2.line(img, (0, 685), (W, 685), (245, 245, 245), 4)
    for x in range(40, W, 5 * YD):
        cv2.line(img, (x, 100), (x, 685), (235, 235, 235), 2)
        for y in (280, 505):
            cv2.line(img, (x + 8 * 2, y - 6), (x + 8 * 2, y + 6), (230, 230, 230), 1)
    return img


def player_pos(role: str, number: int, bx: float, by: float, u: float, sc: dict, idx: int):
    """Feet position + fallen flag for time ``u`` seconds after the snap."""
    t_imp = sc["t_imp"]
    ix, iy = sc["imp"]
    u = min(u, t_imp + 1.5)                        # settle: everything stops
    if u <= 0:
        return bx, by, False
    if role == "RB":
        # takes handoff, then runs to the impact point
        s = ease(u / t_imp)
        x, y = bx + (ix - bx) * s, by + (iy - by) * s
        if u > t_imp:
            k = min((u - t_imp) / 0.4, 1.0)
            x += 14 * k
        return x, y, u > t_imp + 0.3
    if number == 52:
        s = ease(u / t_imp)
        tx, ty = ix + 50, iy + 4                   # arrives from the defensive side, boxes overlap
        x, y = bx + (tx - bx) * s, by + (ty - by) * s
        if u > t_imp:
            x -= 10 * min((u - t_imp) / 0.4, 1.0)
        return x, y, False
    if role == "QB":
        return bx - 30 * ease(u / 0.8), by, False
    if role in ("OL", "TE"):
        return bx + 10 * ease(u / 0.5) + 3 * np.sin(8 * u + idx), by + 4 * np.sin(5 * u + idx), False
    if role == "DL":
        return bx - 14 * ease(u / 0.5) + 3 * np.sin(7 * u + idx), by + 5 * np.sin(6 * u + idx), False
    if role == "WR":
        return bx + 170 * ease(u / 3.0), by + (25 if by < 380 else -25) * ease(u / 3.0), False
    if role == "CB":
        return bx + 150 * ease(u / 3.0), by + (25 if by < 380 else -25) * ease(u / 3.0), False
    # OLB / S pursue at ~55 %
    s = 0.55 * ease(u / (t_imp + 0.8))
    return bx + (ix + 120 - bx) * s, by + (iy - by) * s * 0.7, False


def player_box(x: float, y: float, fallen: bool) -> list[float]:
    if fallen:
        return [x - PH / 2, y - PW + 4, x + PH / 2, y]
    return [x - PW / 2, y - PH, x + PW / 2, y]


def draw_player(img, team: str, number: int, box, fallen: bool):
    k = REBELS if team == "reb" else OPP
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    if fallen:
        cv2.rectangle(img, (x1, y1), (x2, y2), k["body"], -1)
        cv2.circle(img, (x1 - 6, (y1 + y2) // 2), 9, (120, 160, 200), -1)
        cv2.rectangle(img, (x1, y1), (x2, y2), k["trim"], 2)
        return
    head_r = 10
    cv2.circle(img, ((x1 + x2) // 2, y1 + head_r), head_r, (120, 160, 200), -1)       # helmet/face
    cv2.rectangle(img, (x1, y1 + 2 * head_r), (x2, y2 - 22), k["body"], -1)           # torso
    cv2.rectangle(img, (x1 + 4, y2 - 22), (x2 - 4, y2), (90, 90, 90) if team == "reb" else (200, 200, 200), -1)
    cv2.rectangle(img, (x1, y1 + 2 * head_r), (x2, y2 - 22), k["trim"], 1)
    txt = str(number)
    fs, th = 0.62, 2
    (tw, tht), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
    cx, cy = (x1 + x2) // 2, (y1 + 2 * head_r + y2 - 22) // 2
    cv2.putText(img, txt, (cx - tw // 2, cy + tht // 2), cv2.FONT_HERSHEY_SIMPLEX, fs, k["num"], th, cv2.LINE_AA)


def render_play_frame(bg, u: float, sc: dict):
    img = bg.copy()
    cv2.line(img, (LOS_X, 100), (LOS_X, 685), (230, 170, 40), 2)                    # LOS graphic
    items, box52 = [], None
    for i, (role, team, num, bx, by) in enumerate(ROSTER):
        x, y, fallen = player_pos(role, num, bx, by, u, sc, i)
        box = player_box(x, y, fallen)
        items.append((y, team, num, box, fallen))
        if num == 52 and team == "reb":
            box52 = box
    for _, team, num, box, fallen in sorted(items, key=lambda it: (not it[4], it[0])):
        draw_player(img, team, num, box, fallen)
    cv2.rectangle(img, (20, 20), (330, 70), (20, 20, 20), -1)
    cv2.putText(img, "REBELS 14  VIS 7   Q2", (32, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return img, box52


def card_frame(lines: list[str], color=(35, 22, 18)):
    img = np.zeros((H, W, 3), np.uint8)
    img[:] = color
    cv2.rectangle(img, (160, 200), (W - 160, H - 200), (70, 50, 40), -1)
    for i, t in enumerate(lines):
        (tw, _), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_DUPLEX, 1.3, 2)
        cv2.putText(img, t, ((W - tw) // 2, 300 + 80 * i), cv2.FONT_HERSHEY_DUPLEX, 1.3, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def zoom_about(img, cx: float, cy: float, z: float):
    cw, ch = W / z, H / z
    x0 = min(max(cx - cw / 2, 0), W - cw)
    y0 = min(max(cy - ch / 2, 0), H - ch)
    M = np.float32([[z, 0, -x0 * z], [0, z, -y0 * z]])
    return cv2.warpAffine(img, M, (W, H)), (x0, y0)


def build_schedule(seconds: float):
    segs, t, n = [], 0.0, 0
    segs.append(dict(kind="card", start=0.0, end=CARD_LEN, text=["BUCHAREST REBELS", "vs VISITORS"]))
    t = CARD_LEN
    while n < len(SCENARIOS) and t + PLAY_LEN <= seconds:
        segs.append(dict(kind="play", start=t, end=t + PLAY_LEN, idx=n))
        t += PLAY_LEN
        if n == 1:
            sc = SCENARIOS[n]
            src0 = segs[-1]["start"] + PRESNAP - 1.0
            src1 = segs[-1]["start"] + PRESNAP + sc["t_imp"] + 1.5
            dur = (src1 - src0) / REPLAY_SPEED
            if t + dur <= seconds:
                segs.append(dict(kind="replay", start=t, end=t + dur, idx=n, src0=src0, src1=src1))
                t += dur
        n += 1
        if t + CARD_LEN <= seconds:
            segs.append(dict(kind="card", start=t, end=t + CARD_LEN,
                             text=["SCOREBOARD", f"REBELS 14 - 7 VISITORS", f"DOWN {n + 1} & 10"]))
            t += CARD_LEN
    if t < seconds:
        segs.append(dict(kind="card", start=t, end=seconds, text=["TIMEOUT", "REBELS 14 - 7 VISITORS"]))
    return segs


def audio_filter(plays: list[dict], seconds: float) -> str:
    cheer = "+".join(f"between(t,{p['impact_s']:.2f},{p['impact_s'] + 1.8:.2f})" for p in plays) or "0"
    whistle = "+".join(f"between(t,{p['impact_s'] + 1.6:.2f},{p['impact_s'] + 2.0:.2f})" for p in plays) or "0"
    return (f"[1:a]volume=eval=frame:volume='0.5+1.2*({cheer})'[c];"
            f"[2:a]volume=eval=frame:volume='0.25*({whistle})'[w];"
            f"[c][w]amix=inputs=2:normalize=0,atrim=0:{seconds}[a]")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="data/raw/synthetic.mp4")
    ap.add_argument("--seconds", type=float, default=60.0)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    seconds = float(a.seconds)
    segs = build_schedule(seconds)
    bg = field_background()

    plays, replays, cards = [], [], []
    for s in segs:
        if s["kind"] == "play":
            sc = SCENARIOS[s["idx"]]
            snap = s["start"] + PRESNAP
            plays.append(dict(play_index=s["idx"] + 1, start_s=round(s["start"], 3), end_s=round(s["end"], 3),
                              snap_s=round(snap, 3), impact_s=round(snap + sc["t_imp"], 3),
                              scenario=sc["name"], ball_carrier_number=28, ball_carrier_team="opponent",
                              has_replay=any(r["kind"] == "replay" and r["idx"] == s["idx"] for r in segs)))
        elif s["kind"] == "replay":
            replays.append(dict(of_play_index=s["idx"] + 1, start_s=round(s["start"], 3), end_s=round(s["end"], 3),
                                source_start_s=round(s["src0"], 3), source_end_s=round(s["src1"], 3),
                                speed=REPLAY_SPEED, zoom=REPLAY_ZOOM))
        else:
            cards.append(dict(start_s=round(s["start"], 3), end_s=round(s["end"], 3), text=s["text"]))

    ff = ffmpeg_bin()
    cmd = [ff, "-y", "-hide_banner", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-f", "lavfi", "-i", f"anoisesrc=color=pink:amplitude=0.25:sample_rate=44100:duration={seconds}",
           "-f", "lavfi", "-i", f"sine=frequency=2600:sample_rate=44100:duration={seconds}",
           "-filter_complex", audio_filter(plays, seconds), "-map", "0:v", "-map", "[a]",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "128k", "-shortest", "-movflags", "+faststart", str(out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    track52, track52_replay = [], []
    n_frames = int(round(seconds * FPS))
    for fi in range(n_frames):
        t = fi / FPS
        seg = next((s for s in segs if s["start"] <= t < s["end"]), segs[-1])
        if seg["kind"] == "card":
            img = card_frame(seg["text"])
        elif seg["kind"] == "play":
            sc = SCENARIOS[seg["idx"]]
            img, box = render_play_frame(bg, t - seg["start"] - PRESNAP, sc)
            track52.append(dict(t=round(t, 3), box=[round(v, 1) for v in box]))
        else:
            sc = SCENARIOS[seg["idx"]]
            src_t = seg["src0"] + (t - seg["start"]) * REPLAY_SPEED
            snap = seg["src0"] + 1.0
            base, box = render_play_frame(bg, src_t - snap, sc)
            img, (x0, y0) = zoom_about(base, sc["imp"][0], sc["imp"][1] - 40, REPLAY_ZOOM)
            cv2.rectangle(img, (W - 220, 20), (W - 20, 70), (0, 0, 200), -1)
            cv2.putText(img, "REPLAY", (W - 200, 57), cv2.FONT_HERSHEY_DUPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
            zb = [(box[0] - x0) * REPLAY_ZOOM, (box[1] - y0) * REPLAY_ZOOM,
                  (box[2] - x0) * REPLAY_ZOOM, (box[3] - y0) * REPLAY_ZOOM]
            track52_replay.append(dict(t=round(t, 3), source_t=round(src_t, 3), box=[round(v, 1) for v in zb]))
        proc.stdin.write(img.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        print("ffmpeg failed", file=sys.stderr)
        return 1

    truth = dict(video=str(out), width=W, height=H, fps=FPS, duration_s=seconds,
                 note="Synthetic test video. Scenario names describe the intended motion, "
                      "not a verified football label.",
                 player={"team": "Bucharest Rebels", "number": 52, "position": "MLB",
                         "kit": "dark navy jersey, white numbers"},
                 los_x=LOS_X, yard_px=YD,
                 scene_cuts=[round(s["start"], 3) for s in segs[1:]],
                 segments=[{k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()} for s in segs],
                 plays=plays, replays=replays, cards=cards,
                 player_52_track=track52, player_52_replay_track=track52_replay)
    tp = Path(str(out) + ".truth.json") if not str(out).endswith(".mp4") else out.with_suffix(".truth.json")
    tp.write_text(json.dumps(truth, indent=1))
    print(f"wrote {out} and {tp}: {len(plays)} plays, {len(replays)} replay(s), {len(cards)} cards")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
