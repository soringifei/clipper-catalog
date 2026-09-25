"""Synthetic side-view keypoint sequences (COCO-17) for tests and offline demos.

Proportions are fractions of stature ``H`` px; the athlete faces +x, image y points down.
"""
from __future__ import annotations

import numpy as np

from .pose import KP

SHANK, THIGH, TRUNK, UARM, FARM, NECK = 0.246, 0.245, 0.29, 0.186, 0.146, 0.11
ANKLE_H = 0.039


def _ik(a: np.ndarray, c: np.ndarray, l1: float, l2: float, bend: int) -> np.ndarray:
    """Middle joint of a 2-link chain from a to c; bend=+1/-1 chooses the side."""
    d = c - a
    dist = float(np.clip(np.linalg.norm(d), 1e-6, l1 + l2 - 1e-6))
    u = d / (np.linalg.norm(d) + 1e-9)
    x = (l1 ** 2 - l2 ** 2 + dist ** 2) / (2 * dist)
    h = np.sqrt(max(l1 ** 2 - x ** 2, 0.0))
    perp = np.array([-u[1], u[0]]) * bend
    return a + u * x + perp * h


def _body(ankle, hip, trunk_lean_deg, wrist, H, knee_bend=+1, elbow_bend=-1, far_dx=4.0):
    """Keypoints (17,2) from ankle, hip, trunk lean (deg, forward +) and wrist target."""
    ankle, hip, wrist = (np.asarray(v, float) for v in (ankle, hip, wrist))
    knee = _ik(ankle, hip, SHANK * H, THIGH * H, knee_bend)
    lam = np.radians(trunk_lean_deg)
    sho = hip + TRUNK * H * np.array([np.sin(lam), -np.cos(lam)])
    elb = _ik(sho, wrist, UARM * H, FARM * H, elbow_bend)
    wr = sho + (wrist - sho) * min(1.0, (UARM + FARM) * H * 0.999 / (np.linalg.norm(wrist - sho) + 1e-9))
    nose = sho + NECK * H * np.array([np.sin(lam) + 0.35, -np.cos(lam)])
    k = np.zeros((17, 2))
    near = {"sho": sho, "elb": elb, "wri": wr, "hip": hip, "knee": knee, "ank": ankle}
    for n, p in near.items():
        k[KP[f"l_{n}"]] = p
        k[KP[f"r_{n}"]] = p + np.array([far_dx, 0.0])
    k[KP["nose"]] = nose
    k[KP["l_eye"]] = nose + np.array([-0.01, -0.012]) * H
    k[KP["r_eye"]] = nose + np.array([-0.005, -0.012]) * H
    k[KP["l_ear"]] = nose + np.array([-0.06, -0.005]) * H
    k[KP["r_ear"]] = nose + np.array([-0.055, -0.005]) * H
    return k


def _with_conf(frames: list[np.ndarray], near_conf=0.92, far_conf=0.6) -> np.ndarray:
    kp = np.stack(frames)
    conf = np.full(kp.shape[:2] + (1,), near_conf)
    for n in ("r_sho", "r_elb", "r_wri", "r_hip", "r_knee", "r_ank", "r_eye", "r_ear"):
        conf[:, KP[n]] = far_conf
    return np.concatenate([kp, conf], -1)


def squat(T: int, fps: float, reps: int = 5, H: float = 700, x0: float = 640, ground: float = 1000,
          depth: float = 1.0) -> np.ndarray:
    """Back squat: bar (hands) on the upper back; depth 1.0 ~ hips just below knees."""
    out = []
    for t in range(T):
        d = depth * 0.5 * (1 - np.cos(2 * np.pi * reps * t / T))
        ank = np.array([x0, ground - ANKLE_H * H])
        leg = (SHANK + THIGH) * H
        hip = ank + np.array([-0.16 * d * H, -leg * (1 - 0.62 * d) + 0.0])
        lean = 12 + 33 * d
        lam = np.radians(lean)
        sho = hip + TRUNK * H * np.array([np.sin(lam), -np.cos(lam)])
        wrist = sho + np.array([-0.03 * H, 0.10 * H])
        out.append(_body(ank, hip, lean, wrist, H))
    return _with_conf(out)


def deadlift(T: int, fps: float, reps: int = 4, H: float = 700, x0: float = 640,
             ground: float = 1000) -> np.ndarray:
    out = []
    for t in range(T):
        d = 0.5 * (1 - np.cos(2 * np.pi * reps * t / T))
        ank = np.array([x0, ground - ANKLE_H * H])
        leg = (SHANK + THIGH) * H
        hip = ank + np.array([-0.14 * d * H, -leg * (1 - 0.22 * d)])
        lean = 5 + 65 * d
        lam = np.radians(lean)
        sho = hip + TRUNK * H * np.array([np.sin(lam), -np.cos(lam)])
        wrist = sho + np.array([0.0, (UARM + FARM) * H * 0.98])
        out.append(_body(ank, hip, lean, wrist, H))
    return _with_conf(out)


def bench(T: int, fps: float, reps: int = 5, H: float = 700, x0: float = 400,
          bench_y: float = 800) -> np.ndarray:
    """Lying on a bench, head to +x, bar pressed vertically above the shoulders."""
    out = []
    for t in range(T):
        d = 0.5 * (1 - np.cos(2 * np.pi * reps * t / T))
        hip = np.array([x0, bench_y])
        sho = hip + np.array([TRUNK * H, 0.0])
        knee = hip + np.array([-THIGH * H * 0.9, -THIGH * H * 0.3])
        ank = knee + np.array([-0.05 * H, SHANK * H * 0.98])
        reach = (UARM + FARM) * H
        wrist = sho + np.array([0.0, -reach * (0.97 - 0.62 * d)])
        elb = _ik(sho, wrist, UARM * H, FARM * H, +1)
        nose = sho + np.array([NECK * H, -0.04 * H])
        k = np.zeros((17, 2))
        for n, p in {"sho": sho, "elb": elb, "wri": wrist, "hip": hip, "knee": knee, "ank": ank}.items():
            k[KP[f"l_{n}"]] = p
            k[KP[f"r_{n}"]] = p + np.array([0.0, 4.0])
        for n in ("nose", "l_eye", "r_eye", "l_ear", "r_ear"):
            k[KP[n]] = nose
        out.append(k)
    return _with_conf(out)


def jump(T: int, fps: float, jumps: int = 3, H: float = 700, x0: float = 640, ground: float = 1000,
         flight_s: float = 0.5) -> np.ndarray:
    """Countermovement jumps with ``flight_s`` of flight each (h = g t^2/8)."""
    out = []
    period = T / jumps
    for t in range(T):
        ph = (t % period) / period                    # 0..1 within a jump cycle
        ft = flight_s * fps / period                  # flight fraction of the cycle
        start = 0.45
        ank = np.array([x0, ground - ANKLE_H * H])
        leg = (SHANK + THIGH) * H
        d = 0.0
        lift = 0.0
        if 0.2 < ph < start:
            d = 0.6 * np.sin(np.pi * (ph - 0.2) / (start - 0.2))   # dip then extend
        elif start <= ph < start + ft:
            tt = (ph - start) * period / fps
            v0 = 9.81 * flight_s / 2
            m_per_px = 1.85 / H
            lift = (v0 * tt - 0.5 * 9.81 * tt ** 2) / m_per_px
        hip = ank + np.array([-0.12 * d * H, -leg * (1 - 0.5 * d)])
        lean = 8 + 30 * d
        lam = np.radians(lean)
        sho = hip + TRUNK * H * np.array([np.sin(lam), -np.cos(lam)])
        wrist = sho + np.array([-0.05 * H, (UARM + FARM) * H * 0.95])
        k = _body(ank, hip, lean, wrist, H)
        k[:, 1] -= lift
        out.append(k)
    return _with_conf(out)


def sprint(T: int, fps: float, H: float = 500, x_start: float = 100, speed_hps: float = 3.0,
           cadence_hz: float = 2.2, ground: float = 900) -> np.ndarray:
    """Running across the frame (static camera)."""
    out = []
    for t in range(T):
        tt = t / fps
        hx = x_start + speed_hps * H * tt
        hip = np.array([hx, ground - (SHANK + THIGH + ANKLE_H) * H * 0.95])
        k = np.zeros((17, 2))
        ph = 2 * np.pi * cadence_hz * tt
        for side, off in (("l", 0.0), ("r", np.pi)):
            th = np.radians(10 + 45 * np.sin(ph + off))            # thigh angle fwd of vertical
            knee = hip + THIGH * H * np.array([np.sin(th), np.cos(th)])
            flex = np.radians(25 + 85 * (0.5 + 0.5 * np.sin(ph + off - 1.2)))
            sh = th - flex
            ank = knee + SHANK * H * np.array([np.sin(sh), np.cos(sh)])
            k[KP[f"{side}_hip"]] = hip
            k[KP[f"{side}_knee"]] = knee
            k[KP[f"{side}_ank"]] = ank
            arm = np.radians(-40 * np.sin(ph + off))
            lam = np.radians(15)
            sho = hip + TRUNK * H * np.array([np.sin(lam), -np.cos(lam)])
            elb = sho + UARM * H * np.array([np.sin(arm), np.cos(arm)])
            wr = elb + FARM * H * np.array([np.sin(arm + 1.4), np.cos(arm + 1.4)])
            k[KP[f"{side}_sho"]] = sho
            k[KP[f"{side}_elb"]] = elb
            k[KP[f"{side}_wri"]] = wr
        n = k[KP["l_sho"]] + NECK * H * np.array([0.45, -1.0])
        for nm in ("nose", "l_eye", "r_eye", "l_ear", "r_ear"):
            k[KP[nm]] = n
        out.append(k)
    return _with_conf(out, 0.9, 0.85)


def overhead_press(T: int, fps: float, reps: int = 5, H: float = 700, x0: float = 640,
                   ground: float = 1000) -> np.ndarray:
    out = []
    for t in range(T):
        d = 0.5 * (1 - np.cos(2 * np.pi * reps * t / T))   # 0 rack -> 1 lockout
        ank = np.array([x0, ground - ANKLE_H * H])
        hip = ank + np.array([0.0, -(SHANK + THIGH) * H * 0.995])
        sho = hip + np.array([0.0, -TRUNK * H])
        wrist = sho + np.array([0.02 * H, -(0.02 + 0.30 * d) * H])
        out.append(_body(ank, hip, 0.0, wrist, H, elbow_bend=+1))
    return _with_conf(out)


def clean(T: int, fps: float, reps: int = 3, H: float = 700, x0: float = 640,
          ground: float = 1000) -> np.ndarray:
    """Power clean: bar from mid-shin to the shoulders, fast hip extension."""
    out = []
    period = T / reps
    for t in range(T):
        ph = (t % period) / period
        ank = np.array([x0, ground - ANKLE_H * H])
        leg = (SHANK + THIGH) * H
        if ph < 0.35:          # set position
            d, bar = 1.0, 0.0
        elif ph < 0.5:          # pull: extend fast
            u = (ph - 0.35) / 0.15
            d, bar = 1.0 - u, u
        elif ph < 0.8:          # catch in quarter squat, rack
            d, bar = 0.35 * np.sin(np.pi * (ph - 0.5) / 0.3), 1.0
        else:                   # drop bar and reset
            u = (ph - 0.8) / 0.2
            d, bar = u, 1.0 - u
        hip = ank + np.array([-0.14 * d * H, -leg * (1 - 0.25 * d)])
        lean = 5 + 55 * d if bar < 1 else 8
        lam = np.radians(lean)
        sho = hip + TRUNK * H * np.array([np.sin(lam), -np.cos(lam)])
        hang = sho + np.array([0.0, (UARM + FARM) * H * 0.98])
        rack = sho + np.array([0.08 * H, 0.0])
        wrist = hang + (rack - hang) * bar
        out.append(_body(ank, hip, lean, wrist, H, elbow_bend=+1))
    return _with_conf(out)


def lunge(T: int, fps: float, reps: int = 4, H: float = 700, x0: float = 640,
          ground: float = 1000) -> np.ndarray:
    out = []
    for t in range(T):
        d = 0.5 * (1 - np.cos(2 * np.pi * reps * t / T))
        front = np.array([x0 + 0.25 * H, ground - ANKLE_H * H])
        back = np.array([x0 - 0.25 * H, ground - ANKLE_H * H])
        hip = np.array([x0, ground - (SHANK + THIGH) * H * (0.97 - 0.33 * d)])
        kf = _ik(front, hip, SHANK * H, THIGH * H, +1)
        kb = _ik(back, hip, SHANK * H, THIGH * H, +1)
        sho = hip + np.array([0.0, -TRUNK * H])
        wr = sho + np.array([0.0, (UARM + FARM) * H * 0.97])
        elb = _ik(sho, wr, UARM * H, FARM * H, -1)
        k = np.zeros((17, 2))
        for s, (kn, an) in {"l": (kf, front), "r": (kb, back)}.items():
            k[KP[f"{s}_hip"]] = hip
            k[KP[f"{s}_knee"]] = kn
            k[KP[f"{s}_ank"]] = an
            k[KP[f"{s}_sho"]] = sho
            k[KP[f"{s}_elb"]] = elb
            k[KP[f"{s}_wri"]] = wr
        n = sho + np.array([0.04 * H, -NECK * H])
        for nm in ("nose", "l_eye", "r_eye", "l_ear", "r_ear"):
            k[KP[nm]] = n
        out.append(k)
    return _with_conf(out, 0.9, 0.85)


def agility(T: int, fps: float, H: float = 600, x0: float = 640, amp_h: float = 0.9,
            period_s: float = 1.6, ground: float = 950) -> np.ndarray:
    """Lateral shuffle (athletic stance) moving back and forth across the frame."""
    out = []
    for t in range(T):
        tt = t / fps
        x = x0 + amp_h * H * np.sin(2 * np.pi * tt / period_s)
        ank = np.array([x, ground - ANKLE_H * H])
        hip = ank + np.array([-0.06 * H, -(SHANK + THIGH) * H * 0.82])
        k = _body(ank, hip, 25, hip + np.array([0.25 * H, -0.05 * H]), H)
        k[KP["r_ank"], 0] += 0.2 * H * np.sin(2 * np.pi * tt * 2)
        out.append(k)
    return _with_conf(out)


def pushup(T: int, fps: float, reps: int = 5, H: float = 700, x0: float = 300,
           floor: float = 1000) -> np.ndarray:
    """Push-up in side view: hands on the floor under the shoulders, rigid body line."""
    out = []
    for t in range(T):
        d = 0.5 * (1 - np.cos(2 * np.pi * reps * t / T))
        reach = (UARM + FARM) * H
        ank = np.array([x0, floor - 0.03 * H])
        body_len = (SHANK + THIGH + TRUNK) * H
        sho_h = reach * (0.95 - 0.6 * d)
        ang = np.arcsin(np.clip(sho_h / body_len, 0, 1))
        sho = ank + body_len * np.array([np.cos(ang), -np.sin(ang)])
        hip = ank + (SHANK + THIGH) * H * np.array([np.cos(ang), -np.sin(ang)])
        knee = ank + SHANK * H * np.array([np.cos(ang), -np.sin(ang)])
        wr = np.array([sho[0], floor - 0.02 * H])
        elb = _ik(sho, wr, UARM * H, FARM * H, -1)
        k = np.zeros((17, 2))
        for n, p in {"sho": sho, "elb": elb, "wri": wr, "hip": hip, "knee": knee, "ank": ank}.items():
            k[KP[f"l_{n}"]] = p
            k[KP[f"r_{n}"]] = p + np.array([0.0, 3.0])
        nose = sho + np.array([NECK * H, 0.02 * H])
        for nm in ("nose", "l_eye", "r_eye", "l_ear", "r_ear"):
            k[KP[nm]] = nose
        out.append(k)
    return _with_conf(out)


GENERATORS = {"pushup": pushup, "squat": squat, "deadlift": deadlift, "bench": bench, "jump": jump,
              "sprint": sprint, "overhead_press": overhead_press, "olympic": clean,
              "lunge": lunge, "agility": agility}


def render_stick_video(out_path, kps: np.ndarray, width: int, height: int, fps: float,
                       audio: bool = True, rotate_meta: int = 0) -> str:
    """Draw the keypoint sequence as a solid 'mannequin' over a gym-like background and
    write an mp4 (H.264 + optional 440 Hz AAC tone). ``rotate_meta`` adds display-matrix
    rotation metadata (e.g. 90) to mimic phone footage; frames are then stored rotated."""
    import cv2
    from pathlib import Path
    from ..core.media import run_ffmpeg
    from .video import FrameWriter
    out_path = Path(out_path)
    tmp = out_path.with_name(out_path.stem + ".noaudio.mp4")
    bones = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
             (11, 13), (13, 15), (12, 14), (14, 16)]
    yy = np.linspace(0, 1, height)[:, None, None]
    bg = (np.array([40, 34, 30]) * (1 - yy) + np.array([70, 64, 58]) * yy).astype(np.uint8)
    bg = np.repeat(bg, width, axis=1)
    cv2.rectangle(bg, (0, int(height * 0.9)), (width, height), (35, 45, 55), -1)
    sw, sh = (height, width) if rotate_meta in (90, 270) else (width, height)
    wr = FrameWriter(tmp, sw, sh, fps, 23, "ultrafast")
    try:
        for f in range(len(kps)):
            img = bg.copy()
            k = kps[f]
            for a, b in bones:
                pa, pb = k[a, :2], k[b, :2]
                cv2.line(img, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), (180, 150, 120),
                         max(6, width // 90), cv2.LINE_AA)
            n = k[0, :2]
            cv2.circle(img, (int(n[0]), int(n[1])), max(10, width // 45), (170, 140, 115), -1, cv2.LINE_AA)
            w = (k[9, :2] + k[10, :2]) / 2
            cv2.circle(img, (int(w[0]), int(w[1])), max(12, width // 40), (40, 40, 200), -1, cv2.LINE_AA)
            if rotate_meta == 90:
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif rotate_meta == 270:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            wr.write(img)
    finally:
        wr.close()
    dur = len(kps) / fps
    args = []
    if rotate_meta:
        args += ["-display_rotation:v:0", str(-rotate_meta if rotate_meta == 90 else 90)]
    args += ["-i", str(tmp)]
    if audio:
        args += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={dur:.3f}:sample_rate=48000",
                 "-c:a", "aac", "-shortest"]
    args += ["-c:v", "copy", str(out_path)]
    run_ffmpeg(args)
    tmp.unlink(missing_ok=True)
    return str(out_path)
