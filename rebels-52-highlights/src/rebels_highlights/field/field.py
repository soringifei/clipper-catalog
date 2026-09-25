"""Field geometry: yard lines, approximate homography and line of scrimmage.

Coordinates: the "field axis" runs along the length of the field
(perpendicular to the yard lines). ``project_along_field`` maps an image
point onto that axis - through the homography when one could be built
(units ~ yards, assuming detected lines are 5 yd apart), else by a dot
product with the image-space axis direction (units = pixels). Only compare
projections produced with the same ``field`` dict.
"""
from __future__ import annotations

import math
from typing import Any, Optional

import cv2
import numpy as np

DEFAULTS: dict[str, Any] = {
    "green_hsv_lo": [32, 40, 40],
    "green_hsv_hi": [90, 255, 255],
    "white_v_min": 170,
    "white_s_max": 70,
    "min_line_frac": 0.12,        # min segment length / min(h, w)
    "angle_tol_deg": 10.0,        # max deviation from dominant yard-line angle
    "merge_dist_frac": 0.02,      # merge collinear segments within this * width
    "yard_gap_yd": 5.0,           # assumed spacing between adjacent detected lines
    "min_players": 6,
    "los_min_return": 0.30,       # below this los_x is None
}


def field_cfg(cfg: dict) -> dict:
    """``cfg['field']`` merged over module defaults."""
    return {**DEFAULTS, **((cfg or {}).get("field") or {})}


def _empty(reason: str) -> dict:
    return {"yard_lines": [], "los_x": None, "los_confidence": 0.0, "homography": None,
            "field_axis": [1.0, 0.0], "line_angle_deg": None, "los_point": None,
            "los_along": None, "los_line": None, "notes": [reason]}


# ----------------------------------------------------------------- yard lines
def _grass_mask(frame: np.ndarray, fc: dict) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    g = cv2.inRange(hsv, np.array(fc["green_hsv_lo"], np.uint8),
                    np.array(fc["green_hsv_hi"], np.uint8))
    k = max(5, int(min(frame.shape[:2]) * 0.03) | 1)
    # close over white lines and players so they count as "on the field"
    return cv2.morphologyEx(g, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))


def _white_mask(frame: np.ndarray, field_mask: np.ndarray, fc: dict) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    w = cv2.inRange(hsv, np.array([0, 0, fc["white_v_min"]], np.uint8),
                    np.array([180, fc["white_s_max"], 255], np.uint8))
    return cv2.bitwise_and(w, field_mask)


def _seg_angle(s: np.ndarray) -> float:
    """Undirected angle in degrees [0, 180)."""
    return math.degrees(math.atan2(s[3] - s[1], s[2] - s[0])) % 180.0


def _ang_diff(a: float, b: float) -> float:
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def detect_yard_lines(frame: np.ndarray, fc: dict) -> tuple[list[dict], Optional[float]]:
    """Long, near-parallel white lines on grass. Returns (lines, dominant angle)."""
    h, w = frame.shape[:2]
    fm = _grass_mask(frame, fc)
    if np.count_nonzero(fm) < 0.15 * fm.size:
        return [], None
    wm = _white_mask(frame, fm, fc)
    min_len = fc["min_line_frac"] * min(h, w)
    segs = cv2.HoughLinesP(wm, 1, np.pi / 180, threshold=max(20, int(min_len * 0.5)),
                           minLineLength=min_len, maxLineGap=max(5, int(min_len * 0.25)))
    if segs is None:
        return [], None
    segs = np.asarray(segs, dtype=float).reshape(-1, 4)
    lens = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    angs = np.array([_seg_angle(s) for s in segs])
    # dominant angle: length-weighted histogram (sidelines are long too, but
    # yard lines are many; weight count + length)
    hist = np.zeros(180)
    for a, L in zip(angs, lens):
        for da in range(-3, 4):
            hist[int(round(a + da)) % 180] += L * (1.0 - abs(da) / 4.0)
    dom = float(np.argmax(hist))
    keep = [i for i in range(len(segs)) if _ang_diff(angs[i], dom) <= fc["angle_tol_deg"]]
    if not keep:
        return [], None
    # merge collinear segments by their offset along the normal
    th = math.radians(dom)
    nx, ny = -math.sin(th), math.cos(th)
    rhos = [(segs[i][0] + segs[i][2]) / 2 * nx + (segs[i][1] + segs[i][3]) / 2 * ny for i in keep]
    order = np.argsort(rhos)
    tol = fc["merge_dist_frac"] * w
    groups: list[list[int]] = []
    for oi in order:
        if groups and rhos[oi] - rhos[groups[-1][-1]] <= tol:
            groups[-1].append(oi)
        else:
            groups.append([oi])
    dx, dy = math.cos(th), math.sin(th)
    lines = []
    for g in groups:
        idx = [keep[k] for k in g]
        pts = np.concatenate([segs[idx, :2], segs[idx, 2:]])
        cx, cy = pts.mean(axis=0)
        proj = (pts[:, 0] - cx) * dx + (pts[:, 1] - cy) * dy
        a, b = proj.min(), proj.max()
        total = float(lens[idx].sum())
        lines.append({"x1": float(cx + a * dx), "y1": float(cy + a * dy),
                      "x2": float(cx + b * dx), "y2": float(cy + b * dy),
                      "rho": float(np.mean([rhos[k] for k in g])),
                      "length": float(b - a), "support": total})
    long_min = 1.5 * min_len
    lines = [ln for ln in lines if ln["length"] >= long_min] or \
        sorted(lines, key=lambda ln: -ln["length"])[:1]
    lines.sort(key=lambda ln: ln["rho"])
    return lines, dom


def _homography(lines: list[dict], dom: float, fc: dict) -> Optional[np.ndarray]:
    """Map image -> (u yards along field, v in [0,1] across) from >= 2 lines."""
    if len(lines) < 2:
        return None
    th = math.radians(dom)
    # use the parametrisation along the dominant axis of the lines
    vertical = abs(math.sin(th)) >= abs(math.cos(th))
    a, b = lines[0], lines[-1]

    def span(ln: dict) -> tuple[float, float]:
        vals = (ln["y1"], ln["y2"]) if vertical else (ln["x1"], ln["x2"])
        return min(vals), max(vals)

    lo = max(span(a)[0], span(b)[0])
    hi = min(span(a)[1], span(b)[1])
    if hi - lo < 10:
        lo, hi = min(span(a)[0], span(b)[0]), max(span(a)[1], span(b)[1])
    if hi - lo < 10:
        return None

    def at(ln: dict, s: float) -> list[float]:
        if vertical:
            dyl = (ln["y2"] - ln["y1"]) or 1e-6
            return [ln["x1"] + (s - ln["y1"]) * (ln["x2"] - ln["x1"]) / dyl, s]
        dxl = (ln["x2"] - ln["x1"]) or 1e-6
        return [s, ln["y1"] + (s - ln["x1"]) * (ln["y2"] - ln["y1"]) / dxl]

    rh = [ln["rho"] for ln in lines]
    gaps = np.diff(rh)
    med = float(np.median(gaps)) if len(gaps) else 0.0
    k = max(1, int(round((rh[-1] - rh[0]) / med))) if med > 0 else 1
    u = k * float(fc["yard_gap_yd"])
    src = np.float32([at(a, lo), at(b, lo), at(a, hi), at(b, hi)])
    dst = np.float32([[0, 0], [u, 0], [0, 1], [u, 1]])
    try:
        H = cv2.getPerspectiveTransform(src, dst)
    except cv2.error:
        return None
    return H if np.all(np.isfinite(H)) else None


# -------------------------------------------------------------- projections
def _to_field(x: float, y: float, field: dict) -> tuple[float, float]:
    H = field.get("homography")
    if H is not None:
        H = np.asarray(H, dtype=float)
        p = H @ np.array([x, y, 1.0])
        if abs(p[2]) > 1e-9:
            return float(p[0] / p[2]), float(p[1] / p[2])
    ax, ay = (field.get("field_axis") or [1.0, 0.0])
    return float(x * ax + y * ay), float(-x * ay + y * ax)


def _from_field(u: float, v: float, field: dict) -> tuple[float, float]:
    H = field.get("homography")
    if H is not None:
        try:
            Hi = np.linalg.inv(np.asarray(H, dtype=float))
            p = Hi @ np.array([u, v, 1.0])
            if abs(p[2]) > 1e-9:
                return float(p[0] / p[2]), float(p[1] / p[2])
        except np.linalg.LinAlgError:
            pass
    ax, ay = (field.get("field_axis") or [1.0, 0.0])
    return float(u * ax - v * ay), float(u * ay + v * ax)


def project_along_field(x: float, y: float, field: dict) -> float:
    """Position of image point (x, y) along the field axis (see module doc)."""
    return _to_field(x, y, field)[0]


# ---------------------------------------------------------------------- LOS
def _estimate_los(boxes: list[list[float]], field: dict, fc: dict,
                  line_factor: float) -> dict:
    feet = [((b[0] + b[2]) / 2.0, float(b[3])) for b in boxes]
    uv = np.array([_to_field(x, y, field) for x, y in feet])
    # player width in field units -> bandwidth for clustering
    widths = [abs(_to_field(b[2], b[3], field)[0] - _to_field(b[0], b[3], field)[0])
              for b in boxes]
    bw = max(float(np.median(widths)), 1e-3)
    u = uv[:, 0]
    grid = np.linspace(u.min() - bw, u.max() + bw, 256)
    dens = np.exp(-0.5 * ((grid[:, None] - u[None, :]) / bw) ** 2).sum(axis=1)
    peak = float(grid[int(np.argmax(dens))])
    in_cl = np.abs(u - peak) <= 2.5 * bw
    cu = np.sort(u[in_cl])
    n_cl = len(cu)
    density = n_cl / len(u)
    los_u, sep = peak, 0.0
    if n_cl >= 4:
        gaps = np.diff(cu)
        cand = [(gaps[i], i) for i in range(1, len(gaps) - 1)]  # >= 2 players each side
        if cand:
            g, i = max(cand)
            others = np.delete(gaps, i)
            ref = float(np.median(others)) if len(others) else bw
            sep = float(g / max(ref, 0.25 * bw))
            los_u = float((cu[i] + cu[i + 1]) / 2.0)
    # linemen form a line across the field: elongated cluster supports the LOS
    # (measured in image space along the axis / along the yard-line direction)
    ax, ay = field.get("field_axis") or [1.0, 0.0]
    fx = np.array([f for f, keep in zip(feet, in_cl) if keep], dtype=float).reshape(-1, 2)
    along_spread = float(np.std(fx @ np.array([ax, ay]))) + 1e-6 if n_cl else 1.0
    across_spread = float(np.std(fx @ np.array([-ay, ax]))) if n_cl >= 2 else 0.0
    elong = across_spread / along_spread
    sep_s = min(1.0, max(0.0, (sep - 1.2) / 2.8))        # 1.2 -> 0, 4 -> 1
    elong_s = min(1.0, max(0.0, (elong - 0.8) / 2.2))   # 0.8 -> 0, 3 -> 1
    dens_s = min(1.0, max(0.0, (density - 0.2) / 0.4))  # 20% -> 0, 60% -> 1
    conf = (0.45 * sep_s + 0.25 * elong_s + 0.30 * dens_s) * line_factor
    v_c = float(np.median(uv[in_cl, 1])) if n_cl else float(np.median(uv[:, 1]))
    px, py = _from_field(los_u, v_c, field)
    th = math.radians(field.get("line_angle_deg") if field.get("line_angle_deg") is not None
                      else 90.0)
    half = 2.0 * bw if field.get("homography") is None else 100.0
    return {"los_u": los_u, "point": [px, py], "confidence": float(conf),
            "line": [[px - half * math.cos(th), py - half * math.sin(th)],
                     [px + half * math.cos(th), py + half * math.sin(th)]],
            "features": {"separation": sep, "elongation": elong, "density": density,
                         "cluster_size": n_cl}}


def estimate_field(frame_bgr: Optional[np.ndarray], tracks_at_snap: Optional[list[list[float]]],
                   cfg: dict) -> dict:
    """Yard lines, optional homography and line of scrimmage at the snap.

    Returns ``{"yard_lines": [[x1,y1,x2,y2]...], "los_x": float|None,
    "los_confidence": float, "homography": 3x3 list|None, ...}`` plus
    ``field_axis``, ``line_angle_deg``, ``los_point``, ``los_along`` (value in
    ``project_along_field`` space), ``los_line`` and ``notes``.
    """
    fc = field_cfg(cfg)
    if frame_bgr is None or getattr(frame_bgr, "size", 0) == 0:
        return _empty("no_frame")
    lines, dom = detect_yard_lines(frame_bgr, fc)
    out = _empty("ok")
    out["notes"] = []
    out["yard_lines"] = [[round(ln["x1"], 1), round(ln["y1"], 1), round(ln["x2"], 1),
                          round(ln["y2"], 1)] for ln in lines]
    if dom is not None:
        th = math.radians(dom)
        # field axis = normal to yard lines, oriented to point image-right
        ax, ay = -math.sin(th), math.cos(th)
        if ax < 0 or (abs(ax) < 1e-6 and ay < 0):
            ax, ay = -ax, -ay
        out["field_axis"] = [round(ax, 5), round(ay, 5)]
        out["line_angle_deg"] = dom
    H = _homography(lines, dom, fc) if dom is not None else None
    if H is not None:
        # orient so that u grows to the image right like the axis mode
        h, w = frame_bgr.shape[:2]
        tmp = {"homography": H.tolist()}
        if project_along_field(w * 0.75, h / 2, tmp) < project_along_field(w * 0.25, h / 2, tmp):
            H = np.diag([-1.0, 1.0, 1.0]) @ H
        out["homography"] = [[float(v) for v in r] for r in H]
    line_factor = 1.0 if len(lines) >= 2 else (0.75 if lines else 0.5)
    if not lines:
        out["notes"].append("no_yard_lines")
    boxes = [list(map(float, b[:4])) for b in (tracks_at_snap or []) if len(b) >= 4]
    if len(boxes) < fc["min_players"]:
        out["notes"].append("too_few_players")
        return out
    los = _estimate_los(boxes, out, fc, line_factor)
    out["los_confidence"] = round(los["confidence"], 3)
    out["los_features"] = los["features"]
    if los["confidence"] >= fc["los_min_return"]:
        out["los_x"] = round(los["point"][0], 1)
        out["los_point"] = [round(v, 1) for v in los["point"]]
        out["los_along"] = round(los["los_u"], 4)
        out["los_line"] = [[round(v, 1) for v in p] for p in los["line"]]
    else:
        out["notes"].append("los_uncertain")
    return out
