"""Audio filter graphs: loudness normalisation + music ducking.

Graph convention (used by rendering/render.py and rendering/mix.py): the
caller provides a labelled game-audio pad ``[game]`` (and ``[music]`` when
``has_music``); the graph produces ``[aout]`` as 48 kHz stereo float audio
ready for the AAC encoder. When the source has no audio the graph generates
its own silent ``[game]`` pad with ``anullsrc`` so every clip carries AAC.
"""
from __future__ import annotations

import math
from typing import Optional

SAMPLE_RATE = 48000
_FMT = f"aresample={SAMPLE_RATE},aformat=sample_fmts=fltp:sample_rates={SAMPLE_RATE}:channel_layouts=stereo"


def silent_source(duration_s: float, label: str = "game") -> str:
    """Filter chain producing ``duration_s`` of 48k stereo silence into ``[label]``."""
    return (f"anullsrc=r={SAMPLE_RATE}:cl=stereo,atrim=duration={max(duration_s, 0.05):.4f},"
            f"asetpts=N/SR/TB[{label}]")


def audio_filter_graph(has_music: bool, cfg: dict, has_game_audio: bool = True,
                       duration_s: Optional[float] = None) -> str:
    """Return a filter_complex fragment ``[game](+[music]) -> [aout]``.

    * game audio: EBU R128 ``loudnorm`` with ``render.loudnorm`` params.
    * music (only licensed local files): lowered, ducked by a sidechain
      compressor keyed on the (normalised) game audio, then ``amix``-ed under
      the game audio with a limiter so peaks stay below the TP ceiling.
    * ``has_game_audio=False``: a silent ``[game]`` is generated first (needs
      ``duration_s``); loudnorm is skipped for pure silence.
    """
    rc = cfg.get("render", {}) if cfg else {}
    mc = cfg.get("audio", {}) if cfg else {}
    ln = rc.get("loudnorm") or "I=-14:TP=-1.5:LRA=11"
    music_vol = float(mc.get("music_volume", 0.30))
    parts: list[str] = []
    if not has_game_audio:
        parts.append(silent_source(duration_s or 1.0, "game"))
        game_chain = f"[game]{_FMT}"
    else:
        game_chain = f"[game]loudnorm={ln},{_FMT}"
    if not has_music:
        parts.append(f"{game_chain}[aout]")
        return ";".join(parts)
    parts.append(f"{game_chain},asplit=2[gkey][gmain]")
    parts.append(f"[music]{_FMT},volume={music_vol:.3f}[mvol]")
    parts.append("[mvol][gkey]sidechaincompress=threshold=0.03:ratio=8:attack=15:release=350:"
                 "makeup=1[mduck]")
    parts.append("[gmain][mduck]amix=inputs=2:duration=first:dropout_transition=0:normalize=0,"
                 f"alimiter=limit=0.85:level=0,{_FMT}[aout]")
    return ";".join(parts)


# =============================================================================
# numpy audio toolkit (shared by football + gym renderers)
# =============================================================================
import subprocess
import wave
from functools import lru_cache
from pathlib import Path

import numpy as np

SFX_VERSION = 2
SFX_NAMES = ("boom", "whoosh", "whoosh_short", "subdrop", "riser", "rewind")


def decode_audio(path: str, start: float = 0.0, duration: Optional[float] = None,
                 sr: int = SAMPLE_RATE, channels: int = 2) -> np.ndarray:
    """Decode (a window of) any media file to float32 ``(n, channels)`` via ffmpeg."""
    from ..core.media import ffmpeg_bin
    cmd = [ffmpeg_bin(), "-hide_banner", "-loglevel", "error"]
    if start > 0:
        cmd += ["-ss", f"{start:.4f}"]
    if duration is not None:
        cmd += ["-t", f"{max(duration, 0.01):.4f}"]
    cmd += ["-i", str(path), "-vn", "-f", "f32le", "-acodec", "pcm_f32le", "-ac", str(channels),
            "-ar", str(sr), "pipe:1"]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        return np.zeros((0, channels), np.float32)
    x = np.frombuffer(r.stdout, np.float32)
    return x[: len(x) // channels * channels].reshape(-1, channels).copy()


def write_wav(path: str | Path, x: np.ndarray, sr: int = SAMPLE_RATE) -> str:
    x = np.asarray(x, np.float32)
    if x.ndim == 1:
        x = np.stack([x, x], 1)
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(pcm.shape[1])
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return str(path)


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        n, ch, sr = w.getnframes(), w.getnchannels(), w.getframerate()
        pcm = np.frombuffer(w.readframes(n), "<i2").reshape(-1, ch)
    return pcm.astype(np.float32) / 32767.0, sr


def lowpass_fft(x: np.ndarray, sr: int, fc: float, order: float = 4.0) -> np.ndarray:
    """Zero-phase Butterworth-shaped low-pass in the frequency domain."""
    if len(x) < 8:
        return x.copy()
    n = len(x)
    X = np.fft.rfft(x, axis=0)
    f = np.fft.rfftfreq(n, 1.0 / sr)
    h = 1.0 / np.sqrt(1.0 + (f / max(fc, 1.0)) ** (2 * order))
    return np.fft.irfft(X * (h[:, None] if X.ndim == 2 else h), n=n, axis=0).astype(np.float32)


def highpass_fft(x: np.ndarray, sr: int, fc: float, order: float = 2.0) -> np.ndarray:
    n = len(x)
    X = np.fft.rfft(x, axis=0)
    f = np.fft.rfftfreq(n, 1.0 / sr)
    h = 1.0 / np.sqrt(1.0 + (max(fc, 1.0) / np.maximum(f, 1e-3)) ** (2 * order))
    return np.fft.irfft(X * (h[:, None] if X.ndim == 2 else h), n=n, axis=0).astype(np.float32)


def _env(n: int, sr: int, attack: float, decay: float) -> np.ndarray:
    t = np.arange(n) / sr
    return (np.clip(t / max(attack, 1e-4), 0, 1) * np.exp(-np.maximum(t - attack, 0) / decay)).astype(np.float32)


def _sweep_noise(n: int, sr: int, f0: float, f1: float, bw: float, rng, nfft: int = 2048) -> np.ndarray:
    """White noise through a band-pass whose centre glides f0 -> f1 (log), STFT/OLA."""
    hop = nfft // 4
    noise = rng.normal(0, 1, n + nfft).astype(np.float32)
    out = np.zeros(n + nfft, np.float32)
    win = np.hanning(nfft).astype(np.float32)
    freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
    lf = np.log(np.maximum(freqs, 1.0))
    nb = max(1, (n + hop - 1) // hop)
    for i in range(nb):
        p = i / max(1, nb - 1)
        fc = math.exp(math.log(f0) + (math.log(f1) - math.log(f0)) * p)
        m = np.exp(-0.5 * ((lf - math.log(fc)) / bw) ** 2)
        s = i * hop
        seg = np.fft.irfft(np.fft.rfft(noise[s:s + nfft] * win) * m, n=nfft)
        out[s:s + nfft] += seg * win
    out = out[:n]
    return out / (np.abs(out).max() + 1e-9)


def synth_sfx(name: str, sr: int = SAMPLE_RATE, seed: int = 52) -> np.ndarray:
    """Generate an SFX (float32 stereo, peak ~0.9). No samples, no licences."""
    rng = np.random.default_rng(seed + SFX_NAMES.index(name) if name in SFX_NAMES else seed)
    if name == "boom":
        n = int(1.4 * sr)
        t = np.arange(n) / sr
        f = 35 + 45 * np.exp(-t / 0.11)                  # 80 -> 35 Hz drop
        ph = 2 * np.pi * np.cumsum(f) / sr
        body = np.sin(ph) * _env(n, sr, 0.003, 0.42)
        sub2 = np.sin(ph * 0.5) * _env(n, sr, 0.01, 0.6) * 0.35
        nz = lowpass_fft(rng.normal(0, 1, n).astype(np.float32), sr, 2600) * _env(n, sr, 0.0005, 0.03)
        nz = nz / (np.abs(nz).max() + 1e-9)
        click = lowpass_fft(rng.normal(0, 1, n).astype(np.float32), sr, 7000) * _env(n, sr, 0.0002, 0.006)
        click = click / (np.abs(click).max() + 1e-9)
        x = body + sub2 + 0.55 * nz + 0.25 * click
        x = np.tanh(2.4 * x) / np.tanh(2.4)                # slight distortion / weight
        st = np.stack([x, x], 1)
    elif name in ("whoosh", "whoosh_short"):
        dur = 0.75 if name == "whoosh" else 0.32
        n = int(dur * sr)
        t = np.arange(n) / sr / dur
        x = _sweep_noise(n, sr, 350, 4200, 0.55, rng)
        env = (t ** 2.2) * np.clip((1.0 - t) / 0.12, 0, 1)   # swell, quick release at the end
        env = env / (env.max() + 1e-9)
        x = x * env
        pan = 0.5 + 0.35 * (t - 0.5)
        st = np.stack([x * np.sqrt(1 - pan), x * np.sqrt(pan)], 1) * 1.3
    elif name == "rewind":
        n = int(0.5 * sr)
        t = np.arange(n) / sr / 0.5
        x = _sweep_noise(n, sr, 3800, 500, 0.5, rng)
        env = np.sin(np.pi * np.clip(t, 0, 1)) ** 1.5
        wob = np.sin(2 * np.pi * np.cumsum(180 + 900 * (1 - t)) / sr) * 0.25
        x = (x + wob) * env
        st = np.stack([x, x * 0.9], 1)
    elif name == "subdrop":
        n = int(1.6 * sr)
        t = np.arange(n) / sr
        f = 28 + 34 * np.exp(-t / 0.35)                    # 62 -> 28 Hz glide
        ph = 2 * np.pi * np.cumsum(f) / sr
        x = np.sin(ph) * _env(n, sr, 0.006, 0.7)
        x = np.tanh(1.8 * x) / np.tanh(1.8)
        st = np.stack([x, x], 1)
    elif name == "riser":
        dur = 0.9
        n = int(dur * sr)
        t = np.arange(n) / sr / dur
        x = _sweep_noise(n, sr, 500, 7000, 0.45, rng)
        tone = np.sin(2 * np.pi * np.cumsum(180 * 4.0 ** t) / sr) * 0.35
        env = (np.exp(3.2 * t) - 1) / (math.exp(3.2) - 1)
        x = (x + tone) * env
        st = np.stack([x, x], 1)
    else:
        raise ValueError(f"unknown sfx {name}")
    st = st.astype(np.float32)
    fade = min(len(st), int(0.004 * sr))
    st[-fade:] *= np.linspace(1, 0, fade)[:, None]
    return st / (np.abs(st).max() + 1e-9) * 0.9


def sfx_dir(cfg: Optional[dict]) -> Path:
    from ..rendering.style import PROJECT_ROOT
    root = Path((cfg or {}).get("root") or PROJECT_ROOT)
    cache = Path(((cfg or {}).get("paths") or {}).get("cache", "cache"))
    return (cache if cache.is_absolute() else root / cache) / "sfx"


def sfx(name: str, cfg: Optional[dict] = None, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Cached SFX: ``<cache>/sfx/<name>_v<N>.wav`` (generated on first use)."""
    return _sfx_cached(name, str(sfx_dir(cfg)), sr).copy()


@lru_cache(maxsize=32)
def _sfx_cached(name: str, d: str, sr: int) -> np.ndarray:
    p = Path(d) / f"{name}_v{SFX_VERSION}_{sr}.wav"
    if p.is_file():
        try:
            x, fsr = read_wav(p)
            if fsr == sr and len(x):
                return x
        except (OSError, wave.Error, ValueError):
            pass
    x = synth_sfx(name, sr)
    try:
        write_wav(p, x, sr)
    except OSError:
        pass
    return x


# ------------------------------------------------------------------- mixing
def varispeed(src: np.ndarray, src_t0: float, times: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Sample ``src`` (starting at absolute time ``src_t0``) at absolute source
    ``times`` (NaN -> silence). Slower-than-real-time maps pitch down like tape."""
    out = np.zeros((len(times), src.shape[1] if src.ndim == 2 else 1), np.float32)
    if len(src) < 2:
        return out
    idx = (np.asarray(times, np.float64) - src_t0) * sr
    ok = np.isfinite(idx)
    ok[ok] = (idx[ok] >= 0) & (idx[ok] <= len(src) - 1)
    idx_ok = idx[ok]
    base = np.arange(len(src))
    for c in range(out.shape[1]):
        out[ok, c] = np.interp(idx_ok, base, src[:, c] if src.ndim == 2 else src)
    return out


def moving_average(x: np.ndarray, n: int) -> np.ndarray:
    n = max(1, int(n))
    c = np.cumsum(np.concatenate([[0.0], x]))
    y = (c[n:] - c[:-n]) / n
    return np.concatenate([np.full(n - 1, y[0] if len(y) else 0.0), y])


def duck_gain(sfx_track: np.ndarray, sr: int = SAMPLE_RATE, depth: float = 0.45) -> np.ndarray:
    """Gain curve (per sample) that ducks game audio under loud SFX."""
    a = np.abs(sfx_track).max(axis=1) if sfx_track.ndim == 2 else np.abs(sfx_track)
    env = moving_average(a, int(0.02 * sr))
    g = 1 - depth * np.clip(env * 3.0, 0, 1)
    return moving_average(g, int(0.12 * sr)).astype(np.float32)


def place(track: np.ndarray, clip: np.ndarray, t: float, gain: float = 1.0, sr: int = SAMPLE_RATE,
          align_end: bool = False) -> None:
    """Add ``clip`` into ``track`` at time ``t`` (or ending at ``t``)."""
    i0 = int(round(t * sr)) - (len(clip) if align_end else 0)
    j0 = max(0, -i0)
    i0 = max(0, i0)
    n = min(len(track) - i0, len(clip) - j0)
    if n > 0:
        track[i0:i0 + n] += clip[j0:j0 + n] * gain


# ------------------------------------------------------------------- beats
def onset_envelope(x: np.ndarray, sr: int, hop: int = 256, nfft: int = 1024) -> tuple[np.ndarray, float]:
    """Spectral-flux onset strength (log-magnitude, half-wave rectified)."""
    x = np.asarray(x, np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1)
    if len(x) < nfft:
        return np.zeros(1, np.float32), sr / hop
    n = 1 + (len(x) - nfft) // hop
    idx = np.arange(nfft)[None, :] + hop * np.arange(n)[:, None]
    frames = x[idx] * np.hanning(nfft).astype(np.float32)
    mag = np.log1p(100 * np.abs(np.fft.rfft(frames, axis=1)))
    flux = np.maximum(0, np.diff(mag, axis=0)).sum(axis=1)
    flux = np.concatenate([[0.0], flux])
    flux = flux - moving_average(flux, int(sr / hop * 0.5))
    flux = np.maximum(flux, 0)
    return (flux / (flux.max() + 1e-9)).astype(np.float32), sr / hop


def estimate_tempo(env: np.ndarray, env_sr: float, lo_bpm: float = 70, hi_bpm: float = 180) -> float:
    """Tempo (BPM) from the onset autocorrelation with a mild 120 BPM prior."""
    e = env - env.mean()
    if len(e) < 8 or not np.any(e):
        return 120.0
    ac = np.correlate(e, e, "full")[len(e) - 1:]
    lags = np.arange(len(ac))
    lo, hi = int(env_sr * 60 / hi_bpm), int(env_sr * 60 / lo_bpm) + 1
    hi = min(hi, len(ac) - 1)
    if hi <= lo:
        return 120.0
    L = lags[lo:hi]
    bpm = 60 * env_sr / np.maximum(L, 1)
    prior = np.exp(-0.5 * (np.log2(bpm / 120.0) / 0.9) ** 2)
    score = ac[lo:hi] * prior
    k = int(np.argmax(score))
    # parabolic refinement
    if 0 < k < len(score) - 1:
        a, b, c = score[k - 1], score[k], score[k + 1]
        den = a - 2 * b + c
        off = 0.5 * (a - c) / den if den != 0 else 0.0
    else:
        off = 0.0
    return float(60 * env_sr / (L[k] + off))


def detect_beats(src, sr: int = 22050, max_s: float = 300.0) -> tuple[list[float], float]:
    """Beat times (s) + tempo for a file path or mono/stereo array (at ``sr``)."""
    x = decode_audio(src, 0.0, max_s, sr, 1)[:, 0] if isinstance(src, (str, Path)) else np.asarray(src)
    env, esr = onset_envelope(x, sr)
    if len(env) < 16 or env.max() <= 0:
        return [], 120.0
    bpm = estimate_tempo(env, esr)
    P = 60.0 * esr / bpm
    best, best_off = -1.0, 0.0
    for off in np.arange(0, P, 0.5):
        pos = np.arange(off, len(env) - 1, P)
        s = float(env[np.round(pos).astype(int)].sum())
        if s > best:
            best, best_off = s, off
    beats = []
    w = max(1, int(P * 0.1))
    for p in np.arange(best_off, len(env) - 1, P):
        i = int(round(p))
        a, b = max(0, i - w), min(len(env), i + w + 1)
        j = a + int(np.argmax(env[a:b])) if env[a:b].max() > 0.05 else i
        beats.append(j / esr + 512.0 / sr)   # frame centre (nfft/2)
    return beats, bpm


def tile_beats(beats: list[float], period_s: float, total_s: float) -> list[float]:
    """Repeat beat times of a looped track (``-stream_loop``) over ``total_s``."""
    if not beats or period_s <= 0:
        return list(beats)
    out, k = [], 0
    while k * period_s < total_s + 1:
        out += [b + k * period_s for b in beats if b < period_s]
        k += 1
    return [b for b in out if b <= total_s + 1]


def snap_to_beat(t: float, beats: list[float], tol: float = 0.12) -> float:
    """Nearest beat within ``tol`` seconds, else ``t`` unchanged."""
    if not beats:
        return t
    b = np.asarray(beats, float)
    i = int(np.argmin(np.abs(b - t)))
    return float(b[i]) if abs(b[i] - t) <= tol else t
