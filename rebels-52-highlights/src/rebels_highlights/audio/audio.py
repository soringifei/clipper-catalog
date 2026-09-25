"""Audio filter graphs: loudness normalisation + music ducking.

Graph convention (used by rendering/render.py and rendering/mix.py): the
caller provides a labelled game-audio pad ``[game]`` (and ``[music]`` when
``has_music``); the graph produces ``[aout]`` as 48 kHz stereo float audio
ready for the AAC encoder. When the source has no audio the graph generates
its own silent ``[game]`` pad with ``anullsrc`` so every clip carries AAC.
"""
from __future__ import annotations

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
