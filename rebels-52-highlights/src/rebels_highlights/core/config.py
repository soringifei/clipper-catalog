"""Config loading: default.yaml + player.yaml + games.yaml merged into one dict."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Optional

import yaml

from .models import Game

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(games_path: Optional[str] = None, overrides: Optional[dict] = None,
                root: Optional[Path] = None) -> dict[str, Any]:
    root = Path(root or PROJECT_ROOT)
    cfg = yaml.safe_load((root / "config/default.yaml").read_text())
    cfg["player"] = yaml.safe_load((root / "config/player.yaml").read_text())
    gp = Path(games_path) if games_path else root / "config/games.yaml"
    if not gp.is_absolute():
        gp = root / gp
    cfg["games"] = (yaml.safe_load(gp.read_text()) or {}).get("games", [])
    cfg["root"] = str(root)
    return _deep_merge(cfg, overrides or {})


def games_from_config(cfg: dict, only: Optional[list[str]] = None) -> list[Game]:
    games = [Game.from_dict(g) for g in cfg.get("games", [])]
    return [g for g in games if not only or g.game_id in only]


def game_hint(cfg: dict, game_id: str) -> Optional[float]:
    for g in cfg.get("games", []):
        if g.get("game_id") == game_id:
            return g.get("hint_start_s")
    return None
