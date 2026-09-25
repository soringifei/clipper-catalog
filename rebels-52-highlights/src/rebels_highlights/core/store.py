"""Per-game, per-stage JSON cache + resumable state + error log."""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Optional


class Store:
    def __init__(self, cfg: dict):
        self.root = Path(cfg["root"])
        p = cfg["paths"]
        self.cache = self.root / p["cache"]
        self.state = self.root / p["state"]
        self.raw = self.root / p["raw"]
        self.proxies = self.root / p["proxies"]
        self.outputs = self.root / p["outputs"]
        self.review = self.root / p["review"]
        self.reports = self.root / p["reports"]
        for d in (self.cache, self.state, self.raw, self.proxies, self.outputs,
                  self.review, self.reports):
            d.mkdir(parents=True, exist_ok=True)
        for sub in ("candidates", "approved", "single_plays", "mixes",
                    "thumbnails", "manifests"):
            (self.outputs / sub).mkdir(parents=True, exist_ok=True)

    # --- stage cache -----------------------------------------------------
    def stage_path(self, game_id: str, stage: str) -> Path:
        p = self.cache / game_id / f"{stage}.json"  # stage may be "tracking/<play_id>"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def has(self, game_id: str, stage: str) -> bool:
        return self.stage_path(game_id, stage).exists()

    def load(self, game_id: str, stage: str, default: Any = None) -> Any:
        p = self.stage_path(game_id, stage)
        return json.loads(p.read_text()) if p.exists() else default

    def save(self, game_id: str, stage: str, data: Any) -> Path:
        p = self.stage_path(game_id, stage)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1, default=_json_default))
        tmp.replace(p)
        self.mark(game_id, stage, "done")
        return p

    # --- resumable state -------------------------------------------------
    def _state_file(self, game_id: str) -> Path:
        return self.state / f"{game_id}.json"

    def mark(self, game_id: str, stage: str, status: str, **extra: Any) -> None:
        f = self._state_file(game_id)
        st = json.loads(f.read_text()) if f.exists() else {}
        st[stage] = {"status": status, "ts": time.time(), **extra}
        f.write_text(json.dumps(st, indent=1))

    def status(self, game_id: str) -> dict:
        f = self._state_file(game_id)
        return json.loads(f.read_text()) if f.exists() else {}

    # --- errors ------------------------------------------------------------
    def log_error(self, game_id: Optional[str], stage: str, err: BaseException | str,
                  **extra: Any) -> None:
        rec = {"ts": time.time(), "game_id": game_id, "stage": stage,
               "error": str(err), **extra}
        if isinstance(err, BaseException):
            rec["traceback"] = "".join(traceback.format_exception(err))[-4000:]
        with (self.reports / "errors.jsonl").open("a") as fh:
            fh.write(json.dumps(rec, default=_json_default) + "\n")
        if game_id:
            self.mark(game_id, stage, "failed", error=str(err)[:500])

    def run_stage(self, game_id: str, stage: str, fn: Callable[[], Any],
                  resume: bool = True) -> Any:
        """Run ``fn`` unless cached; never raises - failures go to errors.jsonl."""
        if resume and self.has(game_id, stage):
            return self.load(game_id, stage)
        try:
            data = fn()
            self.save(game_id, stage, data)
            return data
        except Exception as e:  # noqa: BLE001 - one stage must not kill the run
            self.log_error(game_id, stage, e)
            return None


def _json_default(o: Any) -> Any:
    if hasattr(o, "to_dict"):
        return o.to_dict()
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")
