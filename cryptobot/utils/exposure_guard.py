# exposure_guard.py
from __future__ import annotations
import os, json
from pathlib import Path
from datetime import datetime, timedelta, timezone

STATE_PATH = Path(os.getenv("EXPOSURE_STATE", "data/exposure_state.json"))
STATE_PATH.parent.mkdir(parents=True, exist_ok=True)

MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT_POS", "5"))
TTL_HOURS = float(os.getenv("EXPOSURE_TTL_HOURS", "24"))

def _now():
    return datetime.now(timezone.utc)

class ExposureGuard:
    def __init__(self, max_positions: int = MAX_CONCURRENT):
        self.max_positions = max_positions
        self.state = self._load()

    def _load(self):
        if not STATE_PATH.exists(): return {}
        try: return json.loads(STATE_PATH.read_text(encoding="utf-8") or "{}")
        except Exception: return {}

    def _save(self):
        try: STATE_PATH.write_text(json.dumps(self.state), encoding="utf-8")
        except Exception: pass

    def _purge(self):
        cutoff = _now() - timedelta(hours=TTL_HOURS)
        to_del = [s for s, iso in self.state.items() if datetime.fromisoformat(iso) < cutoff]
        for s in to_del: self.state.pop(s, None)

    def can_open(self, symbol: str) -> bool:
        self._purge()
        open_syms = set(self.state.keys())
        if symbol in open_syms: return False
        return len(open_syms) < self.max_positions

    def on_open(self, symbol: str) -> None:
        self.state[symbol] = _now().isoformat(); self._save()

    def on_close(self, symbol: str) -> None:
        if symbol in self.state:
            self.state.pop(symbol, None); self._save()
