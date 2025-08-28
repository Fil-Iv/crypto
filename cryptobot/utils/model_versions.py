# model_versions.py
from __future__ import annotations

import os
import json
import time
from typing import Any, Dict, List, Optional
from pathlib import Path

from .logger import log

VERSIONS_PATH = Path(os.getenv("MODEL_VERSIONS_PATH", "models/versions.json"))
VERSIONS_PATH.parent.mkdir(parents=True, exist_ok=True)

def _load_versions() -> Dict[str, Any]:
    if VERSIONS_PATH.exists():
        try:
            return json.loads(VERSIONS_PATH.read_text(encoding="utf-8"))
        except Exception as e:
            log(f"[model_versions] failed to read: {e}")
            return {}
    return {}

def _save_versions(data: Dict[str, Any]) -> None:
    try:
        VERSIONS_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception as e:
        log(f"[model_versions] failed to save: {e}")

def register_version(name: str, metrics: Optional[Dict[str, Any]] = None) -> None:
    """Register a new version with timestamp and optional metrics."""
    vers = _load_versions()
    vers.setdefault("history", [])
    entry = {
        "name": name,
        "ts": time.time(),
        "metrics": metrics or {},
    }
    vers["history"].insert(0, entry)
    # Keep only last N
    vers["history"] = vers["history"][:20]
    _save_versions(vers)
    log(f"[model_versions] registered {name}")

def list_versions() -> List[Dict[str, Any]]:
    vers = _load_versions()
    return vers.get("history", [])

def get_latest() -> Optional[str]:
    vers = list_versions()
    return vers[0]["name"] if vers else None

def rollback_to(name: str) -> bool:
    """Mark given version as latest by reordering history."""
    vers = _load_versions()
    hist = vers.get("history", [])
    idx = next((i for i, v in enumerate(hist) if v.get("name") == name), None)
    if idx is None:
        return False
    # Move to front
    vers["history"].insert(0, vers["history"].pop(idx))
    _save_versions(vers)
    log(f"[model_versions] rolled back to {name}")
    return True
