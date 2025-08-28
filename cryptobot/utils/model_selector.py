# model_selector.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List
from pathlib import Path

from .logger import log

MODELS_DIR = Path(os.getenv("MODELS_DIR", "models"))
MODELS_DIR.mkdir(parents=True, exist_ok=True)

@dataclass
class SelectorConfig:
    force_model: str = os.getenv("FORCE_MODEL", "").strip()
    fallback: str = os.getenv("FALLBACK_MODEL", "baseline")
    allow_untrained_baseline: bool = True

def _list_candidates() -> List[str]:
    names: List[str] = []
    try:
        for p in MODELS_DIR.glob("*.joblib"):
            if p.is_file():
                names.append(p.stem)
    except Exception:
        pass
    # Ensure baseline is present as a logical candidate
    if "baseline" not in names:
        names.append("baseline")
    # sort by mtime (newest first) if files exist
    try:
        names = sorted(names, key=lambda n: (-(MODELS_DIR / f"{n}.joblib").stat().st_mtime if (MODELS_DIR / f"{n}.joblib").exists() else 0))
    except Exception:
        names = list(dict.fromkeys(names))  # keep order
    return names

def select_best_model() -> Dict[str, str]:
    """Return {'name': <model_name>} with simple, reliable logic:
       1) FORCE_MODEL env overrides
       2) newest *.joblib in MODELS_DIR
       3) fallback -> 'baseline'
    """
    cfg = SelectorConfig()
    if cfg.force_model:
        log(f"[selector] FORCE_MODEL={cfg.force_model}")
        return {"name": cfg.force_model}

    cands = _list_candidates()
    for name in cands:
        # Prefer a saved artifact if exists; else accept baseline (train will occur on demand)
        if (MODELS_DIR / f"{name}.joblib").exists():
            log(f"[selector] selected saved model: {name}")
            return {"name": name}

    log(f"[selector] fallback: {cfg.fallback}")
    return {"name": cfg.fallback}
