# model_utils.py
from __future__ import annotations

import os
import io
import json
import hashlib
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
from pathlib import Path

try:
    import joblib  # type: ignore
except Exception:
    joblib = None
import pickle

from .logger import log

MODELS_DIR = Path(os.getenv("MODELS_DIR", "models"))
MODELS_DIR.mkdir(parents=True, exist_ok=True)

@dataclass
class ModelMeta:
    name: str
    created_ts: float
    size_bytes: int
    sha256: str
    features: List[str]

def _model_path(name: str) -> Path:
    return MODELS_DIR / f"{name}.joblib"

def _tmp_path(name: str) -> Path:
    return MODELS_DIR / f".{name}.tmp"

def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def atomic_save(name: str, obj: Any, meta_extra: Optional[Dict[str, Any]] = None) -> Path:
    """Atomically save model object with checksum metadata. Returns final path."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(name)
    final = _model_path(name)

    # serialize
    if joblib is not None:
        buf = io.BytesIO()
        joblib.dump(obj, buf)
        data = buf.getvalue()
    else:
        data = pickle.dumps(obj)

    sha = _sha256_bytes(data)
    meta = {
        "name": name,
        "sha256": sha,
        "features": getattr(obj, "features", None) or getattr(obj, "_feature_cols", None) or [],
    }
    if meta_extra:
        meta.update(meta_extra)

    # write tmp
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, final)

    # sidecar metadata
    sidecar = final.with_suffix(final.suffix + ".meta.json")
    try:
        info = {
            "name": name,
            "sha256": sha,
            "size_bytes": final.stat().st_size,
            "created_ts": final.stat().st_mtime,
            "features": meta.get("features", []),
        }
        if meta_extra:
            info.update(meta_extra)
        sidecar.write_text(json.dumps(info, indent=2), encoding="utf-8")
    except Exception as e:
        log(f"[model_utils] meta write failed: {e}")

    return final

def load(name: str) -> Any:
    p = _model_path(name)
    if not p.exists():
        raise FileNotFoundError(p)
    if joblib is not None:
        return joblib.load(p)
    with open(p, "rb") as f:
        return pickle.load(f)

def exists(name: str) -> bool:
    return _model_path(name).exists()

def list_models(order_by_mtime_desc: bool = True) -> List[str]:
    files = list(MODELS_DIR.glob("*.joblib"))
    if not files:
        return []
    if order_by_mtime_desc:
        files = sorted(files, key=lambda x: -x.stat().st_mtime)
    return [p.stem for p in files]

def prune_models(keep: int = 5) -> List[str]:
    """Keep newest N models; delete older ones; returns deleted names."""
    files = list(MODELS_DIR.glob("*.joblib"))
    files = sorted(files, key=lambda x: -x.stat().st_mtime)
    to_delete = files[keep:]
    deleted: List[str] = []
    for p in to_delete:
        try:
            p.unlink(missing_ok=True)  # py3.8+: use try/except
        except TypeError:
            try:
                p.unlink()
            except Exception:
                pass
        try:
            (p.with_suffix(p.suffix + ".meta.json")).unlink(missing_ok=True)
        except Exception:
            pass
        deleted.append(p.stem)
    if deleted:
        log(f"[model_utils] pruned: {deleted}")
    return deleted

def ensure_baseline_artifact(obj: Any, name: str = "baseline") -> Path:
    """Ensure at least one model artifact exists by saving the given object."""
    p = _model_path(name)
    if p.exists():
        return p
    return atomic_save(name, obj, meta_extra={"created_by": "ensure_baseline_artifact"})
