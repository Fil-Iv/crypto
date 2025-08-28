# adaptive_strategy.py
from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from typing import Dict, Optional
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .utils.data_cache import OHLCVCache
from .utils.exchange_factory import get_exchange
from .logger import log

# ----------------------------- config ----------------------------------------

def _get_env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except Exception:
        return default

@dataclass
class AdaptConfig:
    base_threshold: float = _get_env_float("THRESHOLD", 0.55)
    threshold_min: float = _get_env_float("THRESHOLD_MIN", 0.52)
    threshold_max: float = _get_env_float("THRESHOLD_MAX", 0.70)

    tp_base: float = _get_env_float("TP_BASE", 0.02)   # 2%
    sl_base: float = _get_env_float("SL_BASE", 0.01)   # 1%
    tp_min: float = _get_env_float("TP_MIN", 0.005)
    tp_max: float = _get_env_float("TP_MAX", 0.06)
    sl_min: float = _get_env_float("SL_MIN", 0.003)
    sl_max: float = _get_env_float("SL_MAX", 0.04)

    vol_lookback: int = int(os.getenv("VOL_LOOKBACK", "96"))  # ~1 day on 15m
    baseline_vol: float = _get_env_float("BASELINE_VOL", 0.012)  # 1.2% daily (approx on 15m agg)
    trading_interval_minutes: int = int(os.getenv("TRADING_INTERVAL_MINUTES", "15"))

    # aggressiveness multiplier bounds (affects size externally)
    aggr_min: float = _get_env_float("AGGR_MIN", 0.6)
    aggr_max: float = _get_env_float("AGGR_MAX", 1.4)

# ----------------------------- helpers ---------------------------------------

def _to_tf(minutes: int) -> str:
    return f"{minutes//60}h" if minutes % 60 == 0 else f"{minutes}m"

def _clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))

def _realized_vol(df: pd.DataFrame, lookback: int) -> Optional[float]:
    if df is None or df.empty:
        return None
    # log-returns for better aggregation
    ret = np.log(df["close"]).diff()
    vol = float(ret.rolling(lookback).std().dropna().iloc[-1]) if len(ret) >= lookback else float(ret.std())
    if np.isnan(vol) or not np.isfinite(vol):
        return None
    return abs(vol)

# ----------------------------- core API --------------------------------------

def adaptive_parameters() -> Dict[str, float]:
    """Compute dynamic threshold/TP/SL/aggressiveness based on recent market volatility.
    Conservative when vol increases, more aggressive when vol decreases.
    """
    cfg = AdaptConfig()
    tf = _to_tf(cfg.trading_interval_minutes)

    # Use BTC/USDC and ETH/USDC as market proxies
    ex = get_exchange()
    cache = OHLCVCache()

    vols = []
    for sym in (f"BTC/USDC", f"ETH/USDC"):
        try:
            df = cache.fetch(ex, sym, tf, max(cfg.vol_lookback * 3, 300))
            v = _realized_vol(df, cfg.vol_lookback)
            if v is not None:
                vols.append(v)
        except Exception as e:
            log(f"[adaptive] vol fetch error {sym}: {e}")

    # Fallback if nothing fetched
    vol_now = float(np.median(vols)) if vols else cfg.baseline_vol

    # Normalize to baseline
    vol_norm = max(0.2, min(3.0, vol_now / max(1e-6, cfg.baseline_vol)))  # clamp 0.2..3

    # Aggressiveness inversely to volatility (high vol -> lower aggr)
    aggressiveness = _clamp(1.0 / (0.5 + 0.5 * vol_norm), cfg.aggr_min, cfg.aggr_max)

    # Threshold: increase slightly when aggressiveness is low (be picky in high vol)
    threshold = cfg.base_threshold + (1.0 - aggressiveness) * 0.06  # up to +0.06
    threshold = _clamp(threshold, cfg.threshold_min, cfg.threshold_max)

    # TP/SL scale with volatility (but keep sensible caps)
    tp = cfg.tp_base * (0.6 + 0.5 * vol_norm)   # scale up with vol
    sl = cfg.sl_base * (0.6 + 0.4 * vol_norm)   # scale up, a bit less than TP
    tp = _clamp(tp, cfg.tp_min, cfg.tp_max)
    sl = _clamp(sl, cfg.sl_min, cfg.sl_max)

    out = {
        "threshold": float(round(threshold, 4)),
        "tp": float(round(tp, 4)),
        "sl": float(round(sl, 4)),
        "aggressiveness": float(round(aggressiveness, 3)),
    }
    try:
        log(f"[adaptive] vol_now={vol_now:.5f}, norm={vol_norm:.3f} -> {out}")
    except Exception:
        pass
    return out
