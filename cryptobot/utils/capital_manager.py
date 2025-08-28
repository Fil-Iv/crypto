# capital_manager.py
from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Dict, Any, Optional
from datetime import datetime, timezone, date

import numpy as np
import pandas as pd

# Light, self-contained volatility read (no TA dep)
from .utils.data_cache import OHLCVCache
from .logger import log

STATE_PATH = Path(os.getenv("CAPITAL_STATE_PATH", "data/capital_state.json"))
STATE_PATH.parent.mkdir(parents=True, exist_ok=True)

def _load_state() -> Dict[str, Any]:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def _save_state(st: Dict[str, Any]) -> None:
    try:
        STATE_PATH.write_text(json.dumps(st, indent=2), encoding="utf-8")
    except Exception:
        pass

def _today_str() -> str:
    return date.today().isoformat()

@dataclass
class RiskConfig:
    quote_ccy: str = os.getenv("BASE_QUOTE", "USDC")
    trade_alloc_pct: float = float(os.getenv("TRADE_ALLOC_PCT", "0.10"))  # 10% default
    max_exposure_pct: float = float(os.getenv("MAX_EXPOSURE_PCT", "0.60"))  # max 60% total
    min_quote_alloc: float = float(os.getenv("MIN_QUOTE_ALLOC", "10.0"))
    kelly_cap: float = float(os.getenv("KELLY_CAP", "0.25"))  # cap Kelly at 25% of equity
    daily_stop_pct: float = float(os.getenv("DAILY_STOP_PCT", "0.05"))  # -5% stop-out
    vol_floor: float = float(os.getenv("VOL_FLOOR", "0.003"))  # 0.3%
    vol_lookback: int = int(os.getenv("VOL_LOOKBACK_CAPM", "48"))  # ~12h on 15m
    aggressiveness: float = float(os.getenv("AGGR_DEFAULT", "1.0"))

class CapitalManager:
    """Position sizing with volatility scaling, Kelly fraction cap, and daily stop-out guard.
    Public API:
      - size_quote(symbol, probability) -> allocation in quote currency (e.g., USDC)
      - get_equity(exchange) -> free quote balance
    Back-compat helpers (no-ops if unused):
      - get_available_capital(), load_capital_state(), update_after_trade()
    """
    def __init__(self):
        self.cfg = RiskConfig()
        self.cache = OHLCVCache()
        self.state = _load_state()

    # ------------------------------ Equity & Exposure ------------------------------
    def get_equity(self, exchange) -> float:
        """Free quote balance (e.g., USDC)."""
        try:
            bal = exchange.fetch_balance()
            free = bal.get("free", {}) or bal.get("total", {})
            return float(free.get(self.cfg.quote_ccy, 0.0))
        except Exception as e:
            log(f"[capital] balance error: {e}")
            return 0.0

    def _current_exposure_quote(self) -> float:
        """Total currently allocated (tracked only via our local state)."""
        st = self.state.get("open_alloc_quote", 0.0)
        try:
            return float(st)
        except Exception:
            return 0.0

    def _can_allocate(self, amount_quote: float, equity_quote: float) -> bool:
        total_after = self._current_exposure_quote() + amount_quote
        cap = equity_quote * self.cfg.max_exposure_pct
        return total_after <= cap

    # ------------------------------ Vol & Kelly ------------------------------------
    def _symbol_vol(self, exchange, symbol: str, timeframe: str = "15m") -> float:
        """Return recent close-to-close volatility (stdev of log-returns)."""
        try:
            df = self.cache.fetch(exchange, symbol, timeframe, max(self.cfg.vol_lookback * 3, 180))
            if df.empty or len(df) < 20:
                return self.cfg.vol_floor
            ret = np.log(df["close"]).diff()
            vol = float(ret.tail(self.cfg.vol_lookback).std())
            if not np.isfinite(vol) or vol <= 0:
                vol = self.cfg.vol_floor
            return max(self.cfg.vol_floor, vol)
        except Exception as e:
            log(f"[capital] vol error {symbol}: {e}")
            return self.cfg.vol_floor

    def _kelly_fraction(self, p: float, rr: float = 1.8) -> float:
        """Kelly fraction for win prob p and reward/risk ratio rr (tp/sl)."""
        p = max(0.0, min(1.0, float(p)))
        q = 1.0 - p
        # Kelly = (p - q/rr)
        k = p - (q / max(1e-6, rr))
        return max(0.0, min(self.cfg.kelly_cap, float(k)))

    # ------------------------------ Public API -------------------------------------
    def size_quote(self, exchange=None, symbol: str = "", probability: float = 0.55,
                   timeframe: str = "15m", aggressiveness: Optional[float] = None,
                   tp: float = 0.02, sl: float = 0.01) -> float:
        """Compute quote allocation for a potential trade."""
        aggr = float(aggressiveness if aggressiveness is not None else self.cfg.aggressiveness)
        if exchange is None:
            from .utils.exchange_factory import get_exchange
            exchange = get_exchange()

        equity = self.get_equity(exchange)
        if equity <= 0:
            return 0.0

        # Daily stop-out guard
        today = _today_str()
        day_pnl = float(self.state.get("daily_pnl", {}).get(today, 0.0))
        if day_pnl <= -abs(self.cfg.daily_stop_pct) * max(1.0, equity):
            log("[capital] daily stop-out reached; sizing=0")
            return 0.0

        # Base allocation
        base_alloc = equity * self.cfg.trade_alloc_pct * aggr

        # Volatility scaling (higher vol -> smaller size)
        vol = self._symbol_vol(exchange, symbol, timeframe=timeframe)
        # Map vol (e.g., 0.01) to scaler ~ between 0.4 and 1.2
        vol_scale = float(np.clip(0.02 / max(1e-6, vol), 0.4, 1.2))
        alloc_vol = base_alloc * vol_scale

        # Kelly cap adjustment by predicted probability and tp/sl (reward/risk)
        rr = max(1e-6, tp / max(1e-6, sl))
        kelly = self._kelly_fraction(probability, rr=rr)
        alloc_kelly = equity * kelly

        # Final allocation is min of volatility-adjusted and Kelly-capped
        alloc = min(alloc_vol, alloc_kelly if alloc_kelly > 0 else alloc_vol)

        # Exposure cap
        if not self._can_allocate(alloc, equity):
            log("[capital] exposure cap hit; sizing=0")
            return 0.0

        # Ensure minimum
        alloc = max(self.cfg.min_quote_alloc, float(alloc))
        return float(alloc)

    # ------------------------------ Back-compat helpers ----------------------------
    def get_available_capital(self) -> float:
        """Deprecated: kept for compatibility with older code."""
        return float(self.state.get("available_quote", 0.0))

    def load_capital_state(self) -> Dict[str, Any]:
        return dict(self.state)

    def update_after_trade(self, symbol: str, quote_used: float, pnl_quote: float = 0.0) -> None:
        """Update local accounting after a trade (call optionally)."""
        self.state.setdefault("open_alloc_quote", 0.0)
        self.state["open_alloc_quote"] = max(0.0, float(self.state["open_alloc_quote"]) + float(quote_used))
        # Track daily PnL
        today = _today_str()
        self.state.setdefault("daily_pnl", {})
        self.state["daily_pnl"][today] = float(self.state["daily_pnl"].get(today, 0.0)) + float(pnl_quote)
        _save_state(self.state)
