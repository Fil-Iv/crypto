# utils/backtester.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import numpy as np
import pandas as pd

from utils.add_indicators import add_indicators
from utils.data_cache import OHLCVCache
from utils.logger import log  # важен фикс: няма конфликт с външния пакет "logger"

@dataclass
class BTResult:
    symbol: str
    trades: int
    winrate: float
    pnl_pct: float
    sharpe: float

class Backtester:
    def __init__(self, exchange, model, params: Dict[str, float], timeframe: str = "15m"):
        self.exchange = exchange
        self.model = model
        self.params = params
        self.tf = timeframe
        self.cache = OHLCVCache()

    def run_symbol(self, symbol: str, limit: int = 1500) -> BTResult:
        df = self.cache.fetch(self.exchange, symbol, self.tf, limit)
        df = add_indicators(df)
        if df.empty or len(df) < 200:
            return BTResult(symbol, 0, 0.0, 0.0, 0.0)

        # вероятности walk-forward (без leakage)
        probs = []
        for i in range(len(df)):
            window = df.iloc[: i + 1]
            try:
                p = float(self.model.predict_proba(window))
            except Exception:
                p = np.nan
            probs.append(p)
        df["prob"] = probs
        df = df.dropna(subset=["prob"])

        thr = float(self.params.get("threshold", 0.55))
        tp = float(self.params.get("tp", 0.02))
        sl = float(self.params.get("sl", 0.01))

        df["entry"] = (df["prob"] >= thr).astype(int)

        pnl_list, wins, total = [], 0, 0
        for ts, row in df.iterrows():
            if row["entry"] != 1:
                pnl_list.append(0.0)
                continue
            total += 1
            entry_px = row["close"]
            hit = 0.0
            # прост 5-свещен хоризонт
            for _, r2 in df.loc[ts:].iloc[1:6].iterrows():
                chg = (r2["close"] - entry_px) / max(1e-9, entry_px)
                if chg >= tp:
                    hit = tp
                    wins += 1
                    break
                if chg <= -sl:
                    hit = -sl
                    break
                hit = chg
            pnl_list.append(hit)

        pnl = np.array(pnl_list, dtype=float)
        pnl_pct = float(np.nansum(pnl))
        winrate = float(wins / total) if total else 0.0
        sharpe = float(np.nanmean(pnl) / (np.nanstd(pnl) + 1e-9) * (252.0 ** 0.5)) if total else 0.0
        return BTResult(symbol, total, winrate, pnl_pct, sharpe)

    def run(self, symbols: List[str], limit: int = 1500) -> List[BTResult]:
        out = []
        for s in symbols:
            try:
                out.append(self.run_symbol(s, limit=limit))
            except Exception as e:
                log(f"[Backtest] {s} error: {e}")
        return out

