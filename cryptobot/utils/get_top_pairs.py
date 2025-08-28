from __future__ import annotations
from typing import List, Set
import sys, pathlib

# ВИНАГИ добавяме корена на проекта в sys.path и ползваме абсолютен import
ROOT = pathlib.Path(__file__).resolve().parents[1]  # .../cryptobot
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.logger import log
# <- вече работи при python auto_bot.py

def get_top_usdc_pairs(exchange, limit: int = 12, min_volume_usd: float = 1_000_000.0, blacklist: Set[str] | None = None) -> List[str]:
    blacklist = blacklist or set()
    symbols: List[str] = []
    try:
        exchange.load_markets()
        for s, m in (exchange.markets or {}).items():
            if not isinstance(s, str): continue
            if "/USDC" not in s: continue
            if m.get("spot") is False: continue
            if s in blacklist: continue
            symbols.append(s)
    except Exception as e:
        log(f"get_top_usdc_pairs load_markets error: {e}")

    try:
        if hasattr(exchange, "fetch_tickers") and callable(getattr(exchange, "fetch_tickers")):
            tickers = exchange.fetch_tickers(symbols or None)
            def volume_usd(sym: str) -> float:
                t = tickers.get(sym, {}) if isinstance(tickers, dict) else {}
                qv = t.get("quoteVolume") or t.get("baseVolume") or 0.0
                try: return float(qv)
                except Exception: return 0.0
            symbols = sorted(set(symbols), key=lambda x: volume_usd(x), reverse=True)
            symbols = [s for s in symbols if volume_usd(s) >= min_volume_usd]
    except Exception as e:
        log(f"get_top_usdc_pairs fetch_tickers error: {e}")

    if not symbols:
        symbols = ["BTC/USDC","ETH/USDC","SOL/USDC","XRP/USDC","ADA/USDC","AVAX/USDC"]
    return symbols[:limit]
