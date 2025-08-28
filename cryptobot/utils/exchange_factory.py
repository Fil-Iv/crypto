# utils/exchange_factory.py
from __future__ import annotations

# --- bootstrap for script run (python auto_bot.py) ---
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]  # .../cryptobot
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os, threading, time
from typing import Any, Dict, Callable

# ccxt е опционален — ако го няма, падаме към DummyExchange
try:
    import ccxt  # type: ignore
except Exception:
    ccxt = None  # type: ignore

# абсолютни импорти (config_loader е в utils/)
from utils.config_loader import load_api_credentials  # << FIXED
from utils.rate_limiter import RateLimiter

# ------------------ Dummy fallback ------------------
class DummyExchange:
    def __init__(self):
        self.has = {"fetchOHLCV": True, "createMarketOrder": True, "fetchTicker": True, "fetchBalance": True}
        self.timeframes = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600}
        self._last_prices: Dict[str, float] = {}
        self.markets: Dict[str, dict] = {}
        self.symbols = []

    def set_sandbox_mode(self, v: bool):
        return None

    def load_markets(self, reload: bool = False):
        syms = ["BTC/USDC", "ETH/USDC", "SOL/USDC", "XRP/USDC", "ADA/USDC"]
        self.symbols = syms
        self.markets = {
            s: {
                "symbol": s,
                "precision": {"price": 2, "amount": 6},
                "limits": {"cost": {"min": 5.0}},
                "spot": True,
            }
            for s in syms
        }
        return self.markets

    def _gen(self, last: float, n: int, tf_sec: int):
        import numpy as np, time as _t
        out, t = [], int(_t.time()) // tf_sec * tf_sec
        price = last or 100.0
        for i in range(n):
            price += np.random.normal(0, max(0.001 * price, 0.5))
            o = price
            h = o * (1 + abs(np.random.normal(0, 0.001)))
            l = o * (1 - abs(np.random.normal(0, 0.001)))
            c = price
            v = abs(np.random.normal(50, 10))
            ts = (t - (n - i) * tf_sec) * 1000
            out.append([ts, float(o), float(h), float(l), float(c), float(v)])
        return out

    def fetch_ohlcv(self, symbol: str, timeframe: str = "15m", limit: int = 500):
        tf_sec = self.timeframes.get(timeframe, 900)
        last = self._last_prices.get(symbol, 100.0)
        data = self._gen(last, min(max(limit, 50), 1500), tf_sec)
        if data:
            self._last_prices[symbol] = float(data[-1][4])
        return data

    def fetch_ticker(self, symbol: str):
        price = float(self._last_prices.get(symbol, 100.0))
        return {"symbol": symbol, "last": price, "close": price}

    def fetch_balance(self):
        return {"free": {"USDC": 1000.0}}

    def create_market_buy_order(self, symbol: str, amount: float):
        price = float(self._last_prices.get(symbol, 100.0))
        return {"symbol": symbol, "side": "buy", "amount": float(amount), "price": price}

    def create_market_sell_order(self, symbol: str, amount: float):
        price = float(self._last_prices.get(symbol, 100.0))
        return {"symbol": symbol, "side": "sell", "amount": float(amount), "price": price}

# ------------------------- Retry + Safe wrapper -------------------------
_EXCHANGE = None
_LOCK = threading.Lock()
DRY_RUN = False  # публичен флаг

if ccxt:
    _CCXT_RETRY_ERRORS = (
        getattr(ccxt, "NetworkError", Exception),
        getattr(ccxt, "DDoSProtection", Exception),
        getattr(ccxt, "ExchangeNotAvailable", Exception),
        getattr(ccxt, "RequestTimeout", Exception),
        getattr(ccxt, "RateLimitExceeded", Exception),
    )
    _CCXT_FATAL_ORDER_ERRORS = (
        getattr(ccxt, "InvalidOrder", Exception),
        getattr(ccxt, "InsufficientFunds", Exception),
        getattr(ccxt, "OrderNotFound", Exception),
        getattr(ccxt, "NotSupported", Exception),
    )
else:
    _CCXT_RETRY_ERRORS = (Exception,)
    _CCXT_FATAL_ORDER_ERRORS = (Exception,)

def _retry(fn: Callable, *, tries: int = 5, backoff: float = 0.25, max_backoff: float = 8.0):
    def wrapped(*args, **kwargs):
        delay, last = backoff, None
        for _ in range(tries):
            try:
                return fn(*args, **kwargs)
            except _CCXT_RETRY_ERRORS as e:
                last = e
                time.sleep(delay)
                delay = min(max_backoff, delay * 2.0)
            except _CCXT_FATAL_ORDER_ERRORS:
                raise
        if last:
            raise last
        return fn(*args, **kwargs)
    return wrapped

class SafeExchange:
    """Proxy: rate limit + retries върху ccxt/Dummy."""
    def __init__(self, ex: Any, *, rate: int = 10, per: float = 1.0):
        self._ex = ex
        self._rl = RateLimiter(rate=rate, per=per)
        self.has = getattr(ex, "has", {})
        self.markets = getattr(ex, "markets", {})
        self.symbols = getattr(ex, "symbols", [])
        self.timeframes = getattr(ex, "timeframes", {})

    def load_markets(self, reload: bool = False):
        with self._rl.limit():
            res = _retry(self._ex.load_markets)(reload)
            self.markets = getattr(self._ex, "markets", {})
            self.symbols = getattr(self._ex, "symbols", [])
            self.timeframes = getattr(self._ex, "timeframes", {})
            return res

    def fetch_ohlcv(self, symbol: str, timeframe: str = "15m", limit: int = 500):
        with self._rl.limit():
            return _retry(self._ex.fetch_ohlcv)(symbol, timeframe=timeframe, limit=limit)

    def fetch_ticker(self, symbol: str):
        with self._rl.limit():
            return _retry(self._ex.fetch_ticker)(symbol)

    def fetch_tickers(self, symbols=None):
        if not hasattr(self._ex, "fetch_tickers"):
            raise AttributeError("Underlying exchange lacks fetch_tickers")
        with self._rl.limit():
            return _retry(self._ex.fetch_tickers)(symbols)

    def fetch_balance(self):
        with self._rl.limit():
            return _retry(self._ex.fetch_balance)()

    def create_market_buy_order(self, symbol: str, amount: float):
        with self._rl.limit():
            return _retry(self._ex.create_market_buy_order)(symbol, amount)

    def create_market_sell_order(self, symbol: str, amount: float):
        with self._rl.limit():
            return _retry(self._ex.create_market_sell_order)(symbol, amount)

    # helpers
    def market(self, symbol: str) -> Dict[str, Any]:
        if hasattr(self._ex, "market"):
            return self._ex.market(symbol)
        return self.markets.get(symbol, {})

    def price_precision(self, symbol: str) -> int:
        m = self.market(symbol) or {}
        p = (m.get("precision") or {}).get("price")
        return int(p) if isinstance(p, (int, float)) else 8

    def amount_precision(self, symbol: str) -> int:
        m = self.market(symbol) or {}
        p = (m.get("precision") or {}).get("amount")
        return int(p) if isinstance(p, (int, float)) else 8

    def min_notional(self, symbol: str) -> float:
        m = self.market(symbol) or {}
        limits = m.get("limits") or {}
        cost = (limits.get("cost") or {}).get("min")
        return float(cost) if isinstance(cost, (int, float)) else 5.0

    def __getattr__(self, item: str):
        return getattr(self._ex, item)

# ------------------------------ Factory --------------------------------
def get_exchange(force_reload: bool = False, *, sandbox: bool = False) -> SafeExchange:
    """Връща SafeExchange. Ако няма ключове/ccxt/мрежа → Dummy + DRY_RUN=True."""
    global _EXCHANGE, DRY_RUN
    with _LOCK:
        if _EXCHANGE is not None and not force_reload:
            return _EXCHANGE

        api_key, api_secret = "", ""
        try:
            creds = load_api_credentials()
            api_key = creds.get("apiKey") or creds.get("BINANCE_API_KEY") or ""
            api_secret = creds.get("secret") or creds.get("BINANCE_API_SECRET") or ""
        except Exception:
            pass

        DRY_RUN = bool(os.getenv("DRY_RUN") == "1" or not api_key or not api_secret or ccxt is None)

        if not DRY_RUN:
            try:
                ex = ccxt.binance({
                    "apiKey": api_key,
                    "secret": api_secret,
                    "enableRateLimit": True,
                    "options": {"defaultType": "spot"},
                })
                if sandbox and hasattr(ex, "set_sandbox_mode"):
                    ex.set_sandbox_mode(True)
            except Exception:
                ex = DummyExchange()
                DRY_RUN = True
        else:
            ex = DummyExchange()

        safe = SafeExchange(ex, rate=10, per=1.0)
        try:
            safe.load_markets()
        except Exception:
            ex = DummyExchange()
            safe = SafeExchange(ex, rate=10, per=1.0)
            DRY_RUN = True

        _EXCHANGE = safe
        print(f"[EXCHANGE] Ready. DRY_RUN={DRY_RUN}, sandbox={sandbox}")
        return _EXCHANGE
