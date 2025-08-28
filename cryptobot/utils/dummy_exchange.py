# dummy_exchange.py
from __future__ import annotations
from datetime import datetime, timezone
import numpy as np

class DummyExchange:
    def __init__(self):
        self.has = {
            "fetchOHLCV": True,
            "createMarketOrder": True,
            "fetchTicker": True,
            "fetchBalance": True,
        }
        self.timeframes = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400}
        self._last_prices = {}
        self.markets = {}
        self.symbols = []

    def set_sandbox_mode(self, v: bool):
        return None

    def load_markets(self, reload: bool = False):
        symbols = ["BTC/USDC", "ETH/USDC", "SOL/USDC", "XRP/USDC", "ADA/USDC"]
        self.symbols = symbols
        self.markets = {
            s: {"symbol": s, "precision": {"price": 2, "amount": 6}, "limits": {"cost": {"min": 5.0}}, "taker": 0.001}
            for s in symbols
        }
        return self.markets

    def _gen_ohlcv(self, last: float, n: int, tf_sec: int):
        out = []
        t = int(datetime.now(timezone.utc).timestamp()) // tf_sec * tf_sec
        price = last
        for i in range(n):
            price += np.random.normal(0, max(0.001 * price, 0.5))
            o = price + np.random.normal(0, 0.2)
            h = max(o, price) + np.random.random() * 0.3
            l = min(o, price) - np.random.random() * 0.3
            c = price
            v = abs(np.random.normal(50, 10))
            ts = (t - (n - i) * tf_sec) * 1000
            out.append([ts, float(o), float(h), float(l), float(c), float(v)])
        return out

    def fetch_ohlcv(self, symbol: str, timeframe: str = "15m", limit: int = 500):
        tf_sec = self.timeframes.get(timeframe, 900)
        last = self._last_prices.get(symbol, 100.0)
        data = self._gen_ohlcv(last, min(max(limit, 50), 1500), tf_sec)
        if data:
            self._last_prices[symbol] = float(data[-1][4])
        return data

    def fetch_ticker(self, symbol: str):
        price = float(self._last_prices.get(symbol, 100.0))
        return {"symbol": symbol, "last": price}

    def fetch_balance(self):
        return {"free": {"USDC": 1000.0}}

    def create_market_buy_order(self, symbol: str, amount: float):
        price = float(self._last_prices.get(symbol, 100.0))
        fee = price * amount * 0.001
        return {"symbol": symbol, "side": "buy", "amount": float(amount), "price": price, "fee": fee}

    def create_market_sell_order(self, symbol: str, amount: float):
        price = float(self._last_prices.get(symbol, 100.0))
        fee = price * amount * 0.001
        return {"symbol": symbol, "side": "sell", "amount": float(amount), "price": price, "fee": fee}
